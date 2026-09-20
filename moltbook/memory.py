"""Durable local memory store for Schranz.

SQLite is used instead of a single unbounded JSON document.  The store is
transactional, indexed, bounded at query time, and automatically migrates the
old project's memory.json format on first use.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import os
import threading
from typing import Any, Mapping
import uuid


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+.#:-]{2,}", re.IGNORECASE)


class MemoryError(RuntimeError):
    """Base exception for memory operations."""


class MemoryStore:
    def __init__(
        self,
        path: str | Path,
        *,
        legacy_json_path: str | Path | None = None,
        max_search_candidates: int = 200,
    ) -> None:
        self.path = Path(path)
        self.legacy_json_path = Path(legacy_json_path) if legacy_json_path else None
        self.max_search_candidates = max_search_candidates
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()
        if os.name == "posix":
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
        self._migrate_legacy_json_once()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    type TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    author TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL,
                    source_platform TEXT NOT NULL DEFAULT '',
                    source_id TEXT NOT NULL DEFAULT '',
                    confidence REAL,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_memories_created_at
                    ON memories(created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_memories_type
                    ON memories(type);

                CREATE INDEX IF NOT EXISTS idx_memories_source
                    ON memories(source_platform, source_id);
                """
            )

    def _migrate_legacy_json_once(self) -> None:
        if not self.legacy_json_path or not self.legacy_json_path.exists():
            return
        with self._lock:
            with self._connect() as connection:
                count = connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                if count:
                    return

            try:
                raw = json.loads(self.legacy_json_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise MemoryError("Could not migrate legacy memory.json") from exc

            if not isinstance(raw, list):
                raise MemoryError("Legacy memory.json must contain a JSON list")

            for item in raw:
                if isinstance(item, Mapping):
                    self.add_memory(
                        {
                            "id": item.get("id"),
                            "created_at": item.get("created_at"),
                            "type": item.get("type", "investigation"),
                            "title": item.get("title", ""),
                            "author": item.get("author", ""),
                            "content": item.get("investigation") or item.get("content") or "",
                            "source": item.get("source") or {},
                            "metadata": {"migrated_from": str(self.legacy_json_path)},
                        }
                    )

    @staticmethod
    def _normalize_memory(memory: Mapping[str, Any]) -> dict[str, Any]:
        content = str(memory.get("content") or memory.get("investigation") or "").strip()
        if not content:
            raise MemoryError("Memory content must not be empty")

        source = memory.get("source") or {}
        if not isinstance(source, Mapping):
            source = {}

        metadata = memory.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {}

        tags = memory.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        if not isinstance(tags, list):
            tags = []

        confidence = memory.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
            except (TypeError, ValueError) as exc:
                raise MemoryError("Memory confidence must be numeric") from exc
            confidence = max(0.0, min(1.0, confidence))

        created_at = memory.get("created_at")
        if not created_at:
            created_at = datetime.now(timezone.utc).isoformat()

        memory_id = str(memory.get("id") or uuid.uuid4())
        title = str(memory.get("title") or "").strip()
        author = str(memory.get("author") or "").strip()
        source_platform = str(source.get("platform") or "").strip()
        source_id = str(source.get("id") or source.get("post_id") or source.get("comment_id") or "").strip()

        return {
            "id": memory_id,
            "created_at": str(created_at),
            "type": str(memory.get("type") or "note"),
            "title": title,
            "author": author,
            "content": content,
            "source_platform": source_platform,
            "source_id": source_id,
            "confidence": confidence,
            "tags": [str(tag) for tag in tags[:32]],
            "metadata": dict(metadata),
        }

    def add_memory(self, memory: Mapping[str, Any]) -> dict[str, Any]:
        normalized = self._normalize_memory(memory)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memories (
                    id, created_at, type, title, author, content,
                    source_platform, source_id, confidence, tags_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    created_at=excluded.created_at,
                    type=excluded.type,
                    title=excluded.title,
                    author=excluded.author,
                    content=excluded.content,
                    source_platform=excluded.source_platform,
                    source_id=excluded.source_id,
                    confidence=excluded.confidence,
                    tags_json=excluded.tags_json,
                    metadata_json=excluded.metadata_json
                """,
                (
                    normalized["id"],
                    normalized["created_at"],
                    normalized["type"],
                    normalized["title"],
                    normalized["author"],
                    normalized["content"],
                    normalized["source_platform"],
                    normalized["source_id"],
                    normalized["confidence"],
                    json.dumps(normalized["tags"], ensure_ascii=False),
                    json.dumps(normalized["metadata"], ensure_ascii=False),
                ),
            )
        return self._row_to_dict(normalized)

    @staticmethod
    def _row_to_dict(row: Mapping[str, Any] | sqlite3.Row) -> dict[str, Any]:
        get = row.__getitem__
        keys = set(row.keys()) if isinstance(row, sqlite3.Row) else set(row.keys())
        if "tags_json" in keys:
            try:
                tags = json.loads(get("tags_json"))
            except (TypeError, ValueError, json.JSONDecodeError):
                tags = []
        else:
            tags = list(get("tags") or [])

        if "metadata_json" in keys:
            try:
                metadata = json.loads(get("metadata_json"))
            except (TypeError, ValueError, json.JSONDecodeError):
                metadata = {}
        else:
            metadata = dict(get("metadata") or {})

        if "source_platform" in keys:
            source = {
                "platform": get("source_platform"),
                "id": get("source_id"),
            }
        else:
            source = dict(get("source") or {})

        return {
            "id": get("id"),
            "created_at": get("created_at"),
            "type": get("type"),
            "title": get("title"),
            "author": get("author"),
            "content": get("content"),
            "source": source,
            "confidence": get("confidence"),
            "tags": tags,
            "metadata": metadata,
        }

    def get_recent_memories(self, limit: int = 6) -> list[dict[str, Any]]:
        limit = max(1, min(100, int(limit)))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM memories ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def search_memories(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        terms = []
        for token in _TOKEN_RE.findall(query.lower()):
            if token not in terms:
                terms.append(token)
            if len(terms) >= 12:
                break

        if not terms:
            return self.get_recent_memories(limit)

        clauses: list[str] = []
        values: list[str] = []
        for term in terms:
            pattern = f"%{term}%"
            clauses.append(
                "(lower(title) LIKE ? OR lower(author) LIKE ? OR lower(content) LIKE ? OR lower(tags_json) LIKE ?)"
            )
            values.extend([pattern, pattern, pattern, pattern])

        sql = (
            "SELECT * FROM memories WHERE "
            + " OR ".join(clauses)
            + " ORDER BY created_at DESC LIMIT ?"
        )
        values.append(self.max_search_candidates)

        with self._lock, self._connect() as connection:
            rows = connection.execute(sql, values).fetchall()

        candidates = [self._row_to_dict(row) for row in rows]
        q_lower = query.lower()

        def score(memory: dict[str, Any]) -> tuple[float, str]:
            text = (
                f"{memory['title']} {memory['author']} {memory['content']} "
                f"{' '.join(memory.get('tags', []))}"
            ).lower()
            token_hits = sum(text.count(term) for term in terms)
            phrase_bonus = 3.0 if q_lower and q_lower in text else 0.0
            recency_bonus = 0.25 if memory.get("created_at") else 0.0
            return token_hits + phrase_bonus + recency_bonus, str(memory.get("created_at", ""))

        candidates.sort(key=score, reverse=True)
        return candidates[: max(1, min(100, int(limit)))]

    def has_source_memory(self, source_id: str, *, memory_type: str | None = None) -> bool:
        source_id = str(source_id).strip()
        if not source_id:
            return False
        query = "SELECT 1 FROM memories WHERE source_id = ?"
        values: list[Any] = [source_id]
        if memory_type:
            query += " AND type = ?"
            values.append(memory_type)
        query += " LIMIT 1"
        with self._lock, self._connect() as connection:
            return connection.execute(query, values).fetchone() is not None

    def count(self) -> int:
        with self._lock, self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    def close(self) -> None:
        # Connections are intentionally short-lived; nothing persistent to close.
        return None


# Backwards-compatible function-style API.
_DEFAULT_STORE: MemoryStore | None = None
_DEFAULT_LOCK = threading.Lock()


def _default_store() -> MemoryStore:
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_STORE is None:
                from config import load_settings

                settings = load_settings(require_moltbook_key=False)
                _DEFAULT_STORE = MemoryStore(
                    settings.memory_db_path,
                    legacy_json_path=settings.legacy_memory_json_path,
                    max_search_candidates=settings.max_memory_search_candidates,
                )
    return _DEFAULT_STORE


def load_memory() -> list[dict[str, Any]]:
    store = _default_store()
    with store._lock, store._connect() as connection:
        rows = connection.execute("SELECT * FROM memories ORDER BY created_at ASC").fetchall()
    return [store._row_to_dict(row) for row in rows]


def save_memory(memory: list[Mapping[str, Any]]) -> None:
    """Compatibility helper; replaces the database contents transactionally."""
    store = _default_store()
    with store._lock, store._connect() as connection:
        connection.execute("DELETE FROM memories")
        for item in memory:
            normalized = store._normalize_memory(item)
            connection.execute(
                """
                INSERT INTO memories (
                    id, created_at, type, title, author, content,
                    source_platform, source_id, confidence, tags_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized["id"], normalized["created_at"], normalized["type"],
                    normalized["title"], normalized["author"], normalized["content"],
                    normalized["source_platform"], normalized["source_id"],
                    normalized["confidence"], json.dumps(normalized["tags"], ensure_ascii=False),
                    json.dumps(normalized["metadata"], ensure_ascii=False),
                ),
            )


def add_memory(memory: Mapping[str, Any]) -> dict[str, Any]:
    return _default_store().add_memory(memory)


def get_recent_memories(limit: int = 6) -> list[dict[str, Any]]:
    return _default_store().get_recent_memories(limit)


def search_memories(query: str, limit: int = 8) -> list[dict[str, Any]]:
    return _default_store().search_memories(query, limit)
