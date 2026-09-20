"""Guarded Moltbook write-action policy and audit ledger.

The LLM never receives direct access to this layer. Schranz calls it only
from explicit Python-controlled paths. Writes are disabled by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any

from moltbook.api import MoltbookClient


class ActionPolicyError(RuntimeError):
    """Raised when a Moltbook write is not permitted by local policy."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _hash_payload(
    action: str,
    target_id: str | None,
    payload: dict[str, Any],
) -> str:
    canonical = json.dumps(
        {
            "action": action,
            "target_id": target_id,
            "payload": payload,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ActionPolicy:
    write_enabled: bool = False
    dry_run: bool = True
    require_approval: bool = True

    post_cooldown_seconds: int = 1800
    comment_hourly_limit: int = 50

    max_title_chars: int = 300
    max_content_chars: int = 10000
    max_submolt_chars: int = 100

    # CHANGED: Added "philosophy" because Schranz is explicitly
    # intended to be able to publish philosophy posts.
    allowed_submolts: tuple[str, ...] = (
        "cybersecurity",
        "technology",
        "programming",
        "ai",
        "general",
        "philosophy",
    )


class ActionLedger:
    """Small SQLite ledger used for duplicate prevention and local budgets."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = threading.RLock()

        self._init_db()

        try:
            import os

            if self.path.exists() and hasattr(os, "chmod"):
                os.chmod(
                    self.path,
                    0o600,
                )

        except OSError:
            pass

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=10,
        )

        connection.row_factory = sqlite3.Row

        return connection

    def _init_db(self) -> None:
        with self._connect() as db:

            db.execute(
                """
                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target_id TEXT,
                    payload_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_json TEXT
                )
                """
            )

            db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_actions_created
                ON actions(created_at)
                """
            )

            db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_actions_hash
                ON actions(payload_hash)
                """
            )

    def record(
        self,
        *,
        action: str,
        target_id: str | None,
        payload_hash: str,
        status: str,
        response: Any = None,
    ) -> None:

        response_json = (
            json.dumps(
                response,
                ensure_ascii=False,
                default=str,
            )[:50000]
            if response is not None
            else None
        )

        with self._lock, self._connect() as db:

            db.execute(
                """
                INSERT INTO actions(
                    created_at,
                    action,
                    target_id,
                    payload_hash,
                    status,
                    response_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    _iso(_now()),
                    action,
                    target_id,
                    payload_hash,
                    status,
                    response_json,
                ),
            )

    def seen_hash(
        self,
        payload_hash: str,
    ) -> bool:

        with self._lock, self._connect() as db:

            row = db.execute(
                """
                SELECT 1
                FROM actions
                WHERE payload_hash = ?
                AND status = 'executed'
                LIMIT 1
                """,
                (payload_hash,),
            ).fetchone()

            return row is not None

    def count_since(
        self,
        action: str,
        since: datetime,
    ) -> int:

        with self._lock, self._connect() as db:

            row = db.execute(
                """
                SELECT COUNT(*) AS n
                FROM actions
                WHERE action = ?
                AND status = 'executed'
                AND created_at >= ?
                """,
                (
                    action,
                    _iso(since),
                ),
            ).fetchone()

            return int(
                row["n"]
                if row
                else 0
            )

    def last_executed(
        self,
        action: str,
    ) -> datetime | None:

        with self._lock, self._connect() as db:

            row = db.execute(
                """
                SELECT created_at
                FROM actions
                WHERE action = ?
                AND status = 'executed'
                ORDER BY id DESC
                LIMIT 1
                """,
                (action,),
            ).fetchone()

            if not row:
                return None

            try:
                return datetime.fromisoformat(
                    row["created_at"]
                )

            except ValueError:
                return None


class MoltbookActionLayer:
    """Validate and execute Moltbook writes behind explicit policy gates."""

    def __init__(
        self,
        client: MoltbookClient,
        ledger: ActionLedger,
        policy: ActionPolicy,
    ) -> None:

        self.client = client
        self.ledger = ledger
        self.policy = policy

    def _check_common(
        self,
        action: str,
        target_id: str | None,
        payload: dict[str, Any],
        *,
        approved: bool,
    ) -> str:

        if not self.policy.write_enabled:
            raise ActionPolicyError(
                "Moltbook writes are disabled "
                "(MOLTBOOK_WRITE_ENABLED=false)"
            )

        if self.policy.dry_run:

            self.ledger.record(
                action=action,
                target_id=target_id,
                payload_hash=_hash_payload(
                    action,
                    target_id,
                    payload,
                ),
                status="dry_run",
                response=payload,
            )

            raise ActionPolicyError(
                "Dry-run mode: action was recorded "
                "but not published"
            )

        if (
            self.policy.require_approval
            and not approved
        ):

            raise ActionPolicyError(
                "Action requires explicit approval"
            )

        payload_hash = _hash_payload(
            action,
            target_id,
            payload,
        )

        if self.ledger.seen_hash(payload_hash):
            raise ActionPolicyError(
                "Duplicate action blocked "
                "by the local ledger"
            )

        return payload_hash

    def create_post(
        self,
        *,
        title: str,
        content: str,
        submolt: str,
        post_type: str = "text",
        approved: bool = False,
    ) -> dict[str, Any]:

        if not self.policy.write_enabled:
            raise ActionPolicyError(
                "Moltbook writes are disabled "
                "(MOLTBOOK_WRITE_ENABLED=false)"
            )

        title = title.strip()
        content = content.strip()
        submolt = submolt.strip()

        if not title or len(title) > self.policy.max_title_chars:
            raise ValueError(
                f"Post title must be 1-"
                f"{self.policy.max_title_chars} characters"
            )

        if (
            not content
            or len(content) > self.policy.max_content_chars
        ):
            raise ValueError(
                f"Post content must be 1-"
                f"{self.policy.max_content_chars} characters"
            )

        if (
            not submolt
            or len(submolt) > self.policy.max_submolt_chars
        ):
            raise ValueError(
                f"Submolt must be 1-"
                f"{self.policy.max_submolt_chars} characters"
            )

        normalized_submolt = (
            submolt.lower().lstrip("m/")
        )

        allowed = {
            item.lower().lstrip("m/")
            for item in self.policy.allowed_submolts
        }

        if normalized_submolt not in allowed:

            raise ActionPolicyError(
                f"Submolt '{submolt}' is not in "
                f"MOLTBOOK_ALLOWED_SUBMOLTS: "
                f"{', '.join(sorted(allowed))}"
            )

        submolt = normalized_submolt

        if post_type != "text":
            raise ValueError(
                "Schranz currently allows "
                "text posts only"
            )

        # CHANGED: Keep "type" in the ledger/API payload representation,
        # but do NOT unpack it directly into MoltbookClient.create_post().
        # MoltbookClient calls this argument "post_type".
        payload = {
            "title": title,
            "content": content,
            "submolt": submolt,
            "type": post_type,
        }

        payload_hash = self._check_common(
            "post",
            None,
            payload,
            approved=approved,
        )

        last = self.ledger.last_executed(
            "post"
        )

        if (
            last
            and _now()
            < last + timedelta(
                seconds=self.policy.post_cooldown_seconds
            )
        ):

            raise ActionPolicyError(
                "Local post cooldown is active"
            )

        try:

            # CHANGED: Explicitly translate the action-layer
            # name "post_type" to the API client's parameter name.
            response = self.client.create_post(
                title=title,
                content=content,
                submolt=submolt,
                post_type=post_type,
            )

        except Exception as exc:

            self.ledger.record(
                action="post",
                target_id=None,
                payload_hash=payload_hash,
                status="failed",
                response={
                    "error": str(exc)
                },
            )

            raise

        self.ledger.record(
            action="post",
            target_id=(
                str(response.get("id"))
                if (
                    isinstance(response, dict)
                    and response.get("id")
                )
                else None
            ),
            payload_hash=payload_hash,
            status="executed",
            response=response,
        )

        return response

    def create_comment(
        self,
        *,
        post_id: str,
        content: str,
        parent_id: str | None = None,
        approved: bool = False,
    ) -> dict[str, Any]:

        post_id = post_id.strip()
        content = content.strip()

        if not post_id:
            raise ValueError(
                "post_id must not be empty"
            )

        if (
            not content
            or len(content) > self.policy.max_content_chars
        ):
            raise ValueError(
                f"Comment content must be 1-"
                f"{self.policy.max_content_chars} characters"
            )

        if parent_id is not None:
            parent_id = (
                parent_id.strip()
                or None
            )

        payload = {
            "content": content
        }

        if parent_id:
            payload["parent_id"] = parent_id

        payload_hash = self._check_common(
            "comment",
            post_id,
            payload,
            approved=approved,
        )

        since = _now() - timedelta(
            hours=1
        )

        if (
            self.ledger.count_since(
                "comment",
                since,
            )
            >= self.policy.comment_hourly_limit
        ):

            raise ActionPolicyError(
                "Local hourly comment budget is exhausted"
            )

        try:

            response = self.client.create_comment(
                post_id,
                content=content,
                parent_id=parent_id,
            )

        except Exception as exc:

            self.ledger.record(
                action="comment",
                target_id=post_id,
                payload_hash=payload_hash,
                status="failed",
                response={
                    "error": str(exc)
                },
            )

            raise

        self.ledger.record(
            action="comment",
            target_id=(
                str(response.get("id"))
                if (
                    isinstance(response, dict)
                    and response.get("id")
                )
                else post_id
            ),
            payload_hash=payload_hash,
            status="executed",
            response=response,
        )

        return response