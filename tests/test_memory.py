from pathlib import Path

from moltbook.memory import MemoryStore


def test_memory_store_persists_and_searches(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    store.add_memory(
        {
            "id": "one",
            "type": "investigation",
            "title": "JWT audience validation",
            "author": "alice",
            "content": "Validate issuer and audience before accepting a token.",
            "source": {"platform": "moltbook", "id": "post-1"},
            "tags": ["jwt", "authentication"],
        }
    )

    assert store.count() == 1
    assert store.has_source_memory("post-1", memory_type="investigation")
    results = store.search_memories("JWT audience", limit=1)
    assert len(results) == 1
    assert results[0]["id"] == "one"


def test_legacy_json_migration(tmp_path: Path) -> None:
    legacy = tmp_path / "memory.json"
    legacy.write_text(
        '[{"type":"investigation","title":"Old","author":"bob","investigation":"Old finding"}]',
        encoding="utf-8",
    )
    store = MemoryStore(tmp_path / "data" / "memory.db", legacy_json_path=legacy)
    memories = store.get_recent_memories(10)
    assert len(memories) == 1
    assert memories[0]["title"] == "Old"
    assert memories[0]["content"] == "Old finding"
