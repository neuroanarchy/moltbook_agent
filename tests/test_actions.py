from pathlib import Path

import pytest

from moltbook.actions import ActionLedger, ActionPolicy, ActionPolicyError, MoltbookActionLayer


class FakeClient:
    def __init__(self):
        self.posts = []
        self.comments = []

    def create_post(self, **payload):
        self.posts.append(payload)
        return {"id": "post-1", **payload}

    def create_comment(self, post_id, **payload):
        self.comments.append((post_id, payload))
        return {"id": "comment-1", "post_id": post_id, **payload}


def test_writes_fail_closed_by_default(tmp_path: Path):
    layer = MoltbookActionLayer(FakeClient(), ActionLedger(tmp_path / "actions.db"), ActionPolicy())
    with pytest.raises(ActionPolicyError, match="disabled"):
        layer.create_post(title="x", content="y", submolt="m")


def test_dry_run_does_not_publish(tmp_path: Path):
    client = FakeClient()
    layer = MoltbookActionLayer(
        client,
        ActionLedger(tmp_path / "actions.db"),
        ActionPolicy(write_enabled=True, dry_run=True),
    )
    with pytest.raises(ActionPolicyError, match="Dry-run"):
        layer.create_comment(post_id="p", content="hello", approved=True)
    assert client.comments == []


def test_explicit_write_and_duplicate_block(tmp_path: Path):
    client = FakeClient()
    layer = MoltbookActionLayer(
        client,
        ActionLedger(tmp_path / "actions.db"),
        ActionPolicy(write_enabled=True, dry_run=False, require_approval=True, post_cooldown_seconds=0, allowed_submolts=("m",)),
    )
    layer.create_post(title="x", content="y", submolt="m", approved=True)
    assert len(client.posts) == 1
    with pytest.raises(ActionPolicyError, match="Duplicate"):
        layer.create_post(title="x", content="y", submolt="m", approved=True)


def test_disallowed_submolt_is_blocked(tmp_path: Path):
    client = FakeClient()
    layer = MoltbookActionLayer(
        client,
        ActionLedger(tmp_path / "actions.db"),
        ActionPolicy(write_enabled=True, dry_run=False, require_approval=True, post_cooldown_seconds=0, allowed_submolts=("cybersecurity",)),
    )
    with pytest.raises(ActionPolicyError, match="not in MOLTBOOK_ALLOWED_SUBMOLTS"):
        layer.create_post(title="x", content="y", submolt="general", approved=True)
