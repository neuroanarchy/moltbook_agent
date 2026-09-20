from pathlib import Path

from agents.agent import SchranzAgent, InvestigationResult, CommentAnalysisResult, PostDraft
from config import Settings
from llm.base import ModelProvider
from moltbook.memory import MemoryStore
from moltbook.schemas import Author, Comment, Post


class FakeProvider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.structured_calls = 0

    def chat(self, messages, *, temperature):
        return "hello"

    def chat_structured(self, messages, schema, *, temperature):
        self.structured_calls += 1
        if schema.__name__ == "PostSelection":
            return schema(post_number=1)
        if schema.__name__ == "InvestigationResult":
            return InvestigationResult(
                security_issue="Example issue",
                why_it_matters="Example impact",
                technical_concepts=["authentication"],
                claims=[],
                uncertainties=[],
            )
        if schema.__name__ == "CommentSelection":
            return schema(comment_numbers=[1])
        if schema.__name__ == "CommentAnalysisResult":
            return CommentAnalysisResult(
                comments=[],
                common_themes=["theme"],
                new_insights=["insight"],
                questions=[],
            )
        raise AssertionError(schema)

    def healthcheck(self):
        return "fake ok"


class FakeMoltbook:
    def __init__(self):
        self.post = Post(id="p1", title="Auth issue", content="text", author=Author(name="alice"))
        self.comment = Comment(id="c1", content="claim", author=Author(name="bob"), post_id="p1")

    def get_status(self):
        return {"status": "claimed"}

    def get_home(self):
        return {"ok": True}

    def get_feed(self, *, sort, limit, max_pages):
        return [self.post]

    def get_comments(self, post_id, *, sort, limit):
        return [self.comment]

    def close(self):
        pass


def test_one_full_cycle_uses_memory_and_structured_provider(tmp_path: Path) -> None:
    settings = Settings(
        moltbook_api_key="test",
        memory_db_path=tmp_path / "memory.db",
        legacy_memory_json_path=tmp_path / "missing.json",
        relevant_memory_limit=2,
        max_memory_context_chars=4000,
    )
    provider = FakeProvider()
    api = FakeMoltbook()
    memory = MemoryStore(settings.memory_db_path)
    agent = SchranzAgent(settings, provider=provider, moltbook=api, memory=memory)

    result = agent.run_once()
    assert result["selected_post"].id == "p1"
    assert isinstance(result["investigation"], InvestigationResult)
    assert isinstance(result["comment_analysis"], CommentAnalysisResult)
    assert memory.count() == 2
    assert provider.structured_calls == 3


def test_post_draft_isolated_from_moltbook_feed(tmp_path: Path) -> None:
    settings = Settings(
        moltbook_api_key="test",
        memory_db_path=tmp_path / "memory.db",
        legacy_memory_json_path=tmp_path / "missing.json",
        moltbook_allowed_submolts=("general",),
        llm_fast_model="qwen3:4b",
    )

    class FastPostProvider(FakeProvider):
        def chat_structured(self, messages, schema, *, temperature):
            user_text = "\n".join(m["content"] for m in messages)
            assert "<UNTRUSTED_DATA" not in user_text
            assert "posthuman cybernetic philosophy" in user_text
            return PostDraft(
                should_post=True,
                reason="Useful original topic.",
                submolt="general",
                title="The Cybernetic Self",
                content="What happens to the idea of a human when feedback loops, machines, and cognition become one system?",
            )

    provider = FastPostProvider()
    agent = SchranzAgent(settings, provider=provider, moltbook=FakeMoltbook(), memory=MemoryStore(settings.memory_db_path))
    draft = agent.generate_post_draft("posthuman cybernetic philosophy")
    assert draft.submolt == "general"
    assert draft.title == "The Cybernetic Self"
