import json
from unittest.mock import Mock

from llm.providers import OpenAICompatibleProvider
from agents.agent import PostSelection


def test_openai_compatible_structured_output_is_pydantic_validated() -> None:
    provider = OpenAICompatibleProvider(
        model="test-model",
        base_url="https://example.com/v1",
        api_key="secret",
        timeout_seconds=5,
    )
    provider.session.post = Mock()
    provider.session.post.return_value.status_code = 200
    provider.session.post.return_value.json.return_value = {
        "choices": [
            {"message": {"content": json.dumps({"post_number": 2})}}
        ]
    }

    result = provider.chat_structured(
        [{"role": "system", "content": "test"}],
        PostSelection,
        temperature=0,
    )
    assert result.post_number == 2
    body = provider.session.post.call_args.kwargs["json"]
    assert body["response_format"]["type"] == "json_object"
