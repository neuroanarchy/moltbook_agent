import pytest

from config import ConfigurationError, load_settings, validate_endpoint_url


def test_remote_http_moltbook_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOLTBOOK_API_KEY", "test-key")
    monkeypatch.setenv("MOLTBOOK_BASE_URL", "http://example.com/api/v1")
    with pytest.raises(ConfigurationError):
        load_settings()


def test_local_http_ollama_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOLTBOOK_API_KEY", "test-key")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    settings = load_settings()
    assert settings.ollama_base_url == "http://localhost:11434"


def test_remote_http_llm_endpoint_is_rejected():
    with pytest.raises(ConfigurationError):
        validate_endpoint_url("http://example.com:11434", "LLM", require_https=True, allow_local_http=True)


def test_local_http_endpoint_is_allowed():
    assert validate_endpoint_url("http://127.0.0.1:11434", "LLM", require_https=True, allow_local_http=True) == "http://127.0.0.1:11434"
