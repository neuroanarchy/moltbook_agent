"""Provider factory."""

from __future__ import annotations

from config import Settings
from llm.base import LLMError, ModelProvider
from llm.providers import OllamaProvider, OpenAICompatibleProvider


def create_provider(settings: Settings, *, model: str | None = None) -> ModelProvider:
    selected_model = model or settings.llm_model
    if settings.llm_provider == "ollama":
        return OllamaProvider(
            model=selected_model,
            base_url=settings.ollama_base_url,
            timeout_seconds=settings.ollama_timeout_seconds,
        )

    if settings.llm_provider == "openai_compatible":
        return OpenAICompatibleProvider(
            model=selected_model,
            base_url=settings.openai_compatible_base_url,
            api_key=settings.openai_compatible_api_key,
            timeout_seconds=settings.openai_compatible_timeout_seconds,
            use_json_schema=settings.openai_compatible_json_schema,
        )

    raise LLMError(f"Unsupported LLM provider: {settings.llm_provider}")
