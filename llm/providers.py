"""Built-in LLM provider adapters."""

from __future__ import annotations

import json
from typing import Sequence, Type, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from config import validate_endpoint_url
from llm.base import ChatMessage, LLMError, ModelProvider, _strip_json_fences
T = TypeVar("T", bound=BaseModel)


class OllamaProvider(ModelProvider[T]):
    name = "ollama"

    def __init__(self, *, model: str, base_url: str, timeout_seconds: float) -> None:
        try:
            import ollama
        except ImportError as exc:
            raise LLMError(
                "The ollama package is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self.model = model
        base_url = validate_endpoint_url(
            base_url,
            "Ollama base URL",
            require_https=True,
            allow_local_http=True,
        )
        try:
            self.client = ollama.Client(host=base_url, timeout=timeout_seconds)
        except TypeError:
            # Compatibility with older clients that do not expose timeout in Client().
            self.client = ollama.Client(host=base_url)
            self.timeout_seconds = timeout_seconds
        else:
            self.timeout_seconds = timeout_seconds

    def chat(self, messages: Sequence[ChatMessage], *, temperature: float) -> str:
        try:
            response = self.client.chat(
                model=self.model,
                messages=list(messages),
                options={"temperature": temperature},
                stream=False,
            )
            content = response.message.content
        except Exception as exc:
            raise LLMError(f"Ollama request failed for model '{self.model}'") from exc

        if not isinstance(content, str) or not content.strip():
            raise LLMError("Ollama returned an empty response")
        return content.strip()

    def chat_structured(
        self,
        messages: Sequence[ChatMessage],
        schema: Type[T],
        *,
        temperature: float,
    ) -> T:
        try:
            response = self.client.chat(
                model=self.model,
                messages=list(messages),
                format=schema.model_json_schema(),
                options={"temperature": temperature},
                stream=False,
            )
            raw = response.message.content
            return schema.model_validate_json(_strip_json_fences(raw))
        except ValidationError as exc:
            raise LLMError(f"Ollama returned schema-invalid JSON for '{self.model}'") from exc
        except Exception as exc:
            raise LLMError(f"Ollama structured request failed for model '{self.model}'") from exc

    def healthcheck(self) -> str:
        try:
            result = self.client.show(self.model)
            model_name = getattr(result, "model", None) or self.model
            return f"Ollama OK: {model_name}"
        except Exception as exc:
            raise LLMError(
                f"Ollama is unavailable or model '{self.model}' is not installed"
            ) from exc


class OpenAICompatibleProvider(ModelProvider[T]):
    """Adapter for servers implementing /chat/completions-compatible APIs."""

    name = "openai_compatible"

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
        use_json_schema: bool = False,
    ) -> None:
        self.model = model
        self.base_url = validate_endpoint_url(
            base_url,
            "OpenAI-compatible base URL",
            require_https=True,
            allow_local_http=True,
        )
        self.timeout_seconds = timeout_seconds
        self.use_json_schema = use_json_schema
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "Schranz/1.0",
            }
        )
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def _request(self, body: dict[str, object]) -> dict[str, object]:
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=body,
                timeout=self.timeout_seconds,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise LLMError("OpenAI-compatible provider request failed") from exc

        if 300 <= response.status_code < 400:
            raise LLMError("OpenAI-compatible provider returned an unexpected redirect")
        if response.status_code >= 400:
            raise LLMError(
                f"OpenAI-compatible provider returned HTTP {response.status_code}"
            )
        try:
            payload = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise LLMError("OpenAI-compatible provider returned invalid JSON") from exc

        if not isinstance(payload, dict):
            raise LLMError("OpenAI-compatible provider returned an unexpected response")
        return payload

    @staticmethod
    def _extract_content(payload: dict[str, object]) -> str:
        try:
            choices = payload["choices"]
            first = choices[0]  # type: ignore[index]
            message = first["message"]  # type: ignore[index]
            content = message["content"]  # type: ignore[index]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("OpenAI-compatible provider response lacked message content") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("OpenAI-compatible provider returned empty content")
        return content.strip()

    def chat(self, messages: Sequence[ChatMessage], *, temperature: float) -> str:
        payload = self._request(
            {
                "model": self.model,
                "messages": list(messages),
                "temperature": temperature,
                "stream": False,
            }
        )
        return self._extract_content(payload)

    def chat_structured(
        self,
        messages: Sequence[ChatMessage],
        schema: Type[T],
        *,
        temperature: float,
    ) -> T:
        schema_json = schema.model_json_schema()
        augmented_messages = list(messages) + [
            {
                "role": "user",
                "content": (
                    "Return ONLY valid JSON matching the schema below. "
                    "No markdown, no prose.\n\n"
                    f"Schema:\n{json.dumps(schema_json, ensure_ascii=False)}"
                ),
            }
        ]
        response_format: dict[str, object] = {"type": "json_object"}
        if self.use_json_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": schema_json,
                },
            }

        payload = self._request(
            {
                "model": self.model,
                "messages": augmented_messages,
                "temperature": temperature,
                "stream": False,
                "response_format": response_format,
            }
        )
        raw = self._extract_content(payload)
        try:
            return schema.model_validate_json(_strip_json_fences(raw))
        except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise LLMError(
                f"OpenAI-compatible provider returned schema-invalid JSON for '{self.model}'"
            ) from exc

    def healthcheck(self) -> str:
        try:
            response = self.session.get(
                f"{self.base_url}/models",
                timeout=self.timeout_seconds,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise LLMError("OpenAI-compatible provider health check failed") from exc
        if response.status_code >= 400:
            raise LLMError(
                f"OpenAI-compatible provider health check returned HTTP {response.status_code}"
            )
        return f"OpenAI-compatible provider reachable: {self.model}"

    def close(self) -> None:
        self.session.close()


