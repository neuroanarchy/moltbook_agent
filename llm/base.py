"""Provider interface used by Schranz.

The agent depends on this interface rather than on Ollama/OpenAI-specific
classes, making model/provider replacement a configuration + adapter change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import re
from typing import Generic, Literal, Sequence, TypeVar, TypedDict, Type

from pydantic import BaseModel, ValidationError


Role = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    role: Role
    content: str


T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when an LLM provider cannot complete a request."""


class ModelProvider(ABC, Generic[T]):
    name: str

    @abstractmethod
    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float,
    ) -> str:
        raise NotImplementedError

    def chat_structured(
        self,
        messages: Sequence[ChatMessage],
        schema: Type[T],
        *,
        temperature: float,
    ) -> T:
        """Portable fallback: ask for JSON, then validate it with Pydantic."""
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        augmented = list(messages) + [
            {
                "role": "user",
                "content": (
                    "Return ONLY one valid JSON object matching this JSON Schema. "
                    "Do not use markdown fences and do not include commentary.\n\n"
                    f"JSON Schema:\n{schema_json}"
                ),
            }
        ]
        raw = self.chat(augmented, temperature=temperature)
        try:
            return schema.model_validate_json(_strip_json_fences(raw))
        except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise LLMError(f"Provider '{self.name}' returned invalid structured output") from exc

    @abstractmethod
    def healthcheck(self) -> str:
        raise NotImplementedError


def _strip_json_fences(value: str) -> str:
    text = value.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    return fenced.group(1).strip() if fenced else text
