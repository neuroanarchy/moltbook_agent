"""Configuration loading and validation for Schranz.

All secrets come from environment variables/.env.  This module deliberately
keeps provider-specific settings separate so changing LLM providers is a
configuration change rather than an agent rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent


class ConfigurationError(ValueError):
    """Raised when required or invalid configuration is detected."""


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}")
    return value


def _float_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc
    if not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}")
    return value


def _bool_env(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    normalized = raw.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be true/false")


def validate_endpoint_url(
    raw_url: str,
    name: str,
    *,
    require_https: bool = True,
    allow_local_http: bool = False,
) -> str:
    try:
        parsed = urlparse(raw_url)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a valid URL") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigurationError(f"{name} must be an absolute http(s) URL")
    if parsed.username or parsed.password:
        raise ConfigurationError(f"{name} must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ConfigurationError(f"{name} must not contain a query string or fragment")
    hostname = (parsed.hostname or "").lower()
    is_local = hostname in {"localhost", "127.0.0.1", "::1"}
    if require_https and parsed.scheme != "https" and not (allow_local_http and is_local):
        raise ConfigurationError(f"{name} must use HTTPS")
    return raw_url.rstrip("/")


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings with secrets excluded from repr()."""

    moltbook_api_key: str = field(repr=False)
    moltbook_base_url: str = "https://www.moltbook.com/api/v1"
    moltbook_timeout_seconds: float = 15.0
    moltbook_retries: int = 3
    moltbook_retry_backoff_seconds: float = 0.75
    moltbook_feed_sort: str = "new"
    moltbook_feed_limit: int = 25
    moltbook_feed_pages: int = 1
    moltbook_comment_sort: str = "new"
    moltbook_comment_limit: int = 100

    llm_provider: str = "ollama"
    llm_model: str = "qwen3:8b"
    llm_fast_model: str = "qwen3:4b"
    llm_temperature: float = 0.2

    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout_seconds: float = 600.0

    openai_compatible_base_url: str = "https://api.openai.com/v1"
    openai_compatible_api_key: str = field(default="", repr=False)
    openai_compatible_timeout_seconds: float = 120.0
    openai_compatible_json_schema: bool = False

    memory_db_path: Path = PROJECT_ROOT / "data" / "memory.db"
    legacy_memory_json_path: Path = PROJECT_ROOT / "memory.json"
    recent_memory_limit: int = 6
    relevant_memory_limit: int = 8
    max_memory_search_candidates: int = 200

    max_post_candidates: int = 20
    max_post_preview_chars: int = 1000
    max_post_chars: int = 10000
    max_comment_chars: int = 6000
    max_comment_preview_chars: int = 700
    max_memory_context_chars: int = 10000
    max_comments_for_selection: int = 100
    max_selected_comments: int = 3
    max_chat_history_messages: int = 20
    max_chat_input_chars: int = 8000

    allow_local_moltbook_http: bool = False

    # Write actions are deliberately fail-closed.
    moltbook_write_enabled: bool = False
    moltbook_dry_run: bool = True
    moltbook_require_approval: bool = True
    moltbook_post_cooldown_seconds: int = 1800
    moltbook_comment_hourly_limit: int = 50
    moltbook_action_db_path: Path = PROJECT_ROOT / "data" / "actions.db"
    moltbook_max_write_chars: int = 10000
    moltbook_allowed_submolts: tuple[str, ...] = ("cybersecurity", "technology", "programming", "ai", "general")

    log_level: str = "INFO"


def load_settings(*, require_moltbook_key: bool = True) -> Settings:
    """Load .env/environment variables and return validated settings."""

    load_dotenv(PROJECT_ROOT / ".env")

    api_key = _env("MOLTBOOK_API_KEY", "") or ""
    if require_moltbook_key and not api_key:
        raise ConfigurationError("MOLTBOOK_API_KEY is required")

    allow_local_http = _bool_env("ALLOW_LOCAL_MOLTBOOK_HTTP", False)

    provider = (_env("LLM_PROVIDER", "ollama") or "ollama").lower()
    if provider not in {"ollama", "openai_compatible"}:
        raise ConfigurationError("LLM_PROVIDER must be 'ollama' or 'openai_compatible'")

    moltbook_base_url = validate_endpoint_url(
        _env("MOLTBOOK_BASE_URL", "https://www.moltbook.com/api/v1") or "",
        "MOLTBOOK_BASE_URL",
        require_https=True,
        allow_local_http=allow_local_http,
    )
    ollama_base_url = validate_endpoint_url(
        _env("OLLAMA_BASE_URL", "http://localhost:11434") or "",
        "OLLAMA_BASE_URL",
        require_https=True,
        allow_local_http=True,
    )
    openai_base_url = validate_endpoint_url(
        _env("OPENAI_COMPATIBLE_BASE_URL", "https://api.openai.com/v1") or "",
        "OPENAI_COMPATIBLE_BASE_URL",
        require_https=True,
        allow_local_http=True,
    )

    openai_key = _env("OPENAI_COMPATIBLE_API_KEY", "") or ""
    if provider == "openai_compatible" and not openai_key:
        raise ConfigurationError(
            "OPENAI_COMPATIBLE_API_KEY is required when LLM_PROVIDER=openai_compatible"
        )

    memory_db = Path(_env("MEMORY_DB_PATH", str(PROJECT_ROOT / "data" / "memory.db")) or "")
    if not memory_db.is_absolute():
        memory_db = PROJECT_ROOT / memory_db

    legacy_json = Path(_env("LEGACY_MEMORY_JSON_PATH", str(PROJECT_ROOT / "memory.json")) or "")
    if not legacy_json.is_absolute():
        legacy_json = PROJECT_ROOT / legacy_json

    action_db = Path(_env("MOLTBOOK_ACTION_DB_PATH", str(PROJECT_ROOT / "data" / "actions.db")) or "")

    allowed_submolts_raw = _env("MOLTBOOK_ALLOWED_SUBMOLTS", "cybersecurity,technology,programming,ai,general") or ""
    allowed_submolts = tuple(dict.fromkeys(
        item.strip().lower().lstrip("m/")
        for item in allowed_submolts_raw.split(",")
        if item.strip()
    ))
    if not allowed_submolts:
        raise ConfigurationError("MOLTBOOK_ALLOWED_SUBMOLTS must contain at least one submolt")
    if not action_db.is_absolute():
        action_db = PROJECT_ROOT / action_db

    # CHANGED: Validate LOG_LEVEL here so invalid logging configuration fails
    # as a normal ConfigurationError instead of crashing later in main().
    log_level = (_env("LOG_LEVEL", "INFO") or "INFO").upper()
    if log_level not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}:
        raise ConfigurationError(
            "LOG_LEVEL must be one of CRITICAL, ERROR, WARNING, INFO, DEBUG, NOTSET"
        )

    return Settings(
        moltbook_api_key=api_key,
        moltbook_base_url=moltbook_base_url,
        moltbook_timeout_seconds=_float_env("MOLTBOOK_TIMEOUT_SECONDS", 15.0, minimum=1.0, maximum=120.0),
        moltbook_retries=_int_env("MOLTBOOK_RETRIES", 3, minimum=0, maximum=8),
        moltbook_retry_backoff_seconds=_float_env(
            "MOLTBOOK_RETRY_BACKOFF_SECONDS", 0.75, minimum=0.0, maximum=10.0
        ),
        moltbook_feed_sort=_env("MOLTBOOK_FEED_SORT", "new") or "new",
        moltbook_feed_limit=_int_env("MOLTBOOK_FEED_LIMIT", 25, minimum=1, maximum=100),
        moltbook_feed_pages=_int_env("MOLTBOOK_FEED_PAGES", 1, minimum=1, maximum=10),
        moltbook_comment_sort=_env("MOLTBOOK_COMMENT_SORT", "new") or "new",
        moltbook_comment_limit=_int_env("MOLTBOOK_COMMENT_LIMIT", 100, minimum=1, maximum=500),
        llm_provider=provider,
        llm_model=_env("LLM_MODEL", "qwen3:8b") or "qwen3:8b",
        llm_fast_model=_env("LLM_FAST_MODEL", "qwen3:4b") or "qwen3:4b",
        llm_temperature=_float_env("LLM_TEMPERATURE", 0.2, minimum=0.0, maximum=2.0),
        ollama_base_url=ollama_base_url,
        ollama_timeout_seconds=_float_env("OLLAMA_TIMEOUT_SECONDS", 600.0, minimum=5.0, maximum=1800.0),
        openai_compatible_base_url=openai_base_url,
        openai_compatible_api_key=openai_key,
        openai_compatible_timeout_seconds=_float_env(
            "OPENAI_COMPATIBLE_TIMEOUT_SECONDS", 120.0, minimum=5.0, maximum=900.0
        ),
        openai_compatible_json_schema=_bool_env("OPENAI_COMPATIBLE_JSON_SCHEMA", False),
        memory_db_path=memory_db,
        legacy_memory_json_path=legacy_json,
        recent_memory_limit=_int_env("RECENT_MEMORY_LIMIT", 6, minimum=1, maximum=50),
        relevant_memory_limit=_int_env("RELEVANT_MEMORY_LIMIT", 8, minimum=1, maximum=50),
        max_memory_search_candidates=_int_env(
            "MAX_MEMORY_SEARCH_CANDIDATES", 200, minimum=20, maximum=5000
        ),
        max_post_candidates=_int_env("MAX_POST_CANDIDATES", 20, minimum=1, maximum=50),
        max_post_preview_chars=_int_env("MAX_POST_PREVIEW_CHARS", 1000, minimum=100, maximum=5000),
        max_post_chars=_int_env("MAX_POST_CHARS", 10000, minimum=500, maximum=50000),
        max_comment_chars=_int_env("MAX_COMMENT_CHARS", 6000, minimum=500, maximum=50000),
        max_comment_preview_chars=_int_env("MAX_COMMENT_PREVIEW_CHARS", 700, minimum=100, maximum=3000),
        max_memory_context_chars=_int_env("MAX_MEMORY_CONTEXT_CHARS", 10000, minimum=1000, maximum=50000),
        max_comments_for_selection=_int_env(
            "MAX_COMMENTS_FOR_SELECTION", 100, minimum=1, maximum=500
        ),
        max_selected_comments=_int_env("MAX_SELECTED_COMMENTS", 3, minimum=1, maximum=10),
        max_chat_history_messages=_int_env(
            "MAX_CHAT_HISTORY_MESSAGES", 20, minimum=4, maximum=100
        ),
        max_chat_input_chars=_int_env("MAX_CHAT_INPUT_CHARS", 8000, minimum=100, maximum=50000),
        allow_local_moltbook_http=allow_local_http,
        moltbook_write_enabled=_bool_env("MOLTBOOK_WRITE_ENABLED", False),
        moltbook_dry_run=_bool_env("MOLTBOOK_DRY_RUN", True),
        moltbook_require_approval=_bool_env("MOLTBOOK_REQUIRE_APPROVAL", True),
        moltbook_post_cooldown_seconds=_int_env("MOLTBOOK_POST_COOLDOWN_SECONDS", 1800, minimum=0, maximum=86400),
        moltbook_comment_hourly_limit=_int_env("MOLTBOOK_COMMENT_HOURLY_LIMIT", 50, minimum=1, maximum=500),
        moltbook_action_db_path=action_db,
        moltbook_max_write_chars=_int_env("MOLTBOOK_MAX_WRITE_CHARS", 10000, minimum=100, maximum=50000),
        moltbook_allowed_submolts=allowed_submolts,
        # CHANGED: Use the already-validated log level.
        log_level=log_level,
    )
