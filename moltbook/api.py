"""Hardened Moltbook API client.

Read and write HTTP primitives live here, but writes are only reachable through
the separate Python-controlled action layer. The LLM never receives this client
or the API credential.
"""

from __future__ import annotations

import json
import logging
import random
import time
from typing import Any
from urllib.parse import quote

import requests

from config import validate_endpoint_url
from moltbook.schemas import Comment, Post


logger = logging.getLogger(__name__)


class MoltbookAPIError(RuntimeError):
    """Base exception for Moltbook API failures."""


class MoltbookAuthenticationError(MoltbookAPIError):
    """Raised when Moltbook rejects credentials."""


class MoltbookRateLimitError(MoltbookAPIError):
    """Raised after a bounded retry budget is exhausted by HTTP 429."""


class MoltbookNetworkError(MoltbookAPIError):
    """Raised when the network request cannot be completed."""


_TRANSIENT_STATUSES = {408, 429, 500, 502, 503, 504}


def _safe_path_segment(value: str, name: str) -> str:
    value = str(value).strip()
    if not value or any(ord(char) < 32 for char in value):
        raise ValueError(f"Invalid {name}")
    return quote(value, safe="")


def _retry_delay(response: requests.Response | None, base: float, attempt: int) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(30.0, max(0.0, float(retry_after)))
            except ValueError:
                pass
    exponential = base * (2 ** attempt)
    jitter = random.uniform(0.0, min(base, 1.0))
    return min(30.0, exponential + jitter)


def _extract_items(payload: Any, key: str) -> tuple[list[Any], dict[str, Any]]:
    if isinstance(payload, list):
        return payload, {}
    if not isinstance(payload, dict):
        raise MoltbookAPIError("Moltbook returned an unexpected JSON shape")

    items = payload.get(key)
    if isinstance(items, list):
        return items, payload

    nested = payload.get("data")
    if isinstance(nested, dict) and isinstance(nested.get(key), list):
        return nested[key], nested

    raise MoltbookAPIError(f"Moltbook response did not contain a '{key}' list")


class MoltbookClient:
    """Small, bounded-retry, redirect-disabled Moltbook client."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://www.moltbook.com/api/v1",
        timeout_seconds: float = 15.0,
        retries: int = 3,
        retry_backoff_seconds: float = 0.75,
        user_agent: str = "Schranz/1.0",
        session: requests.Session | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Moltbook API key must not be empty")
        self.base_url = validate_endpoint_url(
            base_url,
            "Moltbook base URL",
            require_https=True,
            allow_local_http=True,
        )
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "User-Agent": user_agent,
            }
        )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        method = method.upper()

        for attempt in range(self.retries + 1):
            try:
                response = self.session.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    timeout=self.timeout_seconds,
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                if attempt >= self.retries:
                    raise MoltbookNetworkError(
                        f"Moltbook request failed after {self.retries + 1} attempts: {method} {path}"
                    ) from exc
                delay = _retry_delay(None, self.retry_backoff_seconds, attempt)
                logger.warning("Moltbook network error; retrying in %.2fs", delay)
                time.sleep(delay)
                continue

            if 200 <= response.status_code < 300:
                try:
                    return response.json()
                except (ValueError, json.JSONDecodeError) as exc:
                    raise MoltbookAPIError(
                        f"Moltbook returned invalid JSON for {method} {path}"
                    ) from exc

            if 300 <= response.status_code < 400:
                # Never follow redirects because the Authorization header is a credential.
                raise MoltbookAPIError(
                    f"Unexpected redirect ({response.status_code}) from Moltbook {method} {path}"
                )

            if response.status_code in {401, 403}:
                raise MoltbookAuthenticationError(
                    f"Moltbook authentication/authorization failed ({response.status_code})"
                )

            if response.status_code == 429:
                if attempt >= self.retries:
                    raise MoltbookRateLimitError(
                        f"Moltbook rate limit persisted after {self.retries + 1} attempts"
                    )
                delay = _retry_delay(response, self.retry_backoff_seconds, attempt)
                logger.warning("Moltbook rate-limited; retrying in %.2fs", delay)
                time.sleep(delay)
                continue

            if response.status_code in _TRANSIENT_STATUSES:
                if attempt >= self.retries:
                    raise MoltbookAPIError(
                        f"Moltbook temporary failure persisted ({response.status_code}) for {method} {path}"
                    )
                delay = _retry_delay(response, self.retry_backoff_seconds, attempt)
                logger.warning(
                    "Moltbook returned %s; retrying in %.2fs",
                    response.status_code,
                    delay,
                )
                time.sleep(delay)
                continue

            raise MoltbookAPIError(
                f"Moltbook request failed with HTTP {response.status_code} for {method} {path}"
            )

        raise AssertionError("unreachable")

    def get_status(self) -> dict[str, Any]:
        payload = self._request_json("GET", "/agents/status")
        if not isinstance(payload, dict):
            raise MoltbookAPIError("Unexpected status response")
        return payload

    def get_home(self) -> dict[str, Any]:
        payload = self._request_json("GET", "/home")
        if not isinstance(payload, dict):
            raise MoltbookAPIError("Unexpected home response")
        return payload

    def get_feed(
        self,
        *,
        sort: str = "new",
        limit: int = 25,
        max_pages: int = 1,
    ) -> list[Post]:
        """Read the personalized feed using cursor pagination when available."""

        posts: list[Post] = []
        seen_ids: set[str] = set()
        cursor: str | None = None

        for _ in range(max(1, max_pages)):
            params: dict[str, Any] = {"sort": sort, "limit": limit}
            if cursor:
                params["cursor"] = cursor

            payload = self._request_json("GET", "/feed", params=params)
            raw_posts, meta = _extract_items(payload, "posts")

            for raw_post in raw_posts:
                if not isinstance(raw_post, dict):
                    continue
                try:
                    post = Post.model_validate(_normalize_post(raw_post))
                except Exception:
                    logger.warning("Skipping malformed Moltbook post")
                    continue
                if post.id not in seen_ids:
                    posts.append(post)
                    seen_ids.add(post.id)

            next_cursor = meta.get("next_cursor") if isinstance(meta, dict) else None
            if not isinstance(next_cursor, str) or not next_cursor:
                break
            cursor = next_cursor

        return posts

    def get_comments(
        self,
        post_id: str,
        *,
        sort: str = "new",
        limit: int = 100,
    ) -> list[Comment]:
        safe_id = _safe_path_segment(post_id, "post id")
        payload = self._request_json(
            "GET",
            f"/posts/{safe_id}/comments",
            # Moltbook documents `sort` for this endpoint; apply our limit locally
            # so a server-version change cannot turn a supported read into a 400.
            params={"sort": sort},
        )
        raw_comments, _ = _extract_items(payload, "comments")
        flattened: list[Comment] = []
        seen_ids: set[str] = set()

        def walk(raw: Any, parent_id: str | None = None) -> None:
            if len(flattened) >= limit or not isinstance(raw, dict):
                return
            normalized = dict(raw)
            if parent_id and not normalized.get("parent_id"):
                normalized["parent_id"] = parent_id
            try:
                comment = Comment.model_validate(_normalize_comment(normalized, post_id))
            except Exception:
                logger.warning("Skipping malformed Moltbook comment")
                return
            if comment.id not in seen_ids:
                flattened.append(comment)
                seen_ids.add(comment.id)
            for reply in raw.get("replies", []) or []:
                walk(reply, comment.id)

        for raw_comment in raw_comments:
            walk(raw_comment)
            if len(flattened) >= limit:
                break

        return flattened

    def create_post(
        self,
        *,
        title: str,
        content: str,
        submolt: str,
        post_type: str = "text",
    ) -> dict[str, Any]:
        """Create a text post. Policy/approval belongs in moltbook.actions."""
        payload = {"title": title, "content": content, "submolt": submolt, "type": post_type}
        response = self._request_json("POST", "/posts", json_body=payload)
        if not isinstance(response, dict):
            raise MoltbookAPIError("Unexpected create-post response")
        return response

    def create_comment(
        self,
        post_id: str,
        *,
        content: str,
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a comment/reply. The synchronous response may contain verification data."""
        safe_id = _safe_path_segment(post_id, "post id")
        payload: dict[str, Any] = {"content": content}
        if parent_id:
            payload["parent_id"] = parent_id
        response = self._request_json(
            "POST",
            f"/posts/{safe_id}/comments",
            json_body=payload,
        )
        if not isinstance(response, dict):
            raise MoltbookAPIError("Unexpected create-comment response")
        return response

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "MoltbookClient":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _normalize_author(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return {
            "id": str(raw["id"]) if raw.get("id") is not None else None,
            "name": str(raw.get("name") or raw.get("username") or "unknown"),
        }
    if raw is not None:
        return {"name": str(raw)}
    return {"name": "unknown"}


def _normalize_post(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(raw.get("id") or raw.get("post_id") or "").strip(),
        "title": str(raw.get("title") or "Untitled"),
        "content": str(raw.get("content") or ""),
        "author": _normalize_author(raw.get("author")),
        "url": raw.get("url"),
        "created_at": raw.get("created_at"),
        "submolt": raw.get("submolt") or raw.get("submolt_name"),
    }


def _normalize_comment(raw: dict[str, Any], post_id: str) -> dict[str, Any]:
    return {
        "id": str(raw.get("id") or raw.get("comment_id") or "").strip(),
        "content": str(raw.get("content") or ""),
        "author": _normalize_author(raw.get("author")),
        "parent_id": str(raw["parent_id"]) if raw.get("parent_id") is not None else None,
        "post_id": str(raw.get("post_id") or post_id),
        "created_at": raw.get("created_at"),
        "replies": [],
    }


# Backwards-compatible function-style API used by older code.
def _default_client() -> MoltbookClient:
    from config import load_settings

    settings = load_settings()
    return MoltbookClient(
        settings.moltbook_api_key,
        base_url=settings.moltbook_base_url,
        timeout_seconds=settings.moltbook_timeout_seconds,
        retries=settings.moltbook_retries,
        retry_backoff_seconds=settings.moltbook_retry_backoff_seconds,
    )


def get_status() -> dict[str, Any]:
    with _default_client() as client:
        return client.get_status()


def get_home() -> dict[str, Any]:
    with _default_client() as client:
        return client.get_home()


def get_feed() -> list[Post]:
    from config import load_settings

    settings = load_settings()
    with MoltbookClient(
        settings.moltbook_api_key,
        base_url=settings.moltbook_base_url,
        timeout_seconds=settings.moltbook_timeout_seconds,
        retries=settings.moltbook_retries,
        retry_backoff_seconds=settings.moltbook_retry_backoff_seconds,
    ) as client:
        return client.get_feed(
            sort=settings.moltbook_feed_sort,
            limit=settings.moltbook_feed_limit,
            max_pages=settings.moltbook_feed_pages,
        )


def get_comments(post_id: str) -> list[Comment]:
    from config import load_settings

    settings = load_settings()
    with MoltbookClient(
        settings.moltbook_api_key,
        base_url=settings.moltbook_base_url,
        timeout_seconds=settings.moltbook_timeout_seconds,
        retries=settings.moltbook_retries,
        retry_backoff_seconds=settings.moltbook_retry_backoff_seconds,
    ) as client:
        return client.get_comments(
            post_id,
            sort=settings.moltbook_comment_sort,
            limit=settings.moltbook_comment_limit,
        )
