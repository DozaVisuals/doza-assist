"""Anthropic provider via the official SDK.

Model selection is hard-coded by ``task_type`` and never exposed to the user:

  - ``profile_creation``  → claude-opus-4-20250514 (My Style synthesis)
  - everything else       → claude-sonnet-4-20250514

The SDK manages SSE streaming, retry/backoff, and typed error classes that
we map to clear user-facing messages.
"""
import time
from typing import Union

from .base import BaseProvider
from . import ProviderError


MODEL_DEFAULT = "claude-sonnet-4-20250514"
MODEL_PROFILE = "claude-opus-4-20250514"


def _model_for_task(task_type: str) -> str:
    return MODEL_PROFILE if task_type == "profile_creation" else MODEL_DEFAULT


def _max_tokens_for_task(task_type: str) -> int:
    if task_type == "profile_creation":
        return 8192
    if task_type == "chat":
        return 2048
    return 4096


def _normalize_messages(user_or_messages: Union[str, list]) -> list:
    if isinstance(user_or_messages, list):
        return list(user_or_messages)
    return [{"role": "user", "content": str(user_or_messages)}]


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, api_key: str):
        if not api_key:
            raise ProviderError(
                "No Anthropic API key configured. Add one in Settings or switch to Local mode.",
                code="missing_key",
            )
        self.api_key = api_key
        # Lazy import: only loaded when this provider is active.
        from anthropic import Anthropic
        self._client = Anthropic(api_key=api_key)

    def _build_body(self, system_prompt, user_or_messages, task_type, kwargs):
        body = {
            "model": _model_for_task(task_type),
            "max_tokens": kwargs.get("max_tokens", _max_tokens_for_task(task_type)),
            "system": system_prompt,
            "messages": _normalize_messages(user_or_messages),
        }
        stop = kwargs.get("stop")
        if stop:
            body["stop_sequences"] = list(stop)
        return body

    def generate(self, system_prompt, user_or_messages, task_type="general", **kwargs):
        from anthropic import APIError
        body = self._build_body(system_prompt, user_or_messages, task_type, kwargs)
        resp = self._call_with_retry(body)
        return "".join(block.text for block in resp.content if hasattr(block, "text"))

    def _call_with_retry(self, body):
        from anthropic import RateLimitError, AuthenticationError, APIError
        try:
            return self._client.messages.create(**body)
        except RateLimitError:
            time.sleep(2)
            try:
                return self._client.messages.create(**body)
            except RateLimitError:
                raise ProviderError(
                    "API rate limited, try again in a moment.",
                    code="rate_limited",
                )
        except AuthenticationError:
            raise ProviderError(
                "Anthropic API key is invalid or expired. Update it in Settings.",
                code="invalid_key",
            )
        except APIError as e:
            raise ProviderError(f"Anthropic API error: {e}")

    def generate_stream(self, system_prompt, user_or_messages, task_type="general", **kwargs):
        from anthropic import RateLimitError, AuthenticationError, APIError
        body = self._build_body(system_prompt, user_or_messages, task_type, kwargs)
        try:
            with self._client.messages.stream(**body) as stream:
                for piece in stream.text_stream:
                    if piece:
                        yield piece
        except AuthenticationError:
            raise ProviderError(
                "Anthropic API key is invalid or expired. Update it in Settings.",
                code="invalid_key",
            )
        except RateLimitError:
            raise ProviderError(
                "API rate limited, try again in a moment.",
                code="rate_limited",
            )
        except APIError as e:
            raise ProviderError(f"Anthropic API error: {e}")

    def test_connection(self) -> dict:
        try:
            text = self.generate(
                "You are a helpful assistant.",
                "Say hello in exactly one word.",
                task_type="general",
                max_tokens=20,
            )
            return {"success": True, "response": (text or "").strip()}
        except ProviderError as e:
            return {"success": False, "error": str(e), "code": e.code}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {e}"}
