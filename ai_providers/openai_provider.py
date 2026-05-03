"""OpenAI provider via the official SDK.

Uses gpt-4o for every ``task_type`` (Sonnet/Opus tiering is an Anthropic
concept; OpenAI's equivalent balance point is gpt-4o for everything).

The SDK manages SSE streaming, retries, and typed error classes that we
map to clear user-facing messages.
"""
import time
from typing import Union

from .base import BaseProvider
from . import ProviderError


MODEL = "gpt-4o"


def _max_tokens_for_task(task_type: str) -> int:
    if task_type == "profile_creation":
        return 8192
    if task_type == "chat":
        return 2048
    return 4096


def _to_openai_messages(system_prompt: str, user_or_messages: Union[str, list]) -> list:
    msgs = [{"role": "system", "content": system_prompt}]
    if isinstance(user_or_messages, list):
        msgs.extend(user_or_messages)
    else:
        msgs.append({"role": "user", "content": str(user_or_messages)})
    return msgs


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, api_key: str):
        if not api_key:
            raise ProviderError(
                "No OpenAI API key configured. Add one in Settings or switch to Local mode.",
                code="missing_key",
            )
        self.api_key = api_key
        # Lazy import: only loaded when this provider is active.
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)

    def _build_body(self, system_prompt, user_or_messages, task_type, kwargs):
        body = {
            "model": MODEL,
            "max_tokens": kwargs.get("max_tokens", _max_tokens_for_task(task_type)),
            "messages": _to_openai_messages(system_prompt, user_or_messages),
        }
        stop = kwargs.get("stop")
        if stop:
            # OpenAI's chat completions API accepts up to 4 stop sequences.
            body["stop"] = list(stop)[:4]
        return body

    def generate(self, system_prompt, user_or_messages, task_type="general", **kwargs):
        body = self._build_body(system_prompt, user_or_messages, task_type, kwargs)
        resp = self._call_with_retry(body)
        return resp.choices[0].message.content or ""

    def _call_with_retry(self, body):
        from openai import RateLimitError, AuthenticationError, APIError
        try:
            return self._client.chat.completions.create(**body)
        except RateLimitError:
            time.sleep(2)
            try:
                return self._client.chat.completions.create(**body)
            except RateLimitError:
                raise ProviderError(
                    "API rate limited, try again in a moment.",
                    code="rate_limited",
                )
        except AuthenticationError:
            raise ProviderError(
                "OpenAI API key is invalid or expired. Update it in Settings.",
                code="invalid_key",
            )
        except APIError as e:
            raise ProviderError(f"OpenAI API error: {e}")

    def generate_stream(self, system_prompt, user_or_messages, task_type="general", **kwargs):
        from openai import RateLimitError, AuthenticationError, APIError
        body = self._build_body(system_prompt, user_or_messages, task_type, kwargs)
        body["stream"] = True
        try:
            for event in self._client.chat.completions.create(**body):
                choice = event.choices[0] if event.choices else None
                if choice and choice.delta and choice.delta.content:
                    yield choice.delta.content
        except AuthenticationError:
            raise ProviderError(
                "OpenAI API key is invalid or expired. Update it in Settings.",
                code="invalid_key",
            )
        except RateLimitError:
            raise ProviderError(
                "API rate limited, try again in a moment.",
                code="rate_limited",
            )
        except APIError as e:
            raise ProviderError(f"OpenAI API error: {e}")

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
