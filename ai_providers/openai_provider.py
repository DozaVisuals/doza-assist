"""OpenAI provider via the official SDK.

Model selection mirrors the Anthropic tiering, keyed by ``task_type``:

  - ``analysis``          → gpt-5.4 (deep story / Story Brief analysis)
  - ``profile_creation``  → gpt-5.4 (My Style synthesis)
  - everything else       → gpt-5.4-mini (chat, selects, general)

gpt-5.4 / gpt-5.4-mini are reasoning models, which changes two things vs.
the former gpt-4o path:
  - the output cap is ``max_completion_tokens`` (``max_tokens`` is rejected
    with a 400);
  - hidden reasoning tokens are spent from that same budget, so a too-small
    cap returns an empty answer. We floor the cap well above the requested
    value and set ``reasoning_effort`` per task.
``temperature`` is never set — reasoning models reject non-default values.

The SDK manages SSE streaming, retries, and typed error classes that we
map to clear user-facing messages.
"""
import time
from typing import Union

from .base import BaseProvider
from . import ProviderError


MODEL_DEFAULT = "gpt-5.4-mini"      # chat, selects, general
MODEL_FLAGSHIP = "gpt-5.4"          # analysis (incl. Story Brief) + My Style

# Task types routed to the flagship model — mirrors anthropic_provider.
_FLAGSHIP_TASKS = {"analysis", "profile_creation"}

# Reasoning models spend part of the output budget on hidden reasoning
# tokens; if max_completion_tokens is too small the visible answer comes
# back empty. Floor every request above this (also covers test_connection,
# which asks for max_tokens=20).
_MIN_COMPLETION_TOKENS = 2048


def _model_for_task(task_type: str) -> str:
    return MODEL_FLAGSHIP if task_type in _FLAGSHIP_TASKS else MODEL_DEFAULT


def _reasoning_effort_for_task(task_type: str) -> str:
    # Balanced reasoning for flagship analysis/synthesis; low effort for
    # interactive/cheap tasks to keep latency and cost down.
    return "medium" if task_type in _FLAGSHIP_TASKS else "low"


def _max_tokens_for_task(task_type: str) -> int:
    if task_type in ("profile_creation", "analysis"):
        return 16384
    if task_type == "chat":
        return 8192
    return 8192


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
        requested = kwargs.get("max_tokens")
        if requested:
            max_out = max(int(requested), _MIN_COMPLETION_TOKENS)
        else:
            max_out = _max_tokens_for_task(task_type)
        body = {
            "model": _model_for_task(task_type),
            # Reasoning models use max_completion_tokens, not max_tokens.
            "max_completion_tokens": max_out,
            "messages": _to_openai_messages(system_prompt, user_or_messages),
            "reasoning_effort": _reasoning_effort_for_task(task_type),
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
