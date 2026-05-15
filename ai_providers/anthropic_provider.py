"""Anthropic provider via the official SDK.

Model selection is hard-coded by ``task_type`` and never exposed to the user:

  - ``profile_creation``, ``analysis``, ``story_brief``
        -> claude-opus-4-7 (highest quality for editorial work)
  - everything else (chat, general)
        -> claude-sonnet-4-6 (fast, cost-effective)

These IDs were bumped from ``claude-opus-4-20250514`` /
``claude-sonnet-4-20250514`` in May 2026 ahead of the June 15 2026
deprecation EOL on the 4.0 IDs.

The SDK manages SSE streaming, retry/backoff, and typed error classes that
we map to clear user-facing messages.

PROMPT CACHING
--------------
Heavy editorial calls reuse the same transcript across many requests in a
single session (Story analysis runs four passes per chunk; chat resends
the transcript on every user turn). To avoid re-billing the same input
tokens on every call, the wrapper layer in ``ai_analysis.py`` can pass:

  - ``system_prompt`` as a list of structured text blocks. Blocks marked
    with ``cache_control`` are cached server-side. The stable system
    text and Editorial DNA examples share an ephemeral 5-minute cache;
    the transcript block gets the extended 1-hour TTL (requires the
    ``extended-cache-ttl-2025-04-11`` beta header, which we send on
    every request so callers do not need to think about it).
  - User messages with content blocks that carry ``cache_control``. The
    chat path uses this so the per-turn transcript message hits the
    cache on every follow-up turn.

Each response.usage is logged with the cache hit rate, and we warn when
the same in-process prefix re-creates a cache entry without reading from
the existing one (a sign that the cache_control breakpoint moved
between calls).
"""
import logging
import time
from typing import Any, Dict, List, Optional, Union

from .base import BaseProvider
from . import ProviderError


MODEL_DEFAULT = "claude-sonnet-4-6"
MODEL_OPUS = "claude-opus-4-7"

# Task types that benefit from Opus-level reasoning: deep editorial
# analysis (story beats, soundbites, social clips) and long-form
# synthesis (story briefs, My Style profiles).
_OPUS_TASK_TYPES = frozenset({"profile_creation", "analysis", "story_brief"})

# Anthropic enforces a per-block minimum on cached prefixes: Sonnet and
# Opus need at least 1024 tokens in a block before the API will cache
# it. We approximate with len(text) // 4, which is a small overestimate
# of tokens for English prose and a safe bound for the minimum check.
_MIN_CACHE_TOKENS = 1024
_CHARS_PER_TOKEN = 4

# Extended 1h TTL on ephemeral cache requires this beta header. Sending
# it only on requests that actually use cache markers keeps non-cached
# calls byte-identical to the pre-caching behavior.
_EXTENDED_TTL_HEADER = "extended-cache-ttl-2025-04-11"

logger = logging.getLogger(__name__)

# Per-process record of prefixes we have already paid to cache. A second
# call that lands on the same prefix is expected to return a non-zero
# cache_read_input_tokens; if it does not, log a WARNING because the
# breakpoint moved silently.
_seen_cache_prefixes: set = set()


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _model_for_task(task_type: str) -> str:
    return MODEL_OPUS if task_type in _OPUS_TASK_TYPES else MODEL_DEFAULT


def _max_tokens_for_task(task_type: str) -> int:
    if task_type in ("profile_creation", "analysis", "story_brief"):
        return 8192
    if task_type == "chat":
        return 2048
    return 4096


def _normalize_messages(user_or_messages: Union[str, list]) -> list:
    if isinstance(user_or_messages, list):
        return list(user_or_messages)
    return [{"role": "user", "content": str(user_or_messages)}]


def _prefix_signature(system_param: Any, messages: list) -> str:
    """Cheap content hash for the cacheable prefix.

    Used only to decide whether to WARN about a missed cache hit on the
    second call to the same prefix in one process. Hashes the stable
    system text plus the first user message text (which carries the
    transcript on the chat path).
    """
    import hashlib
    h = hashlib.sha1()
    if isinstance(system_param, list):
        for block in system_param:
            if isinstance(block, dict) and block.get("cache_control"):
                h.update((block.get("text") or "").encode("utf-8", "replace"))
    elif isinstance(system_param, str):
        h.update(system_param.encode("utf-8", "replace"))
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("cache_control"):
                    h.update((block.get("text") or "").encode("utf-8", "replace"))
        elif isinstance(content, str) and m.get("role") == "user":
            h.update(content.encode("utf-8", "replace"))
            break
    return h.hexdigest()


def _has_cache_markers(system_param: Any, messages: list) -> bool:
    if isinstance(system_param, list):
        for block in system_param:
            if isinstance(block, dict) and block.get("cache_control"):
                return True
    for m in messages:
        if not isinstance(m, dict):
            continue
        content = m.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("cache_control"):
                    return True
    return False


def _log_usage(usage, model: str, call_site: str, signature: Optional[str]):
    """Log Anthropic usage stats and warn on suspected cache misses.

    Hit rate is computed as cache_read / (cache_read + input_tokens),
    matching the way Anthropic bills: fresh input tokens are cheaper
    than cache_creation but ten times more expensive than cache_read.

    The Flask app uses ``print()`` for diagnostics (no ``logging``
    configuration), so the human-readable line also goes through
    ``print()`` to land in the server log alongside per-request lines.
    The module-level ``logger`` is still wired up so pytest's caplog
    fixture can assert on warnings in unit tests.
    """
    if usage is None:
        return
    cache_create = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    in_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    out_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    total_in = cache_read + in_tokens
    hit_rate = (cache_read / total_in) if total_in else 0.0
    line = (
        f"[anthropic.usage] site={call_site} model={model} "
        f"in={in_tokens} out={out_tokens} "
        f"cache_create={cache_create} cache_read={cache_read} "
        f"hit_rate={hit_rate:.2f}"
    )
    print(line, flush=True)
    logger.info(
        "anthropic.usage site=%s model=%s in=%d out=%d cache_create=%d "
        "cache_read=%d hit_rate=%.2f",
        call_site, model, in_tokens, out_tokens,
        cache_create, cache_read, hit_rate,
    )
    if signature and signature in _seen_cache_prefixes:
        if cache_create > 0 and cache_read == 0:
            warn_line = (
                f"[anthropic.cache-miss-on-repeat] site={call_site} model={model} "
                f"signature={signature[:12]} cache_create={cache_create} "
                "expected non-zero cache_read"
            )
            print(warn_line, flush=True)
            logger.warning(
                "anthropic.cache-miss-on-repeat site=%s model=%s signature=%s "
                "cache_create=%d expected non-zero cache_read",
                call_site, model, signature[:12], cache_create,
            )
    if signature and (cache_create > 0 or cache_read > 0):
        _seen_cache_prefixes.add(signature)


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
        # ``system_prompt`` may be a string (legacy single-text path) or
        # a list of structured content blocks (cached-prefix path). The
        # SDK accepts either, so we pass through without coercion.
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
        body = self._build_body(system_prompt, user_or_messages, task_type, kwargs)
        call_site = kwargs.get("call_site") or "generate"
        resp = self._call_with_retry(body, call_site=call_site)
        return "".join(block.text for block in resp.content if hasattr(block, "text"))

    def _request_headers(self, body) -> Dict[str, str]:
        """Return per-request headers. The extended-TTL beta header is
        sent only when the body actually uses cache markers, so
        non-cached calls do not change behavior.
        """
        if _has_cache_markers(body.get("system"), body.get("messages") or []):
            return {"anthropic-beta": _EXTENDED_TTL_HEADER}
        return {}

    def _call_with_retry(self, body, call_site: str = "anthropic"):
        from anthropic import RateLimitError, AuthenticationError, APIError
        extra_headers = self._request_headers(body)
        signature = None
        if extra_headers:
            signature = _prefix_signature(body.get("system"), body.get("messages") or [])
        try:
            resp = self._client.messages.create(extra_headers=extra_headers, **body)
        except RateLimitError:
            time.sleep(2)
            try:
                resp = self._client.messages.create(extra_headers=extra_headers, **body)
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
        _log_usage(getattr(resp, "usage", None), body.get("model"), call_site, signature)
        return resp

    def generate_stream(self, system_prompt, user_or_messages, task_type="general", **kwargs):
        from anthropic import RateLimitError, AuthenticationError, APIError
        body = self._build_body(system_prompt, user_or_messages, task_type, kwargs)
        call_site = kwargs.get("call_site") or "generate_stream"
        extra_headers = self._request_headers(body)
        signature = None
        if extra_headers:
            signature = _prefix_signature(body.get("system"), body.get("messages") or [])
        try:
            with self._client.messages.stream(extra_headers=extra_headers, **body) as stream:
                for piece in stream.text_stream:
                    if piece:
                        yield piece
                try:
                    final = stream.get_final_message()
                    _log_usage(
                        getattr(final, "usage", None),
                        body.get("model"), call_site, signature,
                    )
                except Exception:
                    # Usage is advisory; never let a logging failure
                    # break the streaming call itself.
                    pass
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


# Public helpers used by the wrapper layer to assemble cached payloads.
# Kept here so the call-site logic in ai_analysis.py stays inline and
# does not grow a separate caching abstraction class.

def build_cached_system_blocks(
    system_prompt: str,
    dna_block: Optional[str] = None,
    transcript_block: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return a structured ``system`` list with cache_control markers.

    Block order depends on whether a transcript is present, because
    Anthropic computes the cache key byte-by-byte from the FRONT of the
    prefix. Anything that varies between calls must sit AFTER everything
    cached, otherwise it shifts the prefix and every call misses.

    With a transcript (analysis path):
      1. transcript text, ``cache_control`` 1h TTL. Sits first because
         it is the only piece guaranteed identical across the four
         story-analysis passes (soundbites / beats / overview / social),
         each of which ships a DIFFERENT ``system_prompt``.
      2. dna_block, ``cache_control`` 5m TTL, if supplied.
      3. system_prompt, NO ``cache_control``. Varies per pass on the
         four-pass path, so it must not be inside the cached prefix.

    Without a transcript (chat path):
      1. system_prompt + dna_block joined, ``cache_control`` 5m TTL.
         The system text is the stable prefix here; the transcript
         lives in a user message instead (see
         :func:`build_cached_user_messages`).

    Each block omits ``cache_control`` when it would not meet the 1024
    token minimum. The block is still sent, it just is not a cache
    boundary.
    """
    blocks: List[Dict[str, Any]] = []
    has_transcript = bool((transcript_block or "").strip())

    if has_transcript:
        t = transcript_block.strip()
        tblock: Dict[str, Any] = {"type": "text", "text": t}
        if _approx_tokens(t) >= _MIN_CACHE_TOKENS:
            tblock["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
        blocks.append(tblock)

        if dna_block and dna_block.strip():
            d = dna_block.strip()
            dblock: Dict[str, Any] = {"type": "text", "text": d}
            if _approx_tokens(d) >= _MIN_CACHE_TOKENS:
                # 1h TTL throughout: Anthropic enforces longer-TTL
                # blocks must precede shorter ones in the global order
                # (tools, system, messages). Mixing 5m and 1h causes
                # HTTP 400 when the request also has a 1h transcript
                # block downstream in messages. Uniform TTL sidesteps
                # the rule entirely.
                dblock["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
            blocks.append(dblock)

        sys_text = (system_prompt or "").strip()
        if sys_text:
            blocks.append({"type": "text", "text": sys_text})
        return blocks

    stable_text = (system_prompt or "").strip()
    if dna_block:
        dna_text = dna_block.strip()
        if dna_text:
            stable_text = (
                stable_text + "\n\n" + dna_text
            ).strip() if stable_text else dna_text
    if stable_text:
        block: Dict[str, Any] = {"type": "text", "text": stable_text}
        if _approx_tokens(stable_text) >= _MIN_CACHE_TOKENS:
            # Same uniform-1h rationale as above; see the with-transcript
            # branch comment.
            block["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
        blocks.append(block)
    return blocks


def build_cached_user_messages(
    transcript_block: str,
    other_user_messages: List[Dict[str, Any]],
    extra_user_blocks: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Wrap the chat-path messages so the transcript turn is cached.

    The chat layer sends the transcript as the FIRST user message. To
    cache it, that message's content becomes a list of text blocks with
    the transcript block carrying cache_control. Subsequent messages
    (the assistant ack, history, the new user turn) are left as plain
    strings because they change every turn.
    """
    cached_blocks: List[Dict[str, Any]] = []
    if extra_user_blocks:
        cached_blocks.extend(extra_user_blocks)
    transcript_text = (transcript_block or "").strip()
    if transcript_text:
        block: Dict[str, Any] = {"type": "text", "text": transcript_text}
        if _approx_tokens(transcript_text) >= _MIN_CACHE_TOKENS:
            block["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
        cached_blocks.append(block)

    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": cached_blocks},
    ]
    messages.extend(other_user_messages or [])
    return messages
