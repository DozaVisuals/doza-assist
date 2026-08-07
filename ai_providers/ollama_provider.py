"""Ollama provider — wraps the local /api/generate and /api/chat endpoints.

This is the existing call path lifted out of ``ai_analysis.py``. Behavior is
preserved: ``task_type='analysis'`` uses ``/api/generate`` with
``format='json'`` for structured output; everything else uses ``/api/chat``
with the messages array contract.

``task_type`` is otherwise ignored — local Ollama runs whichever model the
user has configured (the ``model_resolver`` callable resolves it lazily so
this module doesn't import ``model_config`` at load time).
"""
import json
import os
import time

import requests

from .base import BaseProvider
from . import ProviderError


def _tiered_keep_alive():
    # RAM-tiered residency: 30m of idle gemma is free on a big machine and
    # a display-corrupting liability on an 8 GB one (memory_budget governs).
    try:
        from memory_budget import ollama_keep_alive
        return ollama_keep_alive()
    except Exception:
        return "30m"


_KEEP_ALIVE = _tiered_keep_alive()


def _budget_num_ctx(kwargs):
    """Requested num_ctx, clamped to the memory governor's ceiling on
    tight machines only. Chat callers clamp upstream (_sticky_chat_num_ctx)
    — this catches the analysis/story/pro callers that default or pass
    32768 explicitly. On >=12 GB the ceiling is 32768 and any requested
    value passes through untouched (today's behavior)."""
    requested = kwargs.get("num_ctx", 32768)
    try:
        from memory_budget import analysis_num_ctx_ceiling
        ceiling = analysis_num_ctx_ceiling()
    except Exception:
        return requested
    if ceiling >= 32768:
        return requested
    return min(requested, ceiling)

# Connection-refused recovery. The bundled Ollama can die mid-batch (an OOM
# on a heavy model load is the usual culprit on lower-RAM machines); the
# Electron supervisor relaunches it on the same port, but that leaves a few
# seconds where the socket is refused. Without this, the in-flight analyze
# call would fail the whole interview with "[Errno 61] Connection refused".
# We retry ONLY connection errors (read timeouts are deliberately left to the
# caller's model-aware timeout) with backoff, bridging the restart window.
_CONNECT_BACKOFF = (2, 4, 6, 8)  # seconds between attempts → ~20s total bridge


def _post_with_reconnect(url, *, json=None, timeout=None, stream=False):
    """``requests.post`` that survives a brief Ollama restart.

    Retries on ``ConnectionError`` only (a refused socket while the supervisor
    relaunches Ollama), never on read timeouts. When Ollama never comes back
    within the backoff budget, raises a TYPED ProviderError — the old raw
    urllib3 re-raise meant the analysis loop couldn't fast-abort and the
    user saw a connection-pool repr instead of "Ollama isn't running".
    Timeouts are likewise wrapped so callers can count and abort on them.
    """
    last_err = None
    for attempt in range(len(_CONNECT_BACKOFF) + 1):
        try:
            return requests.post(url, json=json, timeout=timeout, stream=stream)
        except requests.exceptions.ConnectionError as e:
            last_err = e
            if attempt < len(_CONNECT_BACKOFF):
                time.sleep(_CONNECT_BACKOFF[attempt])
        except requests.exceptions.Timeout as e:
            raise ProviderError(
                "Ollama timed out mid-request — the model may be overloaded "
                "or still loading. Try again in a moment.",
                code="timeout",
            ) from e
    raise ProviderError(
        "Ollama isn't reachable — it may have stopped. Relaunch Doza Assist "
        "to restart it automatically.",
        code="unreachable",
    ) from last_err


def _raise_ollama_error(response, model):
    """Turn a non-200 Ollama response into a typed, user-facing error.

    Returning ``""`` here (the old behavior) made every failure — a deleted
    model, an out-of-memory load, a bad request — indistinguishable from a
    legitimately empty reply. Users saw "Analysis came back empty… try a
    shorter section" when the real problem was "model not installed", and the
    chat stream just went silent. The cloud providers raise typed
    ProviderErrors; this brings Ollama in line.

    ``code`` lets callers react structurally: the analysis chunk loop
    fast-aborts on permanent conditions instead of burning one long timeout
    per chunk, and 'insufficient_memory' matters specifically on 8GB
    machines where the model warm-loads fine at a small num_ctx but OOMs at
    the analysis call's 32768 — so chat works while analysis fails.
    """
    try:
        detail = (response.json().get("error") or "").strip()
    except (ValueError, json.JSONDecodeError):
        detail = (response.text or "").strip()[:300]
    lower = detail.lower()
    if response.status_code == 404 or "not found" in lower:
        raise ProviderError(
            f"Ollama model '{model}' is not installed. "
            f"Open AI Model settings to download it, or pick a different model.",
            code="model_missing",
        )
    if "memory" in lower and ("requires more" in lower or "available" in lower):
        raise ProviderError(
            f"The AI model '{model}' needs more memory than this Mac has free. "
            f"Close other apps, or switch to the smaller gemma4:e2b variant in "
            f"AI Model settings.",
            code="insufficient_memory",
        )
    raise ProviderError(
        f"Ollama error (HTTP {response.status_code}): {detail or 'no details'}",
        code="server_error",
    )


# Reasoning-suppression markers — we don't want models to emit these
# verbatim. All three providers honor stop sequences, so this list is
# portable across backends.
DEFAULT_STOP_TOKENS = [
    "\n[Thoughts]",
    "\n[Thought Process]",
    "\n[Reasoning]",
    "\n<think>",
    "\n<thinking>",
    "[/Response]\n[Thoughts]",
    "[/Response]\n[Thought",
    # NOTE: the prose-like "[No specific answer" / "[No answer" stops were
    # removed — as STOP sequences they hard-cut generation mid-reply and
    # kept the partial text. The placeholder lines they targeted are
    # handled after generation by _strip_no_answer_placeholders, which can
    # tell a placeholder from a substantive negative answer.
]


def _log_timing(tag, model, num_ctx, payload):
    """Log Ollama's per-call timing metrics (returned on every response and
    previously discarded). One line per call, grep-able:

        [ai-timing] tag=chat model=gemma4:e4b num_ctx=16384 \
            prompt_eval=11873tok/9.42s eval=412tok/13.07s

    prompt_eval_count is the definitive prefix-cache oracle: a turn that
    reuses the cached transcript prefix reports hundreds of tokens; a cold
    turn reports the whole payload. Never raises — telemetry must not be
    able to take a reply down.
    """
    try:
        pe_count = payload.get("prompt_eval_count")
        pe_dur = payload.get("prompt_eval_duration")
        ev_count = payload.get("eval_count")
        ev_dur = payload.get("eval_duration")
        if pe_count is None and ev_count is None:
            return
        def _fmt(count, dur_ns):
            secs = (dur_ns or 0) / 1e9
            return f"{count if count is not None else '?'}tok/{secs:.2f}s"
        print(f"[ai-timing] tag={tag} model={model} num_ctx={num_ctx} "
              f"prompt_eval={_fmt(pe_count, pe_dur)} "
              f"eval={_fmt(ev_count, ev_dur)}", flush=True)
    except Exception:
        pass


def _ollama_messages(system_prompt, user_or_messages):
    """Normalize input into Ollama's /api/chat messages array."""
    msgs = [{"role": "system", "content": system_prompt}]
    if isinstance(user_or_messages, list):
        msgs.extend(user_or_messages)
    else:
        msgs.append({"role": "user", "content": str(user_or_messages)})
    return msgs


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self, base_url: str = "", model_resolver=None):
        from ollama_url import ollama_base_url
        self.base_url = (base_url or ollama_base_url()).rstrip("/")
        # ``model_resolver`` is a zero-arg callable returning the Ollama
        # model tag. Injected so this provider stays decoupled from
        # ``model_config`` at import time.
        self._model_resolver = model_resolver

    def _resolve_model(self, override=None):
        if override:
            return override
        if self._model_resolver:
            try:
                resolved = self._model_resolver()
                if resolved:
                    return resolved
            except Exception:
                pass
        # Single source of truth: the Electron wrapper passes the tag it
        # downloaded + blocked startup on (DOZA_OLLAMA_MODEL=gemma4:e4b).
        # Trust it as the last-resort default so the wrapper, not the core,
        # owns which model ships. Gemma 4 only — NEVER Gemma 3.
        return os.environ.get("DOZA_OLLAMA_MODEL") or "gemma4:e4b"

    def generate(self, system_prompt, user_or_messages, task_type="general", **kwargs):
        model = self._resolve_model(kwargs.get("model_override"))

        # Structured-output path: caller passed a single user prompt string
        # and wants JSON-shaped output. /api/generate with format='json'
        # constrains decoding to a valid JSON token tree.
        if task_type == "analysis" and not isinstance(user_or_messages, list):
            # Larger local models (gemma4:26b/31b) can degenerate under the
            # strict format='json' grammar — they emit an empty or ``{}``
            # body and the whole analysis comes back blank ("Analysis
            # incomplete…"). When the caller asks for free-form output
            # (force_json=False) we drop the grammar and add the
            # reasoning-suppression stop tokens; the model then answers
            # normally and the caller's tolerant parser extracts the JSON.
            force_json = kwargs.get("force_json", True)
            payload = {
                "model": model,
                "prompt": str(user_or_messages),
                "system": system_prompt,
                "stream": False,
                "keep_alive": _KEEP_ALIVE,
                "options": {
                    "temperature": kwargs.get("temperature", 0.1),
                    # Bumped from 768 → 4096. The story-analyze schema
                    # (7 beats + 7 soundbites + 5 themes + 5 b-roll +
                    # summary + title) easily needs 1k+ output tokens;
                    # 768 was forcing format='json' to close the JSON
                    # early, producing syntactically valid but mostly
                    # empty dicts. That masked the truncation as
                    # "Gemma returned content-free response" — the
                    # AI Analysis tab would show only a summary or
                    # only social clips with no story beats. 4096
                    # gives every realistic schema room to finish.
                    "num_predict": kwargs.get("num_predict", 4096),
                    # Bumped 12288 → 32768 to match the chat path.
                    # The previous 12288 left only ~8192 input tokens
                    # after num_predict was reserved. With the
                    # storytelling foundation (~7K tokens) prepended
                    # to the system prompt, barely 1K remained for
                    # the transcript — Ollama silently truncated the
                    # overflow and the model hallucinated timecodes.
                    # Even with the foundation now skipped for analysis
                    # calls, a generous context window prevents
                    # truncation on long transcripts (25-min Gemma
                    # chunk ≈ 5K tokens of transcript text alone).
                    "num_ctx": _budget_num_ctx(kwargs),
                    # Classic sampler pin — mirrors the chat paths; the
                    # analysis/story JSON calls are just as exposed to the
                    # runtime/manifest default drift.
                    "top_k": kwargs.get("top_k", 40),
                    "top_p": kwargs.get("top_p", 0.9),
                    "min_p": kwargs.get("min_p", 0.05),
                },
            }
            if force_json:
                payload["format"] = "json"
            else:
                # Free-form fallback: no grammar, so suppress any reasoning
                # preamble the model would otherwise stream before the JSON.
                payload["options"]["stop"] = DEFAULT_STOP_TOKENS

            # Suppress model-side "thinking" on analysis. A reasoning-capable
            # local model (gemma4:26b/31b) otherwise burns its num_predict
            # budget on a hidden reasoning pass and returns an empty/truncated
            # body — the "Analysis incomplete" blank-out, attacked at the
            # source rather than only salvaged by the force_json fallback.
            # `think` is a recent Ollama field; older daemons (and some
            # non-thinking models) reject the request with a non-200, so on
            # failure we retry once without it. That keeps the default
            # gemma4:e4b path identical to before whenever `think` isn't
            # accepted — this can only help, never regress.
            payload["think"] = False
            response = _post_with_reconnect(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=kwargs.get("timeout", 180),
            )
            if response.status_code != 200 and "think" in payload:
                # The think-retry exists for daemons/models that reject the
                # `think` field — NOT for failures the retry can't fix.
                # Re-sending a full 32k-ctx analysis call after an
                # out-of-memory or model-not-found error doubles a load the
                # 8GB machine just proved it can't take.
                try:
                    _detail = (response.json().get("error") or "").lower()
                except (ValueError, json.JSONDecodeError):
                    _detail = ""
                _permanent = (
                    response.status_code == 404
                    or "not found" in _detail
                    or ("memory" in _detail
                        and ("requires more" in _detail or "available" in _detail))
                )
                if _permanent:
                    _raise_ollama_error(response, model)
                payload.pop("think", None)
                response = _post_with_reconnect(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=kwargs.get("timeout", 180),
                )
            if response.status_code != 200:
                _raise_ollama_error(response, model)
            payload = response.json()
            _log_timing(kwargs.get("timing_tag", task_type), model,
                        _budget_num_ctx(kwargs), payload)
            return payload.get("response", "")

        # Chat / general path: /api/chat with messages array.
        messages = _ollama_messages(system_prompt, user_or_messages)
        response = _post_with_reconnect(
            f"{self.base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "keep_alive": _KEEP_ALIVE,
                "options": {
                    "temperature": kwargs.get("temperature", 0.4 if task_type == "chat" else 0.3),
                    "num_predict": kwargs.get("num_predict", 4096 if task_type == "chat" else 16384),
                    "num_ctx": _budget_num_ctx(kwargs),
                    # Chat runs at 1.1 (Ollama's default): the old 1.3 was
                    # added against repetition loops before the server-side
                    # loop guards existed, and at 1.3 the model is punished
                    # for repeating speaker names, timecode digits, and the
                    # verbatim transcript words a grounded answer must copy
                    # — it drifts into vague paraphrase and reaches EOS
                    # early. Non-chat structured tasks keep 1.3.
                    "repeat_penalty": kwargs.get(
                        "repeat_penalty", 1.1 if task_type == "chat" else 1.3),
                    "repeat_last_n": kwargs.get("repeat_last_n", 128),
                    # Classic sampler pin — see module comment above the
                    # stream variant. Runtime/manifest defaults drifted
                    # loose (gemma4 bakes top_k 64/top_p 0.95/min_p 0)
                    # when Ollama 0.31 started honoring model-baked
                    # params; all quality tuning assumed the tight
                    # classic values.
                    "top_k": kwargs.get("top_k", 40),
                    "top_p": kwargs.get("top_p", 0.9),
                    "min_p": kwargs.get("min_p", 0.05),
                    "stop": kwargs.get("stop", DEFAULT_STOP_TOKENS),
                },
            },
            timeout=kwargs.get("timeout", 900 if task_type != "chat" else 300),
        )
        if response.status_code != 200:
            _raise_ollama_error(response, model)
        payload = response.json()
        _log_timing(kwargs.get("timing_tag", task_type), model,
                    _budget_num_ctx(kwargs), payload)
        return (payload.get("message") or {}).get("content", "")

    def generate_stream(self, system_prompt, user_or_messages, task_type="general", **kwargs):
        model = self._resolve_model(kwargs.get("model_override"))
        messages = _ollama_messages(system_prompt, user_or_messages)
        # Tuple-form timeout: (connect, read). The read leg applies between
        # bytes from the server, not to the whole response — so 60s here
        # means "abort if Ollama goes silent for a full minute mid-stream",
        # not "abort after 60s total". The previous single-value 300s was
        # producing the stuck-chat symptom: Ollama could go quiet for up
        # to 5 minutes (model warmup + silent-thinking) without raising,
        # leaving the editor staring at the typing dots. 60s of complete
        # silence on a chat request is well past "something is wrong" on
        # any local model the app supports.
        connect_timeout = kwargs.get("connect_timeout", 15)
        read_timeout = kwargs.get("read_timeout", kwargs.get("timeout", 60))
        # Through the reconnect bridge: a supervisor-relaunched Ollama used
        # to fail streamed chat instantly with a raw ConnectionError while
        # non-stream calls survived the same blip.
        with _post_with_reconnect(
            f"{self.base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "keep_alive": _KEEP_ALIVE,
                "options": {
                    "temperature": kwargs.get("temperature", 0.4),
                    "num_predict": kwargs.get("num_predict", 4096),
                    "num_ctx": _budget_num_ctx(kwargs),
                    # Mirrors the non-stream chat setting — see the comment
                    # there. The stream path is chat-only in practice.
                    "repeat_penalty": kwargs.get(
                        "repeat_penalty", 1.1 if task_type == "chat" else 1.3),
                    "repeat_last_n": kwargs.get("repeat_last_n", 128),
                    "top_k": kwargs.get("top_k", 40),
                    "top_p": kwargs.get("top_p", 0.9),
                    "min_p": kwargs.get("min_p", 0.05),
                    "stop": kwargs.get("stop", DEFAULT_STOP_TOKENS),
                },
            },
            timeout=(connect_timeout, read_timeout),
            stream=True,
        ) as response:
            if response.status_code != 200:
                _raise_ollama_error(response, model)
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except (ValueError, json.JSONDecodeError):
                    continue
                msg = chunk.get("message") or {}
                piece = msg.get("content", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    # The final chunk carries the run's timing metrics —
                    # the only ground truth for prefill (prompt_eval) vs
                    # decode (eval) cost, and the oracle for whether the
                    # KV prefix cache was reused (a warm turn reports a
                    # tiny prompt_eval_count; a cold one reports the whole
                    # payload).
                    _log_timing(kwargs.get("timing_tag", task_type), model,
                                _budget_num_ctx(kwargs), chunk)
                    break

    def test_connection(self) -> dict:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m.get("name") for m in (r.json().get("models") or [])]
                return {"success": True, "models": models}
            return {"success": False, "error": f"Ollama responded {r.status_code}"}
        except requests.exceptions.ConnectionError:
            return {"success": False, "error": f"Ollama not reachable at {self.base_url}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
