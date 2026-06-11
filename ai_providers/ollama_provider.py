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
import requests

from .base import BaseProvider
from . import ProviderError


_KEEP_ALIVE = "30m"

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
    "[No specific answer",
    "[No answer",
]


def _raise_ollama_error(response, model):
    """Turn a non-200 Ollama response into a typed, user-facing error.

    Returning ``""`` here (the old behavior) made every failure — a deleted
    model, an out-of-memory load, a bad request — indistinguishable from a
    legitimately empty reply. Users saw "Analysis came back empty… try a
    shorter section" when the real problem was "model not installed", and
    the chat stream just went silent. The cloud providers raise typed
    ProviderErrors; this brings Ollama in line.

    ``code`` lets callers react structurally: the analysis chunk loop
    fast-aborts on permanent conditions instead of burning one 300s timeout
    per chunk, and 'insufficient_memory' matters specifically on 8GB
    machines where the model warm-loads fine at a small num_ctx but OOMs at
    the analysis call's 32768 — so chat works while analysis fails
    (issue #39).
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


def _post(url, *, json=None, timeout=None, stream=False, base_url=''):
    """requests.post that converts transport failures into typed
    ProviderErrors. A refused socket used to surface as a raw urllib3 repr
    (or burn the full per-call timeout once per analysis chunk); the typed
    codes let the analysis loop fast-abort and the UI name the real fix."""
    try:
        return requests.post(url, json=json, timeout=timeout, stream=stream)
    except requests.exceptions.ConnectionError as e:
        raise ProviderError(
            f"Ollama isn't reachable at {base_url or url} — is Ollama "
            f"running? Relaunch Doza Assist to start it automatically.",
            code="unreachable",
        ) from e
    except requests.exceptions.Timeout as e:
        raise ProviderError(
            "Ollama timed out mid-request — the model may be overloaded or "
            "still loading. Try again in a moment.",
            code="timeout",
        ) from e


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

    def __init__(self, base_url: str = "http://localhost:11434", model_resolver=None):
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        # ``model_resolver`` is a zero-arg callable returning the Ollama
        # model tag. Injected so this provider stays decoupled from
        # ``model_config`` at import time.
        self._model_resolver = model_resolver

    def _resolve_model(self, override=None):
        if override:
            return override
        if self._model_resolver:
            try:
                return self._model_resolver()
            except Exception:
                pass
        return "gemma3:4b"  # last-resort default

    def generate(self, system_prompt, user_or_messages, task_type="general", **kwargs):
        model = self._resolve_model(kwargs.get("model_override"))

        # Structured-output path: caller passed a single user prompt string
        # and wants JSON-shaped output. /api/generate with format='json'
        # constrains decoding to a valid JSON token tree.
        if task_type == "analysis" and not isinstance(user_or_messages, list):
            response = _post(
                f"{self.base_url}/api/generate",
                base_url=self.base_url,
                json={
                    "model": model,
                    "prompt": str(user_or_messages),
                    "system": system_prompt,
                    "stream": False,
                    "format": "json",
                    "keep_alive": _KEEP_ALIVE,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.1),
                        # Bumped 768 → 4096 → 8192. The story-analyze schema asks
                        # for up to 7 items each carrying a VERBATIM transcript
                        # quote; in a token-verbose language (German runs ~40-50%
                        # longer than English for the same words) the JSON routinely
                        # ran past 4096 output tokens, so format='json' returned
                        # done_reason='length' with the object cut mid-element and
                        # the analysis came back empty (the German "produced no
                        # results" report). The three-pass split keeps each call's
                        # schema small, so a higher ceiling rarely adds latency on
                        # normal inputs but removes the truncation trigger.
                        "num_predict": kwargs.get("num_predict", 8192),
                        # Bumped 12288 → 32768. Generous context window
                        # prevents transcript truncation on long interviews.
                        "num_ctx": kwargs.get("num_ctx", 32768),
                    },
                },
                timeout=kwargs.get("timeout", 300),
            )
            if response.status_code != 200:
                _raise_ollama_error(response, model)
            return response.json().get("response", "")

        # Chat / general path: /api/chat with messages array.
        messages = _ollama_messages(system_prompt, user_or_messages)
        response = _post(
            f"{self.base_url}/api/chat",
            base_url=self.base_url,
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "keep_alive": _KEEP_ALIVE,
                "options": {
                    "temperature": kwargs.get("temperature", 0.4 if task_type == "chat" else 0.3),
                    "num_predict": kwargs.get("num_predict", 4096 if task_type == "chat" else 16384),
                    "num_ctx": kwargs.get("num_ctx", 32768),
                    "repeat_penalty": kwargs.get("repeat_penalty", 1.3),
                    "repeat_last_n": kwargs.get("repeat_last_n", 128),
                    "stop": kwargs.get("stop", DEFAULT_STOP_TOKENS),
                },
            },
            timeout=kwargs.get("timeout", 900 if task_type != "chat" else 300),
        )
        if response.status_code != 200:
            _raise_ollama_error(response, model)
        return (response.json().get("message") or {}).get("content", "")

    def generate_stream(self, system_prompt, user_or_messages, task_type="general", **kwargs):
        model = self._resolve_model(kwargs.get("model_override"))
        messages = _ollama_messages(system_prompt, user_or_messages)
        with _post(
            f"{self.base_url}/api/chat",
            base_url=self.base_url,
            json={
                "model": model,
                "messages": messages,
                "stream": True,
                "keep_alive": _KEEP_ALIVE,
                "options": {
                    "temperature": kwargs.get("temperature", 0.4),
                    "num_predict": kwargs.get("num_predict", 4096),
                    "num_ctx": kwargs.get("num_ctx", 32768),
                    "repeat_penalty": kwargs.get("repeat_penalty", 1.3),
                    "repeat_last_n": kwargs.get("repeat_last_n", 128),
                    "stop": kwargs.get("stop", DEFAULT_STOP_TOKENS),
                },
            },
            timeout=kwargs.get("timeout", 300),
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
