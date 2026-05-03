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
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": str(user_or_messages),
                    "system": system_prompt,
                    "stream": False,
                    "format": "json",
                    "keep_alive": _KEEP_ALIVE,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.1),
                        "num_predict": kwargs.get("num_predict", 768),
                        "num_ctx": kwargs.get("num_ctx", 12288),
                    },
                },
                timeout=kwargs.get("timeout", 180),
            )
            if response.status_code != 200:
                return ""
            return response.json().get("response", "")

        # Chat / general path: /api/chat with messages array.
        messages = _ollama_messages(system_prompt, user_or_messages)
        response = requests.post(
            f"{self.base_url}/api/chat",
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
            return ""
        return (response.json().get("message") or {}).get("content", "")

    def generate_stream(self, system_prompt, user_or_messages, task_type="general", **kwargs):
        model = self._resolve_model(kwargs.get("model_override"))
        messages = _ollama_messages(system_prompt, user_or_messages)
        with requests.post(
            f"{self.base_url}/api/chat",
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
                return
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
