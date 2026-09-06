"""Provider abstraction layer: factory + active-provider lookup.

All AI calls in the codebase route through ``get_active_provider()``. Which
provider answers is decided per project (``meta['ai_provider']``) and carried
to the call site through a context variable:

- request-scoped work (chat, story builds, quote sheets, speaker naming on
  demand) gets it from the app's before_request hook, which reads the
  project named in the URL;
- background jobs (analysis, story briefs, collection builds, deferred
  speaker naming) read the project's meta at job start and set it inside the
  worker thread, because a new thread starts with an empty context.

With nothing set, the answer is local Ollama. ``provider_config.json`` keeps
API keys and endpoints only; its ``active_provider`` field is left in place
for older builds but nothing here reads it.
"""
import contextvars
from contextlib import contextmanager

from .base import BaseProvider
from .config import (
    load_provider_config,
    save_provider_config,
    has_api_key,
    mask_key,
    masked_config,
)


class ProviderError(RuntimeError):
    """User-facing AI provider error.

    Routes catch this and return a clean message + ``settings_url`` so the
    UI can show a clickable link to /settings instead of dumping a stack
    trace. Mid-stream consumers (the SSE chat layer) catch it inside the
    generator and yield a friendly error event.

    ``code`` is one of:
      - ``"missing_key"``  — provider needs a key and none is configured
      - ``"invalid_key"``  — saved key was rejected by the API (auth failure)
      - ``"rate_limited"`` — repeated rate-limit response
      - ``"unreachable"``  — provider endpoint not reachable
      - ``""`` (default)   — generic provider error
    """

    def __init__(self, message: str, code: str = ""):
        super().__init__(message)
        self.code = code


__all__ = [
    "BaseProvider",
    "ProviderError",
    "get_provider",
    "get_active_provider",
    "load_provider_config",
    "save_provider_config",
    "has_api_key",
    "mask_key",
    "masked_config",
    "PROVIDER_NAMES",
    "LOCAL_PROVIDER",
    "normalize_provider_name",
    "provider_for_project",
    "current_provider_name",
    "set_active_provider_name",
    "reset_active_provider_name",
    "clear_active_provider_name",
    "using_provider",
    "using_project_provider",
]

PROVIDER_NAMES = ("ollama", "anthropic", "openai")
LOCAL_PROVIDER = "ollama"

# The provider for the work in progress on this thread. Unset means local.
_ACTIVE_PROVIDER_NAME = contextvars.ContextVar("doza_active_provider", default=None)


def normalize_provider_name(name) -> str:
    """A known provider name, or local when the value is missing or unknown."""
    name = (name or "").strip().lower() if isinstance(name, str) else ""
    return name if name in PROVIDER_NAMES else LOCAL_PROVIDER


def provider_for_project(meta) -> str:
    """Rule 1: a project's provider is its own ``ai_provider`` field, and a
    project without one is local. No other project and no app-wide setting
    has a say."""
    if not isinstance(meta, dict):
        return LOCAL_PROVIDER
    return normalize_provider_name(meta.get("ai_provider"))


def current_provider_name() -> str:
    """The provider the current context resolved to (local when unset)."""
    return _ACTIVE_PROVIDER_NAME.get() or LOCAL_PROVIDER


def set_active_provider_name(name):
    """Set the provider for the current context; returns a reset token."""
    return _ACTIVE_PROVIDER_NAME.set(normalize_provider_name(name))


def reset_active_provider_name(token) -> None:
    try:
        _ACTIVE_PROVIDER_NAME.reset(token)
    except (ValueError, LookupError):
        # A token from another context (streamed responses finish in a
        # different frame than the one that set it): fall back to clearing.
        _ACTIVE_PROVIDER_NAME.set(None)


def clear_active_provider_name() -> None:
    """Back to local for this context. Request teardown uses this rather
    than a token: Flask may run the hooks in different context copies."""
    _ACTIVE_PROVIDER_NAME.set(None)


@contextmanager
def using_provider(name):
    token = set_active_provider_name(name)
    try:
        yield normalize_provider_name(name)
    finally:
        reset_active_provider_name(token)


@contextmanager
def using_project_provider(meta):
    """Run a block with the provider a project's meta names (local when the
    field is missing). For background workers: call at job start, inside the
    worker thread."""
    with using_provider(provider_for_project(meta)) as name:
        yield name


def get_provider(
    name: str,
    *,
    api_key: str = "",
    base_url: str = "",
    model_resolver=None,
) -> BaseProvider:
    """Build a fresh provider instance by name. Raises on unknown name."""
    if name == "ollama":
        from .ollama_provider import OllamaProvider
        from ollama_url import ollama_base_url
        return OllamaProvider(
            base_url=base_url or ollama_base_url(),
            model_resolver=model_resolver,
        )
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(api_key=api_key)
    if name == "openai":
        from .openai_provider import OpenAIProvider
        return OpenAIProvider(api_key=api_key)
    raise ValueError(f"Unknown provider: {name!r}")


def get_active_provider(model_resolver=None) -> BaseProvider:
    """Construct the provider for the work in progress.

    The name comes from the context variable set per project (see the module
    docstring); nothing is read from ``provider_config.json`` except the
    chosen provider's key and endpoint. With no context set this is local
    Ollama, so a call path that forgets to set the project's provider can
    only ever fall back to local, never to a cloud API.

    ``model_resolver`` is an optional zero-arg callable returning the Ollama
    model tag, passed through to ``OllamaProvider`` so it stays decoupled
    from ``model_config`` at import time.
    """
    cfg = load_provider_config()
    name = current_provider_name()
    sub = cfg.get(name) or {}
    base = sub.get("base_url") or ""
    # Legacy/default literal — fall through to the wrapper's OLLAMA_HOST so the
    # bundled Ollama (dynamic port) is reached, not the unbundled default :11434.
    if base in ("http://localhost:11434", "http://127.0.0.1:11434"):
        base = ""
    return get_provider(
        name,
        api_key=sub.get("api_key") or "",
        base_url=base,
        model_resolver=model_resolver,
    )
