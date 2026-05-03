"""Provider abstraction layer — factory + active-provider lookup.

All AI calls in the codebase route through ``get_active_provider()`` so
swapping between local Ollama and a cloud API is a config flip with no
call-site changes.
"""
from .base import BaseProvider
from .config import (
    load_provider_config,
    save_provider_config,
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
    "mask_key",
    "masked_config",
]


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
        return OllamaProvider(
            base_url=base_url or "http://localhost:11434",
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
    """Construct a provider instance from the saved config.

    ``model_resolver`` is an optional zero-arg callable returning the Ollama
    model tag — passed through to ``OllamaProvider`` so it stays decoupled
    from ``model_config`` at import time.
    """
    cfg = load_provider_config()
    name = cfg.get("active_provider") or "ollama"
    sub = cfg.get(name) or {}
    return get_provider(
        name,
        api_key=sub.get("api_key") or "",
        base_url=sub.get("base_url") or "",
        model_resolver=model_resolver,
    )
