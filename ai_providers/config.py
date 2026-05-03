"""Provider config load/save.

Stored at ``$DOZA_DATA_DIR/provider_config.json`` (defaults to
``~/Library/Application Support/DozaAssist/provider_config.json``). Contains
the active provider name plus per-provider settings, including any API keys.

The file lives on the user's machine and is never transmitted off-box. API
keys have the same security posture as Ollama's local config — they sit in
the user's local data dir and never leave it except in calls to the
respective provider's API endpoint.

Note: preferences.py uses 'Doza Assist' (with space), everything else uses
'DozaAssist'. Unify later.
"""
import os
import json
import tempfile

DEFAULT_DATA_DIR = os.path.expanduser("~/Library/Application Support/DozaAssist")
CONFIG_FILENAME = "provider_config.json"

_DEFAULT_CONFIG = {
    "active_provider": "ollama",
    "ollama":    {"model": "", "base_url": "http://localhost:11434"},
    "anthropic": {"api_key": ""},
    "openai":    {"api_key": ""},
}


def _config_path() -> str:
    data_dir = os.environ.get("DOZA_DATA_DIR") or DEFAULT_DATA_DIR
    return os.path.join(data_dir, CONFIG_FILENAME)


def load_provider_config() -> dict:
    """Load config from disk; merge with defaults; return."""
    path = _config_path()
    cfg = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                cfg = loaded
        except (OSError, json.JSONDecodeError):
            cfg = {}

    out = dict(_DEFAULT_CONFIG)
    if cfg.get("active_provider") in ("ollama", "anthropic", "openai"):
        out["active_provider"] = cfg["active_provider"]
    for sub in ("ollama", "anthropic", "openai"):
        out[sub] = {**_DEFAULT_CONFIG[sub], **(cfg.get(sub) or {})}
    return out


def save_provider_config(config: dict) -> None:
    """Atomic write to the config path."""
    path = _config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".prov-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(config, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def mask_key(key: str) -> str:
    """Display-safe representation: 'sk-ant...a1b2', or '' for empty."""
    if not key:
        return ""
    if len(key) <= 10:
        return "***"
    return f"{key[:6]}...{key[-4:]}"


def masked_config(cfg: dict) -> dict:
    """Return a copy of ``cfg`` with API keys replaced by their masked form."""
    out = json.loads(json.dumps(cfg))
    for sub in ("anthropic", "openai"):
        if isinstance(out.get(sub), dict) and "api_key" in out[sub]:
            out[sub]["api_key"] = mask_key(out[sub]["api_key"])
    return out
