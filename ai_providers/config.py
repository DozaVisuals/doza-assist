"""Provider config load/save.

Stored at ``$DOZA_DATA_DIR/provider_config.json`` (defaults to
``~/Library/Application Support/DozaAssist/provider_config.json``). Holds
the active provider name plus per-provider settings.

API keys live in the macOS Keychain via the ``keyring`` library, NOT in
the JSON file. ``load_provider_config`` reads the file and hydrates each
provider's ``api_key`` from Keychain so callers see one merged dict.
``save_provider_config`` inverts that: incoming ``api_key`` values are
routed to Keychain and the JSON field is blanked before write.

Migration: a plaintext ``api_key`` found in the JSON (older builds, or a
Keychain-unavailable fallback save) moves to Keychain on the next
successful load and the JSON field is cleared. One-time, silent.

Fallback: if Keychain is unreachable (locked account, headless CI, etc.)
the storage layer drops back to plaintext JSON with a stderr warning so
the app keeps working rather than dying on the first AI call.
"""
import os
import sys
import json
import tempfile

try:
    import keyring
    import keyring.errors
    _KEYRING_AVAILABLE = True
except ImportError:
    keyring = None  # type: ignore[assignment]
    _KEYRING_AVAILABLE = False

DEFAULT_DATA_DIR = os.path.expanduser("~/Library/Application Support/DozaAssist")
CONFIG_FILENAME = "provider_config.json"

# Service name shared across all entries; per-provider username distinguishes
# them. Keep this stable — changing it orphans existing Keychain entries.
_KEYCHAIN_SERVICE = "DozaAssist"

_PROVIDERS_WITH_KEYS = ("anthropic", "openai")

_DEFAULT_CONFIG = {
    "active_provider": "ollama",
    "ollama":    {"model": "", "base_url": "http://localhost:11434"},
    "anthropic": {"api_key": ""},
    "openai":    {"api_key": ""},
}


def _config_path() -> str:
    data_dir = os.environ.get("DOZA_DATA_DIR") or DEFAULT_DATA_DIR
    return os.path.join(data_dir, CONFIG_FILENAME)


def _keychain_username(provider: str) -> str:
    return f"{provider}_api_key"


def _keychain_get(provider: str) -> str:
    """Return the stored key, or '' if absent or Keychain unavailable."""
    if not _KEYRING_AVAILABLE:
        return ""
    try:
        return keyring.get_password(_KEYCHAIN_SERVICE, _keychain_username(provider)) or ""
    except Exception as e:
        print(f"[provider_config] Keychain read failed for {provider}: {e}", file=sys.stderr)
        return ""


def _keychain_set(provider: str, key: str) -> bool:
    """Persist a key. Returns True iff it landed in Keychain."""
    if not _KEYRING_AVAILABLE:
        return False
    try:
        keyring.set_password(_KEYCHAIN_SERVICE, _keychain_username(provider), key)
        return True
    except Exception as e:
        print(f"[provider_config] Keychain write failed for {provider}: {e}", file=sys.stderr)
        return False


def _keychain_delete(provider: str) -> bool:
    """Remove a key. Returns True for both 'deleted' and 'wasn't there';
    False only if Keychain itself is unreachable."""
    if not _KEYRING_AVAILABLE:
        return False
    try:
        keyring.delete_password(_KEYCHAIN_SERVICE, _keychain_username(provider))
        return True
    except keyring.errors.PasswordDeleteError:
        return True  # already absent
    except Exception as e:
        print(f"[provider_config] Keychain delete failed for {provider}: {e}", file=sys.stderr)
        return False


def _atomic_write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".prov-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_provider_config() -> dict:
    """Load config from disk; merge with defaults; hydrate keys from Keychain."""
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

    # One-time migration: any plaintext api_key in the JSON moves to Keychain
    # and gets blanked from the file. Migration only blanks the JSON field
    # after a successful Keychain write, so a Keychain outage leaves the file
    # intact and the app keeps working against the plaintext fallback.
    needs_rewrite = False
    for provider in _PROVIDERS_WITH_KEYS:
        plaintext = (out.get(provider) or {}).get("api_key") or ""
        if plaintext and _keychain_set(provider, plaintext):
            out[provider]["api_key"] = ""
            needs_rewrite = True
    if needs_rewrite:
        try:
            _atomic_write_json(path, out)
        except OSError as e:
            print(f"[provider_config] Failed to rewrite migrated config: {e}", file=sys.stderr)

    # Hydrate from Keychain so callers see api_key as a regular dict field.
    # Keychain returning "" with the JSON also "" is the normal "no key"
    # state. Keychain unreachable + JSON still has plaintext = the fallback
    # path: out[provider]["api_key"] keeps the JSON value untouched.
    for provider in _PROVIDERS_WITH_KEYS:
        kc = _keychain_get(provider)
        if kc:
            out[provider]["api_key"] = kc

    return out


def save_provider_config(config: dict) -> None:
    """Persist config. API keys go to Keychain; the JSON keeps everything else.

    A non-empty api_key in the input is stored in Keychain. An empty string
    means "remove" — the Keychain entry is deleted. Either way the api_key
    field is blanked in the JSON before write so secrets never hit disk.

    If Keychain is unreachable, the api_key falls through to JSON as plaintext
    (matching the pre-migration behavior) and a warning is printed. The next
    successful save migrates it back out.
    """
    cfg = json.loads(json.dumps(config))  # defensive deep copy

    for provider in _PROVIDERS_WITH_KEYS:
        sub = cfg.get(provider)
        if not isinstance(sub, dict) or "api_key" not in sub:
            continue
        key = (sub.get("api_key") or "").strip()
        if key:
            persisted = _keychain_set(provider, key)
        else:
            persisted = _keychain_delete(provider)
        if persisted:
            sub["api_key"] = ""

    _atomic_write_json(_config_path(), cfg)


def has_api_key(provider: str) -> bool:
    """True iff a non-empty key is stored for this provider.

    Checks Keychain first; falls back to ``load_provider_config`` (which
    covers the Keychain-unavailable plaintext fallback case).
    """
    if provider not in _PROVIDERS_WITH_KEYS:
        return False
    if _keychain_get(provider):
        return True
    return bool((load_provider_config().get(provider) or {}).get("api_key"))


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
    for sub in _PROVIDERS_WITH_KEYS:
        if isinstance(out.get(sub), dict) and "api_key" in out[sub]:
            out[sub]["api_key"] = mask_key(out[sub]["api_key"])
    return out
