"""
App-wide user preferences for Doza Assist.

Stored at ~/Library/Application Support/Doza Assist/preferences.json.
Currently tracks the user's default editing platform (FCP / Premiere / Resolve).
All IO is best-effort — if the file is unreadable or the directory can't be
created, we fall back to the in-memory default rather than breaking the app.
"""

import os
import json
import tempfile

PREFS_DIR = os.path.expanduser("~/Library/Application Support/Doza Assist")
PREFS_PATH = os.path.join(PREFS_DIR, "preferences.json")

VALID_PLATFORMS = ("fcp", "premiere", "resolve")
DEFAULT_PLATFORM = "fcp"


def load_preferences() -> dict:
    """Load preferences. Returns {} on any error."""
    try:
        with open(PREFS_PATH, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_preferences(prefs: dict) -> bool:
    """Atomic write. Returns True on success, False otherwise."""
    try:
        os.makedirs(PREFS_DIR, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=".prefs-", dir=PREFS_DIR)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(prefs, f, indent=2)
            os.replace(tmp_path, PREFS_PATH)
            return True
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return False
    except OSError:
        return False


def get_default_platform() -> str:
    prefs = load_preferences()
    p = prefs.get("default_platform")
    return p if p in VALID_PLATFORMS else DEFAULT_PLATFORM


def set_default_platform(platform: str) -> bool:
    if platform not in VALID_PLATFORMS:
        return False
    prefs = load_preferences()
    prefs["default_platform"] = platform
    return save_preferences(prefs)
