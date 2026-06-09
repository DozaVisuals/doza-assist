"""Atomic JSON file IO shared across the app.

Every project-state file (meta.json, story_builds.json, segment_vectors.json,
activity.json, …) used to be written with a plain ``open(path, 'w')`` +
``json.dump``. A crash, force-quit, or full disk mid-write leaves a truncated
file behind, and a truncated meta.json used to 500 the dashboard for every
project. Writing to a temp file in the same directory and ``os.replace``-ing
it into place makes the swap atomic on POSIX — readers see either the old
complete file or the new complete file, never a partial one.

This mirrors the pattern already proven in preferences.save_preferences and
ai_providers.config.
"""

import json
import os
import tempfile


def atomic_write_json(path, data, indent=2):
    """Write ``data`` as JSON to ``path`` atomically.

    The temp file is created in the destination directory so os.replace is a
    same-filesystem rename. On any failure the temp file is removed and the
    original file (if any) is left untouched. Raises on unexpected errors so
    callers that need to surface failures still can.
    """
    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix='.tmp-', dir=directory)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_json(path, default=None):
    """Read JSON from ``path``, returning ``default`` when the file is
    missing, unreadable, or corrupt. Tolerant counterpart to
    atomic_write_json for callers that should degrade instead of 500."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return default
