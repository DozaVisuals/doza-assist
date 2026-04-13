"""
Profile storage for My Style (Editorial DNA Level 1).
Single JSON file at ~/.doza-assist/editorial_dna/profile.json.
"""

import os
import json
from pathlib import Path


PROFILE_DIR = os.path.join(Path.home(), '.doza-assist', 'editorial_dna')
PROFILE_PATH = os.path.join(PROFILE_DIR, 'profile.json')


def load_profile():
    """Load the style profile from disk. Returns dict or None."""
    if not os.path.isfile(PROFILE_PATH):
        return None
    try:
        with open(PROFILE_PATH, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def save_profile(profile):
    """Write the style profile to disk."""
    os.makedirs(PROFILE_DIR, exist_ok=True)
    with open(PROFILE_PATH, 'w') as f:
        json.dump(profile, f, indent=2)


def delete_profile():
    """Remove the profile file."""
    if os.path.isfile(PROFILE_PATH):
        os.remove(PROFILE_PATH)


def export_profile():
    """Return the profile dict for download (same as load)."""
    return load_profile()


def set_active(active):
    """Toggle the active field without touching anything else."""
    profile = load_profile()
    if profile is None:
        return False
    profile['active'] = bool(active)
    save_profile(profile)
    return True


def is_active():
    """Return True if a profile exists and is active."""
    profile = load_profile()
    if profile is None:
        return False
    return profile.get('active', True)
