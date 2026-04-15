"""
Multi-profile storage for Editorial DNA v2.1.

Filesystem layout (all JSON, matching the rest of the app):

    ~/.doza-assist/editorial_dna/
        profile.json                       # LEGACY v1, kept for migration
        profiles_index.json                # { active_profile_id, profiles: [...] }
        profiles/
            <profile_id>/
                profile.json               # full profile: v1 metrics + system prompt + user refinements
                summary.json               # StyleProfileSummary (structured, for dashboard)
                system_prompt.txt          # the generated long-form prompt
                source_files.json          # per-file metadata, also holds stored transcripts
                snapshots/
                    <snapshot_id>.json

Every function returns plain dicts. No SQLite, no ORM — this is intentional.
"""

import os
import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path

from editorial_dna.models import (
    PROFILE_SCHEMA_VERSION,
    empty_style_profile_summary,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EDNA_ROOT = os.path.join(Path.home(), '.doza-assist', 'editorial_dna')
LEGACY_PROFILE_PATH = os.path.join(EDNA_ROOT, 'profile.json')
PROFILES_DIR = os.path.join(EDNA_ROOT, 'profiles')
INDEX_PATH = os.path.join(EDNA_ROOT, 'profiles_index.json')


def _profile_dir(profile_id):
    return os.path.join(PROFILES_DIR, profile_id)


def _ensure_dirs(profile_id=None):
    os.makedirs(EDNA_ROOT, exist_ok=True)
    os.makedirs(PROFILES_DIR, exist_ok=True)
    if profile_id:
        pd = _profile_dir(profile_id)
        os.makedirs(pd, exist_ok=True)
        os.makedirs(os.path.join(pd, 'snapshots'), exist_ok=True)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def _load_index():
    if not os.path.isfile(INDEX_PATH):
        return {'active_profile_id': None, 'profiles': []}
    try:
        with open(INDEX_PATH, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {'active_profile_id': None, 'profiles': []}


def _save_index(index):
    _ensure_dirs()
    with open(INDEX_PATH, 'w') as f:
        json.dump(index, f, indent=2)


def _index_entry(profile_id, name, created_at):
    return {'id': profile_id, 'name': name, 'created_at': created_at}


# ---------------------------------------------------------------------------
# Low-level file IO per profile
# ---------------------------------------------------------------------------

def _read_json(path, default=None):
    if not os.path.isfile(path):
        return default
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return default


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def _write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text or '')


def _read_text(path, default=''):
    if not os.path.isfile(path):
        return default
    try:
        with open(path, 'r') as f:
            return f.read()
    except IOError:
        return default


# ---------------------------------------------------------------------------
# Public API: profile lifecycle
# ---------------------------------------------------------------------------

def list_profiles():
    """Return the index list with active flag resolved per entry."""
    _maybe_migrate_legacy()
    index = _load_index()
    active_id = index.get('active_profile_id')
    out = []
    for entry in index.get('profiles', []):
        out.append({
            'id': entry['id'],
            'name': entry['name'],
            'created_at': entry.get('created_at'),
            'is_active': entry['id'] == active_id,
        })
    return out


def get_profile(profile_id):
    """Return the full in-memory representation of a profile, or None."""
    _maybe_migrate_legacy()
    pd = _profile_dir(profile_id)
    if not os.path.isdir(pd):
        return None
    profile = _read_json(os.path.join(pd, 'profile.json'), default=None)
    if profile is None:
        return None
    profile['summary'] = _read_json(os.path.join(pd, 'summary.json'), default={})
    profile['system_prompt'] = _read_text(os.path.join(pd, 'system_prompt.txt'))
    profile['source_files'] = _read_json(
        os.path.join(pd, 'source_files.json'), default=[]
    )
    profile['id'] = profile_id
    return profile


def get_active_profile():
    """Return the currently active profile dict or None.

    Respects the legacy `active` boolean — if the active profile is explicitly
    toggled off, this returns None so the injector skips it (matches v1
    semantics of the master toggle).
    """
    _maybe_migrate_legacy()
    index = _load_index()
    active_id = index.get('active_profile_id')
    if not active_id:
        return None
    profile = get_profile(active_id)
    if profile is None:
        return None
    if not profile.get('active', True):
        return None
    return profile


def get_active_profile_id():
    _maybe_migrate_legacy()
    return _load_index().get('active_profile_id')


def set_active(profile_id):
    """Make profile_id the active one."""
    index = _load_index()
    if profile_id is not None and not any(e['id'] == profile_id for e in index.get('profiles', [])):
        return False
    index['active_profile_id'] = profile_id
    _save_index(index)
    return True


def create_profile(name, description=''):
    """Create an empty profile folder + index entry. Does NOT activate it."""
    _ensure_dirs()
    profile_id = uuid.uuid4().hex[:12]
    now = datetime.now().isoformat()
    _ensure_dirs(profile_id)

    profile = {
        'profile_version': PROFILE_SCHEMA_VERSION,
        'feature_name': 'My Style',
        'name': name,
        'description': description,
        'created_at': now,
        'updated_at': now,
        'active': True,  # toggle state within the profile
        # v1 metric fields (populated by importer)
        'speech_pacing': {},
        'structural_rhythm': {},
        'soundbite_craft': {},
        'story_shape': {},
        'content_patterns': {},
        'natural_language_summary': '',
    }
    _write_json(os.path.join(_profile_dir(profile_id), 'profile.json'), profile)
    _write_json(os.path.join(_profile_dir(profile_id), 'summary.json'),
                empty_style_profile_summary(profile_id, name))
    _write_text(os.path.join(_profile_dir(profile_id), 'system_prompt.txt'), '')
    _write_json(os.path.join(_profile_dir(profile_id), 'source_files.json'), [])

    index = _load_index()
    index.setdefault('profiles', []).append(_index_entry(profile_id, name, now))
    if not index.get('active_profile_id'):
        index['active_profile_id'] = profile_id
    _save_index(index)
    return profile_id


def save_profile(profile_id, profile_dict):
    """Persist the metric/prompt portion of a profile back to disk.

    profile_dict is expected to have the v1 metric fields at the top level
    (speech_pacing, structural_rhythm, etc.). The auxiliary files (summary,
    system_prompt, source_files) are written via save_summary/save_system_prompt/
    save_source_files.
    """
    pd = _profile_dir(profile_id)
    if not os.path.isdir(pd):
        return False
    # Don't let callers clobber the name/description silently
    existing = _read_json(os.path.join(pd, 'profile.json'), default={}) or {}
    merged = {**existing, **profile_dict}
    merged['profile_version'] = PROFILE_SCHEMA_VERSION
    merged['updated_at'] = datetime.now().isoformat()
    # Strip the ancillary keys get_profile() attached
    for k in ('summary', 'system_prompt', 'source_files', 'id'):
        merged.pop(k, None)
    _write_json(os.path.join(pd, 'profile.json'), merged)
    return True


def save_summary(profile_id, summary_dict):
    pd = _profile_dir(profile_id)
    if not os.path.isdir(pd):
        return False
    summary_dict['last_updated'] = datetime.now().isoformat()
    _write_json(os.path.join(pd, 'summary.json'), summary_dict)
    return True


def save_system_prompt(profile_id, text):
    pd = _profile_dir(profile_id)
    if not os.path.isdir(pd):
        return False
    _write_text(os.path.join(pd, 'system_prompt.txt'), text or '')
    return True


def save_source_files(profile_id, source_files):
    pd = _profile_dir(profile_id)
    if not os.path.isdir(pd):
        return False
    _write_json(os.path.join(pd, 'source_files.json'), source_files or [])
    return True


def rename_profile(profile_id, new_name):
    pd = _profile_dir(profile_id)
    if not os.path.isdir(pd):
        return False
    profile = _read_json(os.path.join(pd, 'profile.json'), default={}) or {}
    profile['name'] = new_name
    profile['updated_at'] = datetime.now().isoformat()
    _write_json(os.path.join(pd, 'profile.json'), profile)

    summary = _read_json(os.path.join(pd, 'summary.json'), default={}) or {}
    summary['profile_name'] = new_name
    _write_json(os.path.join(pd, 'summary.json'), summary)

    index = _load_index()
    for e in index.get('profiles', []):
        if e['id'] == profile_id:
            e['name'] = new_name
    _save_index(index)
    return True


def set_profile_active_toggle(profile_id, active):
    """Toggle the within-profile on/off flag (matches v1 master toggle semantics)."""
    pd = _profile_dir(profile_id)
    if not os.path.isdir(pd):
        return False
    profile = _read_json(os.path.join(pd, 'profile.json'), default={}) or {}
    profile['active'] = bool(active)
    _write_json(os.path.join(pd, 'profile.json'), profile)
    return True


def delete_profile(profile_id):
    """Remove a profile folder + index entry. If it was active, clear active."""
    pd = _profile_dir(profile_id)
    if os.path.isdir(pd):
        shutil.rmtree(pd, ignore_errors=True)
    index = _load_index()
    index['profiles'] = [e for e in index.get('profiles', []) if e['id'] != profile_id]
    if index.get('active_profile_id') == profile_id:
        index['active_profile_id'] = (index['profiles'][0]['id']
                                      if index['profiles'] else None)
    _save_index(index)
    return True


def export_all():
    """Return a single dict with every profile and its summary/snapshots.

    This is what the 'Export my profile data' button sends as a JSON download.
    """
    _maybe_migrate_legacy()
    index = _load_index()
    profiles = []
    for entry in index.get('profiles', []):
        pid = entry['id']
        pd = _profile_dir(pid)
        snaps = []
        snap_dir = os.path.join(pd, 'snapshots')
        if os.path.isdir(snap_dir):
            for fname in sorted(os.listdir(snap_dir)):
                if fname.endswith('.json'):
                    snap = _read_json(os.path.join(snap_dir, fname), default=None)
                    if snap:
                        snaps.append(snap)
        profiles.append({
            'id': pid,
            'name': entry['name'],
            'profile': _read_json(os.path.join(pd, 'profile.json'), default={}),
            'summary': _read_json(os.path.join(pd, 'summary.json'), default={}),
            'system_prompt': _read_text(os.path.join(pd, 'system_prompt.txt')),
            'source_files': _read_json(os.path.join(pd, 'source_files.json'), default=[]),
            'snapshots': snaps,
        })
    return {
        'schema_version': PROFILE_SCHEMA_VERSION,
        'exported_at': datetime.now().isoformat(),
        'active_profile_id': index.get('active_profile_id'),
        'profiles': profiles,
    }


def import_bundle(bundle, overwrite=False):
    """Restore from an export_all() dict. Returns list of imported profile ids.

    If overwrite=False, profiles whose IDs already exist are renamed with a
    suffix rather than clobbered. The user's existing data is never silently
    overwritten.
    """
    imported = []
    existing_ids = {p['id'] for p in list_profiles()}
    for entry in bundle.get('profiles', []):
        pid = entry.get('id') or uuid.uuid4().hex[:12]
        if pid in existing_ids and not overwrite:
            pid = uuid.uuid4().hex[:12]
        _ensure_dirs(pid)
        pd = _profile_dir(pid)
        _write_json(os.path.join(pd, 'profile.json'), entry.get('profile', {}))
        _write_json(os.path.join(pd, 'summary.json'), entry.get('summary', {}))
        _write_text(os.path.join(pd, 'system_prompt.txt'), entry.get('system_prompt', ''))
        _write_json(os.path.join(pd, 'source_files.json'), entry.get('source_files', []))
        for snap in entry.get('snapshots', []):
            sid = snap.get('snapshot_id') or uuid.uuid4().hex[:12]
            _write_json(os.path.join(pd, 'snapshots', f'{sid}.json'), snap)

        index = _load_index()
        if not any(e['id'] == pid for e in index.get('profiles', [])):
            index.setdefault('profiles', []).append(
                _index_entry(pid, entry.get('name', 'Imported Profile'),
                             entry.get('profile', {}).get('created_at')
                             or datetime.now().isoformat())
            )
        if not index.get('active_profile_id'):
            index['active_profile_id'] = pid
        _save_index(index)
        imported.append(pid)
    return imported


# ---------------------------------------------------------------------------
# Migration from v1 (single-profile.json)
# ---------------------------------------------------------------------------

def _maybe_migrate_legacy():
    """If a v1 profile.json exists and no v2.1 index exists, convert it.

    This runs on demand (every public read goes through it) so the first time
    a v1 user loads the app after updating, their data is seamlessly promoted.
    """
    if os.path.isfile(INDEX_PATH):
        return  # already migrated
    if not os.path.isfile(LEGACY_PROFILE_PATH):
        # First-time user: just write an empty index so we don't re-check every call
        _save_index({'active_profile_id': None, 'profiles': []})
        return

    try:
        with open(LEGACY_PROFILE_PATH, 'r') as f:
            legacy = json.load(f)
    except (json.JSONDecodeError, IOError):
        _save_index({'active_profile_id': None, 'profiles': []})
        return

    # Promote the legacy profile into a v2.1 folder called "My Style"
    profile_id = uuid.uuid4().hex[:12]
    _ensure_dirs(profile_id)
    pd = _profile_dir(profile_id)
    now = datetime.now().isoformat()
    name = 'My Style'

    v2_profile = {
        'profile_version': PROFILE_SCHEMA_VERSION,
        'feature_name': 'My Style',
        'name': name,
        'description': 'Migrated from Editorial DNA v1.',
        'created_at': legacy.get('created_at', now),
        'updated_at': now,
        'active': legacy.get('active', True),
        'speech_pacing': legacy.get('speech_pacing', {}),
        'structural_rhythm': legacy.get('structural_rhythm', {}),
        'soundbite_craft': legacy.get('soundbite_craft', {}),
        'story_shape': legacy.get('story_shape', {}),
        'content_patterns': legacy.get('content_patterns', {}),
        'natural_language_summary': legacy.get('natural_language_summary', ''),
    }
    _write_json(os.path.join(pd, 'profile.json'), v2_profile)

    # Build a StyleProfileSummary that fills in everything we CAN from v1 data
    # and leaves the rest as placeholders for the user to regenerate.
    summary = empty_style_profile_summary(profile_id, name)
    summary['created_at'] = v2_profile['created_at']
    summary['projects_analyzed'] = len(legacy.get('source_files', []))
    summary['total_runtime_seconds'] = int(sum(
        sf.get('duration_seconds', 0) for sf in legacy.get('source_files', [])
    ))
    # Populate what v1 actually knows:
    ss = legacy.get('story_shape', {}) or {}
    if ss.get('opening_style'):
        summary['narrative_patterns']['opening_style'] = ss.get('opening_style')
        summary['structural_preferences']['tends_to_open_with'] = ss.get('opening_style')
    if ss.get('closing_style'):
        summary['narrative_patterns']['resolution_style'] = ss.get('closing_style')
        summary['structural_preferences']['tends_to_close_with'] = ss.get('closing_style')
    sr = legacy.get('structural_rhythm', {}) or {}
    if sr.get('energy_arc'):
        summary['narrative_patterns']['emotional_arc_shape'] = sr['energy_arc']
    sp = legacy.get('speech_pacing', {}) or {}
    if sp.get('rhythm_descriptor'):
        summary['narrative_patterns']['pacing_signature'] = sp['rhythm_descriptor']
    sc = legacy.get('soundbite_craft', {}) or {}
    if sc.get('avg_soundbite_length'):
        summary['narrative_patterns']['average_clip_length_seconds'] = \
            round(sc['avg_soundbite_length'], 2)
    # v1 doesn't know narrative opening style confidence, themes, tone, etc.
    # Those stay as PLACEHOLDER_NARRATIVE for the user to re-run.

    _write_json(os.path.join(pd, 'summary.json'), summary)

    # The "system prompt" for a v1 profile is essentially the natural language
    # summary plus the style block the injector builds from metrics. We'll
    # write the natural language summary here so the new injector has something
    # to load even without re-running analysis.
    _write_text(os.path.join(pd, 'system_prompt.txt'),
                legacy.get('natural_language_summary', ''))

    _write_json(os.path.join(pd, 'source_files.json'),
                legacy.get('source_files', []))

    index = {
        'active_profile_id': profile_id,
        'profiles': [_index_entry(profile_id, name, v2_profile['created_at'])],
    }
    _save_index(index)

    # Take an initial snapshot so evolution tracking has a starting point
    try:
        from editorial_dna.snapshots import create_snapshot
        create_snapshot(profile_id, note='Migrated from Editorial DNA v1')
    except Exception as e:
        print(f"[edna] initial snapshot on migration failed: {e}")

    # Keep the legacy file in place but rename it so load_profile() in
    # storage.py no longer finds it (belt-and-suspenders).
    try:
        os.rename(LEGACY_PROFILE_PATH, LEGACY_PROFILE_PATH + '.migrated')
    except OSError:
        pass
