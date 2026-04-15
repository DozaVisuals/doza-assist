"""
Evolution tracking for Editorial DNA v2.1.

A snapshot is taken every time a profile is analyzed. It stores a deep copy of
the StyleProfileSummary at that moment plus a delta_from_previous block that
describes what changed since the last snapshot in plain language.

Snapshots live at:
    ~/.doza-assist/editorial_dna/profiles/<profile_id>/snapshots/<snapshot_id>.json
"""

import os
import json
import uuid
import copy
from datetime import datetime

from editorial_dna.profiles import _profile_dir, _read_json, _write_json


def _snap_dir(profile_id):
    return os.path.join(_profile_dir(profile_id), 'snapshots')


def list_snapshots(profile_id):
    """Return all snapshots for a profile sorted oldest → newest."""
    sd = _snap_dir(profile_id)
    if not os.path.isdir(sd):
        return []
    snaps = []
    for fname in os.listdir(sd):
        if fname.endswith('.json'):
            snap = _read_json(os.path.join(sd, fname), default=None)
            if snap:
                snaps.append(snap)
    snaps.sort(key=lambda s: s.get('snapshot_date', ''))
    return snaps


def _latest_snapshot(profile_id):
    snaps = list_snapshots(profile_id)
    return snaps[-1] if snaps else None


def create_snapshot(profile_id, note=''):
    """Take a new snapshot of the profile's current summary.

    Computes delta_from_previous against the most recent existing snapshot.
    """
    from editorial_dna.profiles import get_profile
    profile = get_profile(profile_id)
    if profile is None:
        return None

    summary = profile.get('summary') or {}
    source_files = profile.get('source_files') or []

    prev = _latest_snapshot(profile_id)
    delta = _compute_delta(prev.get('summary_json') if prev else None, summary)

    snapshot_id = uuid.uuid4().hex[:12]
    snapshot = {
        'profile_id': profile_id,
        'snapshot_id': snapshot_id,
        'snapshot_date': datetime.now().isoformat(),
        'projects_in_portfolio_at_snapshot': len(source_files),
        'note': note,
        'summary_json': copy.deepcopy(summary),
        'delta_from_previous': delta,
    }
    os.makedirs(_snap_dir(profile_id), exist_ok=True)
    _write_json(os.path.join(_snap_dir(profile_id), f'{snapshot_id}.json'), snapshot)
    return snapshot


def _compute_delta(prev_summary, new_summary):
    """Return a dict describing differences between two StyleProfileSummary dicts.

    First snapshot has no previous, so delta is empty. Missing/placeholder
    fields are ignored — we only report on fields that have real values in
    BOTH snapshots.
    """
    if not prev_summary or not new_summary:
        return {
            'is_first_snapshot': True,
            'changes': [],
        }

    changes = []

    # Numeric deltas
    prev_n = prev_summary.get('narrative_patterns', {}) or {}
    new_n = new_summary.get('narrative_patterns', {}) or {}
    prev_clip = prev_n.get('average_clip_length_seconds') or 0
    new_clip = new_n.get('average_clip_length_seconds') or 0
    avg_clip_change = 0.0
    if prev_clip and new_clip:
        avg_clip_change = round(((new_clip - prev_clip) / prev_clip) * 100, 1)
        if abs(avg_clip_change) >= 5:
            direction = 'longer' if avg_clip_change > 0 else 'shorter'
            changes.append(
                f"Your average clip length got {abs(avg_clip_change):.0f}% {direction}."
            )

    # Opening style change
    prev_open = prev_n.get('opening_style')
    new_open = new_n.get('opening_style')
    opening_changed = (
        prev_open and new_open and prev_open != new_open
        and 'Not yet' not in (prev_open + new_open)
    )
    if opening_changed:
        changes.append(f"Your typical opening shifted from {prev_open} to {new_open}.")

    # Themes — added / removed
    prev_themes = set((prev_summary.get('thematic_patterns', {}) or {}).get('common_themes') or [])
    new_themes = set((new_summary.get('thematic_patterns', {}) or {}).get('common_themes') or [])
    new_detected = sorted(new_themes - prev_themes)
    removed = sorted(prev_themes - new_themes)
    if new_detected:
        if len(new_detected) == 1:
            changes.append(f"A new theme emerged: {new_detected[0]}.")
        else:
            changes.append(f"New themes emerged: {', '.join(new_detected[:3])}.")

    # Tone shift
    prev_tone = (prev_summary.get('voice_characteristics', {}) or {}).get('tone')
    new_tone = (new_summary.get('voice_characteristics', {}) or {}).get('tone')
    tone_shift = None
    if prev_tone and new_tone and prev_tone != new_tone and 'Not yet' not in (prev_tone + new_tone):
        tone_shift = f"{prev_tone} → {new_tone}"
        changes.append(f"Your tone shifted: {tone_shift}.")

    # Projects added
    prev_count = prev_summary.get('projects_analyzed') or 0
    new_count = new_summary.get('projects_analyzed') or 0
    if new_count > prev_count:
        added = new_count - prev_count
        changes.append(f"You added {added} new project{'s' if added != 1 else ''}.")

    return {
        'is_first_snapshot': False,
        'average_clip_length_change': avg_clip_change,
        'new_themes_detected': new_detected,
        'themes_no_longer_present': removed,
        'opening_style_changed': bool(opening_changed),
        'tone_shift': tone_shift,
        'changes': changes,  # plain-language, ready to render
    }
