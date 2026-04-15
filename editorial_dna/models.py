"""
Data models for Editorial DNA v2.1.

Plain dicts are used at the storage layer (JSON round-trips cleanly) but this
module defines the canonical shape of a StyleProfileSummary and a Snapshot so
callers have a single source of truth.

Nothing here is a dataclass — keeping dicts makes export/import trivial and
matches the rest of the app.
"""

from datetime import datetime


PROFILE_SCHEMA_VERSION = "2.1"
PLACEHOLDER_NARRATIVE = "Not yet analyzed — click Regenerate analysis."


def empty_style_profile_summary(profile_id, profile_name):
    """Return an empty StyleProfileSummary dict with placeholders.

    The fields here match the v2.1 spec. Fields that come from v1's pure-Python
    metric pipeline are already populated after analysis. Fields that require an
    LLM pass over raw transcripts start as placeholders and are filled in by
    editorial_dna.analysis.generate_structured_summary().
    """
    now = datetime.now().isoformat()
    return {
        "profile_id": profile_id,
        "profile_name": profile_name,
        "created_at": now,
        "last_updated": now,
        "projects_analyzed": 0,
        "total_runtime_seconds": 0,
        "narrative_patterns": {
            "opening_style": PLACEHOLDER_NARRATIVE,
            "opening_style_confidence": 0.0,
            "average_act_count": 0,
            "average_clip_length_seconds": 0.0,
            "pacing_signature": PLACEHOLDER_NARRATIVE,
            "emotional_arc_shape": PLACEHOLDER_NARRATIVE,
            "resolution_style": PLACEHOLDER_NARRATIVE,
        },
        "thematic_patterns": {
            "common_themes": [],
            "subject_focus": PLACEHOLDER_NARRATIVE,
            "vulnerability_pattern": PLACEHOLDER_NARRATIVE,
        },
        "structural_preferences": {
            "favors_chronological": False,
            "uses_intercutting": False,
            "preferred_position_for_emotional_peak": 0.5,
            "tends_to_open_with": PLACEHOLDER_NARRATIVE,
            "tends_to_close_with": PLACEHOLDER_NARRATIVE,
        },
        "voice_characteristics": {
            "tone": PLACEHOLDER_NARRATIVE,
            "formality_level": PLACEHOLDER_NARRATIVE,
            "uses_narration": False,
            "subject_to_narration_ratio": 1.0,
        },
        # Free-form notes the user has added via "Refine my style". Appended to
        # the system prompt on top of everything the analyzer produced.
        "user_refinements": "",
    }


def is_placeholder(value):
    """True if a summary field hasn't been filled in by a real analysis pass."""
    return isinstance(value, str) and value == PLACEHOLDER_NARRATIVE
