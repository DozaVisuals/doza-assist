"""
Data models for Editorial DNA v2.2.

Plain dicts are used at the storage layer (JSON round-trips cleanly) but this
module defines the canonical shape of a StyleProfileSummary and a Snapshot so
callers have a single source of truth.

v2.2 adds the corpus-grounded fields the PRD calls out as the few-shot moat:
``exemplar_passages``, ``quote_archetypes``, and ``language_patterns``. These
get verbatim text from the imported transcripts and are what the injector
prepends to AI calls so the model anchors on real examples instead of abstract
labels.
"""

from datetime import datetime


PROFILE_SCHEMA_VERSION = "2.2"

# Bump this when the analysis prompt or schema changes in a way that would
# materially shift profile output for existing users. The UI surfaces a
# "this profile was analyzed with an older version, regenerate?" banner when
# a stored summary's analysis_version is below this value.
ANALYSIS_VERSION = 2

PLACEHOLDER_NARRATIVE = "Not yet analyzed — click Regenerate analysis."


def empty_style_profile_summary(profile_id, profile_name):
    """Return an empty StyleProfileSummary dict with placeholders.

    Fields that come from the pure-Python metric pipeline are populated after
    a single import. Fields that require an LLM pass over raw transcripts start
    as placeholders and are filled in by
    ``editorial_dna.analysis.generate_structured_summary()``.
    """
    now = datetime.now().isoformat()
    return {
        "profile_id": profile_id,
        "profile_name": profile_name,
        "created_at": now,
        "last_updated": now,
        "analysis_version": ANALYSIS_VERSION,
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
        # PRD calls these "the most important field" — verbatim 30-90s passages
        # the analyzer pulled from the corpus that exemplify the editor's style.
        # Each is {source_filename, passage, why_this_passage}.
        "exemplar_passages": [],
        # Recurring quote archetypes the editor reaches for, with one verbatim
        # example per archetype. Each is {type, description, example_from_corpus}.
        "quote_archetypes": [],
        # What kind of language the editor favors and what they cut around.
        "language_patterns": {
            "favored_phrasing": PLACEHOLDER_NARRATIVE,
            "filtered_out": PLACEHOLDER_NARRATIVE,
        },
        # Free-form notes the user added via "Refine my style". Prepended to
        # the system prompt on top of everything the analyzer produced.
        "user_refinements": "",
    }


def is_placeholder(value):
    """True if a summary field hasn't been filled in by a real analysis pass."""
    return isinstance(value, str) and value == PLACEHOLDER_NARRATIVE
