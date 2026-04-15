"""
Editorial DNA v2.1 structured analysis pass.

This module builds two outputs from a profile's transcripts + metrics:

  1. A long-form system prompt (the thing that gets injected into Story Builder
     and AI Chat). Reuses editorial_dna.summarizer for the natural-language
     portion, then layers the user refinements on top.

  2. A StyleProfileSummary dict (the structured object the dashboard renders).
     Narrative/thematic fields are filled by asking the LLM to output JSON.
     Fields already present in v1 metrics (pacing, opening/closing) are passed
     through directly. If the LLM call fails, fields are left as placeholders —
     the analyzer is always best-effort, never blocking.

The analyzer is intentionally tolerant: bad JSON from the model falls back to
keeping existing values, so a broken LLM response can never nuke a profile.
"""

import json
import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_analysis import _call_ai  # noqa: E402

from editorial_dna.models import empty_style_profile_summary, PLACEHOLDER_NARRATIVE
from editorial_dna.summarizer import generate_summary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_structured_summary(profile_id, profile_name, metrics_profile, source_files,
                                 transcripts_text='', existing_summary=None):
    """Build a StyleProfileSummary dict for this profile.

    metrics_profile : dict — v1-shaped metric profile (speech_pacing, etc.)
    source_files    : list — from source_files.json
    transcripts_text: str  — concatenated raw transcript text from the source
                             files (may be empty for migrated v1 profiles)
    existing_summary: dict or None — previous summary, used as fallback for
                             fields the LLM can't answer confidently

    Always returns a full summary dict. Never raises.
    """
    # Start from an empty summary + whatever was already there
    summary = empty_style_profile_summary(profile_id, profile_name)
    if existing_summary:
        # Preserve user_refinements and any fields the caller already filled in
        summary['user_refinements'] = existing_summary.get('user_refinements', '')

    summary['projects_analyzed'] = len(source_files or [])
    summary['total_runtime_seconds'] = int(sum(
        sf.get('duration_seconds', 0) for sf in (source_files or [])
    ))

    # --- Pass-through from v1 metrics (these are authoritative, always correct)
    sp = (metrics_profile or {}).get('speech_pacing', {}) or {}
    sr = (metrics_profile or {}).get('structural_rhythm', {}) or {}
    sc = (metrics_profile or {}).get('soundbite_craft', {}) or {}
    ss = (metrics_profile or {}).get('story_shape', {}) or {}

    if ss.get('opening_style'):
        summary['narrative_patterns']['opening_style'] = ss['opening_style']
        summary['structural_preferences']['tends_to_open_with'] = ss['opening_style']
    if ss.get('closing_style'):
        summary['narrative_patterns']['resolution_style'] = ss['closing_style']
        summary['structural_preferences']['tends_to_close_with'] = ss['closing_style']
    if sp.get('rhythm_descriptor'):
        summary['narrative_patterns']['pacing_signature'] = sp['rhythm_descriptor']
    if sr.get('energy_arc'):
        summary['narrative_patterns']['emotional_arc_shape'] = sr['energy_arc']
    if sc.get('avg_soundbite_length'):
        summary['narrative_patterns']['average_clip_length_seconds'] = round(
            sc['avg_soundbite_length'], 2
        )

    # Rough heuristic: longest beat position gives a preferred peak location
    peak_pos_map = {'first_third': 0.25, 'middle': 0.5, 'last_third': 0.8}
    summary['structural_preferences']['preferred_position_for_emotional_peak'] = \
        peak_pos_map.get(sr.get('longest_beat_position', 'middle'), 0.5)

    # --- LLM pass for narrative / thematic / voice fields
    # Only run if we actually have transcript text to analyze
    if transcripts_text and transcripts_text.strip():
        llm_fields = _call_llm_for_narrative_fields(transcripts_text, metrics_profile)
        if llm_fields:
            _merge_llm_fields(summary, llm_fields)

    return summary


def generate_system_prompt(profile_id, profile_name, metrics_profile, summary):
    """Build the long-form system prompt that gets injected into AI calls.

    Combines:
      - The v1 natural language summary (editorial_dna.summarizer)
      - The structured narrative/thematic/voice fields that have been filled
      - Any user refinements the user typed in the "Refine my style" box

    Returns a string.
    """
    # Start with the v1 summarizer's prose if we have metrics
    nl_summary = ''
    if metrics_profile and metrics_profile.get('speech_pacing'):
        try:
            nl_summary = generate_summary(metrics_profile, transcripts_text='')
        except Exception as e:
            print(f"[edna] generate_summary failed: {e}")
            nl_summary = (metrics_profile.get('natural_language_summary') or '')

    parts = [f"EDITORIAL STYLE: {profile_name}", ""]
    if nl_summary:
        parts.append(nl_summary)
        parts.append("")

    # Describe the structured fields in plain language (skip placeholders)
    np_ = summary.get('narrative_patterns', {})
    tp = summary.get('thematic_patterns', {})
    vc = summary.get('voice_characteristics', {})

    def _ok(v):
        return v and v != PLACEHOLDER_NARRATIVE

    structural_lines = []
    if _ok(np_.get('opening_style')):
        structural_lines.append(f"- Typical opening: {np_['opening_style']}")
    if _ok(np_.get('resolution_style')):
        structural_lines.append(f"- Typical resolution: {np_['resolution_style']}")
    if _ok(np_.get('pacing_signature')):
        structural_lines.append(f"- Pacing signature: {np_['pacing_signature']}")
    if _ok(np_.get('emotional_arc_shape')):
        structural_lines.append(f"- Emotional arc: {np_['emotional_arc_shape']}")
    if np_.get('average_clip_length_seconds'):
        structural_lines.append(
            f"- Average clip length: {np_['average_clip_length_seconds']:.1f}s"
        )

    if structural_lines:
        parts.append("Structural tendencies:")
        parts.extend(structural_lines)
        parts.append("")

    themes = tp.get('common_themes') or []
    if themes:
        parts.append(f"Recurring themes in this editor's work: {', '.join(themes[:10])}.")
        parts.append("")

    if _ok(vc.get('tone')):
        parts.append(f"Editorial voice: {vc['tone']}")
    if _ok(vc.get('formality_level')):
        parts.append(f"Formality: {vc['formality_level']}")
    if vc.get('tone') or vc.get('formality_level'):
        parts.append("")

    refinements = (summary.get('user_refinements') or '').strip()
    if refinements:
        parts.append("Editor's own notes on their style (high priority — follow these):")
        parts.append(refinements)
        parts.append("")

    parts.append(
        "When suggesting clips, story structures, or sequences, weight your "
        "choices toward this style. Don't force it, but prefer it when multiple "
        "valid options exist. The goal is to feel like an assistant editor who "
        "has worked with this person for years."
    )
    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# LLM call for narrative fields
# ---------------------------------------------------------------------------

_LLM_PROMPT = """You are analyzing a documentary editor's body of work to describe
their editorial style. Below is concatenated spoken-word content from several of
their finished pieces. Output ONLY a JSON object with the following shape:

{{
  "opening_style": "<2-4 word label, e.g. 'emotional hook', 'scene-setting'>",
  "opening_style_confidence": <0.0 to 1.0>,
  "emotional_arc_shape": "<e.g. 'rising', 'valley', 'peak-and-resolve'>",
  "resolution_style": "<e.g. 'open-ended', 'definitive', 'reflective'>",
  "common_themes": ["<theme 1>", "<theme 2>", "<theme 3>", "<theme 4>", "<theme 5>"],
  "subject_focus": "<e.g. 'personal stories', 'institutional', 'issue-driven'>",
  "vulnerability_pattern": "<e.g. 'early reveal', 'gradual', 'withheld'>",
  "favors_chronological": <true|false>,
  "uses_intercutting": <true|false>,
  "tone": "<short phrase, e.g. 'warm and observational', 'urgent'>",
  "formality_level": "<e.g. 'conversational', 'journalistic', 'literary'>",
  "uses_narration": <true|false>
}}

Rules:
- Be conservative. If you're not confident about a field, use a lower confidence
  number or a more general label. It's better to say "varies" than to overclaim.
- Base everything on what you actually see in the transcripts. Don't guess.
- Output ONLY the JSON object, nothing else. No markdown fences.

TRANSCRIPTS:
{transcripts}
"""


def _call_llm_for_narrative_fields(transcripts_text, metrics_profile):
    """Ask the LLM to fill in narrative/thematic/voice fields. Returns dict or None."""
    # Truncate to something the model can handle
    text = transcripts_text.strip()
    if len(text) > 12000:
        # Take the first 6k and last 6k — opening + closing signal
        text = text[:6000] + "\n...\n" + text[-6000:]

    prompt = _LLM_PROMPT.format(transcripts=text)
    try:
        raw = _call_ai(prompt)
    except Exception as e:
        print(f"[edna] LLM analysis call failed: {e}")
        return None

    return _parse_llm_json(raw)


def _parse_llm_json(raw):
    """Best-effort JSON parse. Strips markdown fences, extracts first {...} block."""
    if not raw:
        return None
    text = raw.strip()
    # Remove ```json ... ``` wrappers if present
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    # Extract first {...} block
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        # Common LLM mistake: trailing commas. Try one fix.
        cleaned = re.sub(r',(\s*[}\]])', r'\1', m.group(0))
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def _merge_llm_fields(summary, llm):
    """Merge an LLM JSON response into the summary dict, field by field.

    Only overwrites placeholders or empty values. Bad types are silently skipped.
    """
    np_ = summary['narrative_patterns']
    tp = summary['thematic_patterns']
    sp = summary['structural_preferences']
    vc = summary['voice_characteristics']

    def _s(v):
        return isinstance(v, str) and v.strip()

    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _b(v):
        return v if isinstance(v, bool) else None

    if _s(llm.get('opening_style')):
        np_['opening_style'] = llm['opening_style']
    conf = _f(llm.get('opening_style_confidence'))
    if conf is not None:
        np_['opening_style_confidence'] = max(0.0, min(1.0, conf))
    if _s(llm.get('emotional_arc_shape')):
        np_['emotional_arc_shape'] = llm['emotional_arc_shape']
    if _s(llm.get('resolution_style')):
        np_['resolution_style'] = llm['resolution_style']

    themes = llm.get('common_themes')
    if isinstance(themes, list):
        tp['common_themes'] = [str(t) for t in themes if _s(str(t))][:10]
    if _s(llm.get('subject_focus')):
        tp['subject_focus'] = llm['subject_focus']
    if _s(llm.get('vulnerability_pattern')):
        tp['vulnerability_pattern'] = llm['vulnerability_pattern']

    b = _b(llm.get('favors_chronological'))
    if b is not None:
        sp['favors_chronological'] = b
    b = _b(llm.get('uses_intercutting'))
    if b is not None:
        sp['uses_intercutting'] = b

    if _s(llm.get('tone')):
        vc['tone'] = llm['tone']
    if _s(llm.get('formality_level')):
        vc['formality_level'] = llm['formality_level']
    b = _b(llm.get('uses_narration'))
    if b is not None:
        vc['uses_narration'] = b
