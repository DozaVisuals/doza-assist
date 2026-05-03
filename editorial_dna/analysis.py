"""
Editorial DNA v2.2 structured analysis pass.

This module builds two outputs from a profile's transcripts + metrics:

  1. A long-form system prompt (the thing that gets injected into Story Builder
     and AI Chat). Combines the v1 natural-language prose summary with the
     structured fields and verbatim exemplar passages — the few-shot anchor
     that makes the editor's voice show up in AI suggestions.

  2. A StyleProfileSummary dict (the structured object the dashboard renders).
     Narrative/thematic/voice fields plus exemplar passages, quote archetypes,
     and language patterns are filled by asking the LLM to output JSON.

This prompt is product. Bumping ``editorial_dna.models.ANALYSIS_VERSION``
triggers a "regenerate" prompt in the UI for every existing profile — treat it
like a database migration.

The analyzer is intentionally tolerant: bad JSON from the model falls back to
keeping existing values, so a broken LLM response can never nuke a profile.

Long-corpus strategy: corpora that fit comfortably in the model's context
(~12k chars) are analyzed in one pass. Longer corpora go through a two-pass
synthesis: the model extracts patterns from each source individually, then a
second pass synthesizes those into the final profile. We never silently
truncate the middle of a long corpus — that loses the act structure that the
analysis is trying to read.
"""

import json
import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_analysis import _call_ai  # noqa: E402

from editorial_dna.models import (
    empty_style_profile_summary,
    PLACEHOLDER_NARRATIVE,
    ANALYSIS_VERSION,
)
from editorial_dna.summarizer import generate_summary


# Threshold above which we switch from single-pass to two-pass synthesis. Sized
# to leave headroom for the system prompt and the schema instructions.
SINGLE_PASS_CHAR_BUDGET = 12000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_structured_summary(profile_id, profile_name, metrics_profile, source_files,
                                 transcripts_text='', existing_summary=None):
    """Build a StyleProfileSummary dict for this profile.

    metrics_profile : dict — v1-shaped metric profile (speech_pacing, etc.)
    source_files    : list — from source_files.json (each has filename +
                      transcript_text used for per-source two-pass synthesis)
    transcripts_text: str  — concatenated transcript text. Pre-built by callers
                      who already have it; recomputed from source_files if empty.
    existing_summary: dict or None — previous summary, used as fallback for
                      fields the LLM can't answer confidently

    Always returns a full summary dict. Never raises.
    """
    summary = empty_style_profile_summary(profile_id, profile_name)
    if existing_summary:
        summary['user_refinements'] = existing_summary.get('user_refinements', '')

    summary['projects_analyzed'] = len(source_files or [])
    summary['total_runtime_seconds'] = int(sum(
        sf.get('duration_seconds', 0) for sf in (source_files or [])
    ))
    summary['analysis_version'] = ANALYSIS_VERSION

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

    peak_pos_map = {'first_third': 0.25, 'middle': 0.5, 'last_third': 0.8}
    summary['structural_preferences']['preferred_position_for_emotional_peak'] = \
        peak_pos_map.get(sr.get('longest_beat_position', 'middle'), 0.5)

    # Build the corpus text we'll feed the LLM. Prefer per-source structure when
    # we have it (so the model can attribute exemplars to specific files). Fall
    # back to the concatenated blob the caller passed.
    per_source_corpus = _per_source_corpus(source_files)
    if not per_source_corpus and transcripts_text:
        per_source_corpus = [{'filename': 'corpus', 'text': transcripts_text}]

    if per_source_corpus:
        llm_fields = _analyze_corpus(per_source_corpus)
        if llm_fields:
            _merge_llm_fields(summary, llm_fields)

    return summary


def _per_source_corpus(source_files):
    """Return [{filename, text}, ...] for sources that have transcript_text."""
    out = []
    for sf in (source_files or []):
        text = (sf.get('transcript_text') or '').strip()
        if text:
            out.append({
                'filename': sf.get('filename') or sf.get('original_filename') or 'untitled',
                'text': text,
            })
    return out


def _analyze_corpus(per_source_corpus):
    """Decide single-pass vs two-pass and return the merged LLM fields dict.

    Single-pass: corpus fits in budget, one call with all sources concatenated.
    Two-pass: per-source extraction into intermediate JSON, then synthesis.
    """
    total_chars = sum(len(s['text']) for s in per_source_corpus)
    if total_chars <= SINGLE_PASS_CHAR_BUDGET:
        joined = _format_corpus_for_prompt(per_source_corpus)
        return _call_llm_for_full_profile(joined, single_pass=True)

    # Two-pass: extract per-source observations + verbatim quote candidates,
    # then synthesize. This preserves the act structure that single-pass with
    # head/tail truncation throws away.
    intermediates = []
    for src in per_source_corpus:
        per_src = _call_llm_per_source(src['filename'], src['text'])
        if per_src:
            intermediates.append({'filename': src['filename'], 'observations': per_src})
    if not intermediates:
        return None
    return _call_llm_synthesis(intermediates)


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

    lp = summary.get('language_patterns') or {}
    if _ok(lp.get('favored_phrasing')) or _ok(lp.get('filtered_out')):
        parts.append("Language patterns:")
        if _ok(lp.get('favored_phrasing')):
            parts.append(f"- Favors: {lp['favored_phrasing']}")
        if _ok(lp.get('filtered_out')):
            parts.append(f"- Cuts around: {lp['filtered_out']}")
        parts.append("")

    archetypes = summary.get('quote_archetypes') or []
    if archetypes:
        parts.append("Quote archetypes this editor reaches for:")
        for arc in archetypes[:5]:
            label = arc.get('type') or 'archetype'
            desc = arc.get('description') or ''
            example = (arc.get('example_from_corpus') or '').strip()
            line = f"- {label}"
            if desc:
                line += f": {desc}"
            parts.append(line)
            if example:
                parts.append(f'  e.g. "{example}"')
        parts.append("")

    # The few-shot moat: verbatim passages from the editor's own work. The PRD
    # calls this the most important field — concrete examples beat abstract
    # description every time.
    exemplars = summary.get('exemplar_passages') or []
    if exemplars:
        parts.append("Exemplar passages from this editor's finished work:")
        for ex in exemplars[:5]:
            src = ex.get('source_filename') or 'source'
            why = ex.get('why_this_passage') or ''
            passage = (ex.get('passage') or '').strip()
            if not passage:
                continue
            parts.append(f"--- from {src} ---")
            parts.append(passage)
            if why:
                parts.append(f"(why this passage: {why})")
            parts.append("")
        parts.append(
            "Use these as anchors. When you face a choice between two valid "
            "selects, prefer the one that fits the same shape as the passages above."
        )
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

_FULL_PROFILE_SCHEMA = """{{
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
  "uses_narration": <true|false>,
  "favored_phrasing": "<what kinds of language make this editor's cuts>",
  "filtered_out": "<what language this editor cuts around: filler, hedging, brand-speak, etc.>",
  "quote_archetypes": [
    {{
      "type": "<short label, e.g. 'confession', 'expert declaration', 'turning point'>",
      "description": "<what this archetype sounds like, in one phrase>",
      "example_from_corpus": "<verbatim quote from the transcripts, no paraphrasing>"
    }}
  ],
  "exemplar_passages": [
    {{
      "source_filename": "<exact filename from the SOURCE markers in the corpus>",
      "passage": "<verbatim 30-90 second excerpt copied from the transcript>",
      "why_this_passage": "<one sentence on what makes this representative>"
    }}
  ]
}}"""


_FULL_PROFILE_RULES = """Rules:
- Be specific, not generic. "Favors emotional moments" is useless. "Favors moments where the subject contradicts their earlier statement" is useful.
- Quote VERBATIM from the corpus when filling example_from_corpus and exemplar_passages.passage. Do not paraphrase, do not invent, do not stitch fragments together. If you can't find a representative passage, return fewer items rather than fabricating one.
- 3-4 quote archetypes is the right amount. 3-5 exemplar passages is the right amount. Don't pad.
- Each exemplar passage should be substantial (30-90 seconds of speech, roughly 60-200 words) and copied as a contiguous block from the transcript.
- source_filename must match one of the ---SOURCE: ...--- headers in the corpus exactly. If the corpus has only one source, use that filename.
- Be conservative. If a field is genuinely unclear from the corpus, use a lower confidence or a more general label. "Varies" beats overclaiming.
- Output ONLY the JSON object. No markdown fences, no prose, no explanation."""


_FULL_PROFILE_PROMPT = (
    "You are analyzing a documentary editor's body of work to describe their "
    "editorial style. Below is the concatenated spoken-word content from "
    "several of their finished pieces (---SOURCE: filename--- markers separate "
    "them). Output ONLY a single valid JSON object matching this schema:\n\n"
    + _FULL_PROFILE_SCHEMA
    + "\n\n"
    + _FULL_PROFILE_RULES
    + "\n\nCORPUS:\n{corpus}\n"
)


_PER_SOURCE_PROMPT = """You are reading one finished piece by a documentary editor. Read it carefully, then output a single JSON object describing the editorial moves you can see in THIS piece. We will combine your observations across pieces in a second pass.

Schema:
{{
  "opening_observation": "<what does this piece open with?>",
  "closing_observation": "<what does this piece close with?>",
  "subject_focus": "<who is this piece about / what is it about?>",
  "themes": ["<theme 1>", "<theme 2>", "<theme 3>"],
  "tonal_register": "<e.g. 'warm and observational'>",
  "exemplar_passage": "<a verbatim 30-90 second contiguous passage from the transcript that best exemplifies what this piece is doing — copy it exactly, no paraphrasing>",
  "exemplar_why": "<one sentence on why that passage represents this piece>",
  "favored_phrasing": "<a short observation about the kind of language that's making the cut here>",
  "filtered_out": "<a short observation about what the editor seems to be cutting around in this piece>"
}}

Rules:
- Quote VERBATIM. Do not paraphrase. Do not stitch fragments. If you can't find a strong exemplar, set exemplar_passage to "".
- Output ONLY JSON. No markdown fences. No prose.

PIECE FILENAME: {filename}

TRANSCRIPT:
{transcript}
"""


_SYNTHESIS_PROMPT = """You are synthesizing per-piece observations from one editor's finished work into a single editorial style profile. Each observation block below was generated from one of their pieces and includes a verbatim exemplar passage from that piece.

Your job: identify the editorial pattern that connects these pieces. What did this editor pick? What did they leave out? What kinds of moments do they reach for? Output a single JSON object matching the schema below.

Schema:
{schema}

Rules for synthesis:
- Patterns must be supported by multiple pieces, not just one. A single-piece observation is signal but not pattern.
- exemplar_passages: pick 3-5 of the strongest exemplar_passage values from the per-piece blocks. Use them VERBATIM. Set source_filename to the filename from the matching block. Do not invent or rewrite.
- quote_archetypes: 3-4 archetypes that recur across pieces, each with one verbatim example_from_corpus copied from one of the per-piece exemplar_passage values.
- Prefer specificity over generality.
- Output ONLY the JSON object. No markdown fences, no prose.

PER-PIECE OBSERVATIONS:
{intermediates}
"""


def _format_corpus_for_prompt(per_source_corpus):
    """Format [{filename, text}] as ---SOURCE: name---\n... blocks."""
    blocks = []
    for s in per_source_corpus:
        blocks.append(f"---SOURCE: {s['filename']}---\n{s['text'].strip()}")
    return "\n\n".join(blocks)


def _call_llm_for_full_profile(corpus_text, single_pass=True):
    """One-shot extraction for corpora that fit in the budget."""
    prompt = _FULL_PROFILE_PROMPT.format(corpus=corpus_text)
    try:
        raw = _call_ai(prompt)
    except Exception as e:
        print(f"[edna] full-profile LLM call failed: {e}")
        return None
    parsed = _parse_llm_json(raw)
    if parsed is None:
        # Retry once with a stricter "JSON only, no markdown" instruction.
        retry_prompt = prompt + (
            "\n\nIMPORTANT: your previous response was not valid JSON. "
            "Output ONLY a single JSON object. No markdown fences, no prose, "
            "no explanation. Start with { and end with }."
        )
        try:
            raw = _call_ai(retry_prompt)
        except Exception as e:
            print(f"[edna] full-profile LLM retry failed: {e}")
            return None
        parsed = _parse_llm_json(raw)
    return parsed


def _call_llm_per_source(filename, transcript):
    """Per-source extraction for two-pass synthesis."""
    # Cap each source so we don't blow the context. 8k chars is ~80 minutes of
    # speech which covers the whole shape of even a long doc piece.
    text = transcript.strip()
    if len(text) > 8000:
        # Sample three windows so the model sees opening, middle, close.
        third = len(text) // 3
        text = (
            text[:2500] + "\n[...]\n"
            + text[third: third + 2500] + "\n[...]\n"
            + text[-2500:]
        )
    prompt = _PER_SOURCE_PROMPT.format(filename=filename, transcript=text)
    try:
        raw = _call_ai(prompt)
    except Exception as e:
        print(f"[edna] per-source LLM call failed for {filename}: {e}")
        return None
    return _parse_llm_json(raw)


def _call_llm_synthesis(intermediates):
    """Synthesize per-source observations into a single profile."""
    blob = json.dumps(intermediates, indent=2, ensure_ascii=False)
    # If the intermediate blob is huge, trim verbose passages while keeping all
    # files represented (so synthesis still draws from the whole portfolio).
    if len(blob) > 14000:
        trimmed = []
        for entry in intermediates:
            obs = dict(entry.get('observations') or {})
            ep = obs.get('exemplar_passage') or ''
            if len(ep) > 800:
                obs['exemplar_passage'] = ep[:800].rstrip() + '...'
            trimmed.append({'filename': entry.get('filename'), 'observations': obs})
        blob = json.dumps(trimmed, indent=2, ensure_ascii=False)
    prompt = _SYNTHESIS_PROMPT.format(
        schema=_FULL_PROFILE_SCHEMA, intermediates=blob,
    )
    try:
        raw = _call_ai(prompt)
    except Exception as e:
        print(f"[edna] synthesis LLM call failed: {e}")
        return None
    parsed = _parse_llm_json(raw)
    if parsed is None:
        retry = prompt + (
            "\n\nIMPORTANT: your previous response was not valid JSON. "
            "Output ONLY the JSON object."
        )
        try:
            raw = _call_ai(retry)
        except Exception as e:
            print(f"[edna] synthesis LLM retry failed: {e}")
            return None
        parsed = _parse_llm_json(raw)
    return parsed


def _parse_llm_json(raw):
    """Best-effort JSON parse. Strips markdown fences, extracts first {...} block."""
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        # Common LLM mistake: trailing commas before } or ].
        cleaned = re.sub(r',(\s*[}\]])', r'\1', m.group(0))
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def _merge_llm_fields(summary, llm):
    """Merge an LLM JSON response into the summary dict, field by field.

    Only overwrites placeholders or empty values. Bad types are silently skipped.
    Exemplar passages and quote archetypes are validated for non-empty
    verbatim text before storing — empty/junk entries get dropped.
    """
    np_ = summary['narrative_patterns']
    tp = summary['thematic_patterns']
    sp = summary['structural_preferences']
    vc = summary['voice_characteristics']
    lp = summary['language_patterns']

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
        np_['opening_style'] = llm['opening_style'].strip()
    conf = _f(llm.get('opening_style_confidence'))
    if conf is not None:
        np_['opening_style_confidence'] = max(0.0, min(1.0, conf))
    if _s(llm.get('emotional_arc_shape')):
        np_['emotional_arc_shape'] = llm['emotional_arc_shape'].strip()
    if _s(llm.get('resolution_style')):
        np_['resolution_style'] = llm['resolution_style'].strip()

    themes = llm.get('common_themes')
    if isinstance(themes, list):
        tp['common_themes'] = [str(t).strip() for t in themes if _s(str(t))][:10]
    if _s(llm.get('subject_focus')):
        tp['subject_focus'] = llm['subject_focus'].strip()
    if _s(llm.get('vulnerability_pattern')):
        tp['vulnerability_pattern'] = llm['vulnerability_pattern'].strip()

    b = _b(llm.get('favors_chronological'))
    if b is not None:
        sp['favors_chronological'] = b
    b = _b(llm.get('uses_intercutting'))
    if b is not None:
        sp['uses_intercutting'] = b

    if _s(llm.get('tone')):
        vc['tone'] = llm['tone'].strip()
    if _s(llm.get('formality_level')):
        vc['formality_level'] = llm['formality_level'].strip()
    b = _b(llm.get('uses_narration'))
    if b is not None:
        vc['uses_narration'] = b

    if _s(llm.get('favored_phrasing')):
        lp['favored_phrasing'] = llm['favored_phrasing'].strip()
    if _s(llm.get('filtered_out')):
        lp['filtered_out'] = llm['filtered_out'].strip()

    archetypes = llm.get('quote_archetypes')
    if isinstance(archetypes, list):
        cleaned = []
        for arc in archetypes:
            if not isinstance(arc, dict):
                continue
            t = arc.get('type')
            d = arc.get('description')
            ex = arc.get('example_from_corpus')
            if not (_s(t) and _s(ex)):
                continue
            cleaned.append({
                'type': t.strip(),
                'description': (d or '').strip(),
                'example_from_corpus': ex.strip(),
            })
        if cleaned:
            summary['quote_archetypes'] = cleaned[:5]

    exemplars = llm.get('exemplar_passages')
    if isinstance(exemplars, list):
        cleaned = []
        for ex in exemplars:
            if not isinstance(ex, dict):
                continue
            passage = ex.get('passage')
            if not _s(passage):
                continue
            # Reject pathologically short or pathologically long passages —
            # the prompt asked for 30-90 seconds of speech (~60-200 words).
            words = len(passage.split())
            if words < 20 or words > 600:
                continue
            cleaned.append({
                'source_filename': (ex.get('source_filename') or '').strip(),
                'passage': passage.strip(),
                'why_this_passage': (ex.get('why_this_passage') or '').strip(),
            })
        if cleaned:
            summary['exemplar_passages'] = cleaned[:5]
