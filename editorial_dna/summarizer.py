"""
Natural language summary generator for My Style (Editorial DNA Level 1).
Produces the 3-5 sentence description that gets injected into AI prompts.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_analysis import _call_ai


def generate_summary(profile, transcripts_text=''):
    """
    Generate a short natural language summary of how this editor shapes
    spoken stories. This is the most important field in the entire schema —
    it's what the AI reads to understand the editor's voice, and it's what
    the user sees on the My Style dashboard.

    If transcripts_text is provided, the model grounds its observations in the
    actual content of the editor's cuts. Without it, it falls back to metrics
    only (which tends to read more generic).
    """
    sp = profile.get('speech_pacing', {})
    sr = profile.get('structural_rhythm', {})
    sc = profile.get('soundbite_craft', {})
    ss = profile.get('story_shape', {})

    # Trim transcripts to a manageable window — take opening + closing of
    # each piece where possible. This gives the model enough to ground on
    # without overflowing context.
    snippet = (transcripts_text or '').strip()
    if len(snippet) > 10000:
        snippet = snippet[:5000] + "\n...\n" + snippet[-5000:]

    # Build a compact, human metric block — no WPM, no ratios, no stddev,
    # just the two or three observations that actually matter for prose.
    metric_bits = []
    if sc.get('avg_soundbite_length'):
        metric_bits.append(f"holds most speaking clips around {sc['avg_soundbite_length']:.0f} seconds before cutting")
    if sp.get('rhythm_descriptor'):
        metric_bits.append(f"pacing reads as {sp['rhythm_descriptor']}")
    if sr.get('energy_arc'):
        metric_bits.append(f"energy arc across the piece tends to {sr['energy_arc']}")
    if ss.get('opening_style') and ss['opening_style'] != 'other':
        metric_bits.append(f"openings tend toward {ss['opening_style'].replace('_', ' ')}")
    if ss.get('closing_style') and ss['closing_style'] != 'other':
        metric_bits.append(f"endings tend toward {ss['closing_style'].replace('_', ' ')}")
    metric_summary = '; '.join(metric_bits) if metric_bits else 'no reliable metric signal yet'

    if snippet:
        prompt = f"""You are writing a short paragraph that describes how a specific video editor shapes spoken stories, for that editor to read on their own dashboard.

Below is transcript text pulled straight from finished pieces this editor cut. Read it carefully — notice what the pieces are actually ABOUT (the subject matter, who speaks, what they say) — and write 3-4 sentences that describe this editor's sensibility in a way ONLY someone who looked at their real work could write.

Hard rules:
- Ground everything in what you actually see in the transcript below. Reference specific subject matter where it reveals something (e.g. "gravitates to athletes talking about pressure", not "covers sports topics").
- NO generic editor vocabulary. Forbidden words: cinematic, emotive, craft, masterful, measured, polished, observational, thoughtful, rhythmic, dynamic, engaging, compelling, captivating, resonant, nuanced. If you find yourself reaching for one of those, stop and describe what you actually see instead.
- NO sentences that could apply to any documentary editor. Every sentence should feel TRUE of these specific pieces and FALSE of a random other editor's work.
- Don't mention numbers or metrics. Write like you're describing a friend's work to another editor.
- 3-4 sentences total. No bullet points. No intro like "This editor...". Start in the middle of the thought.

Metric notes (for context only — don't quote these numbers): {metric_summary}

TRANSCRIPT EXCERPTS:
{snippet}

Now write the 3-4 sentences."""
    else:
        # No transcript text available (e.g. migrated v1 profile) — fall back
        # to metrics, but still forbid the jargon vocabulary.
        prompt = f"""Write 2-3 short sentences describing how a video editor shapes spoken stories. You have no transcript to work with — only these high-level observations: {metric_summary}.

Hard rules:
- NO generic editor vocabulary. Forbidden: cinematic, emotive, craft, masterful, measured, polished, observational, thoughtful, rhythmic, dynamic, engaging, compelling, captivating, resonant, nuanced.
- Describe concrete tendencies in plain language. If you don't have enough signal to say something specific, say so honestly ("not enough material imported yet to read a clear voice").
- No intro like "This editor...". No bullet points. 2-3 sentences."""

    response = _call_ai(prompt)
    # Clean up: remove quotes, markdown artifacts
    summary = response.strip().strip('"').strip("'")
    # Remove any markdown headers
    lines = []
    for line in summary.split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            lines.append(line)
    return ' '.join(lines)
