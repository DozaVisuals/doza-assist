"""
AI-powered classifications for My Style (Editorial DNA Level 1).
Uses the existing Ollama/Claude dual backend from ai_analysis.py.
"""

import sys
import os

# Add parent directory so we can import ai_analysis
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_analysis import _call_ai


def classify_opening(first_15s_text):
    """Classify opening style from the first 15 seconds of transcript.
    Returns one of: cold_quote, narration, scene_setting, question, other
    """
    if not first_15s_text.strip():
        return 'other'

    prompt = f"""Classify the opening style of this documentary/interview piece based on the first 15 seconds of spoken content.

TEXT: "{first_15s_text}"

Categories:
- cold_quote: Opens with a direct quote or soundbite from a subject, no setup
- narration: Opens with narrator/reporter voice-over providing context
- scene_setting: Opens by describing a place, situation, or atmosphere
- question: Opens with a question (rhetorical or direct)
- other: Doesn't clearly fit any of the above

Respond with ONLY the category name, nothing else."""

    response = _call_ai(prompt).strip().lower().replace('"', '').replace("'", '')
    valid = {'cold_quote', 'narration', 'scene_setting', 'question', 'other'}
    # Extract valid category from response
    for cat in valid:
        if cat in response:
            return cat
    return 'other'


def classify_closing(last_15s_text):
    """Classify closing style from the last 15 seconds of transcript.
    Returns one of: callback, statement, question, button, other
    """
    if not last_15s_text.strip():
        return 'other'

    prompt = f"""Classify the closing style of this documentary/interview piece based on the last 15 seconds of spoken content.

TEXT: "{last_15s_text}"

Categories:
- callback: Echoes or references something from earlier in the piece
- statement: Ends with a definitive declaration or observation
- question: Ends with a question (rhetorical or direct), leaving the viewer thinking
- button: Ends with a short punchy line or emotional moment (a "button" ending)
- other: Doesn't clearly fit any of the above

Respond with ONLY the category name, nothing else."""

    response = _call_ai(prompt).strip().lower().replace('"', '').replace("'", '')
    valid = {'callback', 'statement', 'question', 'button', 'other'}
    for cat in valid:
        if cat in response:
            return cat
    return 'other'


def classify_rhythm(beat_stats):
    """Generate a rhythm descriptor from beat statistics.
    beat_stats: dict with avg_beat_length, stddev_beat_length, words_per_minute
    Returns a short descriptive phrase like 'measured and deliberate'.
    """
    prompt = f"""Based on these speech rhythm metrics from a finished documentary/interview piece, write a 2-4 word rhythm descriptor.

Metrics:
- Average speaking beat length: {beat_stats.get('avg_beat_length', 0):.1f} seconds
- Beat length variability (std dev): {beat_stats.get('stddev_beat_length', 0):.1f} seconds
- Words per minute: {beat_stats.get('words_per_minute', 0):.0f}

Examples of good descriptors: "measured and deliberate", "rapid-fire and punchy", "variable and dynamic", "slow burn with bursts", "steady and conversational"

Respond with ONLY the descriptor phrase (2-4 words), nothing else."""

    response = _call_ai(prompt).strip().strip('"').strip("'").lower()
    # Sanity check: should be short
    if len(response.split()) > 8:
        return 'conversational'
    return response


def classify_energy_arc(pacing_first, pacing_middle, pacing_last):
    """Describe the energy arc based on pacing per third.
    Returns a short descriptive phrase like 'builds to the end'.
    """
    prompt = f"""Based on the pacing across three acts of a documentary/interview piece, write a 2-5 word energy arc descriptor.

Pacing (words per minute):
- First third: {pacing_first:.0f} WPM
- Middle third: {pacing_middle:.0f} WPM
- Final third: {pacing_last:.0f} WPM

Examples: "builds to the end", "front-loaded energy", "steady throughout", "dips then surges", "peaks in the middle", "slow start fast finish"

Respond with ONLY the descriptor phrase (2-5 words), nothing else."""

    response = _call_ai(prompt).strip().strip('"').strip("'").lower()
    if len(response.split()) > 8:
        return 'balanced'
    return response


def detect_callbacks(opening_third_text, closing_third_text):
    """Check if the closing third references or echoes themes from the opening third.
    Returns True/False.
    """
    if not opening_third_text.strip() or not closing_third_text.strip():
        return False

    # Quick keyword overlap check first (fast path)
    opening_words = set(w.lower() for w in opening_third_text.split() if len(w) > 4)
    closing_words = set(w.lower() for w in closing_third_text.split() if len(w) > 4)
    overlap = opening_words & closing_words
    # Filter out very common words
    common = {'about', 'would', 'could', 'should', 'there', 'their', 'these', 'those',
              'which', 'where', 'going', 'being', 'really', 'think', 'people', 'things'}
    meaningful_overlap = overlap - common

    if len(meaningful_overlap) >= 3:
        return True

    # If keyword overlap is ambiguous, use AI
    prompt = f"""Does the closing section of this piece echo, reference, or callback to themes or phrases from the opening section?

OPENING (first third):
"{opening_third_text[:500]}"

CLOSING (final third):
"{closing_third_text[:500]}"

Respond with ONLY "yes" or "no"."""

    response = _call_ai(prompt).strip().lower()
    return 'yes' in response


def estimate_topic_count(full_text):
    """Estimate how many distinct topics/subjects are covered in the piece."""
    if not full_text.strip():
        return 1

    prompt = f"""How many distinct topics or subjects are covered in this documentary/interview transcript? Count major topic shifts, not minor tangents.

TRANSCRIPT (excerpt):
"{full_text[:2000]}"

Respond with ONLY a number (e.g. "3"), nothing else."""

    response = _call_ai(prompt).strip()
    # Extract first number from response
    import re
    match = re.search(r'\d+', response)
    if match:
        count = int(match.group())
        return max(1, min(count, 20))  # Clamp to reasonable range
    return 1
