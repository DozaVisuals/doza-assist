"""
Natural language summary generator for My Style (Editorial DNA Level 1).
Produces the 3-5 sentence description that gets injected into AI prompts.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_analysis import _call_ai


def generate_summary(profile):
    """
    Generate a 3-5 sentence natural language summary of the editor's style
    based on the full profile metrics. This is the most important field in the
    entire schema — it's what the AI reads to understand the editor's voice.
    """
    sp = profile.get('speech_pacing', {})
    sr = profile.get('structural_rhythm', {})
    sc = profile.get('soundbite_craft', {})
    ss = profile.get('story_shape', {})
    cp = profile.get('content_patterns', {})

    prompt = f"""Based on these narrative editing metrics from finished projects by a documentary editor, write a 3-5 sentence description of how this editor shapes spoken stories. Be specific and observational, not generic. Mention concrete tendencies.

Example of the tone and specificity I want:
"Opens cold with a question almost every time. Lets primary speakers run for 12-18 seconds before cutting away. Cuts cleanly at sentence boundaries, rarely mid-thought. Builds intensity in the final third. Comfortable letting silence land before reveals."

Now write a similar description for an editor with these metrics:

Speech Pacing:
- Words per minute: {sp.get('words_per_minute', 0)}
- Average speaking beat: {sp.get('avg_beat_length', 0):.1f} seconds
- Median beat: {sp.get('median_beat_length', 0):.1f} seconds
- Beat variability (std dev): {sp.get('stddev_beat_length', 0):.1f} seconds
- Longest beat: {sp.get('longest_beat', 0):.1f} seconds
- Shortest beat: {sp.get('shortest_beat', 0):.1f} seconds
- Rhythm: {sp.get('rhythm_descriptor', 'unknown')}

Structure:
- Total duration: {sr.get('total_duration_seconds', 0):.0f} seconds across analyzed pieces
- Speech-to-silence ratio: {sr.get('speech_to_silence_ratio', 0):.0%}
- Speaker switches per minute: {sr.get('speaker_switches_per_minute', 0):.1f}
- Longest beat position: {sr.get('longest_beat_position', 'middle')}
- First third pacing: {sr.get('pacing_first_third_wpm', 0):.0f} WPM
- Middle third pacing: {sr.get('pacing_middle_third_wpm', 0):.0f} WPM
- Final third pacing: {sr.get('pacing_last_third_wpm', 0):.0f} WPM
- Energy arc: {sr.get('energy_arc', 'unknown')}

Soundbite Craft:
- Average soundbite length: {sc.get('avg_soundbite_length', 0):.1f} seconds
- Clean cuts (at sentence boundaries): {sc.get('clean_cut_ratio', 0):.0%}
- Average gap before next cut: {sc.get('avg_gap_before_cut', 0):.2f} seconds

Story Shape:
- Typical opening: {ss.get('opening_style', 'unknown')}
- Typical closing: {ss.get('closing_style', 'unknown')}
- Uses callbacks: {ss.get('uses_callbacks', False)}

Content:
- Question-to-statement ratio: {cp.get('question_to_statement_ratio', 0):.2f}
- Distinct topics per piece: {cp.get('topic_count', 0)}

Write 3-5 sentences. No bullet points. No intro like "This editor...". Just direct observations. Be specific about numbers when they reveal something about style."""

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
