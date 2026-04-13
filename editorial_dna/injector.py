"""
Prompt injector for My Style (Editorial DNA Level 1).
Prepends the editor's style profile to any AI system prompt.
"""

from editorial_dna.storage import load_profile


def inject_my_style(system_prompt):
    """
    If a My Style profile exists and is active, prepend a style context block
    to the system prompt. Otherwise return the prompt unchanged.
    """
    profile = load_profile()
    if not profile:
        return system_prompt
    if not profile.get('active', True):
        return system_prompt

    summary = profile.get('natural_language_summary', '')
    if not summary:
        return system_prompt

    sp = profile.get('speech_pacing', {})
    sr = profile.get('structural_rhythm', {})
    sc = profile.get('soundbite_craft', {})
    ss = profile.get('story_shape', {})

    style_block = f"""MY STYLE CONTEXT:
The editor you are assisting has a specific narrative voice, learned from their finished work:

{summary}

Specific tendencies:
- Average speaking beat length: {sp.get('avg_beat_length', 0):.1f} seconds
- Pacing: {sp.get('rhythm_descriptor', 'conversational')}
- Energy arc: {sr.get('energy_arc', 'balanced')}
- Cut style: {int(sc.get('clean_cut_ratio', 0) * 100)}% clean (sentence-boundary) cuts
- Typical opening: {ss.get('opening_style', 'varies')}
- Typical closing: {ss.get('closing_style', 'varies')}

When suggesting clips, story structures, or sequences, weight your choices toward this style. Do not force it, but prefer it when multiple valid options exist. The goal is to feel like an assistant editor who has worked with this person for years. The user can toggle this style off if they want generic suggestions.
"""
    return style_block + "\n\n" + system_prompt
