"""
Prompt injector for Editorial DNA v2.1.

Resolves the profile to use in this order:
  1. An explicit profile_id passed by the caller (session override from the
     Story Builder / AI Chat UI — the user can switch styles mid-project).
  2. The currently-active profile from the profiles index.
  3. Nothing — return the system prompt unchanged.

Reads v2.1 multi-profile storage, falls through to the legacy v1 single profile
only as a last resort (the migration path converts v1 profiles on first read,
so this fallback should basically never fire once the user has loaded the app).
"""

from editorial_dna.profiles import get_profile, get_active_profile
from editorial_dna.models import PLACEHOLDER_NARRATIVE


def inject_my_style(system_prompt, profile_id=None):
    """Prepend the active profile's style block to the AI system prompt.

    profile_id : optional explicit override. If given and valid, use that
                 profile instead of the active one. Invalid IDs fall through
                 to active.
    """
    profile = None
    if profile_id:
        profile = get_profile(profile_id)
        if profile and not profile.get('active', True):
            profile = None
    if profile is None:
        profile = get_active_profile()

    if not profile:
        return system_prompt

    block = _build_style_block(profile)
    if not block:
        return system_prompt

    return block + "\n\n" + system_prompt


def _build_style_block(profile):
    """Return the style-context string to prepend, or '' if nothing to inject."""
    name = profile.get('name') or 'My Style'
    summary = profile.get('summary') or {}
    stored_prompt = profile.get('system_prompt') or ''
    nl_summary = profile.get('natural_language_summary') or ''

    # Prefer the stored long-form system prompt (written by analysis.py)
    if stored_prompt.strip():
        return f"MY STYLE CONTEXT — {name}\n\n{stored_prompt}"

    # Fall back to the v1-shaped style block built from raw metrics
    sp = profile.get('speech_pacing', {}) or {}
    sr = profile.get('structural_rhythm', {}) or {}
    sc = profile.get('soundbite_craft', {}) or {}
    ss = profile.get('story_shape', {}) or {}

    if not nl_summary and not sp:
        return ''

    lines = [
        f"MY STYLE CONTEXT — {name}",
        "",
        "The editor you are assisting has a specific narrative voice, learned from their finished work:",
        "",
        nl_summary or "(no natural language summary yet)",
        "",
        "Specific tendencies:",
        f"- Average speaking beat length: {sp.get('avg_beat_length', 0):.1f} seconds",
        f"- Pacing: {sp.get('rhythm_descriptor', 'conversational')}",
        f"- Energy arc: {sr.get('energy_arc', 'balanced')}",
        f"- Cut style: {int(sc.get('clean_cut_ratio', 0) * 100)}% clean (sentence-boundary) cuts",
        f"- Typical opening: {ss.get('opening_style', 'varies')}",
        f"- Typical closing: {ss.get('closing_style', 'varies')}",
    ]

    refinements = (summary.get('user_refinements') or '').strip()
    if refinements:
        lines += ['', "Editor's own notes on their style (high priority):", refinements]

    lines += [
        "",
        "When suggesting clips, story structures, or sequences, weight your "
        "choices toward this style. Do not force it, but prefer it when multiple "
        "valid options exist. The goal is to feel like an assistant editor who "
        "has worked with this person for years. The user can toggle this style "
        "off if they want generic suggestions.",
    ]
    return '\n'.join(lines)
