"""
Prompt injector for Editorial DNA v2.2.

Resolves which profile(s) to apply in this order:

  1. An explicit profile_id passed by the caller (session override from the
     Story Builder / AI Chat UI — the user can switch styles mid-project).
  2. Every profile flagged active in the profiles index. v2.2 supports
     multiple simultaneously-active profiles; their style blocks are
     concatenated with a ``---PROFILE BREAK---`` separator and the model is
     told to lean toward the first listed when they conflict.
  3. Nothing — return the system prompt unchanged.

Active profiles whose individual ``active`` flag is toggled off (the
within-profile master toggle) are skipped, so a user can deactivate a profile
without removing it from the active set.
"""

from editorial_dna.profiles import (
    get_profile,
    get_active_profile,
    get_active_profiles,
)


def inject_my_style(system_prompt, profile_id=None):
    """Prepend My Style context to the AI system prompt.

    profile_id : optional explicit override. If given and valid, use that
                 profile instead of the active set. Invalid IDs fall through
                 to the active set.

    No-profile fallback: when the user has no active My Style profile (fresh
    install, or a Pro user who hasn't built one yet), this prepends a short
    default editor-identity block that anchors the model in documentary-editor
    mode. The storytelling foundation injected later in the call chain
    provides the actual editorial principles; this block tells the model who
    to BE while reading them.
    """
    profiles = []
    if profile_id:
        p = get_profile(profile_id)
        if p and p.get('active', True):
            profiles = [p]
    if not profiles:
        profiles = [p for p in (get_active_profiles() or []) if p.get('active', True)]
        # Belt-and-suspenders for callers that rely on legacy semantics
        if not profiles:
            single = get_active_profile()
            if single:
                profiles = [single]

    if not profiles:
        # No profile active — fall back to a generic editor identity that
        # leans on the bundled storytelling foundation (master.md). This
        # keeps clip selection grounded in editorial principles even when
        # the user hasn't taught the app their personal style yet.
        return _DEFAULT_EDITOR_IDENTITY + "\n\n" + system_prompt

    blocks = [_build_style_block(p) for p in profiles]
    blocks = [b for b in blocks if b]
    if not blocks:
        return _DEFAULT_EDITOR_IDENTITY + "\n\n" + system_prompt

    if len(blocks) == 1:
        return blocks[0] + "\n\n" + system_prompt

    blended = (
        "MULTIPLE STYLE PROFILES ARE ACTIVE — blend them, leaning toward the "
        "first listed when they conflict.\n\n"
        + "\n\n---PROFILE BREAK---\n\n".join(blocks)
        + "\n\n"
    )
    return blended + system_prompt


# Default editor-identity block prepended when no My Style profile is active.
# Short on purpose — the storytelling foundation that follows in the prompt
# does the heavy lifting on editorial principles; this block tells the model
# WHO to be while reading them. Without this fallback, the chat on a fresh
# install was producing reasoning-tag wrappers and unfocused prose because
# the model had nothing anchoring it in "documentary editor" mode.
_DEFAULT_EDITOR_IDENTITY = """DEFAULT EDITOR IDENTITY (no personal style profile active yet)

You are a documentary editor with deep instincts for spoken-story craft. Read every transcript through that lens. The storytelling foundation that follows below contains your operating principles — apply them as a documentary editor would: pick moments that earn their place because of emotional truth, narrative tension, or unexpected revelation. Cut around hedging, filler, brand-speak, and meta-commentary. Favor specificity over generality. When the user asks for clips, name the moments with [CLIP:] markers and a one-sentence why; never substitute summary prose for the actual cited moments.

The user can teach the app their personal editorial voice via My Style — that profile, when active, will replace this default identity with their specific patterns. Until then, default to grounded documentary craft."""


def get_active_style_block(profile_id=None):
    """Return the active My Style content as a single string, or ``None``.

    Same profile-resolution order as :func:`inject_my_style` (explicit ID,
    then active set, then legacy single-active). Unlike that function,
    this one DOES NOT prepend the default editor-identity fallback when
    no profile is active — it returns ``None``, leaving the caller to
    decide whether to skip the style message entirely.

    Used by the chat path to build a separate user-role STYLE CONTEXT
    message when a profile is active. Multi-profile blends are joined
    with the same ``---PROFILE BREAK---`` separator the inline injector
    uses, with the same blend-toward-first-listed instruction prepended.
    """
    profiles = []
    if profile_id:
        p = get_profile(profile_id)
        if p and p.get('active', True):
            profiles = [p]
    if not profiles:
        profiles = [p for p in (get_active_profiles() or []) if p.get('active', True)]
        if not profiles:
            single = get_active_profile()
            if single:
                profiles = [single]
    if not profiles:
        return None

    blocks = [_build_style_block(p) for p in profiles]
    blocks = [b for b in blocks if b]
    if not blocks:
        return None
    if len(blocks) == 1:
        return blocks[0]
    return (
        "MULTIPLE STYLE PROFILES ARE ACTIVE — blend them, leaning toward the "
        "first listed when they conflict.\n\n"
        + "\n\n---PROFILE BREAK---\n\n".join(blocks)
    )


def _build_style_block(profile):
    """Return the style-context string to prepend, or '' if nothing to inject."""
    name = profile.get('name') or 'My Style'
    summary = profile.get('summary') or {}
    stored_prompt = profile.get('system_prompt') or ''
    nl_summary = profile.get('natural_language_summary') or ''

    # Prefer the stored long-form system prompt (written by analysis.py).
    # That prompt already includes exemplar passages, quote archetypes,
    # language patterns, and refinements.
    if stored_prompt.strip():
        return f"MY STYLE CONTEXT — {name}\n\n{stored_prompt}"

    # Fall back to the v1-shaped style block built from raw metrics. This path
    # only fires for very old migrated profiles that haven't been re-analyzed
    # since the v2.x bump.
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
