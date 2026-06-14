"""Per-project AI Output Language: canonical list, resolver, prompt directive.

All AI-generated PROSE (Analysis, Story Brief, Story Builder, Chat, Quote
Sheets, Collections) is written in a per-project Output Language; verbatim
transcript quotes always stay in the source language. This module is the
single source of truth for three things every surface shares:

- the canonical language list (replaces the lists previously duplicated
  across dashboard.html, project.html, and the Pro import queue),
- ``resolve_output_language``: the ONE resolver every surface calls
  (no feature implements its own fallback logic),
- ``language_directive``: the prompt fragment appended at assembly time.
  English resolves to the empty string so English projects produce
  byte-identical prompts to pre-feature behavior.

Core-owned on purpose: whisper.tokenizer.LANGUAGES only exists after the
on-demand Whisper install, so nothing here may depend on it. Pro surfaces
(electron pro/ extensions) import from here and add nothing language-
specific of their own.
"""

# Canonical (code, English name) pairs, in UI display order. 'en' first,
# then alphabetical — mirrors the historical dashboard dropdown exactly.
# 'auto' and 'match' are UI sentinels, not languages, and live outside
# this list.
LANGUAGES = [
    ('en', 'English'),
    ('ar', 'Arabic'),
    ('zh', 'Chinese'),
    ('cs', 'Czech'),
    ('da', 'Danish'),
    ('nl', 'Dutch'),
    ('fi', 'Finnish'),
    ('fr', 'French'),
    ('de', 'German'),
    ('el', 'Greek'),
    ('he', 'Hebrew'),
    ('hi', 'Hindi'),
    ('hu', 'Hungarian'),
    ('id', 'Indonesian'),
    ('it', 'Italian'),
    ('ja', 'Japanese'),
    ('ko', 'Korean'),
    ('ms', 'Malay'),
    ('no', 'Norwegian'),
    ('pl', 'Polish'),
    ('pt', 'Portuguese'),
    ('ro', 'Romanian'),
    ('ru', 'Russian'),
    ('sk', 'Slovak'),
    ('es', 'Spanish'),
    ('sv', 'Swedish'),
    ('th', 'Thai'),
    ('tr', 'Turkish'),
    ('uk', 'Ukrainian'),
    ('vi', 'Vietnamese'),
]

LANGUAGE_NAMES = dict(LANGUAGES)

# Whisper may auto-detect languages outside the UI list (e.g. 'nn'
# Nynorsk). Names for the directive come from here when possible;
# unknown codes fall back to English output rather than emitting a
# directive with a bare ISO code the model may misread.
_EXTRA_DETECTED_NAMES = {
    'nn': 'Norwegian Nynorsk',
    'nb': 'Norwegian Bokmål',
}


def language_name(code):
    """English display name for a canonical or known-detected code."""
    if not code:
        return None
    code = code.strip().lower()
    return LANGUAGE_NAMES.get(code) or _EXTRA_DETECTED_NAMES.get(code)


def resolve_output_language(meta):
    """The one shared resolver. Returns an ISO-639-1 code, never None.

    - output_language explicit code -> that code (unknown codes -> 'en').
    - output_language 'match' (or absent) -> the project's interview
      language; 'auto' resolves through meta['detected_language']
      (persisted at transcription completion); unavailable detection
      falls back to 'en'.

    Reads ONLY project meta — never the transcript blob.
    """
    meta = meta or {}
    out = (meta.get('output_language') or 'match').strip().lower()
    if out != 'match':
        return out if language_name(out) else 'en'
    lang = (meta.get('language') or 'en').strip().lower()
    if lang == 'auto':
        detected = (meta.get('detected_language') or '').strip().lower()
        return detected if language_name(detected) else 'en'
    return lang if language_name(lang) else 'en'


def resolve_collection_output_language(collection_meta, member_metas=None):
    """The output language for a collection's AI prose.

    Resolution order:
      1. An explicit, non-'match' ``output_language`` on the collection
         itself wins (future-proofing a per-collection override UI).
      2. Otherwise INHERIT from the member projects: the majority of the
         members' own resolved output languages, counting only NON-English
         languages (ties broken by member order). A folder of Norwegian
         interviews therefore produces Norwegian collection summaries /
         recommended stories / Story Builder output, matching the
         per-project behaviour — which the original "default English" rule
         got wrong. English is the FALLBACK, not a content language: it is
         deliberately not counted, so members that resolve to English
         (genuinely English, or 'auto' not yet transcribed/detected)
         cannot drag a Norwegian collection back to English.
      3. English only when no member resolves to a non-English language.

    ``member_metas`` is an iterable of member project meta dicts; each is
    run through :func:`resolve_output_language` (so 'match'/'auto'/detected
    all work per member).
    """
    out = ((collection_meta or {}).get('output_language') or '').strip().lower()
    if out and out != 'match' and language_name(out):
        return out

    votes = {}
    order = []
    for meta in (member_metas or []):
        code = resolve_output_language(meta)
        if code == 'en':
            continue  # fallback, never a content vote (see docstring)
        if code not in votes:
            order.append(code)
        votes[code] = votes.get(code, 0) + 1
    if votes:
        # Highest vote count; tie → earliest in member order (deterministic).
        return max(order, key=lambda c: (votes[c], -order.index(c)))
    return 'en'


def language_directive(code, chat=False):
    """Prompt fragment enforcing the resolved output language.

    Returns '' for English (or unknown codes): every assembly site does a
    plain string concat, so English projects produce byte-identical
    prompts to pre-feature behavior.

    The verbatim carve-outs are load-bearing, not stylistic: the citation
    validators do exact matching on speaker labels / IDs / enum values and
    fuzzy quote-vs-transcript matching — translated values silently drop
    citations.
    """
    code = (code or 'en').strip().lower()
    if code == 'en':
        return ''
    name = language_name(code)
    if not name:
        return ''
    directive = (
        f'\n\nOUTPUT LANGUAGE: Write all generated prose — summaries, '
        f'descriptions, explanations, titles, beat labels, headers, and '
        f'reasoning — in {name}. '
        f'Never translate verbatim quotes from the transcript: quote them '
        f'exactly as they appear, in their original language. '
        f'Never translate speaker labels, project or clip IDs, timecodes, '
        f'or fixed field values/enums — copy those exactly as given. '
        # Counterweight to the My Style / STYLE CONTEXT exemplars, which are
        # verbatim passages from the editor's own (often English) finished
        # work and sit at peak-attention positions — without this they pull
        # a small model back toward English regardless of the instruction.
        f'Any MY STYLE or STYLE CONTEXT exemplar passages or example quotes '
        f'shown elsewhere in this prompt are provided for STYLE and STRUCTURE '
        f'ONLY and may be written in a different language — do not let their '
        f'language influence your output. Write your generated prose in '
        f'{name} regardless of the language of those examples.'
    )
    if chat:
        directive += (
            f' If the user writes their message in a language other than '
            f'{name}, reply in the language the user wrote in instead.'
        )
    return directive
