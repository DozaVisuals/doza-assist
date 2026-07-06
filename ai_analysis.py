"""
AI Analysis for Doza Assist.
Uses Ollama (local) or Claude API for story structure and social clip suggestions.
"""

import hashlib
import os
import json
import re
import threading
from collections import OrderedDict

import requests

from editorial_dna.injector import get_active_style_block, inject_my_style
from editorial_dna.storytelling import inject_storytelling_foundation
from doza_assist.output_language import language_directive

# Keep the Ollama model resident in memory between requests. Default is
# 5 minutes — too short for an editor reading a reply before typing the
# next question. 30 minutes covers normal session rhythm. Set on every
# generate call in this module so a single tweak propagates everywhere.
_OLLAMA_KEEP_ALIVE = '30m'


# ── Conversational vs extractive intent ──────────────────────────────────
#
# The new chat orientation says "default to conversation, use clips when
# they earn their place." But two systems fight that default:
#   1. _salvage_clips_if_missing forces clips into ANY clipless response
#   2. Long-interview chat (>60min) uses _chat_layer2_chunked_search which
#      ONLY returns clip cards — no conversational LLM call at all
# This classifier tells those code paths when to step out of the way.
#
# Heuristic, not LLM-based — must be cheap (runs on every chat turn) and
# fail-safe toward "conversational" so the orientation paragraph is the
# default behavior rather than the exception.
_EXTRACTIVE_VERB_STARTS = (
    'find', 'finds', 'pull', 'pulls', 'list', 'show', 'shows', 'show me',
    'give', 'gives', 'give me', 'get', 'gets', 'get me', 'surface',
    'surfaces', 'search', 'searches', 'identify', 'identifies', 'gather',
    'gathers', 'compile', 'compiles', 'fetch', 'extract', 'extracts',
    'point me', 'point out', 'pick',
    # Assembly verbs, "me"-suffixed forms — "build me a 2-minute cut" is
    # unambiguously a clip ask, not discussion. Without these,
    # duration-targeted builds were classified conversational and never
    # reached the clip pipeline (or, on >60-min projects, diverted to
    # prose-only synthesis).
    'build me', 'make me', 'cut me',
)
# Bare assembly verbs are ambiguous English ("make sense of...", "cut to
# the chase...", "create a description of..." are all discussion). They
# flip the classification to extractive ONLY when the message also carries
# an anchored duration target or a deliverable noun (_DURATION_INTENT_NOUNS)
# — "make a 90 second teaser" yes, "make it shorter" no.
_CONDITIONAL_ASSEMBLY_VERB_STARTS = (
    'build', 'builds', 'make', 'makes', 'cut', 'cuts', 'create', 'creates',
    'assemble', 'assembles',
)
_NO_CLIP_SIGNALS = (
    'no clip', 'without clip', 'no markers', 'without markers',
    'just talk', 'just tell me', 'just tell me in general', 'in general',
    "don't pull", "don't find", "don't list", "don't return",
    "don't surface", 'do not pull', 'do not find', 'do not return',
    'no need for clips', 'skip the clips', 'skip clips',
)
# Clip-seeking nouns: an editor whose message names "moments", "clips",
# "quotes", "soundbites", or "highlights" wants playable material
# even when the ask wears question syntax ("What are the strongest
# emotional moments in this interview?"). Verb-start heuristics alone
# routed those questions conversational, so no salvage, no count
# enforcement, and no grounding ran — the card count was whatever the
# model happened to emit (live tester bug: prose said "three moments",
# ONE card rendered). Kept deliberately separate from
# _DURATION_INTENT_NOUNS: that list carries deliverable FORMATS ("video",
# "documentary", "podcast") that appear in genuinely conversational
# questions ("what is this video about?") and must not flip them.
# 'beat'/'beats' is deliberately ABSENT: as a music/pacing homonym, an
# idiom ("beats me"), and a plain verb ("this beats the other take") it
# misfires on craft discussion far more often than it names a retrieval
# target ("the story beats feel off" is a structure question, not a clip
# ask) — and a forced clip card on a conversational message is worse
# than a missed extractive ask.
_CLIP_SEEKING_NOUNS = frozenset({
    'moment', 'moments', 'clip', 'clips', 'quote', 'quotes',
    'soundbite', 'soundbites', 'highlight', 'highlights',
})
# The plural forms also promise SEVERAL cards — see _plural_clip_minimum.
_PLURAL_CLIP_NOUNS = frozenset(
    n for n in _CLIP_SEEKING_NOUNS if n.endswith('s'))
# Domain reading of an unqualified plural ask ("the strongest moments"):
# at least a few — the same bound "a few" already maps to in
# _detect_explicit_clip_count.
_PLURAL_CLIP_MINIMUM = 3

# ── Clip-noun reference guards ────────────────────────────────────────────
#
# GUIDING PRINCIPLE for the clip-noun intent flip (and for the count
# top-up further down): PRECISION over recall. A forced clip card on a
# conversational message is WORSE than a missed extractive ask — a missed
# ask still gets a prose answer and the editor can rephrase, while
# unwanted cards break the conversation (and on >60-min projects the
# misroute swallows the question entirely: chunked search returns ONLY
# clip cards, so "why did you pick those clips?" would never be
# answered). Every guard below therefore errs toward "conversational"
# whenever the noun plausibly refers to clips the conversation ALREADY
# produced, to the app's own choices, to an idiom/filler, or to a
# negated / wound-down ask.

# Negation / wind-down tokens shortly BEFORE a clip noun: "no more clips",
# "that's enough clips", "we don't need more moments", "stop with the
# clips" are stop signals, not asks.
_CLIP_NOUN_NEGATIONS = frozenset({
    "don't", 'dont', "doesn't", 'doesnt', "won't", 'wont',
    'stop', 'enough', 'without',
})
# 'no'/'not' get a TIGHTER window (see the guard): as leading discourse
# markers they also open genuine asks ("no, show me the highlights").
_CLIP_NOUN_SHORT_NEGATIONS = frozenset({'no', 'not'})
# Past-tense delivery verbs — "the quotes you PULLED": the noun refers to
# cards already on the table. PAST forms only: base forms ("can you PULL
# clips about the fire?") are live requests and must stay extractive —
# they count as back-references only behind a past interrogative
# ('did/have/had you', see _clip_noun_is_reference).
_CLIP_NOUN_DELIVERY_PAST = frozenset({
    'picked', 'pulled', 'chose', 'chosen', 'selected', 'showed', 'gave',
    'suggested', 'found', 'said', 'mentioned', 'recommended',
    'highlighted', 'sent', 'listed',
})
# Past-tense state verbs right after a determiner+noun — "those clips
# WERE perfect", "the moments FELT right": assessment of delivered cards.
_CLIP_NOUN_PAST_STATE = frozenset({
    'were', 'was', 'felt', 'seemed', 'looked', 'sounded', 'worked',
    'helped', 'landed', 'are',
})
# App-action verbs before the noun — "can you REMOVE the second clip?":
# a meta request about existing cards, never a retrieval ask.
_CLIP_NOUN_EDIT_VERBS = frozenset({
    'remove', 'delete', 'drop', 'replace', 'swap', 'reorder', 'rearrange',
    'rename', 'discard',
})
# First-person assessment — "I LIKE the highlights so far": gratitude /
# feedback about delivered cards. Bigram-gated on a literal 'i'/'we'
# subject so requests ("I'd like the best highlights") still flip.
_CLIP_NOUN_ASSESSMENT_VERBS = frozenset({
    'like', 'love', 'liked', 'loved', 'enjoy', 'enjoyed',
    'appreciate', 'appreciated',
})
# Retrieval verbs ANYWHERE before a deictic noun override the deictic
# guard — "I WANT that moment where he admits it", "can you LOCATE this
# quote" are fetch asks even though 'that/this' precedes the noun (H7).
_CLIP_NOUN_RETRIEVAL_VERBS = frozenset({
    'find', 'locate', 'want', 'pull', 'grab', 'fetch', 'need', 'get',
})
# Word-numbers that keep a plural deictic extractive: "compare those TWO
# moments" is a selection ask, not a back-reference.
_CLIP_NOUN_COUNT_WORDS = frozenset({
    'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'couple', 'few', 'several',
})


def _clip_noun_is_reference(tokens, i):
    """True when ``tokens[i]`` (a clip-seeking noun) is a CONVERSATIONAL
    REFERENCE — a negated/wound-down ask, a back-reference to clips
    already delivered, a meta question about the app's choices, an
    idiom/filler, or a deictic mention — rather than a retrieval target.

    Shared by ``_is_conversational_query`` (the intent flip) and
    ``_plural_clip_minimum`` (the plural floor) so both stay consistent:
    a noun occurrence that doesn't flip the message extractive must not
    force a 3-card minimum either.
    """
    tok = tokens[i]
    prev = tokens[i - 1] if i > 0 else ''
    prev2 = tokens[i - 2] if i > 1 else ''
    nxt = tokens[i + 1] if i + 1 < len(tokens) else ''
    nxt2 = tokens[i + 2] if i + 2 < len(tokens) else ''
    window = tokens[max(0, i - 5):i]

    # Negation / wind-down shortly before the noun: "no more clips,
    # what's the overall theme?", "don't give me any more clips".
    if any(w in _CLIP_NOUN_NEGATIONS for w in window):
        return True
    if any(w in _CLIP_NOUN_SHORT_NEGATIONS for w in tokens[max(0, i - 3):i]):
        return True
    # Time reference, plural AND singular: "a few moments ago you said…",
    # "a moment ago". Must run BEFORE the plural early-return below.
    if nxt == 'ago':
        return True
    # Idiom fillers around a singular noun: "wait a moment", "hold on one
    # moment", "hang on a moment", "she pauses for a moment".
    if prev in ('a', 'one'):
        if any(w in ('wait', 'hold', 'hang')
               for w in tokens[max(0, i - 4):i]):
            return True
        if prev2 in ('for', 'after', 'in'):
            return True
    # App-action asks about existing cards: "can you remove the second
    # clip?" wants an edit, not new material.
    if any(w in _CLIP_NOUN_EDIT_VERBS for w in window):
        return True
    # Back-references to what the assistant already did: "why DID YOU
    # PICK those clips?", "the quotes YOU PULLED are great". A modal
    # request ("can you find clips about…") stays extractive — only past
    # interrogatives ('did/have/had you' + a choice verb) and past-tense
    # delivery verbs guard. 'find' is deliberately absent from the
    # choice-verb set: "did you find any quotes about the fire?" is a
    # polite retrieval ask, not a meta question.
    for j in range(max(0, i - 5), i):
        if tokens[j] != 'you':
            continue
        j_prev = tokens[j - 1] if j > 0 else ''
        j_next = tokens[j + 1] if j + 1 < len(tokens) else ''
        if j_next in _CLIP_NOUN_DELIVERY_PAST:
            return True
        if j_prev in ('did', 'have', 'had') and j_next in (
                'pick', 'pull', 'choose', 'select', 'show', 'give',
                'suggest', 'recommend', 'list', 'send'):
            return True
    # First-person assessment: "I like the highlights so far."
    for j in range(max(0, i - 4), i):
        if tokens[j] in _CLIP_NOUN_ASSESSMENT_VERBS and j > 0 \
                and tokens[j - 1] in ('i', 'we'):
            return True
    # Determiner + noun + past-tense verb or 'you': "those clips were
    # perfect", "the quotes you pulled".
    if prev in ('those', 'these', 'the'):
        if nxt == 'you' or nxt in _CLIP_NOUN_PAST_STATE \
                or nxt in _CLIP_NOUN_DELIVERY_PAST:
            return True

    if tok in _PLURAL_CLIP_NOUNS:
        # Plural deictics discuss delivered cards ("what do these moments
        # have in common?") — UNLESS the message selects among them with
        # 'which' ("which of those moments is strongest?" wants a
        # re-ranked card) or counts them ("compare those two moments").
        det = ''
        if prev in ('those', 'these'):
            det = prev
        elif prev2 in ('those', 'these') \
                and prev not in _CLIP_NOUN_COUNT_WORDS \
                and not prev.isdigit():
            det = prev2
        if det and 'which' not in tokens[:i]:
            return True
        return False

    # Singular deictic guard ("at that moment she changes — why?"), with
    # a one-adjective gap ("in that same moment") — but CATAPHORIC
    # retrieval overrides it (H7): a relative clause after the noun
    # ("that moment WHERE he admits it") or a retrieval verb anywhere
    # before it ("I WANT that moment…", "can you LOCATE this quote…")
    # marks a fetch ask, not discussion of an already-identified point.
    deictic = prev in ('that', 'this') or (
        prev2 in ('that', 'this') and prev.isalpha())
    if not deictic:
        return False
    if nxt in ('where', 'when') or (nxt == 'in' and nxt2 == 'which'):
        return False
    if any(w in _CLIP_NOUN_RETRIEVAL_VERBS for w in tokens[:i]):
        return False
    return True


def _clip_noun_retrieval_targets(message):
    """The clip-seeking noun tokens in ``message`` that are genuine
    retrieval targets — every occurrence ``_clip_noun_is_reference``
    guards as conversational is skipped. Empty list → no clip ask."""
    tokens = re.findall(r"[a-z0-9']+", (message or '').lower())
    out = []
    for i, tok in enumerate(tokens):
        if tok not in _CLIP_SEEKING_NOUNS:
            continue
        if _clip_noun_is_reference(tokens, i):
            continue
        out.append(tok)
    return out


def _is_conversational_query(message: str, segments=None) -> bool:
    """Return True when the editor's message looks like discussion (themes,
    story, character, craft, opinion, chitchat) rather than clip extraction.

    Heuristic order:
      1. Empty / whitespace → conversational (let model handle gracefully)
      2. Explicit "no clips" instruction → conversational, hard signal
      3. Mentions a known speaker name → extractive (route to chunk search
         which actually scans the transcript for that speaker's content)
      4. Starts with an extractive verb → extractive
      5. Contains a clip-seeking noun (_CLIP_SEEKING_NOUNS) → extractive,
         question phrasing notwithstanding — with a narrow deictic guard
         (see the inline comment at the check)
      6. Default → conversational (matches the orientation: default to talk)

    The speaker-name check matters on long interviews: the conversational
    synthesis path has only a summary + digest, not the full transcript,
    so it can claim "I don't have her interview loaded" when asked about
    one specific speaker. Routing speaker-anchored questions through chunk
    search guarantees real transcript content reaches the answer.
    """
    if not message:
        return True
    msg = message.lower().strip()
    if not msg:
        return True
    for s in _NO_CLIP_SIGNALS:
        if s in msg:
            return True
    # Speaker-name anchor: any token in the message matches a known speaker.
    # Compare on first names too — editors say "do mae" not "do mae babcock".
    if segments:
        speaker_tokens = set()
        for spk in _extract_speaker_names(segments):
            for part in spk.lower().split():
                # Skip generic / role labels — only real names anchor
                if part in ('speaker', 'host', 'guest', 'interviewer',
                            'subject', 'narrator', 'unknown'):
                    continue
                if len(part) >= 3:
                    speaker_tokens.add(part)
        if speaker_tokens:
            # Word-boundary check so "mae" doesn't match "make" or "name"
            for tok in speaker_tokens:
                if re.search(rf'\b{re.escape(tok)}\b', msg):
                    return False
    # Strip leading punctuation/symbols, then look at first word + bigram
    stripped = msg.lstrip('"\'`([{ \t')
    first_token = stripped.split()[0].rstrip(',.!?:;') if stripped.split() else ''
    first_two = ' '.join(stripped.split()[:2]).rstrip(',.!?:;')
    for verb in _EXTRACTIVE_VERB_STARTS:
        # Match either "verb" or "verb me" / "show me" patterns
        if first_token == verb or first_two == verb:
            return False
    # Bare assembly verbs flip only with corroborating deliverable intent:
    # an anchored duration target ("make a 90 second teaser") or a
    # deliverable noun later in the message ("cut a highlight reel").
    # Without that anchor, "make sense of...", "cut to the chase...",
    # "create a description..." stay conversational.
    if first_token in _CONDITIONAL_ASSEMBLY_VERB_STARTS:
        if parse_target_duration_seconds(message) is not None:
            return False
        # Scan tokens AFTER the verb — "cut" itself is a deliverable noun
        # and must not self-anchor ("cut to the chase").
        rest_tokens = re.findall(r"[a-z0-9']+", stripped)[1:]
        if any(t in _DURATION_INTENT_NOUNS for t in rest_tokens):
            return False
    # Clip-seeking nouns flip the ask extractive even when it's phrased as
    # a question ("What are the strongest emotional moments?", "Are there
    # any quotes about the fire?") — those are retrieval asks wearing
    # question syntax, and a clip answer is what the editor wants. But a
    # noun MENTION is not a noun ASK: negated/wound-down asks ("no more
    # clips, what's the theme?"), back-references to cards already
    # delivered ("those clips were perfect", "the quotes you pulled"),
    # meta questions about the app's choices ("why did you pick those
    # clips?"), idioms/fillers ("wait a moment", "a few moments ago you
    # said…"), and deictic singulars ("at that moment she changes") all
    # stay discussion — the guard set lives in _clip_noun_is_reference,
    # which errs conversational by design (see the guiding-principle
    # comment above it). Cataphoric retrieval still flips ("I want that
    # moment where he admits it"). Non-deictic singulars ("what's the
    # best moment?") stay extractive — even mid-discussion uses ("what do
    # you make of the moment she cries?") benefit from a playable card,
    # and on the Layer-1 path the prose answer rides along with it.
    if _clip_noun_retrieval_targets(stripped):
        return False
    return True


# Speech-report / content-lookup questions: "what does she say about X",
# "did he mention the fire", "how does she describe the river". These are
# conversational in VOICE (the editor wants an answer, not a card dump)
# but extractive in SUBSTANCE — the answer must contain what was actually
# said, not a thematic gloss. The negative lookahead keeps craft questions
# aimed at the assistant ("what do you think about…") and interpretive
# follow-ups with demonstrative subjects ("what does that say about the
# piece?", "what do these clips tell us?") conversational — only an
# animate/named subject reports speech.
_CONTENT_LOOKUP_SUBJECT_GUARD = (
    r'(?!you\b|we\b|i\b|that\b|this\b|these\b|those\b|it\b)')
_CONTENT_LOOKUP_WH_RE = re.compile(
    r'\b(?:what|where|when|how)\b[^?.!\n]{0,40}?'
    r'\b(?:do|does|did)\s+' + _CONTENT_LOOKUP_SUBJECT_GUARD +
    r'\w+[^?.!\n]{0,30}?'
    r'\b(?:say|says|said|mention|mentions|mentioned|talk|talks|talked'
    r'|describe|describes|described|tell|tells|told|explain|explains'
    r'|explained|discuss|discusses|discussed)\b',
    re.IGNORECASE)
_CONTENT_LOOKUP_YN_RE = re.compile(
    r'\b(?:do|does|did)\s+' + _CONTENT_LOOKUP_SUBJECT_GUARD +
    r'\w+\s+(?:ever\s+|actually\s+)?'
    r'(?:mention|say|talk\s+about|bring\s+up|discuss|address|explain)\b',
    re.IGNORECASE)
# Shapes without do-support: passives ("what was said about the merger"),
# imperatives ("tell me what she says about X"), and factual-wh asks the
# system prompt names as content questions ("what year did that happen").
_CONTENT_LOOKUP_EXTRA_RES = (
    re.compile(r'\bwhat\s+(?:was|were|is|are)\s+(?:said|mentioned'
               r'|discussed|told|brought\s+up)\b', re.IGNORECASE),
    re.compile(r'\btell\s+me\s+what\b[^?.!\n]{0,40}?'
               r'\b(?:say|says|said|mention|mentions|mentioned|think'
               r'|thinks|thought)\b', re.IGNORECASE),
    re.compile(r'\bwhat\s+(?:year|date|day|month|time)\b[^?.!\n]{0,40}?'
               r'\b(?:do|does|did|was|were|is|are)\b', re.IGNORECASE),
)


def _is_content_lookup_query(message: str) -> bool:
    """True when the message asks what a SPEAKER said about something.

    Suppressed entirely by an explicit no-clips signal — "no clips, just
    tell me what she said" is a hard instruction the grounding tail and
    salvage backstop must not override with bolted-on cards.
    """
    if not message:
        return False
    low = message.lower()
    if any(s in low for s in _NO_CLIP_SIGNALS):
        return False
    return bool(_CONTENT_LOOKUP_WH_RE.search(message)
                or _CONTENT_LOOKUP_YN_RE.search(message)
                or any(r.search(message) for r in _CONTENT_LOOKUP_EXTRA_RES))


# ── Clip-aware chat (editor selections context) ──────────────────────────
#
# When the editor has clips in `labeled_sections`, we replace the
# pre-analyzed moments block with an enriched <editor_selections> block so
# the chat can act as an editorial partner (find gaps, flag overlaps,
# suggest running order) instead of just a transcript search tool.
#
# Toggle off via the env var when something breaks in the wild — one switch
# reverts to the legacy analysis-block behavior without rolling back a build.
def _clip_aware_chat_enabled() -> bool:
    val = os.environ.get('DOZA_DISABLE_CLIP_AWARE_CHAT', '').strip().lower()
    return val not in ('1', 'true', 'yes', 'on')


# Cap the clips block at the size of the analysis block it replaces.
# Measured on a representative 100-min project with full analysis (7 beats,
# 7 soundbites, 7 social clips): ~4,074 chars / ~1,072 tokens. The spec
# says "if the analysis block is smaller than 3,000 tokens, lower the
# clips ceiling to match" — 1,072 < 3,000 so we use that as the ceiling.
_CLIP_AWARE_MAX_CHARS = 4074
_CHARS_PER_TOKEN_EST = 3.8


def _seconds_to_hms_tc(secs) -> str:
    """Format float seconds as HH:MM:SS for editor_selections markers."""
    if not isinstance(secs, (int, float)) or secs < 0:
        return "00:00:00"
    s = int(secs)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{sec:02d}'


def _display_speaker(raw, speaker_names):
    """Resolve a raw speaker label to its display name via the project's
    ``speaker_names`` map (populated by the Pro diarization rename UI).

    Returns the raw label unchanged when no mapping exists, when ``raw`` is
    falsy, or when the mapped value is whitespace-only. Pure function — no
    side effects, never raises.
    """
    if not raw or not speaker_names:
        return raw
    try:
        custom = speaker_names.get(raw)
    except AttributeError:
        return raw
    if isinstance(custom, str) and custom.strip():
        return custom.strip()
    return raw


def _lookup_speaker_for_clip(clip, segments, speaker_names=None) -> str:
    """Return the speaker label for the first transcript segment whose
    timecode falls within the clip's [start, end] range. Returns '' when
    the transcript carries no speaker labels (single-speaker projects)
    or no segment overlaps.

    When ``speaker_names`` is provided, the raw segment speaker is resolved
    through that map before being returned — so renamed speakers surface
    their display name in clip enrichment (Step 6 of the diarization rollout).
    """
    try:
        start = float(clip.get('start') or 0)
        end = float(clip.get('end') or 0)
    except (TypeError, ValueError):
        return ''
    if end <= start:
        return ''
    for seg in segments or []:
        try:
            s = float(seg.get('start') or 0)
        except (TypeError, ValueError):
            continue
        if s >= start and s < end:
            spk = (seg.get('speaker') or '').strip()
            if spk:
                return _display_speaker(spk, speaker_names)
            return ''
    return ''


def _build_editor_selections_block(labeled_sections, segments, speaker_names=None) -> str:
    """Build the <editor_selections> XML block from labeled_sections.

    Each clip is enriched at prompt-assembly time (not at save time):
      1. Speaker looked up from the first transcript segment in the clip's range
      2. All clips sorted by start time
      3. Sequential index (1, 2, 3...) assigned in chronological order

    Token management: the total block is capped at _CLIP_AWARE_MAX_CHARS to
    match the analysis block it replaces. If we're over budget, each clip's
    text is truncated to the first 50 characters + "…".

    Returns "" when labeled_sections is empty so callers can concatenate
    unconditionally.
    """
    if not labeled_sections:
        return ''

    # Normalize + sort by start time
    clips = []
    for c in labeled_sections:
        if not isinstance(c, dict):
            continue
        try:
            start = float(c.get('start') or 0)
            end = float(c.get('end') or 0)
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        clips.append({
            'start': start,
            'end': end,
            'text': str(c.get('text') or '').strip(),
            'speaker': _lookup_speaker_for_clip(c, segments, speaker_names),
        })
    if not clips:
        return ''
    clips.sort(key=lambda x: x['start'])

    def _format(clips_, truncate_text=False) -> str:
        lines = []
        for i, c in enumerate(clips_, start=1):
            text = c['text']
            if truncate_text and len(text) > 50:
                text = text[:50].rstrip() + '…'
            elif not text:
                text = '(no transcript text saved)'
            block_lines = [
                f"[SELECTED {i}] [{_seconds_to_hms_tc(c['start'])}-{_seconds_to_hms_tc(c['end'])}]",
            ]
            if c['speaker']:
                block_lines.append(f"Speaker: {c['speaker']}")
            block_lines.append(f"Content: {text}")
            lines.append('\n'.join(block_lines))
        body = '\n\n'.join(lines)
        return (
            "<editor_selections>\n"
            "The editor has selected the following clips from the transcript, "
            "listed in chronological order:\n\n"
            f"{body}\n\n"
            f"Total selections: {len(clips_)}\n"
            "</editor_selections>"
        )

    full = _format(clips, truncate_text=False)
    if len(full) <= _CLIP_AWARE_MAX_CHARS:
        return full
    # Over budget — truncate clip text to 50 chars + ellipsis
    return _format(clips, truncate_text=True)


def _build_clip_aware_framing(my_style_active: bool) -> str:
    """Framing paragraph appended to the system prompt when clips exist.
    Two variants based on whether a My Style profile is active."""
    if my_style_active:
        return (
            "\n\nYou have three layers of context:\n"
            "1. The STYLE CONTEXT message describes how this editor builds stories based on their past work\n"
            "2. The transcript is the raw source material\n"
            "3. <editor_selections> are the clips the editor has already chosen for this project\n\n"
            "When the editor asks for suggestions, gaps, or sequence advice, always filter your "
            "recommendations through their storytelling foundation. Prioritize moments and structures "
            "that match their editorial patterns. When something breaks from their pattern, note it "
            "as a deliberate departure rather than correcting it."
        )
    return (
        "\n\nYou have two layers of context:\n"
        "1. The transcript is the raw source material\n"
        "2. <editor_selections> are the clips the editor has already chosen for this project\n\n"
        "When the editor asks for suggestions, gaps, or sequence advice, reason about coverage, "
        "redundancy, and narrative arc based on the clips they've selected and the surrounding "
        "transcript context."
    )


def _load_chat_system_prompt():
    """Load the master chat system prompt from prompts/chat-system-prompt.md.

    Runs once at module import. Extracts the content between the first pair
    of triple-backtick fences in the markdown file (so the surrounding
    document metadata — purpose, implementation notes — doesn't leak into
    the LLM context). Returns ``None`` if the file is missing; callers
    must surface the error visibly rather than silently degrading.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, 'prompts', 'chat-system-prompt.md')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return None
    parts = content.split('```')
    # Markdown ```...``` fenced block: parts[1] is the content between the
    # first opening and closing fence. Lower-index parts are the prose
    # before the fence; higher-index parts are everything after.
    if len(parts) >= 3:
        return parts[1].strip()
    # No fence found — return the whole document so the prompt is at least
    # present, even if surrounding prose leaks. Prefer this over silently
    # using nothing.
    return content.strip()


CHAT_SYSTEM_PROMPT = _load_chat_system_prompt()


def _format_duration_seconds(secs):
    """Format a float duration as H:MM:SS or M:SS — for the transcript header."""
    if not secs or secs <= 0:
        return '0:00'
    s = int(secs)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f'{h}:{m:02d}:{sec:02d}'
    return f'{m}:{sec:02d}'


def _extract_speaker_names(segments):
    """Pull a deduplicated, ordered list of speaker labels from a segment list."""
    seen = set()
    out = []
    for seg in segments or []:
        spk = (seg.get('speaker') or '').strip()
        if spk and spk not in seen:
            seen.add(spk)
            out.append(spk)
    return out


def _build_transcript_message(project_name, segments, formatted,
                              analysis_block, relevant_excerpts_block):
    """The user-role transcript message: opening declarative line so the
    LLM treats this as loaded data (not a request to provide a transcript),
    then the project header, then the word-level transcript and any
    analysis / relevant-excerpts blocks.

    The declarative opener matters for Gemma 4B's instruction-following:
    without it, the model sometimes reads the message as "user is about
    to paste a transcript" and replies asking the user to do so.
    """
    duration_sec = segments[-1].get('end', 0) if segments else 0
    parts = [
        "Here is the loaded project. Use this transcript to answer "
        "everything I ask after this message.",
        '',
        f'PROJECT: {project_name}',
        f'DURATION: {_format_duration_seconds(duration_sec)}',
    ]
    speakers = _extract_speaker_names(segments)
    if speakers:
        parts.append(f"SPEAKERS: {', '.join(speakers)}")
    parts.append('')
    parts.append('TRANSCRIPT:')
    parts.append(formatted)
    if analysis_block:
        parts.append(analysis_block)
    if relevant_excerpts_block:
        parts.append(relevant_excerpts_block)
    return '\n'.join(parts)


def _build_transcript_ack(project_name):
    """The fake assistant acknowledgement that follows the transcript
    message. Anchors the model in 'transcript already received' mode so
    the next user turn is treated as a question against the loaded data,
    not as the user pasting source material. Brief on purpose — anything
    longer becomes its own context drag."""
    return f"Transcript loaded for '{project_name}'. What would you like to find?"


# Post-transcript contract restatement, appended to the FINAL user turn.
# The full three-mode orientation lives in CHAT_SYSTEM_PROMPT, which on a long
# transcript sits tens of thousands of tokens above the generation point —
# small local models (Gemma 4B especially) have strong recency bias and drift
# back to prose summaries, the original long-FCPXML chat bug. Restating the
# contract after the transcript AND history puts it in the recency-correct
# position, immediately before generation. Adapted from the OSS v3.5.11
# `_FINAL_REMINDER` but reworded for Pro's richer contract: it restates all
# three modes (extractive / hybrid / conversational) from
# prompts/chat-system-prompt.md instead of a blanket "always emit markers",
# so Pro's conversational mode stays first-class. Constant text on the final
# message keeps the Ollama KV prefix stable across turns; the raw user
# message (without this tail) is what app.py persists to chat_history.
_FINAL_REMINDER = (
    'FINAL REMINDER (the full project context is above): when I ask you to '
    'find, pull, list, rank, or recommend moments — including hybrid '
    'questions like "what\'s the strongest theme and where does it live" — '
    'every specific moment you name must carry a '
    '[CLIP: start=HH:MM:SS end=HH:MM:SS title="short headline" note="one-line '
    'editorial reason"] marker built from the transcript timecodes, one per '
    'line, so I can play it. Never describe a moment as a clip without its '
    'marker, and never substitute a prose summary for requested markers. For '
    'purely conversational questions (story, themes, craft) answer normally — '
    'a marker only when a specific moment directly anchors your point. '
    'Whatever the question, GROUND your answer in this footage: name who is '
    'speaking, cite what they actually said (short verbatim quotes are '
    'welcome), and point at concrete moments — never generic film-speak that '
    'could describe any documentary.'
)


def _compact_history_turn(content, cap=1200):
    """Compact a replayed assistant turn: full [CLIP:] markers collapse to
    one-line "‣ title (start–end)" references and the prose is capped.

    Two independent wins: (a) history stops eating the context window —
    prior card-heavy answers ran to thousands of tokens each and, because
    Ollama evicts oldest-first on overflow, every history token pushed the
    transcript message closer to silent eviction; (b) the model stops
    seeing full marker grammar in history, so it re-emits old cards less
    on "more"-style follow-ups (the spans themselves stay recoverable —
    _history_clip_spans reads the RAW history, not this replay copy).
    """
    import re

    def _as_reference(m):
        full = m.group(0)
        sm = re.search(r'start=([^\s\]]+)', full)
        em = re.search(r'end=([^\s\]]+)', full)
        tm = re.search(r'title\s*=\s*["\'“‘](.*?)["\'”’]', full)
        if sm and em:
            title = (tm.group(1) if tm else 'clip').strip()
            return f'‣ {title} ({sm.group(1)}–{em.group(1)})'
        return full
    compact = re.sub(r'\[CLIP:[^\]]*\]', _as_reference, content)
    if len(compact) > cap:
        compact = compact[:cap].rstrip() + ' …'
    return compact


def _estimate_chat_num_ctx(system_message, messages, num_predict=4096):
    """Pick a num_ctx that holds the FULL assembled chat payload plus the
    reply budget.

    The predecessor (:func:`_estimate_layer1_num_ctx`) budgeted from the
    formatted transcript alone with a fixed 4096-token slack — but the
    chat system prompt by itself is ~4.5K tokens, and the analysis block,
    excerpts, style block, and history were never counted. The result on
    mid-length interviews was a window smaller than the prompt, and
    Ollama resolves that by silently DROPPING the oldest non-system
    message — the transcript itself. The model then answers from the
    system prompt and history alone: fluent, thematic, and completely
    ungrounded. This estimator measures what is actually sent.

    2.8 chars/token matches MEASURED gemma4 tokenization of real
    timecode-formatted English transcript payloads (2.82 observed via
    Ollama prompt_eval_count; plain prose runs looser but the transcript
    dominates the payload). Non-Latin scripts tokenize far denser — CJK
    measured ~1.5 chars/token — so payloads with a meaningful non-ASCII
    share get the tighter divisor. Overestimating num_ctx costs a little
    KV memory; underestimating silently evicts the transcript.
    """
    total_chars = len(system_message or '')
    non_ascii = 0
    for m in messages or []:
        content = m.get('content') or ''
        total_chars += len(content)
        non_ascii += sum(1 for ch in content if ord(ch) > 0x2FFF)
    divisor = 2.8
    if total_chars and (non_ascii / total_chars) > 0.15:
        divisor = 1.6
    prompt_tokens = int(total_chars / divisor) + 256  # +template overhead
    needed = prompt_tokens + num_predict
    for ctx in (8192, 12288, 16384, 24576, 32768):
        if needed <= ctx:
            return ctx
    # Above the 32K ceiling the payload no longer fits; warn loudly so an
    # over-long context stops being an invisible quality cliff.
    print(f"[chat] WARNING: assembled chat payload (~{prompt_tokens} tokens "
          f"+ {num_predict} reply) exceeds the 32768 num_ctx ceiling — "
          f"oldest context will be truncated by the model server", flush=True)
    return 32768


# Per-conversation num_ctx high-water marks. Ollama reloads the model
# runner whenever num_ctx CHANGES (measured ~2.5s reload + full prompt
# re-eval vs ~0.3s warm), so a conversation that bounces between rungs as
# history grows and excerpt blocks come and go pays the reload on every
# flip. Never shrink mid-conversation: grow-only per project.
_NUM_CTX_HWM: "OrderedDict[str, int]" = OrderedDict()
_NUM_CTX_HWM_MAX = 64


def _sticky_chat_num_ctx(project_name, system_message, messages):
    """Payload-aware num_ctx with a grow-only floor per project."""
    est = _estimate_chat_num_ctx(system_message, messages)
    key = str(project_name or '')
    prior = _NUM_CTX_HWM.get(key, 0)
    ctx = max(est, prior)
    _NUM_CTX_HWM[key] = ctx
    _NUM_CTX_HWM.move_to_end(key)
    while len(_NUM_CTX_HWM) > _NUM_CTX_HWM_MAX:
        _NUM_CTX_HWM.popitem(last=False)
    return ctx


def _build_chat_messages(message, history, project_name, segments,
                        formatted, analysis_block, relevant_excerpts_block,
                        profile_id, labeled_sections=None, speaker_names=None,
                        include_final_reminder=True, language_directive_text=''):
    """Construct the (system_message, messages_array) pair for an Ollama
    /api/chat call.

    Layout:
      system        : CHAT_SYSTEM_PROMPT + storytelling foundation
      user (opt.)   : STYLE CONTEXT — only when a My Style profile is active
      user          : PROJECT/DURATION/SPEAKERS/TRANSCRIPT block
      ...history... : prior user/assistant turns (capped at last 6)
      user          : the current user message + _FINAL_REMINDER

    ``include_final_reminder=False`` drops the contract restatement from the
    final turn. The Layer-2 conversational-synthesis divert passes False: its
    router already classified the question as discussion-style and it
    deliberately skips the clip-salvage post-processor, so nudging the model
    toward markers there would bolt clips onto answers Pro wants as prose.

    History cap: 6 entries (3 round-trips). Long transcripts already eat
    most of the context window; older turns rarely contribute editorial
    value beyond what's already in the system + transcript.

    Clip-aware swap: when ``labeled_sections`` is non-empty AND the
    feature flag is on, the analysis_block is replaced by an enriched
    <editor_selections> block in the transcript message, and a framing
    paragraph is appended to the system prompt. When the list is empty
    (or the flag is off), behavior is unchanged.
    """
    if CHAT_SYSTEM_PROMPT is None:
        raise RuntimeError(
            'Chat system prompt missing. Expected at '
            'core/prompts/chat-system-prompt.md.'
        )
    # Use the master chat prompt as the system message verbatim. The
    # storytelling foundation (master.md, ~76 KB) is intentionally NOT
    # appended here: the new chat prompt is self-contained for chat use,
    # and prepending the foundation pushes the system message past
    # Gemma 4B's effective context budget. Foundation continues to be
    # available to Story Builder and AI Analysis via inject_storytelling_foundation
    # in those separate code paths.
    system_message = CHAT_SYSTEM_PROMPT

    style_block = get_active_style_block(profile_id=profile_id)

    # Conditional swap: when the editor has clips, replace the
    # pre-analyzed moments block with the enriched editor_selections
    # block and add a framing paragraph to the system prompt.
    selections_block = ''
    if labeled_sections and _clip_aware_chat_enabled():
        try:
            selections_block = _build_editor_selections_block(labeled_sections, segments, speaker_names)
        except Exception as e:
            print(f"[chat] editor_selections build failed: {e}")
            selections_block = ''
        if selections_block:
            system_message = system_message + _build_clip_aware_framing(bool(style_block))
            # The clips block REPLACES the analysis block — same slot in
            # the transcript message, no double-injection.
            analysis_block = selections_block

    # Output-language directive rides at the very END of the system string
    # (after any clip-aware framing). Empty for English — plain concat
    # keeps English prompts byte-identical to pre-feature behavior.
    if language_directive_text:
        system_message = system_message + language_directive_text

    # History hygiene BEFORE assembly. The client pushes the current user
    # message into chatHistory before it fires the request, so the raw
    # history usually arrives with the current question already at its
    # tail — replaying it would show the model the question twice and burn
    # a history slot. Drop it here (also repairs old persisted histories).
    history = list(history or [])
    if history and history[-1].get('role') == 'user' and \
            (history[-1].get('content') or '').strip() == (message or '').strip():
        history = history[:-1]

    messages = []
    if style_block:
        messages.append({
            'role': 'user',
            'content': f'STYLE CONTEXT (active My Style profile):\n\n{style_block}',
        })

    transcript_msg = _build_transcript_message(
        project_name, segments, formatted, analysis_block, relevant_excerpts_block,
    )
    messages.append({'role': 'user', 'content': transcript_msg})
    # Fake assistant acknowledgement — anchors the model in "transcript
    # already received" mode. Without it Gemma 4B sometimes replies to the
    # next user turn with "I'd love to help, paste the transcript first."
    messages.append({
        'role': 'assistant',
        'content': _build_transcript_ack(project_name),
    })

    if history:
        for turn in history[-6:]:
            role = turn.get('role', 'user')
            content = (turn.get('content') or '').strip()
            if not content:
                continue
            if role not in ('user', 'assistant'):
                role = 'user'
            if role == 'assistant':
                content = _compact_history_turn(content)
            elif len(content) > 600:
                content = content[:600].rstrip() + ' …'
            messages.append({'role': role, 'content': content})

    if include_final_reminder:
        # Recency-correct contract restatement: after the transcript AND the
        # history, immediately before generation. Only the FINAL turn carries
        # it — replayed history turns come from chat_history, which stores the
        # raw message, so the KV prefix stays stable across turns.
        final_tail = _FINAL_REMINDER
        # Duration ask ("give me 2 minutes of selects"): one pre-computed
        # guidance line in the same recency slot. The clip count is derived
        # in code — small local models follow explicit counts but cannot
        # sum timecodes, so the arithmetic never rides on the model.
        target_seconds = parse_target_duration_seconds(message)
        if target_seconds and not _is_conversational_query(message, segments=segments):
            hint_clips = _duration_clip_count_hint(target_seconds)
            final_tail = (
                f'{final_tail}\n\nDURATION TARGET: I asked for about '
                f'{int(round(target_seconds))} seconds of material in total. '
                f'Suggest roughly {hint_clips} clips so their combined runtime '
                f'reaches that total — do not stop early.'
            )
        # Speech-report asks ("what does she say about X") ride the
        # conversational route but their answer contract is extractive:
        # report what was SAID. Same recency slot as the other contract
        # restatements — a Gemma-class model ignores this rule when it
        # only lives tens of thousands of tokens up in the system prompt.
        if _is_content_lookup_query(message):
            final_tail = (
                f'{final_tail}\n\nCONTENT QUESTION: answer with what was '
                f'actually said in the footage — name the speaker, quote '
                f'their exact words briefly (copied verbatim from the '
                f'transcript above), and put a [CLIP: start=HH:MM:SS '
                f'end=HH:MM:SS title="..."] marker after each passage you '
                f'cite so I can play it. No thematic summary without the '
                f'actual words.'
            )
        messages.append({'role': 'user', 'content': f'{message}\n\n{final_tail}'})
    else:
        messages.append({'role': 'user', 'content': message})
    return system_message, messages

# Bounded LRU for Layer 2 chunked-search responses. Layer 2 fires
# many _call_ai_json requests per chat query; if the user re-asks the
# same question against an unchanged transcript, every chunk hits the
# cache instead of re-billing the LLM. Keying on sha1 of each prompt
# plus the model variant means re-analysis (which produces fresh chunk
# text) busts the relevant entries automatically without explicit
# invalidation. 512 × ~3KB ≈ 1.5MB ceiling.
_CHUNK_CACHE_MAX = 512
_CHUNK_CACHE: "OrderedDict[tuple, str]" = OrderedDict()
_CHUNK_CACHE_LOCK = threading.Lock()


def chat_about_transcript(transcript, message, history=None, project_name="Interview",
                          analysis=None, profile_id=None, segment_vectors=None,
                          paragraph_index=None, labeled_sections=None,
                          speaker_names=None, output_language=None):
    """
    Chat with AI about the transcript. Supports follow-up questions.
    Returns the AI reply as a string (may contain embedded clip suggestions).

    ``segment_vectors`` is the optional pre-classified segment list emitted
    by ``generate_segment_vectors``. When provided, theme tags become an
    additional search vocabulary (so "show me the resilience moments" finds
    paragraphs flagged with that theme even when the speaker never said the
    word) and high-narrative-score segments get prioritized as relevant
    excerpts.

    ``paragraph_index`` is an optional :class:`doza_assist.retrieval.TfidfIndex`
    over the transcript's paragraphs. When supplied, the top cosine-similarity
    matches for the user's query feed into the relevant-excerpts pool. This
    is what catches abstract-synthesis queries that share no surface words
    with the transcript ("what's the most revealing moment?") — TF-IDF
    weights distinctive terms higher than common ones, so the result reads
    more like a topic-relevant subset than a substring match.

    Without both signals, behavior falls back to literal keyword matching —
    same as before this change.
    """
    segments = (transcript or {}).get('segments', [])
    duration = segments[-1].get('end', 0) if segments else 0
    # Resolved output-language directive ('' for English → zero diff) and
    # the title-anchor skip: cross-language replies carry translated clip
    # titles whose words won't literally appear in the transcript, so the
    # title-anchor validator sub-check would drop legitimate clips.
    directive = language_directive(output_language, chat=True)
    # The chat clause is for prose replies only — JSON-only sub-calls
    # (layer-2 chunked search, salvage) get the plain directive so the
    # "reply in the user's language" sentence can't fight their
    # "Return ONLY JSON" contracts.
    directive_plain = language_directive(output_language)
    skip_title_anchor = bool(output_language) and \
        output_language != (transcript or {}).get('language', 'en')
    phrases, words = _extract_query_keywords(message)
    theme_phrases = _collect_theme_phrases_from_vectors(segment_vectors, message)
    tfidf_hits = []
    # Only run TF-IDF when the message carries real content words. On
    # keyword-less questions ("whats this all about") the raw query is all
    # stopwords the index doesn't filter, so the "matches" were arbitrary
    # paragraphs — injected as RELEVANT EXCERPTS, they anchored whole-piece
    # questions to noise. Exception: keyword-less EXTRACTIVE count asks
    # ("give me 5 of the strongest moments") still need the ranked pool —
    # the count top-up draws candidates from it on projects that carry no
    # segment vectors.
    _count_ask = (_detect_explicit_clip_count(message) is not None
                  or _plural_clip_minimum(message) is not None)
    if paragraph_index is not None and (phrases or words or _count_ask):
        try:
            tfidf_hits = paragraph_index.query_paragraphs(message, k=8) or []
        except Exception as e:
            print(f"[chat] TF-IDF retrieval failed: {e}")
            tfidf_hits = []

    # ── Routing ──────────────────────────────────────────────────────────
    # > 60 min  → Layer 2 is the engine. Keywords (if any) are passed into
    #             each chunk prompt as a prioritization hint, but every
    #             chunk is searched regardless — the best setup for a topic
    #             may live in a neighboring chunk that doesn't contain the
    #             exact words. Bypasses the single-prompt path entirely.
    # ≤ 60 min  → Single-prompt path. If keywords matched, Layer 1 injects
    #             a RELEVANT EXCERPTS block between transcript and final
    #             reminder so recency bias reinforces the answer.
    # ─────────────────────────────────────────────────────────────────────
    if duration > _LONG_CHAT_SECONDS:
        # Conversational divert: long interviews normally go straight to
        # chunked clip search, which only ever returns clip cards. When the
        # editor's question is conversational ("what's the story", "no
        # clips just tell me", greetings), bypass the chunk search and run
        # a synthesis call against the analysis block + summary instead.
        # Pass segments so the classifier can detect speaker-name anchors
        # ("do mae", "what about Posey") and route those to chunk search.
        if _is_conversational_query(message, segments=segments):
            return _chat_layer2_conversational_synthesis(
                message, history, project_name, segments,
                analysis, profile_id, labeled_sections,
                speaker_names=speaker_names,
                language_directive_text=directive,
                skip_title_anchor=skip_title_anchor,
            )
        paragraphs = _build_paragraphs(transcript)
        return _chat_layer2_chunked_search(
            paragraphs, message, history, project_name,
            phrases, words, profile_id, analysis,
            segment_vectors=segment_vectors, theme_phrases=theme_phrases,
            tfidf_hits=tfidf_hits, speaker_names=speaker_names,
            language_directive_text=directive_plain,
        )

    formatted = _format_transcript_for_ai(transcript, speaker_names)
    relevant_excerpts_block = ''
    matched = []
    if phrases or words or theme_phrases or tfidf_hits:
        # Layer 1 on short interviews: segments have the same shape the
        # paragraph helpers expect, so the matcher runs against segments
        # directly with ±2 context.
        matched = _find_relevant_paragraphs(
            segments, phrases, words, context=2, theme_phrases=theme_phrases,
        )
        # Fold in TF-IDF top hits (over the paragraph corpus, not segments).
        # These come pre-ranked by cosine similarity and bring abstract
        # queries onto a sensible answer pool even with zero literal match.
        matched = _merge_paragraph_lists(matched, tfidf_hits)
        # Augment with high-narrative-score segments overlapping any literal
        # match — when vectors are present, these are pre-curated highlights.
        matched = _augment_with_high_score_vectors(matched, segments, segment_vectors, theme_phrases)
        relevant_excerpts_block = _build_relevant_excerpts_block(
            matched, synthesis=_is_synthesis_query(message),
        )
    analysis_block = _build_chat_analysis_index(analysis)

    system_message, messages = _build_chat_messages(
        message, history, project_name, segments,
        formatted, analysis_block, relevant_excerpts_block, profile_id,
        labeled_sections=labeled_sections, speaker_names=speaker_names,
        language_directive_text=directive,
    )
    num_ctx = _sticky_chat_num_ctx(project_name, system_message, messages)
    response = _call_ai_chat(system_message, messages, num_ctx=num_ctx)
    response = _strip_trailing_repetition(response)
    cleaned = _clean_chat_response(response)
    cleaned = _validate_clip_markers_in_text(
        cleaned, segments, skip_title_anchor=skip_title_anchor,
    )
    # Skip clip salvage when the editor's question is conversational
    # (themes, story, craft, chitchat, or explicit "no clips"). Forcing
    # markers into a discussion answer breaks the orientation contract.
    # Exception: content-lookup asks ("what does she say about X") keep
    # the conversational voice but promised playable passages — salvage
    # backstops them when the model cited nothing. Only when the reply
    # has actual prose, though: salvaging an EMPTY reply on a yes/no
    # content question ("did she ever mention X?") would fabricate an
    # implied 'yes' out of cards for a topic never discussed.
    if not _is_conversational_query(message, segments=segments) \
            or (_is_content_lookup_query(message) and cleaned.strip()):
        cleaned = _salvage_clips_if_missing(
            cleaned, formatted, segments, num_ctx=num_ctx,
            matched_paragraphs=matched, user_message=message,
            language_directive_text=directive_plain,
            skip_title_anchor=skip_title_anchor,
        )
    # Count-vs-duration ownership: a count-less duration ask ("1 minute
    # of selects" says nothing about clip count, and used to be misparsed
    # as clip-count 1 and hard-trimmed to ONE clip) hands the reply to the
    # deterministic duration pass. But when BOTH parse ("give me 5 clips,
    # about 2 minutes total"), the COUNT is the card-by-card promise the
    # user actually made — it owns the reply and the duration degrades to
    # a soft bound (duration enforcement is skipped rather than padding or
    # trimming past the promised count). Mirrored in Layer 2's final_top_k
    # and in collection chat.
    target_seconds = parse_target_duration_seconds(message)
    explicit_count = _detect_explicit_clip_count(message)
    extractive = not _is_conversational_query(message, segments=segments)
    if target_seconds is None or explicit_count is not None:
        # Enforce the clip-count contract from the user message. Gemma 4B
        # routinely ignores "1 clip" / "one more" / "another" and emits 2-3
        # (trim), and just as routinely under-delivers ("Give me 5 more" →
        # three cards) — the top-up half fills the gap deterministically
        # from the ranked pool, skipping moments earlier turns already
        # showed. Plural asks with no explicit count ("the strongest
        # emotional moments") guarantee at least _PLURAL_CLIP_MINIMUM cards
        # the same way. Top-up material is extractive-only: a conversational
        # aside that happens to parse a count ("just one thing — what's her
        # name?") must not grow clip cards, so it keeps the trim-only shape.
        # On "more"-style asks, markers the model RE-EMITS from history are
        # dropped as duplicates before counting ("5 more" = 5 NEW moments).
        cleaned = _enforce_clip_count(
            cleaned, explicit_count,
            candidates=_count_topup_pool(matched, segments, segment_vectors,
                                         message=message, history=history)
            if extractive else None,
            transcript=transcript,
            exclude_spans=_history_clip_spans(history),
            min_count=_plural_clip_minimum(message) if extractive else None,
            drop_reemitted=bool(
                _MORE_CLIPS_RE.search((message or '').lower())),
        )
    elif extractive:
        # History exclusion mirrors the count path: "give me another 2
        # minutes of selects" must top up with NEW footage, never re-issue
        # clips earlier turns already showed.
        cleaned = _enforce_duration_target(
            cleaned, target_seconds, matched, transcript=transcript,
            exclude_spans=_history_clip_spans(history),
        )
    return cleaned


_OLLAMA_STOP_TOKENS = [
    '\n[Thoughts]',
    '\n[Thought Process]',
    '\n[Reasoning]',
    '\n<think>',
    '\n<thinking>',
    '[/Response]\n[Thoughts]',
    '[/Response]\n[Thought',
    '[No specific answer',
    '[No answer',
]


def _call_ai_chat_stream(system_message, messages, num_ctx=32768):
    """Stream chat tokens through the active AI provider.

    ``system_message`` is the master system prompt (chat-system-prompt.md
    plus storytelling foundation). ``messages`` is the user/assistant
    array — STYLE CONTEXT (optional), TRANSCRIPT, history, current message.

    Yields ``str`` pieces as they arrive. On provider error, raises
    ``RuntimeError`` with a user-facing message so the SSE layer can
    surface it. The previous silent fall-back from Ollama to Anthropic via
    ``ANTHROPIC_API_KEY`` was removed — the user picks a provider and
    failures must be visible.
    """
    from ai_providers import get_active_provider
    provider = get_active_provider(model_resolver=_get_ollama_model)
    yield from provider.generate_stream(
        system_message, messages, task_type="chat", num_ctx=num_ctx,
    )


def chat_about_transcript_stream(transcript, message, history=None, project_name="Interview",
                                 analysis=None, profile_id=None, segment_vectors=None,
                                 paragraph_index=None, labeled_sections=None,
                                 speaker_names=None, output_language=None):
    """Streaming variant of :func:`chat_about_transcript`.

    Layer 1 yields ``('token', piece)`` events as Ollama produces them, then
    a single ``('done', cleaned_reply)`` once the response completes and
    the post-processing pipeline (marker normalization, timecode validation,
    markdown stripping) has run on the full text. Layer 2's chunked search
    isn't a single LLM stream, so it yields ``('progress', label)`` updates
    as chunks finish and a final ``('done', reply)`` with the formatted
    clips. Either way the consumer sees a steady stream and can render
    progress immediately instead of waiting on the full reply.

    On hard failure (no AI backend), yields ``('done', '')`` so the SSE
    client can show an error rather than hanging.
    """
    # Tell the UI we're alive before we touch any potentially-slow
    # retrieval helper. With a malformed segment the editor saw the chat
    # handler block silently for 30+ seconds before the ollama call even
    # started — this gives the SSE pump something to flush so the
    # "Thinking…" indicator updates and the client-side watchdog timer
    # has a recent event to anchor on.
    yield ('heartbeat', 'preparing')

    segments = (transcript or {}).get('segments', [])
    duration = segments[-1].get('end', 0) if segments else 0
    # Mirror of the non-streaming path: resolved language directive ('' for
    # English) plus the cross-language title-anchor skip for the validator.
    directive = language_directive(output_language, chat=True)
    # The chat clause is for prose replies only — JSON-only sub-calls
    # (layer-2 chunked search, salvage) get the plain directive so the
    # "reply in the user's language" sentence can't fight their
    # "Return ONLY JSON" contracts.
    directive_plain = language_directive(output_language)
    skip_title_anchor = bool(output_language) and \
        output_language != (transcript or {}).get('language', 'en')
    # Each retrieval step is wrapped so a single misbehaving helper can't
    # take the whole chat down. Failures degrade gracefully — empty result
    # + heartbeat — instead of bubbling up and blocking the SSE.
    try:
        phrases, words = _extract_query_keywords(message)
    except Exception as e:
        print(f"[chat-stream] keyword extraction failed: {e}")
        phrases, words = [], []
    try:
        theme_phrases = _collect_theme_phrases_from_vectors(segment_vectors, message)
    except Exception as e:
        print(f"[chat-stream] theme phrase collection failed: {e}")
        theme_phrases = []
    tfidf_hits = []
    # Keyword-or-count-ask gate mirrors the non-streaming path — see the
    # comment there.
    try:
        _count_ask = (_detect_explicit_clip_count(message) is not None
                      or _plural_clip_minimum(message) is not None)
    except Exception:
        _count_ask = False
    if paragraph_index is not None and (phrases or words or _count_ask):
        try:
            tfidf_hits = paragraph_index.query_paragraphs(message, k=8) or []
        except Exception as e:
            print(f"[chat-stream] tfidf query failed: {e}")
            tfidf_hits = []
    yield ('heartbeat', 'preparing')

    if duration > _LONG_CHAT_SECONDS:
        # Conversational divert (mirror of the non-streaming variant): when
        # the editor asked a discussion-style question, skip chunk search
        # and yield a synthesized prose answer instead. Speaker-name
        # anchors ("do mae", "Posey's story") count as extractive even
        # without a verb start, so they reach the chunk search.
        if _is_conversational_query(message, segments=segments):
            for event in _chat_layer2_conversational_synthesis_stream(
                message, history, project_name, segments,
                analysis, profile_id, labeled_sections,
                speaker_names=speaker_names,
                language_directive_text=directive,
                skip_title_anchor=skip_title_anchor,
            ):
                yield event
            return
        # Layer 2 is parallel by chunk. Token-stream isn't meaningful, but
        # per-chunk progress is — yield a counter as each chunk completes
        # so the UI can show "3/8 chunks searched…" instead of a frozen
        # spinner for 30 seconds. Final reply is the same as the
        # non-streaming variant.
        paragraphs = _build_paragraphs(transcript)
        for event in _chat_layer2_chunked_search_stream(
            paragraphs, message, history, project_name,
            phrases, words, profile_id, analysis,
            segment_vectors=segment_vectors, theme_phrases=theme_phrases,
            tfidf_hits=tfidf_hits, speaker_names=speaker_names,
            language_directive_text=directive_plain,
        ):
            yield event
        return

    # Layer 1: build messages, stream tokens. Single source of truth for
    # prompt construction is _build_chat_messages above; the streaming
    # variant just wraps the same call with token-level filtering.
    try:
        formatted = _format_transcript_for_ai(transcript, speaker_names)
    except Exception as e:
        print(f"[chat-stream] transcript format failed: {e}")
        formatted = ''
    relevant_excerpts_block = ''
    matched = []
    if phrases or words or theme_phrases or tfidf_hits:
        try:
            matched = _find_relevant_paragraphs(
                segments, phrases, words, context=2, theme_phrases=theme_phrases,
            )
            matched = _merge_paragraph_lists(matched, tfidf_hits)
            matched = _augment_with_high_score_vectors(matched, segments, segment_vectors, theme_phrases)
            relevant_excerpts_block = _build_relevant_excerpts_block(
                matched, synthesis=_is_synthesis_query(message),
            )
        except Exception as e:
            print(f"[chat-stream] relevance retrieval failed: {e}")
            matched = []
            relevant_excerpts_block = ''
    try:
        analysis_block = _build_chat_analysis_index(analysis)
    except Exception as e:
        print(f"[chat-stream] analysis index build failed: {e}")
        analysis_block = ''
    yield ('heartbeat', 'preparing')

    system_message, messages = _build_chat_messages(
        message, history, project_name, segments,
        formatted, analysis_block, relevant_excerpts_block, profile_id,
        labeled_sections=labeled_sections, speaker_names=speaker_names,
        language_directive_text=directive,
    )

    pieces = []
    num_ctx = _sticky_chat_num_ctx(project_name, system_message, messages)
    _think_buf = ''
    _in_thinking = False
    _THINK_OPENERS = ('<think>', '<thinking>', '[Thoughts]', '[Thought Process]', '[Reasoning]')
    _THINK_CLOSERS = ('</think>', '</thinking>', '[/Thoughts]', '[/Thought Process]', '[/Reasoning]')

    # Repetition-loop detector: if the last N tokens of visible output
    # repeat the same short phrase, the model has degenerated — stop early.
    _rep_tail = ''
    _REP_WINDOW = 200  # chars to keep
    _REP_MIN_PATTERN = 4  # shortest repeating unit (chars)
    _REP_THRESHOLD = 6   # must repeat this many times to trigger

    def _is_stuck(tail):
        """Return True if *tail* ends with the same short phrase repeated."""
        if len(tail) < _REP_MIN_PATTERN * _REP_THRESHOLD:
            return False
        for plen in range(_REP_MIN_PATTERN, len(tail) // _REP_THRESHOLD + 1):
            pat = tail[-plen:]
            # Punctuation-only units are usually legitimate structure —
            # markdown table rows, '----' dividers, '. . .' ellipses — so
            # they get a 5× trip threshold instead of the letter one. (No
            # real divider repeats a letterless unit 30× back-to-back; a
            # degenerated model does, and with no cutoff at all it would
            # flood until num_predict exhausts.)
            threshold = (_REP_THRESHOLD if any(c.isalpha() for c in pat)
                         else _REP_THRESHOLD * 5)
            count = 0
            pos = len(tail) - plen
            while pos >= 0 and tail[pos:pos + plen] == pat:
                count += 1
                pos -= plen
            if count >= threshold:
                print(f"[chat-stream] repetition cutoff fired on pattern "
                      f"{pat!r} x{count}", flush=True)
                return True
        return False

    # Heartbeat bookkeeping. When the model spends long stretches in a
    # ``<think>...</think>`` phase, this loop consumes pieces silently and
    # the SSE connection has nothing to flush. Browsers see "connection is
    # alive but quiet" — typing dots forever. We emit a synthetic
    # ``heartbeat`` event every ~25 silent pieces so the client knows
    # we're still processing and so any intermediate buffering layer
    # actually flushes bytes.
    _piece_count = 0
    _last_yield_piece = 0
    _HEARTBEAT_PIECES = 25

    # The stream can die mid-generation (read-timeout between bytes, an
    # Ollama restart). Without the catch, the exception escaped the
    # generator BEFORE the ('done', cleaned) event — the client kept the
    # raw, un-postprocessed token tail forever. Now whatever arrived
    # still flows through the full cleaning pipeline below.
    try:
        for piece in _call_ai_chat_stream(system_message, messages, num_ctx=num_ctx):
            _piece_count += 1
            pieces.append(piece)
            _think_buf += piece
            if not _in_thinking:
                if any(op in _think_buf for op in _THINK_OPENERS):
                    _in_thinking = True
                    _think_buf = ''
                    # Tell the UI the model is in an internal reasoning phase
                    # so it can show a less-confusing label than "Thinking…".
                    yield ('heartbeat', 'reasoning')
                    _last_yield_piece = _piece_count
                else:
                    for tag in _THINK_OPENERS:
                        if any(_think_buf.endswith(tag[:i]) for i in range(1, len(tag))):
                            break
                    else:
                        yield ('token', piece)
                        _last_yield_piece = _piece_count
                        _think_buf = _think_buf[-30:] if len(_think_buf) > 30 else _think_buf
                        _rep_tail = (_rep_tail + piece)[-_REP_WINDOW:]
                        if _is_stuck(_rep_tail):
                            break
            else:
                if any(cl in _think_buf for cl in _THINK_CLOSERS):
                    _in_thinking = False
                    _think_buf = ''
                    yield ('heartbeat', 'composing')
                    _last_yield_piece = _piece_count

            if _piece_count - _last_yield_piece >= _HEARTBEAT_PIECES:
                yield ('heartbeat', 'reasoning' if _in_thinking else 'composing')
                _last_yield_piece = _piece_count
    except Exception as e:
        if not pieces:
            # Nothing arrived at all — a pre-first-token failure (Ollama
            # down, model missing, OOM) must PROPAGATE so the SSE layer
            # surfaces its typed error message; swallowing it left the
            # user a permanently frozen 'Thinking…' bubble. Only a
            # mid-stream death with partial text is worth salvaging
            # through the cleaning pipeline below.
            raise
        print(f"[chat-stream] stream aborted mid-generation: {e}", flush=True)

    # Strip any trailing repetition the model produced before we cut it off.
    full = ''.join(pieces)
    full = _strip_trailing_repetition(full)
    cleaned = _clean_chat_response(full)
    cleaned = _validate_clip_markers_in_text(
        cleaned, segments, skip_title_anchor=skip_title_anchor,
    )
    # Salvage pass mirrors the non-streaming path. Adds latency only when
    # the model's first attempt produced zero markers — most calls return
    # immediately. Streamed clients see a brief pause after the prose
    # finishes, then the marker block lands as part of the final message.
    # Skipped on conversational queries — see _is_conversational_query —
    # except content-lookup asks with surviving prose, which promised
    # playable passages (empty-reply guard: see the non-streaming path).
    if not _is_conversational_query(message, segments=segments) \
            or (_is_content_lookup_query(message) and cleaned.strip()):
        cleaned = _salvage_clips_if_missing(
            cleaned, formatted, segments, num_ctx=num_ctx,
            matched_paragraphs=matched, user_message=message,
            language_directive_text=directive_plain,
            skip_title_anchor=skip_title_anchor,
        )
    # Same count-vs-duration fork the non-streaming path applies: an
    # explicit count owns the reply even when a duration also parses (the
    # duration becomes a soft bound); only a count-less duration ask hands
    # the reply to the duration pass. The deterministic passes run in the
    # same post-stream slot the salvage pass already occupies.
    target_seconds = parse_target_duration_seconds(message)
    explicit_count = _detect_explicit_clip_count(message)
    extractive = not _is_conversational_query(message, segments=segments)
    if target_seconds is None or explicit_count is not None:
        # Trim over-delivery AND top up under-delivery against the user's
        # count (plus the plural-ask minimum) — same defense as the
        # non-streaming path. See _enforce_clip_count.
        cleaned = _enforce_clip_count(
            cleaned, explicit_count,
            candidates=_count_topup_pool(matched, segments, segment_vectors,
                                         message=message, history=history)
            if extractive else None,
            transcript=transcript,
            exclude_spans=_history_clip_spans(history),
            min_count=_plural_clip_minimum(message) if extractive else None,
            drop_reemitted=bool(
                _MORE_CLIPS_RE.search((message or '').lower())),
        )
    elif extractive:
        # History exclusion mirrors the count path — see the
        # non-streaming variant.
        cleaned = _enforce_duration_target(
            cleaned, target_seconds, matched, transcript=transcript,
            exclude_spans=_history_clip_spans(history),
        )
    yield ('done', cleaned)


def _call_ai_chat(system_message, messages, num_ctx=32768):
    """Non-streaming chat call through the active provider.

    Same input shape as :func:`_call_ai_chat_stream` — a system string plus
    a messages array. Returns the full reply as a single string.
    """
    from ai_providers import get_active_provider
    provider = get_active_provider(model_resolver=_get_ollama_model)
    return provider.generate(
        system_message, messages, task_type="chat", num_ctx=num_ctx,
    )


def _count_clip_markers(text):
    """How many ``[CLIP:]`` markers (with at least the ``start=`` field set)
    does the text contain? Empty placeholders like ``[CLIP]`` do NOT count —
    those are model-emitted junk that need to be stripped, not preserved.
    """
    if not text:
        return 0
    import re
    return len(re.findall(r'\[CLIP:[^\]]*?start=', text))


# Negative lookahead shared by every count pattern that captures a number:
# a number attached to a time unit is a DURATION, not a clip count ("give
# me 1 minute of selects" must never become clip-count 1 and get trimmed
# to a single clip). parse_target_duration_seconds owns those asks.
_NOT_TIME_UNIT = r'(?![\s-]*(?:minutes?|mins?|seconds?|secs?|hours?|hrs?)\b)'
# Variant that also rejects "N more <unit>" ("give me 30 more seconds of
# selects" is a duration ask, never clip-count 30). Used by the request-
# verb digit pattern, whose bare digit would otherwise stop looking at
# the intervening "more".
_NOT_TIME_UNIT_THROUGH_MORE = (
    r'(?![\s-]*(?:more[\s-]+)?(?:minutes?|mins?|seconds?|secs?|hours?|hrs?)\b)'
)


# Clip-noun alternation shared by the explicit-count patterns — the same
# noun set the intent flip trusts (_CLIP_SEEKING_NOUNS, plus 'excerpts'
# to match _parse_user_clip_count). "compare the two MOMENTS where…"
# promises exactly two cards the same way "two clips" does; parsing only
# next-to-"clips" made Layer 1 fall back to the plural 3-minimum and
# append an unrequested third card.
_COUNT_CLIP_NOUNS = r'(?:clips?|moments?|quotes?|soundbites?|highlights?|excerpts?)'
_COUNT_ADJ = r'(?:great\s+|strong\s+|best\s+)?'
# "…moments AGO" is a time reference, never a count: "a few moments ago
# you said…" must not parse count=3 (and "2 moments ago" not count=2).
_NOT_AGO = r'(?!\s+ago\b)'
# Variant for patterns that end BEFORE the optional noun ("a few",
# "several"): reject when a clip noun + "ago" follows.
_NOT_NOUN_AGO = (
    r'(?!(?:\s+' + _COUNT_CLIP_NOUNS + r')?\s+ago\b)'
)

# Count-RANGE pieces ("8-10 strongest standalone soundbites", "8 to 10
# clips", "eight to ten moments"). A unit after the SECOND number makes
# the range a DURATION ("15 to 20 minute cut", "8-10s") — this lookahead
# rejects it so parse_target_duration_seconds keeps ownership. The bare
# [smh] letters catch the spaced-shorthand forms ("8-10 s") that the
# word units can't; the attached form ("8-10s") is rejected structurally
# by the (?!\d)\b anchor after the second number (see below).
_RANGE_SEP = r'(?:\s*[-–—]\s*|\s+to\s+)'
_RANGE_NOT_TIME_UNIT = (
    r'(?![\s-]*(?:more[\s-]+)?'
    r'(?:minutes?|mins?|seconds?|secs?|hours?|hrs?|[smh]\b)\b)'
)
# The second number must END at a word boundary with no digits left over
# ((?!\d)\b). Without it, regex backtracking re-splits "8-10 second
# clips" as lo=8 hi=1 with "0 second clips" unconsumed — the time-unit
# lookahead then inspects "0…" instead of the unit, the duration ask is
# stolen as count=5, and "15 to 20 minutes of the best material" parses
# hi=2 → count=9 (F1). Every digit-range pattern appends this.
_RANGE_END_ANCHOR = r'(?!\d)\b'
# Up to three filler words between the range and the clip noun ("8-10
# STRONGEST STANDALONE soundbites", "8 to 10 OF THE BEST moments"). Lazy,
# and only reachable when the time-unit lookahead already passed — so the
# filler can never skip over a duration unit.
_RANGE_NOUN_FILLER = r"(?:\s+[\w'-]+){0,3}?"
_RANGE_WORD_NUMBERS = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
}
_RANGE_WORD_ALT = '|'.join(_RANGE_WORD_NUMBERS)
# Hyphen-tolerant ([\s-]+): "eight to ten-minute segments" attaches the
# unit with a hyphen — a duration ask, never count=9 (F1).
_RANGE_WORD_NOT_TIME_UNIT = (
    r'(?![\s-]+(?:more[\s-]+)?(?:minutes?|mins?|seconds?|secs?|hours?|hrs?)\b)'
)


def _detect_clip_count_range(msg):
    """Parse a numeric clip-count RANGE from an already-lowercased message
    and return the midpoint rounded UP (8-10 → 9), or ``None``.

    Live tester bug (1.0.30 screenshot): "Identify the 8-10 strongest
    standalone soundbites…" parsed no count at all, fell back to the
    plural 3-minimum, and shipped THREE cards against an 8-10 ask.
    Recognized shapes — digit and word forms, hyphen/dash/"to"
    separators — anchored to a clip noun ("8-10 … soundbites") or to a
    request verb ("give me 8-10"). Unit-suffixed ranges are durations,
    never counts ("15 to 20 minute cut", "8-10s") — see
    ``_RANGE_NOT_TIME_UNIT``.
    """
    # Digit range + clip noun: "8-10 strongest standalone soundbites".
    m = re.search(
        r'\b(\d{1,2})' + _RANGE_SEP + r'(\d{1,2})' + _RANGE_END_ANCHOR
        + _RANGE_NOT_TIME_UNIT
        + _RANGE_NOUN_FILLER + r'\s+' + _COUNT_CLIP_NOUNS + r'\b' + _NOT_AGO,
        msg,
    )
    if not m:
        # Digit range after a request verb, no noun needed: "give me 8-10".
        m = re.search(
            r'\b(?:find|give|pull|show|get|make|identify|pick|select)\s+'
            r'(?:me\s+|us\s+)?(?:the\s+)?'
            r'(\d{1,2})' + _RANGE_SEP + r'(\d{1,2})' + _RANGE_END_ANCHOR
            + _RANGE_NOT_TIME_UNIT + _NOT_AGO,
            msg,
        )
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return max(1, -(-(lo + hi) // 2))  # midpoint, rounded up

    # Word range + clip noun: "eight to ten moments".
    m = re.search(
        r'\b(' + _RANGE_WORD_ALT + r')\s+to\s+(' + _RANGE_WORD_ALT + r')'
        + _RANGE_WORD_NOT_TIME_UNIT
        + _RANGE_NOUN_FILLER + r'\s+' + _COUNT_CLIP_NOUNS + r'\b' + _NOT_AGO,
        msg,
    )
    if not m:
        # Word range after a request verb: "pull eight to ten".
        m = re.search(
            r'\b(?:find|give|pull|show|get|make|identify|pick|select)\s+'
            r'(?:me\s+|us\s+)?(?:the\s+)?'
            r'(' + _RANGE_WORD_ALT + r')\s+to\s+(' + _RANGE_WORD_ALT + r')'
            + _RANGE_WORD_NOT_TIME_UNIT + _NOT_AGO,
            msg,
        )
    if m:
        lo = _RANGE_WORD_NUMBERS[m.group(1)]
        hi = _RANGE_WORD_NUMBERS[m.group(2)]
        return max(1, -(-(lo + hi) // 2))
    return None


def _detect_explicit_clip_count(message):
    """Parse an explicit clip count from a user message, or return ``None``.

    Pattern coverage matches the prompt's CLIP COUNT rules:
    - "1 clip", "2 moments", "3 quotes" (digits + any clip-seeking noun)
    - "one clip", "two moments" through "five soundbites" (words)
    - "both moments" / "both quotes" → 2
    - "give me one", "the best one", "just one" → 1
    - "one more", "another (clip|one)" → 1
    - "a few more", "some more" → 3 (upper bound of "a few")

    Numbers attached to time units never count ("give me 1 minute",
    "two more minutes", "give me 30 more seconds of selects") — see
    ``_NOT_TIME_UNIT``, which every digit pattern applies AFTER consuming
    an optional intervening "more". Numbers attached to "ago" never count
    either ("a few moments ago you said…" is a memory reference) — see
    ``_NOT_AGO`` / ``_NOT_NOUN_AGO``.

    Returns ``None`` if no explicit count is detectable — the model uses
    its judgment in that case (1-4 typical per the prompt).
    """
    if not message:
        return None
    import re
    msg = message.lower().strip()

    # Ranges FIRST — "8 to 10 clips" would otherwise half-match the
    # single-digit pattern below as count=10 ("10 clips"), and "8-10
    # strongest standalone soundbites" wouldn't match at all (falling to
    # the plural 3-minimum — the 1.0.30 screenshot bug).
    rng = _detect_clip_count_range(msg)
    if rng is not None:
        return rng

    # "1 clip" / "2 moments" / "3 quotes" / etc.
    m = re.search(
        r'\b(\d+)' + _NOT_TIME_UNIT + r'\s+' + _COUNT_ADJ
        + _COUNT_CLIP_NOUNS + r'\b' + _NOT_AGO,
        msg,
    )
    if m:
        return max(1, int(m.group(1)))

    # "1 more" / "2 more" / "3 more" — digit + "more" (with optional clip
    # noun after). The user means N additional clips. Same parse as
    # "1 clip" but with "more" as the noun.
    m = re.search(
        r'\b(\d+)\s+more(?:\s+' + _COUNT_CLIP_NOUNS + r')?\b'
        + _NOT_TIME_UNIT + _NOT_AGO,
        msg,
    )
    if m:
        return max(1, int(m.group(1)))

    # "find me 3" / "give me 2" / "pull 4" — bare digit after a request
    # verb. The time-unit lookahead must see THROUGH an intervening
    # "more": "give me 30 more seconds of selects" is a duration ask, and
    # with the plain lookahead, "seconds" hid behind the "more" and the
    # message parsed count=30 → a 30-card wall from the top-up (H11).
    # The see-through lives INSIDE the lookahead (not as a consumed
    # optional group) because regex backtracking un-consumes an optional
    # "(?:\s+more)?" whenever consuming it would fail the lookahead.
    # A digit that HEADS a range ("give me 8-10 second clips", "give me
    # 15 to 20 minutes…") is never a bare count: the range parser above
    # owns ranges, and when it rejected the message (unit-attached second
    # number = a duration ask) this fallback must not seize the range's
    # first number as count=8/15 (F1).
    m = re.search(
        r'\b(?:find|give|pull|show|get|make)\s+(?:me\s+)?(\d+)\b'
        r'(?!\s*[-–—]\s*\d|\s+to\s+\d)'
        + _NOT_TIME_UNIT_THROUGH_MORE + _NOT_AGO,
        msg,
    )
    if m:
        return max(1, int(m.group(1)))

    word_to_num = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}
    m = re.search(
        r'\b(one|two|three|four|five)\s+' + _COUNT_ADJ
        + _COUNT_CLIP_NOUNS + r'\b' + _NOT_AGO,
        msg,
    )
    if m:
        return word_to_num[m.group(1)]

    # "both moments" / "compare both quotes about the fire" → exactly 2.
    if re.search(
        r'\bboth\s+(?:of\s+the\s+)?' + _COUNT_ADJ + _COUNT_CLIP_NOUNS
        + r'\b' + _NOT_AGO,
        msg,
    ):
        return 2

    # Word-form + "more": "two more", "three more clips", etc.
    m = re.search(
        r'\b(one|two|three|four|five)\s+more(?:\s+'
        + _COUNT_CLIP_NOUNS + r')?\b' + _NOT_TIME_UNIT + _NOT_AGO,
        msg,
    )
    if m:
        return word_to_num[m.group(1)]

    # Single-clip phrases. "another" alone (without "clip") still implies 1
    # in this domain — every chat turn is about clips. "Strongest"/"best"
    # require a singular noun after to avoid hitting "strongest moments"
    # (plural) which means several.
    if re.search(
        r'\b(?:one\s+more(?:\s+clip)?'
        r'|another(?:\s+(?:clip|one))?'
        r'|just\s+one(?:\s+clip)?'
        r'|give\s+me\s+one(?:\s+clip)?'
        r'|find\s+me\s+one(?:\s+clip)?'
        r'|the\s+single\s+(?:best|strongest)'
        r'|the\s+(?:best|strongest)\s+(?:one|clip|moment|single))\b'
        + _NOT_TIME_UNIT,
        msg,
    ):
        return 1

    # "A few more" / "some more" — cap at 3. "A few moments AGO you
    # said…" is a memory reference, not a count (_NOT_NOUN_AGO).
    if re.search(
        r'\b(?:a\s+few(?:\s+more)?|some\s+more|several)\b'
        + _NOT_TIME_UNIT + _NOT_NOUN_AGO,
        msg,
    ):
        return 3

    return None


def _plural_clip_minimum(message):
    """Minimum marker count for a PLURAL clip-noun ask with no explicit
    count ("the strongest emotional moments", "show me the best quotes").

    Plural phrasing promises SEVERAL cards, but without a parsed count the
    reply carried however many markers the model happened to emit — the
    live tester screenshot: prose says "I've pulled three moments" with
    ONE card under it. Returns ``_PLURAL_CLIP_MINIMUM`` when a plural
    clip-seeking noun appears, else ``None``.

    Only consulted when ``_detect_explicit_clip_count`` returned ``None``
    (an explicit count always wins) and the message classified extractive
    — both enforced at the call sites, mirroring how the duration pass is
    gated. The minimum never trims: a model that volunteers five moments
    for a plural ask keeps all five (see ``_enforce_clip_count``).

    Uses the same reference guards as the intent flip
    (``_clip_noun_is_reference``): a plural noun that only BACK-REFERENCES
    delivered cards or sits in an idiom ("give me more of what you showed
    a few moments ago") promises nothing and must not force a 3-card
    floor onto the reply.
    """
    if not message:
        return None
    for tok in _clip_noun_retrieval_targets(message):
        if tok in _PLURAL_CLIP_NOUNS:
            return _PLURAL_CLIP_MINIMUM
    return None


# Marker pattern shared by the server-side enforcement passes. Tolerates
# ']' inside quoted attribute values (note="he said [wow] ..." ) — the
# naive [^\]]* form stops at the first ']' and leaves malformed residue
# behind when such a marker is trimmed. Quoted runs are matched as whole
# units; both alternatives exclude newlines so a marker with a broken
# quote can never swallow the rest of the reply.
_CLIP_MARKER_RE = re.compile(r'\[CLIP:(?:"[^"\n]*"|[^\]"\n])*\]')


def _strip_trimmed_clip_tail(tail):
    """Remove every [CLIP:] marker in ``tail`` along with the per-clip
    prose attached to it: the marker's own line and the explanation
    lines that immediately follow it (up to the next blank line). Prose
    that precedes the first trimmed marker belongs to the last SURVIVING
    clip and is kept, as is unrelated prose after a note paragraph
    ("Overall these show her arc."). Used by both trim passes —
    _enforce_clip_count and _enforce_duration_target — so a trimmed
    reply never describes clips that no longer exist.
    """
    if not tail:
        return tail
    kept = []
    dropping = False  # consuming a trimmed marker's attached note lines
    for line in tail.splitlines():
        if _CLIP_MARKER_RE.search(line):
            dropping = True
            continue
        if not line.strip():
            dropping = False  # a blank line ends the trimmed clip's note
            kept.append(line)
            continue
        if dropping:
            continue
        kept.append(line)
    out = '\n'.join(kept)
    # Drop any leftover "Here's another:" / "Plus:" connector lines that
    # introduced the trimmed clips. A connector is a short line ending in
    # a colon with nothing meaningful on either side.
    out = re.sub(r'\n\s*[A-Z][^.\n]{0,40}:\s*\n', '\n', out)
    # Collapse the blank-line runs the dropped paragraphs leave behind.
    out = re.sub(r'\n[ \t]*\n(?:[ \t]*\n)+', '\n\n', out)
    return out


def _history_clip_spans(history):
    """``(start, end, group)`` triples for every [CLIP:] marker in prior
    conversation turns.

    ``history`` is the chat-history list the pipeline already receives —
    dicts with ``role``/``content``, where assistant turns store the raw
    reply text including its markers. Both roles are scanned: a marker in
    either side means that moment is already on the table. ``group`` is
    the marker's ``project="..."`` field when present (the collection
    pipeline's multi-timeline tag), ``None`` on single-project markers —
    which conservatively collide with every group in
    ``_grouped_spans_overlap``. Feed the result to ``_enforce_clip_count``
    as ``exclude_spans`` so "give me 5 more" yields 5 NEW moments instead
    of re-issuing cards the editor has already seen.
    """
    spans = []
    for turn in (history or []):
        if not isinstance(turn, dict):
            continue
        content = turn.get('content') or ''
        if '[CLIP' not in content:
            continue
        for m in _CLIP_MARKER_RE.finditer(content):
            sm = re.search(r'start=([\d:.]+)', m.group(0))
            em = re.search(r'end=([\d:.]+)', m.group(0))
            if not (sm and em):
                continue
            s = _tc_to_seconds(sm.group(1))
            e = _tc_to_seconds(em.group(1))
            gm = re.search(r'project="([^"]*)"', m.group(0))
            grp = (gm.group(1).strip() if gm else '') or None
            spans.append((s, max(s, e), grp))
    return spans


# Hard cap on the count top-up's append goal. _parse_user_clip_count
# already clamps to 1-10; this is the same bound applied defensively at
# the enforcement layer so a misparsed count (the "give me 30 more
# seconds" family, or any future parser gap) can never append dozens of
# cards. The TRIM half is uncapped — trimming to a large ask is harmless.
_COUNT_TOPUP_MAX = 10


def _normalize_exclude_spans(exclude_spans):
    """``exclude_spans`` input (any iterable of (start, end[, group])) →
    clean ``(start, end, group)`` float triples, malformed entries
    skipped. Shared by the count and duration enforcement passes."""
    out = []
    for span in (exclude_spans or []):
        try:
            s, e = float(span[0]), float(span[1])
        except (TypeError, ValueError, IndexError):
            continue
        grp = span[2] if len(span) > 2 else None
        out.append((s, max(s, e), grp))
    return out


def _drop_reemitted_clip_markers(text, exclude, group_key=None):
    """Remove model-emitted [CLIP:] markers that RE-ISSUE a span already
    shown earlier in the conversation, along with their attached prose
    (the marker's line plus the note lines that follow it, mirroring
    ``_strip_trimmed_clip_tail`` so the reply never describes clips that
    no longer exist).

    Used by ``_enforce_clip_count`` on "more"-style asks only ("give me
    5 more" promises 5 NEW moments): Gemma routinely re-emits previously
    shown clips from the history in its prompt, and counting those toward
    the target shipped repeats as "new" clips. The screen runs BEFORE the
    trim and the have-count so an all-repeats reply doesn't survive the
    trim path either. Overlap here is the plain >50% grouped rule — NO
    adjacency margin: a marker that merely borders a shown clip is new
    material; only a real re-issue is a duplicate. Fresh (non-"more")
    asks never reach this — a fresh re-ask may legitimately repeat a
    previously shown moment.
    """
    if not text or not exclude:
        return text

    def _is_dup(marker_text):
        sm = re.search(r'start=([\d:.]+)', marker_text)
        em = re.search(r'end=([\d:.]+)', marker_text)
        if not (sm and em):
            return False
        s = _tc_to_seconds(sm.group(1))
        e = max(s, _tc_to_seconds(em.group(1)))
        grp = None
        if group_key:
            gm = re.search(r'project="([^"]*)"', marker_text)
            grp = (gm.group(1).strip() if gm else '') or None
        return _grouped_spans_overlap(s, e, grp, exclude)

    # Line walk shared with the quality floor — drops each duplicate
    # marker's line plus its attached note lines.
    return _drop_marker_lines_where(text, _is_dup)


def _enforce_clip_count(text, target, candidates=None, transcript=None,
                        exclude_spans=None, group_key=None, min_count=None,
                        drop_reemitted=False):
    """Hold a chat reply's [CLIP:] marker count to the user's ask — trim
    past ``target`` AND top up model under-delivery from the ranked pool.

    The trim half defends against Gemma 4 ignoring the explicit count rule
    in the prompt — the prompt asks for "EXACTLY 1" but the model emits
    2-3 anyway. The top-up half is the mirror defense: "Give me 5 more"
    answered with three cards used to pass through untouched (live tester
    bug), so when the reply carries FEWER markers than the ask, canonical
    markers are appended from ``candidates`` — the same deterministic
    machinery as ``_enforce_duration_target``, counting markers instead
    of seconds. GUIDING PRINCIPLE: the top-up must be PRECISE — if the
    pool can't fill the gap with genuinely new, non-adjacent material,
    the reply ships with what exists. The honest under-count always beats
    a padded sliver card.

    ``target``        explicit user count (``_detect_explicit_clip_count``);
                      trim and top-up both apply.
    ``min_count``     floor for plural asks with NO explicit count
                      (``_plural_clip_minimum`` — "the strongest momentS"
                      promises several); top-up only, NEVER trims, ignored
                      whenever ``target`` parses.
    ``candidates``    ranked pool for the top-up (matched paragraphs or
                      Layer-2 candidates — ``_duration_candidate_span``
                      shapes). ``None`` keeps the historical trim-only
                      behavior byte-for-byte.
    ``exclude_spans`` ``(start, end, group)`` triples already shown earlier
                      in the conversation (``_history_clip_spans``) — a
                      top-up never re-issues a moment the editor has seen,
                      nor a continuation within
                      ``_TOPUP_ADJACENCY_GAP_SECONDS`` of one.
    ``group_key``     per-source timeline field for multi-project pools,
                      same contract as ``_enforce_duration_target``.
                      Appended markers carry ``project="<group>"`` so the
                      collection pipeline gets exact attribution instead
                      of reconstructing it by string match.
    ``drop_reemitted`` True on "more"-style asks (``_MORE_CLIPS_RE``):
                      model-emitted markers that overlap an excluded span
                      are duplicates — removed (with their prose) so they
                      neither count toward the target nor ship as "new".
    """
    if not text:
        return text
    goal = None
    allow_trim = False
    if target is not None and target >= 1:
        goal = target
        allow_trim = True
    elif min_count is not None and min_count >= 1:
        goal = min_count
    if goal is None:
        return text
    exclude = _normalize_exclude_spans(exclude_spans)
    if drop_reemitted and exclude:
        # An all-repeats reply can drop to empty prose here — the top-up
        # below then rebuilds the cards from the pool, which is exactly
        # the contract: "5 more" never ships repeats as "new".
        text = _drop_reemitted_clip_markers(text, exclude, group_key)
    clips = list(_CLIP_MARKER_RE.finditer(text))
    if allow_trim and len(clips) > goal:
        # Cut at the end of the target-th marker. Anything after gets the
        # CLIP markers stripped along with their attached prose (so any
        # salvageable standalone prose stays, but no excess clip cards —
        # and no orphaned per-clip notes — render).
        cut = clips[goal - 1].end()
        head = text[:cut]
        tail = _strip_trimmed_clip_tail(text[cut:])
        result = (head + tail).rstrip()
        return result
    if len(clips) >= goal or not candidates:
        return text

    # TOP-UP: append ranked candidates until the marker count reaches the
    # ask. Overlap bookkeeping mirrors _enforce_duration_target — emitted
    # markers plus every prior-conversation span count as taken, per
    # timeline group when the pool is multi-source — with the ADJACENCY
    # rule on top: candidates within _TOPUP_ADJACENCY_GAP_SECONDS of a
    # taken span are continuations, not new moments.
    taken = []
    for m in clips:
        sm = re.search(r'start=([\d:.]+)', m.group(0))
        em = re.search(r'end=([\d:.]+)', m.group(0))
        if not (sm and em):
            continue
        s = _tc_to_seconds(sm.group(1))
        e = max(s, _tc_to_seconds(em.group(1)))
        grp = None
        if group_key:
            gm = re.search(r'project="([^"]*)"', m.group(0))
            grp = (gm.group(1).strip() if gm else '') or None
        taken.append((s, e, grp))
    taken.extend(exclude)

    have = len(clips)
    lines = []
    topup_goal = min(goal, _COUNT_TOPUP_MAX)
    for cand in candidates:
        if have >= topup_goal:
            break
        span = _duration_candidate_span(cand, transcript)
        if span is None:
            continue
        s, e, title, why = span
        # Never mint a sliver card: the quality floor just dropped those
        # from the model's reply, so the top-up must not re-introduce one.
        if e - s < _MIN_CLIP_MARKER_SECONDS:
            continue
        # Keep top-up cards clip-sized — same cap _deterministic_clip_markers
        # applies. (The duration pass keeps full spans because it needs the
        # runtime; a COUNT ask wants usable cards.)
        if e - s > 60:
            e = s + 45
        cand_group = None
        if group_key:
            cand_group = str(cand.get(group_key) or '').strip() or None
        if _grouped_spans_overlap(s, e, cand_group, taken,
                                  min_gap=_TOPUP_ADJACENCY_GAP_SECONDS):
            continue
        taken.append((s, e, cand_group))
        have += 1
        start_tc, end_tc = _seconds_to_tc(s), _seconds_to_tc(e)
        attrs = [f'start={start_tc}', f'end={end_tc}']
        if cand_group:
            # Exact source attribution for multi-project pools: the
            # candidate dict is in hand, so emit its timeline directly
            # instead of leaving the collection pipeline to reconstruct
            # it by string-matching capped spans (which went ambiguous
            # exactly when two projects shared a capped span + title).
            attrs.append('project="{0}"'.format(cand_group.replace('"', "'")))
        attrs.append(f'title="{title}"')
        if why:
            attrs.append(f'note="{why}"')
        lines.append('[CLIP: ' + ' '.join(attrs) + ']')
    if not lines:
        return text
    return (text.rstrip() + '\n\n' + '\n'.join(lines)).strip()


def _carryover_clip_queries(history):
    """User turns from ``history`` whose assistant reply produced clip
    cards, MOST RECENT FIRST. These are the themes the conversation is
    actually mining — the seed for keyword carryover on count-only
    follow-ups ("give me 5 more" after "strongest emotional moments").
    """
    turns = [t for t in (history or []) if isinstance(t, dict)]
    out = []
    for i in range(len(turns) - 1, -1, -1):
        if turns[i].get('role') != 'user':
            continue
        for j in range(i + 1, len(turns)):
            if turns[j].get('role') != 'assistant':
                continue
            if '[CLIP' in (turns[j].get('content') or ''):
                content = (turns[i].get('content') or '').strip()
                if content:
                    out.append(content)
            break
    return out


def _vector_window_candidates(segments, segment_vectors, limit=24):
    """Fallback pool for keyword-less count top-ups: ONE candidate per
    curated vector window, spanning the vector's own timecode_in/out —
    never the raw 2-8s whisper segments inside it. Raw segments made the
    old fallback append consecutive 4-second slivers of a single
    highlight window and count each as a separate "more" clip; a padded
    sliver card is worse than an honest under-count (guiding principle).

    Ranked by narrative score — every 'high' window before any 'medium'
    (the old pool was chronological despite the ranked-pool contract),
    chronological within a band. 'low' windows never mint user-facing
    cards. ``text`` carries the first overlapping segment's words so
    ``_duration_candidate_span`` derives a real title.
    """
    if not segment_vectors or not segments:
        return []
    buckets = {'high': [], 'medium': []}
    for v in segment_vectors:
        if not isinstance(v, dict):
            continue
        score = str(v.get('narrative_score', 'medium')).lower()
        if score not in buckets:
            continue
        try:
            start = _tc_to_seconds(v.get('timecode_in'))
            end = _tc_to_seconds(v.get('timecode_out'))
        except Exception:
            continue
        if end <= start:
            continue
        text = ''
        for seg in segments:
            seg_start = float(seg.get('start', 0) or 0)
            seg_end = float(seg.get('end', seg_start) or seg_start)
            if seg_end <= start or seg_start >= end:
                continue
            seg_text = (seg.get('text') or '').strip()
            if seg_text:
                text = seg_text
                break
        buckets[score].append({'start': start, 'end': end, 'text': text})
    return (buckets['high'] + buckets['medium'])[:limit]


def _count_topup_pool(matched, segments, segment_vectors, message=None,
                      history=None):
    """Ranked candidate pool for the clip-count top-up.

    Uses the retrieval matches when the query produced any. Count-only
    follow-ups ("give me 5 more") carry no searchable keywords, so the
    pool is built in two precision-ordered fallbacks:

    1. THEME CARRYOVER — when the current message extracts no keywords,
       re-run retrieval seeded with the most recent prior user turn that
       actually produced clips ("strongest emotional moments" → "give me
       5 more" keeps mining emotional moments instead of going
       theme-blind). Earlier clip-producing turns are tried in turn when
       the latest was itself keyword-less.
    2. CURATED WINDOWS — one candidate per high/medium narrative-score
       vector window (``_vector_window_candidates``), high first, never
       raw whisper slivers.

    Returns ``[]`` when nothing qualifies; the top-up then honestly
    delivers only what the model emitted.
    """
    if matched:
        return matched
    if segments and history and message is not None:
        cur_phrases, cur_words = _extract_query_keywords(message)
        if not (cur_phrases or cur_words):
            for carry in _carryover_clip_queries(history):
                try:
                    phrases, words = _extract_query_keywords(carry)
                    theme_phrases = _collect_theme_phrases_from_vectors(
                        segment_vectors, carry)
                    if not (phrases or words or theme_phrases):
                        continue
                    carried = _find_relevant_paragraphs(
                        segments, phrases, words, context=2,
                        theme_phrases=theme_phrases,
                    )
                    if carried:
                        return carried
                except Exception:
                    continue
    return _vector_window_candidates(segments, segment_vectors)


# ── Duration-target parsing + enforcement ────────────────────────────────
#
# The bundled Gemma models cannot do arithmetic: any "give me N minutes"
# ask must be parsed and enforced in CODE, never delegated to the model.
# parse_target_duration_seconds is the single shared parser — project chat,
# collection chat (pro/collection/chat.py imports it the same way it
# imports _enforce_clip_count), and Story Builder all call it.

_DURATION_WORD_NUMBERS = {
    'one': 1.0, 'two': 2.0, 'three': 3.0, 'four': 4.0, 'five': 5.0,
    'six': 6.0, 'seven': 7.0, 'eight': 8.0, 'nine': 9.0, 'ten': 10.0,
    'a': 1.0, 'an': 1.0,
}

# Tokens immediately BEFORE a duration phrase that mark it as a timeline
# position ("at 14 minutes", "the first 2 minutes") or as the SOURCE
# footage ("from the 40 minute interview") rather than a requested length.
_DURATION_POSITION_BEFORE = frozenset({
    'at', 'after', 'before', 'past', 'from', 'until', 'till', 'by',
    'first', 'last', 'opening', 'final', 'initial', 'closing', 'within',
})
# Filler skipped when walking back to the governing word ("build me a 14
# minute video" → back past 'a'/'me' to 'build').
_DURATION_SKIP_BEFORE = frozenset({
    'a', 'an', 'the', 'this', 'that', 'me', 'my', 'us', 'our', 'another',
    'about', 'approximately', 'roughly', 'around', 'like', 'exactly',
    'only', 'just', 'some',
})
# Tokens immediately AFTER that mark a position, a per-clip length, or a
# relative/comparative adjustment ("14 minutes in", "the 2 minute mark",
# "30 seconds each", "cut 30 seconds off", "make it 2 minutes shorter",
# "make each clip 30 seconds long") — none of these are total-output asks.
_DURATION_POSITION_AFTER = frozenset({
    'in', 'into', 'mark', 'point', 'ago', 'each', 'apiece',
    'shorter', 'longer', 'less', 'off', 'long', 'early', 'late',
})
# Request verbs that anchor a duration as the requested OUTPUT length.
_DURATION_REQUEST_VERBS = frozenset({
    'give', 'make', 'build', 'cut', 'create', 'assemble', 'pull', 'find',
    'get', 'want', 'need', 'do', 'produce', 'edit', 'put', 'string',
    'deliver', 'export', 'compile', 'grab', 'show',
})
# Deliverable nouns that anchor "14 minute X" as the requested OUTPUT length.
_DURATION_INTENT_NOUNS = frozenset({
    'cut', 'video', 'edit', 'version', 'story', 'sequence', 'reel',
    'montage', 'piece', 'film', 'teaser', 'trailer', 'supercut', 'promo',
    'highlight', 'highlights', 'selects', 'stringout', 'assembly', 'rough',
    'draft', 'episode', 'short', 'documentary', 'doc', 'build', 'clip',
    'clips', 'moments',
})

# Plausibility floor for a parsed target: nobody asks the app to build a
# sub-15-second deliverable, but "give me a sec" / "give me 10 seconds"
# (as in: wait a moment) are common chat filler. Below this → None.
_DURATION_MIN_TARGET_SECONDS = 15.0


def parse_target_duration_seconds(message):
    """Parse a requested TOTAL output duration from a user message.

    Pure regex — no LLM. Recognized shapes: "14 minute" / "14-minute" /
    "90 sec" / "1.5 hours", word numbers ("one minute", "a minute", plus
    "and a half"), ranges ("15 to 20 minute" / "15-20 minute" → the
    midpoint), and "m:ss"/"h:mm:ss" literals when clearly anchored to
    a request ("make a 2:30 reel"). Only ANCHORED durations count — a
    request verb governs the number ("build me a 14 minute...") or a
    deliverable noun / "of" follows it ("14 minute cut", "2 minutes of
    selects"). Content durations the speaker merely mentions ("the 3 hour
    rescue", "she spends 2 minutes describing..."), positional references
    ("the moment at 14 minutes in", "the first 2 minutes", "what did she
    say at 1:35"), relative adjustments ("cut 30 seconds off", "make it
    2 minutes shorter"), and messages with competing durations all return
    ``None`` — callers fall back to today's behavior when the ask is
    ambiguous. False positives here are worse than a missed ask: every
    hit activates deterministic duration enforcement downstream.

    Returns float seconds or ``None``.
    """
    if not message:
        return None
    import re
    msg = message.lower()

    def _governing_before(idx):
        tokens = re.findall(r"[a-z0-9']+", msg[:idx])
        for tok in reversed(tokens):
            if tok in _DURATION_SKIP_BEFORE:
                continue
            return tok
        return ''

    def _token_after(idx):
        m = re.match(r"[^a-z0-9]*([a-z0-9']+)", msg[idx:])
        return m.group(1) if m else ''

    def _classify(start, end):
        """'positional' (skip), 'anchored' (a clear ask), or 'plain'."""
        after = _token_after(end)
        if after in _DURATION_POSITION_AFTER:
            return 'positional'
        # Per-clip phrasing can put "each" BEFORE the number too:
        # "make each clip 30 seconds long".
        preceding = re.findall(r"[a-z0-9']+", msg[:start])[-4:]
        if any(t in ('each', 'every', 'apiece') for t in preceding):
            return 'positional'
        governing = _governing_before(start)
        if governing in _DURATION_POSITION_BEFORE:
            return 'positional'
        if governing in _DURATION_REQUEST_VERBS:
            return 'anchored'
        if after == 'of' or after in _DURATION_INTENT_NOUNS:
            return 'anchored'
        # "14 minute paranormal investigation video": a deliverable noun
        # within the next few tokens still anchors the ask.
        tail = re.findall(r"[a-z0-9']+", msg[end:])[:5]
        if any(t in _DURATION_INTENT_NOUNS for t in tail):
            return 'anchored'
        return 'plain'

    candidates = []  # (seconds, classification)

    # Range asks: "15 to 20 minute", "15-20 minute", "15–20 minute", plus
    # word-number lows ("five to ten minute") and mixed units across the
    # connector ("90 second to 2 minute"). The single-number pass below
    # only sees the unit-adjacent number — a "15 to 20 minute video" ask
    # parsed as a hard 1200s target (the range TOP), never the ask's real
    # center. A range targets its MIDPOINT. Matched spans are remembered
    # so the single-number pass skips the "20 minute" inside a consumed
    # range instead of double-counting it.
    _num_word = r'\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten'
    _unit = r'hours?|hrs?|minutes?|mins?|seconds?|secs?'
    range_re = re.compile(
        rf'(?<![\d:.])\b({_num_word})'
        rf'(?:\s*({_unit}))?'
        r'\s*(?:[-–—]|to)\s*'
        rf'({_num_word})[\s-]+'
        rf'({_unit})\b'
    )

    def _range_qty(raw):
        try:
            return float(raw)
        except ValueError:
            return _DURATION_WORD_NUMBERS.get(raw)

    def _unit_mult(unit):
        return 3600.0 if unit[0] == 'h' else (60.0 if unit[0] == 'm' else 1.0)

    range_spans = []
    for m in range_re.finditer(msg):
        lo_qty = _range_qty(m.group(1))
        hi_qty = _range_qty(m.group(3))
        if not lo_qty or not hi_qty:
            continue
        hi_unit = m.group(4)
        lo_secs = lo_qty * _unit_mult(m.group(2) or hi_unit)
        hi_secs = hi_qty * _unit_mult(hi_unit)
        # Consumed BEFORE any plausibility judgment: a recognized range
        # shape must never leak its inner number to the single-number
        # pass — a rejected "10 to 30 second teaser" otherwise parsed as
        # a hard 30s target (the range TOP), the exact misparse this
        # pass exists to fix. Also covers positional ranges ("the 15-20
        # minute mark").
        range_spans.append((m.start(), m.end()))
        # A real range runs low-to-high ("20 to 15 minute" is noise, not
        # an ask), and the MIDPOINT — the value actually enforced — must
        # be a plausible target: a "10 to 30 second teaser" legitimately
        # centers on 20s even though its low end sits under the floor.
        if hi_secs <= lo_secs:
            continue
        mid = (lo_secs + hi_secs) / 2.0
        if mid < _DURATION_MIN_TARGET_SECONDS or hi_secs > 6 * 3600:
            continue
        cls = _classify(m.start(), m.end())
        if cls == 'positional':
            continue
        candidates.append((mid, cls))

    # Range shapes the regex above cannot model (word numbers past ten,
    # "half an hour to 45 minutes", ...) are recognized by their range
    # connector sitting immediately before a number+unit match, and
    # consumed the same way — leaking one would turn the range TOP into
    # a hard target.
    range_orphan_re = re.compile(
        rf'(?:{_num_word}'
        r'|eleven|twelve|thirteen|fifteen|twenty|thirty|forty|fifty'
        r'|sixty|ninety|half(?:\s+an?)?)'
        rf'(?:\s*(?:{_unit}))?'
        r'\s*(?:[-–—]|to)\s*$'
    )

    # Number + unit: "14 minute(s)", "14-minute", "90 sec", "1.5 hours",
    # "one minute", "a minute", "half an hour". An intervening "more" is
    # skippable ("give me 30 MORE seconds of selects", "pull 2 more
    # minutes of selects" — follow-up phrasings of the same duration
    # ask); without it these parsed neither as duration nor count and
    # leaked into the count path's request-verb digit pattern.
    unit_re = re.compile(
        r'(?<![\d:.])\b'
        r'(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten'
        r'|an?|half\s+an?)'
        r'(?:\s+more)?'
        r'[\s-]+'
        r'(hours?|hrs?|minutes?|mins?|seconds?|secs?)\b'
    )
    for m in unit_re.finditer(msg):
        if any(m.start() < r_end and m.end() > r_start
               for r_start, r_end in range_spans):
            continue  # the trailing half of an already-consumed range
        if range_orphan_re.search(msg[:m.start()]):
            continue  # the TOP half of a range the range pass can't model
        raw = m.group(1)
        if raw.startswith('half'):
            qty = 0.5
        else:
            try:
                qty = float(raw)
            except ValueError:
                qty = _DURATION_WORD_NUMBERS.get(raw)
        if not qty:
            continue
        # Bare article + unit ("give me a second", "give me a sec",
        # "hang on, a minute") is chat filler, not an ask. It only counts
        # when it clearly quantifies deliverable material: "a minute OF
        # selects" or a deliverable noun close behind ("a minute and a
        # half of selects").
        if raw in ('a', 'an'):
            article_after = _token_after(m.end())
            article_tail = re.findall(r"[a-z0-9']+", msg[m.end():])[:5]
            if article_after != 'of' and not any(
                    t in _DURATION_INTENT_NOUNS for t in article_tail):
                continue
        unit = m.group(2)
        mult = 3600.0 if unit[0] == 'h' else (60.0 if unit[0] == 'm' else 1.0)
        secs = qty * mult
        # "a minute and a half" / "two minutes and a half"
        if re.match(r'\s+and\s+a\s+half\b', msg[m.end():]):
            secs *= 1.5
        if secs < _DURATION_MIN_TARGET_SECONDS or secs > 6 * 3600:
            continue
        cls = _classify(m.start(), m.end())
        if cls == 'positional':
            continue
        candidates.append((float(secs), cls))

    # "m:ss" / "h:mm:ss" literals — a bare timecode in prose is a position
    # reference, so these only count when anchored to a request.
    tc_re = re.compile(r'(?<![\d:.])(\d{1,2}):(\d{2})(?::(\d{2}))?(?![\d:.])')
    for m in tc_re.finditer(msg):
        if _classify(m.start(), m.end()) != 'anchored':
            continue
        a, b, c = m.group(1), m.group(2), m.group(3)
        if c is not None:
            secs = int(a) * 3600 + int(b) * 60 + int(c)
        else:
            secs = int(a) * 60 + int(b)
        if _DURATION_MIN_TARGET_SECONDS <= secs <= 6 * 3600:
            candidates.append((float(secs), 'anchored'))

    if not candidates:
        return None
    # Anchored-only: a 'plain' duration is a narrative fact the message
    # mentions ("it took 3 hours to set up", "her 5 minute speech"), not
    # a deliverable ask. Honoring a lone plain candidate turned ordinary
    # locate/discuss messages into duration-enforced clip floods, and
    # silently discarded explicit clip counts ("give me 3 clips from her
    # 5 minute speech" must enforce count=3, no duration).
    anchored = {s for s, cls in candidates if cls == 'anchored'}
    if len(anchored) == 1:
        return anchored.pop()
    return None  # no anchored duration, or competing anchored ones


def _duration_clip_count_hint(target_seconds, avg_clip_seconds=35.0):
    """Clip-count guidance for a duration ask, computed in code.

    Small local models follow explicit counts far better than time budgets,
    so prompts state "roughly K clips" with K derived here. 35s is the
    observed average of retrieval paragraphs and chat clip suggestions.
    """
    try:
        k = int(round(float(target_seconds) / max(1.0, float(avg_clip_seconds))))
    except (TypeError, ValueError):
        return 2
    return max(2, min(24, k))


# Band shared by the chat-side duration pass: top up under 0.8× the target,
# trim past 1.25×. Wider than the story band because chat pools are
# retrieval-ranked paragraphs, not curated segments.
_DURATION_TARGET_FLOOR = 0.8
_DURATION_TARGET_CEILING = 1.25


def _duration_candidate_span(cand, transcript=None):
    """Normalize a ranked-pool candidate to ``(start, end, title, why)``.

    Tolerates both pool shapes: Layer 1 matched paragraphs ({'start',
    'end', 'text'}) and Layer 2 aggregated candidates ({'start_sec',
    'end_sec', 'title', 'why'}). ``transcript`` supplies title text when
    the candidate carries neither. Returns ``None`` when no usable
    timespan exists.
    """
    if not isinstance(cand, dict):
        return None
    try:
        if 'start_sec' in cand or 'end_sec' in cand:
            s = float(cand.get('start_sec', 0) or 0)
            e = float(cand.get('end_sec', s) or s)
        else:
            s = float(cand.get('start', 0) or 0)
            e = float(cand.get('end', s) or s)
    except (TypeError, ValueError):
        return None
    if e <= s:
        return None
    title = (cand.get('title') or '').strip()
    if not title:
        text = (cand.get('text') or '').strip()
        if not text and transcript:
            for seg in (transcript.get('segments') or []):
                seg_start = float(seg.get('start', 0) or 0)
                seg_end = float(seg.get('end', seg_start) or seg_start)
                if seg_end >= s and seg_start <= e and (seg.get('text') or '').strip():
                    text = seg['text'].strip()
                    break
        title = ' '.join(text.split()[:5]).rstrip('.,!?;:')[:40] or 'Transcript moment'
    title = title.replace('"', "'").replace('[', '(').replace(']', ')')
    # Brackets in the note would corrupt marker parsing downstream —
    # neutralize them the same way titles are.
    why = (cand.get('why') or '').strip().replace('"', "'")
    why = why.replace('[', '(').replace(']', ')')
    return s, e, title, why


def _spans_overlap(start, end, spans):
    """True when [start, end] overlaps any span by >50% of the shorter."""
    dur = max(1.0, end - start)
    for os_, oe in spans:
        overlap = max(0.0, min(end, oe) - max(start, os_))
        shorter = min(dur, max(1.0, oe - os_))
        if overlap / shorter > 0.5:
            return True
    return False


# Adjacency margin for the deterministic top-ups: a candidate whose span
# overlaps — or merely sits within this many seconds of — a taken/history
# span is a CONTINUATION of material the editor already has (the next
# slice of the same window, the tail of an already-shown clip). Appending
# it would fake variety with near-duplicates, and a padded sliver card is
# worse than an honest under-count (guiding principle).
_TOPUP_ADJACENCY_GAP_SECONDS = 10.0


def _grouped_spans_overlap(start, end, group, spans, min_gap=0.0):
    """Group-aware variant of ``_spans_overlap`` for multi-source pools.

    ``spans`` are ``(start, end, group)`` triples where ``group`` names the
    timeline the span's timestamps are local to (e.g. a collection merges
    several projects, each with its OWN zero-based clock — identical
    numbers on different timelines are DIFFERENT footage). Two spans only
    collide when they share a group. A ``None`` group means the timeline
    is unknown; it conservatively collides with every group (better to
    skip a candidate than emit the same moment twice). With every group
    ``None`` this reduces exactly to ``_spans_overlap``.

    ``min_gap`` > 0 switches to the top-ups' ADJACENCY rule: any overlap,
    or a same-timeline separation under ``min_gap`` seconds, collides —
    no continuations of already-shown clips, no consecutive slices of one
    window (see ``_TOPUP_ADJACENCY_GAP_SECONDS``). The default 0 keeps
    the historical >50%-of-the-shorter-span semantics for every other
    caller (dedupe of the model's own re-emissions).
    """
    dur = max(1.0, end - start)
    for os_, oe, og in spans:
        if group is not None and og is not None and og != group:
            continue  # different timelines — equal numbers, unrelated footage
        if min_gap > 0.0:
            # Separation is negative when the spans overlap, so this one
            # comparison covers both "overlaps at all" and "too close".
            if max(os_ - end, start - oe) < min_gap:
                return True
            continue
        overlap = max(0.0, min(end, oe) - max(start, os_))
        shorter = min(dur, max(1.0, oe - os_))
        if overlap / shorter > 0.5:
            return True
    return False


def _enforce_duration_target(text, target_seconds, candidates, transcript=None,
                             group_key=None, exclude_spans=None):
    """Deterministically hold a chat reply's [CLIP:] total to a duration ask.

    Measures the emitted markers with ``_tc_to_seconds``, tops up from the
    ranked candidate pool when the total lands under 0.8× the target (the
    ``_deterministic_clip_markers`` machinery, minus its 60s cap, skipping
    time-overlaps with clips already emitted), and trims trailing markers
    — the weakest-ranked in every emitting path — when the total passes
    1.25×. The model is never asked to do this arithmetic.

    ``group_key`` (optional, additive): name of the candidate-dict field
    carrying a per-source timeline id — e.g. ``'project_name'`` for
    collection pools that merge several projects, each keeping its OWN
    zero-based clock (identical timestamps on different timelines are
    different footage, not overlaps). When set, the overlap bookkeeping
    runs per group: a marker only blocks candidates on the SAME timeline,
    with an emitted marker's group read from its ``project="..."`` field
    (double-quoted, the shape this pipeline itself writes). Markers or
    candidates without a group conservatively collide with every group.
    Default ``None`` keeps the historical single-timeline behavior
    byte-for-byte for existing callers. Appended markers carry
    ``project="<group>"`` for exact multi-project attribution (same
    contract as ``_enforce_clip_count``).

    ``exclude_spans`` (optional, additive): ``(start, end, group)``
    triples already shown earlier in the conversation
    (``_history_clip_spans``) — same shape and semantics as the count
    path's parameter. "Give me another 2 minutes of selects" must top up
    with NEW footage, never re-issue (or continue — the adjacency rule
    applies) moments the editor already has. Excluded spans block
    candidates but do NOT count toward the time budget: they were
    delivered in prior turns.
    """
    if not text or not target_seconds or target_seconds <= 0:
        return text
    import re
    markers = []
    for m in _CLIP_MARKER_RE.finditer(text):
        sm = re.search(r'start=([\d:.]+)', m.group(0))
        em = re.search(r'end=([\d:.]+)', m.group(0))
        if not (sm and em):
            continue
        s = _tc_to_seconds(sm.group(1))
        e = _tc_to_seconds(em.group(1))
        markers.append((m, s, max(s, e)))
    marker_groups = []
    if group_key:
        # The group of an already-emitted marker is its project="..."
        # field — written double-quoted by every emitting path (model
        # markers via the pro stash/restore, repair markers, and this
        # function's own top-up after pro re-attachment). No field →
        # unknown timeline → collides with every group (conservative).
        for m, _s, _e in markers:
            gm = re.search(r'project="([^"]*)"', m.group(0))
            grp = gm.group(1).strip() if gm else ''
            marker_groups.append(grp or None)
    total = sum(e - s for _m, s, e in markers)
    floor = _DURATION_TARGET_FLOOR * float(target_seconds)
    ceiling = _DURATION_TARGET_CEILING * float(target_seconds)

    if markers and total > ceiling:
        keep = len(markers)
        while keep > 1 and total > ceiling:
            last_dur = markers[keep - 1][2] - markers[keep - 1][1]
            if total - last_dur < floor:
                break
            total -= last_dur
            keep -= 1
        if keep == len(markers):
            return text
        cut = markers[keep - 1][0].end()
        head = text[:cut]
        tail = _strip_trimmed_clip_tail(text[cut:])
        return (head + tail).rstrip()

    if total >= floor:
        return text

    # TOP-UP: extend with ranked candidates the reply didn't already
    # cover — nor prior turns (exclude_spans), with the same adjacency
    # rule as the count top-up: a candidate within
    # _TOPUP_ADJACENCY_GAP_SECONDS of taken/history material is a
    # continuation of what the editor already has, not new footage.
    taken = [
        (s, e, marker_groups[i] if marker_groups else None)
        for i, (_m, s, e) in enumerate(markers)
    ]
    taken.extend(_normalize_exclude_spans(exclude_spans))
    lines = []
    for cand in (candidates or []):
        if total >= floor:
            break
        span = _duration_candidate_span(cand, transcript)
        if span is None:
            continue
        s, e, title, why = span
        dur = e - s
        # No sliver cards from the duration top-up either — same floor
        # the canonicalization pass applies to model-emitted markers.
        if dur < _MIN_CLIP_MARKER_SECONDS:
            continue
        cand_group = None
        if group_key:
            cand_group = str(cand.get(group_key) or '').strip() or None
        if _grouped_spans_overlap(s, e, cand_group, taken,
                                  min_gap=_TOPUP_ADJACENCY_GAP_SECONDS):
            continue
        if total + dur > ceiling:
            continue
        taken.append((s, e, cand_group))
        total += dur
        start_tc, end_tc = _seconds_to_tc(s), _seconds_to_tc(e)
        attrs = [f'start={start_tc}', f'end={end_tc}']
        if cand_group:
            # Exact source attribution — see _enforce_clip_count.
            attrs.append('project="{0}"'.format(cand_group.replace('"', "'")))
        attrs.append(f'title="{title}"')
        if why:
            attrs.append(f'note="{why}"')
        lines.append('[CLIP: ' + ' '.join(attrs) + ']')
    if not lines:
        return text
    return (text.rstrip() + '\n\n' + '\n'.join(lines)).strip()


def _format_clip_title(title: str) -> str:
    """Normalize a clip title to readable sentence case.

    Local models often emit lowercased keyword-stuffed titles ("papermaking
    artist process details") that read like search queries instead of card
    headlines. We can't restructure the phrase for grammar without an LLM
    pass, but we CAN promote the first letter to uppercase and tidy the
    spacing so the card looks like a headline rather than a tag list.

    Existing capitals (proper nouns, acronyms) are preserved — only the
    leading character is forced uppercase.
    """
    if not title:
        return title
    cleaned = ' '.join(title.split())
    if not cleaned:
        return cleaned
    if cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def _strip_raw_json_blobs(text):
    """Strip JSON object literals the model leaks in place of [CLIP:] markers.

    Some local models, when asked for "highlights" or "an emotional arc",
    fall back on a training-data convention and emit shapes like::

        {
          "type": "highlight",
          "content": "...",
          "context": "..."
        }

    interleaved with the real ``[CLIP:]`` markers. The frontend has no
    handler for those blobs, so they render as raw text between the prose
    summary and the clip cards — exactly the screenshot the user reported.

    This pass removes any standalone ``{...}`` block whose first key is one
    of the recognized leak shapes (type / kind / category) and that also
    contains a content/context/quote field. Lone CLIP markers and other
    bracketed UI tokens are unaffected.
    """
    if not text:
        return text
    import re

    leak_pattern = re.compile(
        r'\{\s*"(?:type|kind|category)"\s*:\s*"[^"]*"\s*,'
        r'(?:[^{}]|\{[^{}]*\})*'
        r'"(?:content|text|quote|excerpt|summary|context)"\s*:'
        r'(?:[^{}]|\{[^{}]*\})*\}\s*,?',
        re.DOTALL,
    )
    text = leak_pattern.sub('', text)

    # Drop wrapping array brackets if their items just got stripped.
    text = re.sub(r'^\s*\[\s*\]\s*,?\s*$', '', text, flags=re.MULTILINE)
    # Strip lone trailing commas left dangling after we removed a sibling.
    text = re.sub(r'^\s*,\s*$', '', text, flags=re.MULTILINE)
    # Collapse blank-line runs the removals may have created.
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def _strip_placeholder_timecode_garbage(text):
    """Strip leftover model output that gestures at clips but contains no
    real timecodes — typically JSON-array-of-placeholders shapes like:

        [
          "00:00:00 - 00:00:00",
          "00:00:00 - 00:00:00"
        ]

    or bare lines containing only ``00:00:00 - 00:00:00`` (zero-zero ranges).

    Confused small models emit these when they understand they're supposed
    to output something timecode-shaped but can't actually find moments in
    the transcript. The frontend renders them as adjustable timecode chips
    which is worse than no output at all. Strip server-side.

    Filled-in valid timecodes (e.g. ``"00:12:34 - 00:13:00"``) are
    preserved — only the all-zero/identical-range placeholders go.
    """
    if not text:
        return text
    import re

    # 1. Bare-array of placeholder timecode strings — strip the whole array.
    array_pattern = re.compile(
        r'\[\s*(?:"00:00:00\s*[-–]\s*00:00:00"\s*,?\s*)+\s*\]',
        re.MULTILINE,
    )
    text = array_pattern.sub('', text)

    # 2. Lines containing only an all-zero placeholder range (with optional
    #    quotes, commas, brackets). Drop the line.
    line_pattern = re.compile(
        r'^[\[\s"]*00:00:00\s*[-–]\s*00:00:00[\]\s",]*$',
        re.MULTILINE,
    )
    text = line_pattern.sub('', text)

    # 3. Inline ``00:00:00 - 00:00:00`` ranges embedded in prose — replace
    #    with empty string. Keeps the surrounding text but drops the
    #    meaningless range.
    inline_pattern = re.compile(r'\b00:00:00\s*[-–]\s*00:00:00\b')
    text = inline_pattern.sub('', text)

    # 4. Stray opening/closing brackets left behind after we stripped the
    #    array contents. Only drop bare-line brackets — keep [CLIP: …] etc.
    text = re.sub(r'^\s*\[\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\]\s*$', '', text, flags=re.MULTILINE)

    # 5. Trailing `[` on a labeling line (e.g. ``The Clip: [``) where the
    #    model used brackets as a visual wrapper around a [CLIP:] marker that
    #    follows on the next line. Strip the trailing `[` so the label reads
    #    cleanly. Match the optional space + bracket at end of line.
    text = re.sub(r'^([^\n\[]+):\s*\[\s*$', r'\1:', text, flags=re.MULTILINE)

    return text


def _strip_empty_clip_placeholders(text):
    """Remove ``[CLIP]`` and ``[CLIP:]`` (and any bracketed CLIP shape that
    lacks a ``start=`` field) from a chat response.

    Small models occasionally emit literal placeholder pills (``[CLIP] [CLIP]``)
    when they understand they're supposed to output markers but can't fill in
    the timecode fields. The frontend renders these as empty broken pills.
    Strip them server-side so the UI never sees the junk.
    """
    if not text:
        return text
    import re
    # Strip any [CLIP...] that doesn't contain start= or end=
    return re.sub(r'\[CLIP\b[^\]]*?\](?<!\bstart=\w\])',
                  lambda m: '' if 'start=' not in m.group(0) else m.group(0),
                  text)


# Reasoning-mode tag pairs. Small local models (gemma4:e2b in particular)
# wrap their answers in faux-XML "thinking" tags they learned from
# reasoning-finetuned siblings. None of this is meant for the user — it's
# the model talking to itself. Strip server-side before render.
_REASONING_TAG_PAIRS = [
    ('Thought Process', '/Thought Process'),
    ('Thoughts', '/Thoughts'),
    ('Reasoning', '/Reasoning'),
    ('Reflection', '/Reflection'),
    ('Plan', '/Plan'),
    ('Analysis', '/Analysis'),
    ('Response', '/Response'),
]


def _strip_essay_scaffolding(text):
    """Neutralize essay scaffolding without destroying the content it
    carries. Historical behavior deleted whole lines here — ``> `` quote
    lines and numbered ``1. Header: explanation`` items vanished with
    their text, which is exactly how specific, grounded answers turned
    into dangling headers ("Thematically, we are looking at:" followed by
    nothing). Content is never deleted anymore:

      1. ``> quoted line`` keeps its text as a plain quoted-prose line —
         the blockquote decoration goes, the words stay (conversational
         answers have no clip card to carry the quote for them).
      2. Numbered ``1. Header: explanation`` items are kept verbatim —
         the frontends render numbered lists natively.
      3. Parenthetical confessions of fabrication ("(hypothetical
         selection)", "(paraphrased)") are still removed: they flag
         content the model invented, and the flag itself is noise.
    """
    if not text:
        return text
    import re

    # 1. Markdown blockquote lines: keep the text, drop the "> " prefix,
    # wrap in quotation marks when the model didn't supply its own.
    # Bare ">" separator lines (multi-paragraph blockquotes) drop first so
    # the content rule below can't wrap the FOLLOWING prose line; the
    # content rule matches same-line whitespace only ([ \t], never \n).
    text = re.sub(r'^[ \t]*>[ \t]*$\n?', '', text, flags=re.MULTILINE)

    def _unquote_block(m):
        inner = m.group(1).strip()
        if not inner:
            return ''
        if inner[0] in '"“‘\'' or inner[-1] in '"”’\'':
            return f'{inner}\n'
        return f'“{inner}”\n'
    text = re.sub(r'^[ \t]*>[ \t]+(.*)$\n?', _unquote_block, text,
                  flags=re.MULTILINE)

    # 2. (removed) Numbered "1. Header: explanation" items are legitimate
    # structured answers — both chat frontends render numbered lists.

    # 3. Parenthetical confessions of fabrication: "(A hypothetical
    # selection ...)", "(illustrative quote)", "(paraphrased)", etc.
    text = re.sub(
        r'\([^)]*?(?:hypothetical|fabricated|imagined|placeholder'
        r'|illustrative|paraphrased|made[- ]up|approximation|approximate'
        r'|made up)[^)]*?\)',
        '',
        text,
        flags=re.IGNORECASE,
    )

    return text


def _strip_no_answer_placeholders(text):
    """Strip apologetic placeholder lines like ``[No specific answer
    provided for this prompt.]``, ``[No answer]``, ``[No response]``, or
    ``[No matches found]`` that small Gemma variants emit as a leading
    narration when they understand they should produce prose but can't
    cleanly summarize. The clip markers that follow are usually fine —
    drop only the apologetic preamble.
    """
    if not text:
        return text
    import re
    pattern = re.compile(
        r'^\s*\[?\s*No\s+(?:specific\s+|direct\s+|relevant\s+|explicit\s+|clear\s+|new\s+|additional\s+|further\s+)?'
        r'(?:answer|response|reply|match(?:es)?|result(?:s)?|comment(?:s)?|'
        r'content|output|request|ask|moment(?:s)?|clip(?:s)?|excerpt(?:s)?'
        r'|quote(?:s)?|segment(?:s)?)\b'
        r'[^\]\n]*\]?\s*$',
        re.MULTILINE | re.IGNORECASE,
    )

    def _placeholder_only(m):
        line = m.group(0).strip()
        # A real negative ANSWER ("No quotes about the fire exist." /
        # "No direct quotes, but she circles it at 12:40…") is content,
        # not placeholder narration — deleting it flips an honest 'no'
        # into silence (and downstream salvage could then fabricate an
        # implied 'yes'). Placeholders are bracket-wrapped or terse
        # sentence fragments; keep any bracket-free line that reads like
        # a sentence (ends in punctuation) or carries substance markers
        # (comma / 'but' / timecode / length).
        bracketed = line.startswith('[')
        substantive = (
            ',' in line or ' but ' in line.lower()
            or re.search(r'\d{1,2}:\d{2}', line)
            or line.endswith(('.', '!', '?'))
        )
        if not bracketed and (len(line) >= 60 or substantive):
            return m.group(0)
        return ''
    return pattern.sub(_placeholder_only, text)


def _strip_meta_preamble(text):
    """Strip free-form reasoning narration that small models leak around
    the actual answer.

    Distinct from :func:`_strip_reasoning_tags`, which only catches blocks
    wrapped in registered opener tags like ``[Reasoning]`` or ``<think>``.
    Gemma 4 (4B in particular) often emits the same content shape with no
    opener tag at all — a leading ``[The user is asking…]`` block, a
    ``Suggested Response:`` label, and a trailing ``(Note: …)`` caveat.
    The frontend renders all three as raw text between the user's question
    and the clip cards.

    Three patterns:

    1. **Leading bracketed prose**: a ``[…]`` block at the very start
       of the reply containing 2+ sentences and 80+ chars, that is NOT
       a ``[CLIP:]`` marker. The size + sentence-count thresholds protect
       legitimate short bracketed asides.

    2. **Preamble labels**: lines that are nothing but
       ``Suggested Response:`` / ``Final Answer:`` / ``Final Response:`` /
       ``Suggested Answer:`` / ``My Response:`` / ``My Answer:``. Drop the
       label line; whatever follows is kept. Bare ``Answer:`` or
       ``Response:`` is intentionally not stripped — too generic, would
       hit legitimate content.

    3. **Trailing ``(Note: …)`` parentheticals**: a parenthetical at the
       very end of the reply beginning with ``Note:``. Caveats and
       disambiguations belong in the model's head, not the user's chat.
    """
    if not text:
        return text
    import re

    # 1. Leading [...] reasoning blocks. Negative lookahead skips [CLIP...]
    # variants. Gemma often emits 2-3 of these in a row before the actual
    # answer, so we loop. A bracket qualifies as "reasoning prose" if it's
    # long enough AND either contains 2+ sentence-ending punctuation OR a
    # first-person reasoning verb ("I will/should/need", "Let me", "I'll").
    # Single-sentence bracketed reasoning is the exact pattern the previous
    # 2-sentence floor missed.
    reasoning_kw = re.compile(
        r"\b(?:I (?:will|'ll|should|need to|must|cannot|can'?t|am)"
        r"|Let me|We (?:should|need)|My (?:goal|task)"
        r"|(?:the\s+)?user\s+(?:is|wants|asked|specified|did(?:n'?t| not))"
        r"|assuming|since the (?:user|prompt|context)"
        r"|based on (?:the|this|previous|our)"
        r"|no (?:specific|direct|explicit|clear) (?:request|ask|count|number|moment)"
        r"|previous context|previous turn|previous reply"
        r"|(?:will|let me) (?:select|choose|pick|find|extract))\b",
        re.IGNORECASE,
    )
    while True:
        # 30-char floor: above "[Speaker 1]" / "[music]" / "[applause]"
        # which we want to keep, below typical Gemma reasoning leaks
        # ("[No specific request provided, so no output is generated.]"
        # is 56 chars). The keyword check protects against false
        # positives at this length.
        m = re.match(r'^\s*\[(?!\s*[Cc][Ll][Ii][Pp]\b)([^\[\]]{30,})\]\s*\n*', text)
        if not m:
            break
        inner = m.group(1)
        sentence_punct = inner.count('.') + inner.count('?') + inner.count('!')
        # Tier 1: 2+ sentences → reasoning. Tier 2: contains a known
        # reasoning keyword → reasoning. Tier 3: bracket is >120 chars →
        # almost always reasoning at the start of a chat reply, no
        # keyword check needed.
        if sentence_punct >= 2 or reasoning_kw.search(inner) or len(inner) > 120:
            text = text[m.end():]
        else:
            break

    # 2. Multi-word preamble labels on their own line. Anchored to start
    # of line so a "Final Answer:" buried mid-paragraph is left alone.
    text = re.sub(
        r'^\s*(?:Suggested\s+Response|Suggested\s+Answer|Final\s+Response|'
        r'Final\s+Answer|My\s+Response|My\s+Answer)\s*:\s*\n',
        '',
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # 3. Trailing "(Note: ...)" parenthetical at end of message.
    text = re.sub(
        r'\n*\(\s*Note\s*:\s*[^)]*\)\s*$',
        '',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # 4. (removed) Follow-up offers ("Would you like me to pull those as
    # clips?") used to be deleted wholesale — but an offer to do more
    # editorial work is exactly what a producer says next, and the same
    # regex was eating substantive lines that merely started with "I can
    # pull the moment where…". Offers stay.

    # 5. (removed) "Label: sentence" sub-headers above the first marker
    # used to be deleted WITH their prose — which destroyed real content
    # like "Speakers: Maria Sanchez and her daughter." and structured
    # answers ("Setup: …", "Payoff: …"). Both frontends render these
    # fine as plain lines; they stay.

    return text.strip()


def _strip_reasoning_tags(text):
    """Remove ``[Thought Process]...[/Thought Process]`` / ``[Thoughts]...
    [/Thoughts]`` / ``<think>...</think>`` and similar reasoning-mode wrappers.

    Handles three shapes:
      1. Closed pair: ``[Tag]content[/Tag]`` → strip everything (tags + content)
      2. Open-ended: ``[Tag]...end-of-message`` (no close) → strip from tag onward
      3. Bare openers/closers that survived a partial match → strip just the tag

    The "Response" tag is a special case: real content lives inside it, so for
    a closed [Response]...[/Response] pair we KEEP the inner content but drop
    the wrapping tags. Other tags (Thoughts, Thought Process, etc.) drop both
    tags AND content because they're internal monologue.
    """
    if not text:
        return text
    import re

    # 1. Closed pairs first — handle each tag pair.
    for opener, closer in _REASONING_TAG_PAIRS:
        opener_re = re.escape(opener)
        closer_re = re.escape(closer)
        if opener == 'Response':
            # Keep the inner content of the Response tag; just unwrap.
            pattern = re.compile(
                rf'\[\s*{opener_re}\s*\](.*?)\[\s*{closer_re}\s*\]',
                re.DOTALL | re.IGNORECASE,
            )
            text = pattern.sub(lambda m: m.group(1).strip(), text)
        else:
            # Drop both tags AND content — internal reasoning, not for users.
            pattern = re.compile(
                rf'\[\s*{opener_re}\s*\].*?\[\s*{closer_re}\s*\]',
                re.DOTALL | re.IGNORECASE,
            )
            text = pattern.sub('', text)

    # 2. Open-ended: tag opens but never closes (model ran out of tokens
    #    or got cut off). Strip from the opener to the end of the message
    #    so the UI doesn't show "[Thoughts] half a thought…" garbage.
    #    Strictly-internal tags (Thoughts/Thought Process/Reasoning/
    #    Reflection) strip unconditionally — text after them is monologue
    #    by definition, wherever it starts. Header-ambiguous tags
    #    (Analysis/Plan/Response) only strip when they open near the HEAD
    #    of the reply: a model using "[Analysis]" as a mid-answer section
    #    header must not lose everything after it.
    _HEADER_AMBIGUOUS = {'Analysis', 'Plan', 'Response'}
    for opener, _closer in _REASONING_TAG_PAIRS:
        opener_re = re.escape(opener)
        m = re.search(rf'\[\s*{opener_re}\s*\]', text, flags=re.IGNORECASE)
        if m and (opener not in _HEADER_AMBIGUOUS or m.start() <= 200):
            text = re.sub(
                rf'\[\s*{opener_re}\s*\].*?$',
                '',
                text,
                flags=re.DOTALL | re.IGNORECASE,
            )

    # 3. Bare leftover tags that escaped both passes (e.g. just "[Thoughts]"
    #    on its own line). Drop the marker.
    text = re.sub(r'\[\s*/?\s*(?:Thought Process|Thoughts|Reasoning|'
                  r'Reflection|Plan|Analysis|Response)\s*\]',
                  '', text, flags=re.IGNORECASE)

    # 4. <think>...</think> and <thinking>...</thinking> (XML-style reasoning)
    text = re.sub(r'<\s*think(?:ing)?\s*>.*?<\s*/\s*think(?:ing)?\s*>',
                  '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<\s*think(?:ing)?\s*>.*?$', '', text,
                  flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<\s*/?\s*think(?:ing)?\s*>', '', text, flags=re.IGNORECASE)

    return text


def _truncate_repetitions(text, min_block_chars=120):
    """Detect when a small model has fallen into a generation loop and
    truncate at the first repetition.

    Splits the response into paragraphs (blank-line-delimited blocks). If any
    paragraph longer than ``min_block_chars`` appears more than once, keep
    only the first occurrence and drop everything from the second onward —
    that's where the loop started, and nothing useful comes after.

    A whitespace-collapsed lowercase fingerprint is used for the comparison
    so the detector survives the model adding/removing punctuation between
    iterations.
    """
    if not text or len(text) < min_block_chars * 2:
        return text
    import re
    paragraphs = re.split(r'\n\s*\n', text)
    seen = {}
    keep_until = len(paragraphs)
    for i, p in enumerate(paragraphs):
        p_stripped = p.strip()
        if len(p_stripped) < min_block_chars:
            continue
        fingerprint = re.sub(r'\s+', ' ', p_stripped.lower())
        if fingerprint in seen:
            keep_until = i
            break
        seen[fingerprint] = i
    if keep_until < len(paragraphs):
        return '\n\n'.join(paragraphs[:keep_until]).rstrip()
    return text


# Every number-capturing pattern carries _NOT_TIME_UNIT so duration asks
# ("give me one minute of selects") fall through to the default instead of
# being read as a count.
_CLIP_COUNT_PATTERNS = [
    (r'\b(?:find|give|pull|get|show|grab)\s+(?:me\s+)?(\d+)' + _NOT_TIME_UNIT + r'\s+clips?\b', None),
    (r'\b(\d+)' + _NOT_TIME_UNIT + r'\s+clips?\b', None),
    (r'\bthe\s+(?:single\s+)?best\s+(?:one|clip)\b', 1),
    (r'\bjust\s+one\b' + _NOT_TIME_UNIT
     + r'|\bgive\s+me\s+one\b' + _NOT_TIME_UNIT
     + r'|\bonly\s+one\b' + _NOT_TIME_UNIT, 1),
]


def _extract_clip_count_from_message(message, default=3):
    """Parse an explicit clip count from the user's message ("find 2 clips",
    "the best one", etc). Returns ``default`` when no explicit number is found.
    Caps at 8 so a fat-fingered "find 25 clips" doesn't generate a wall of
    half-relevant moments.
    """
    if not message:
        return default
    import re
    m = message.lower()
    for pat, fixed in _CLIP_COUNT_PATTERNS:
        match = re.search(pat, m)
        if match:
            if fixed is not None:
                return fixed
            try:
                n = int(match.group(1))
                return max(1, min(8, n))
            except (ValueError, IndexError):
                continue
    return default


_SALVAGE_PROMPT = """The previous response did not include any [CLIP: start=HH:MM:SS end=HH:MM:SS title="..."] markers, but the user needs them — without markers the UI shows nothing playable.

Read the previous response and the transcript. Extract {target_count} specific moment(s) from the transcript that best match what the previous response was discussing. For each, emit a single [CLIP:] marker in this EXACT shape:

  [CLIP: start=HH:MM:SS end=HH:MM:SS title="2-6 word title"]

Concrete example using a real timecode shape (DO NOT copy these specific times — pick yours from the transcript below):

  [CLIP: start=00:01:23 end=00:01:45 title="speaker confesses doubt"]

Hard rules:
- Output ONLY [CLIP:] markers, one per line. No prose, no explanation, no headers, no bullets, no placeholder text like "[CLIP]".
- Every timecode MUST be copied from a [HH:MM:SS-HH:MM:SS] segment marker in the transcript below. Do not invent times. Do not output an empty marker.
- Each clip should span 5-60 seconds. Combine adjacent segments if needed for a complete thought.
- Title: 2-6 words, no quotes around it, no markdown.

PREVIOUS RESPONSE (what to find clips for):
{prior_response}

TRANSCRIPT:
{formatted}

Now output exactly {target_count} fully-filled [CLIP:] marker(s) and nothing else."""


def _deterministic_clip_markers(matched_paragraphs, target_count):
    """Build canonical ``[CLIP:]`` markers directly from already-retrieved
    transcript paragraphs. Used as a non-LLM fallback when both the main
    chat call and the LLM salvage call fail to produce valid markers.

    ``matched_paragraphs`` is the list returned by
    :func:`_find_relevant_paragraphs` (and friends) — already ranked so the
    top entries are the most relevant to the user's query. We just take the
    first ``target_count`` and format their ``start``/``end`` as anchored
    markers. Title is the first 5 words of the paragraph text.

    Returns a string of newline-separated markers, or ``''`` if nothing
    formattable was found.
    """
    if not matched_paragraphs:
        return ''
    lines = []
    seen = set()
    for p in matched_paragraphs:
        if len(lines) >= target_count:
            break
        try:
            start_sec = float(p.get('start', 0) or 0)
            end_sec = float(p.get('end', start_sec) or start_sec)
        except (TypeError, ValueError):
            continue
        if end_sec <= start_sec:
            continue
        # No sliver cards from the deterministic fallback — same floor
        # the canonicalization pass applies to model-emitted markers.
        if end_sec - start_sec < _MIN_CLIP_MARKER_SECONDS:
            continue
        # Cap ultra-long paragraphs at 60s to keep clips usable
        if end_sec - start_sec > 60:
            end_sec = start_sec + 45
        # De-duplicate near-identical timecodes from overlapping context expansions
        key = (round(start_sec / 5), round(end_sec / 5))
        if key in seen:
            continue
        seen.add(key)
        # Title from first few words of the paragraph
        text = (p.get('text') or '').strip()
        words = text.split()[:5]
        title = ' '.join(words).rstrip('.,!?;:')[:40] or 'transcript moment'
        # Sanitize title for marker form (no quotes, no brackets)
        title = title.replace('"', "'").replace('[', '(').replace(']', ')')
        start_tc = _seconds_to_tc(start_sec)
        end_tc = _seconds_to_tc(end_sec)
        lines.append(f'[CLIP: start={start_tc} end={end_tc} title="{title}"]')
    return '\n'.join(lines)


def _salvage_clips_if_missing(cleaned_text, formatted_transcript, segments,
                                num_ctx=8192, target_count=3,
                                matched_paragraphs=None, user_message=None,
                                language_directive_text='',
                                skip_title_anchor=False):
    """Force clips into the response when the model's first pass produced
    none. Two-stage salvage:

      1. Focused LLM extractor call with an EXAMPLE marker in the prompt.
         Stricter than the main system prompt; usually rescues mid-tier
         models (gemma4:e2b, e4b) that skipped the contract on attempt one.
      2. Deterministic fallback: build canonical markers directly from
         already-retrieved transcript paragraphs (no LLM). This is the
         floor — even a model that emits pure garbage gets the user
         playable clips drawn from the same paragraphs the LLM was
         shown as RELEVANT EXCERPTS.

    Disabled when ``DOZA_DISABLE_CLIP_SALVAGE`` is set so power users can
    opt out for performance. Returns ``cleaned_text`` either unchanged
    (markers already present) or with appended marker block.
    """
    if os.environ.get('DOZA_DISABLE_CLIP_SALVAGE'):
        return cleaned_text
    if _count_clip_markers(cleaned_text) > 0:
        return cleaned_text
    if not (formatted_transcript and segments):
        return cleaned_text

    # Honor user-stated count when present ("find 2 clips" → 2 markers)
    target_count = _extract_clip_count_from_message(user_message, default=target_count)

    prose_to_keep = (cleaned_text or '').strip()
    # Cap the prior response so the salvage call doesn't burn context on
    # rambly prose from the first pass.
    prior = prose_to_keep
    if len(prior) > 2000:
        prior = prior[:2000].rstrip() + '...'
    if not prior:
        prior = '(the previous response was empty — extract clips that match the transcript)'

    salvage_prompt = _SALVAGE_PROMPT.format(
        target_count=target_count,
        prior_response=prior,
        formatted=formatted_transcript,
    )

    salvage_system = (
        "You are a clip-marker extractor. You output ONLY filled-in "
        '[CLIP: start=HH:MM:SS end=HH:MM:SS title="..."] markers, one per '
        "line, with real timecodes from the transcript. You never emit "
        "an empty [CLIP] placeholder. No prose. No explanation."
    )
    # Output-language directive ('' for English → byte-identical prompt).
    if language_directive_text:
        salvage_system = salvage_system + language_directive_text

    validated = ''
    try:
        raw = _call_ai_chat(
            salvage_system,
            [{'role': 'user', 'content': salvage_prompt}],
            num_ctx=num_ctx,
        )
    except Exception as e:
        print(f"[salvage] LLM extractor call failed: {e}")
        raw = ''

    if raw and raw.strip():
        normalized = _clean_chat_response(raw)
        validated = _validate_clip_markers_in_text(
            normalized, segments, skip_title_anchor=skip_title_anchor,
        )

    # Deterministic fallback: if the LLM extractor produced nothing usable,
    # build markers from the matched paragraphs directly. This guarantees
    # the user sees playable clips even when the model is utter garbage.
    if _count_clip_markers(validated) == 0:
        det = _deterministic_clip_markers(matched_paragraphs, target_count)
        if det:
            print(f"[salvage] LLM extractor failed; using deterministic markers from "
                  f"{len(matched_paragraphs or [])} matched paragraph(s)")
            validated = det
        else:
            print("[salvage] no valid markers and no retrieval matches — "
                  "returning prose only")
            return prose_to_keep

    if not validated.strip():
        return prose_to_keep

    # Keep the prose context AND the markers — prose explains why, markers
    # give the UI playable cards.
    if prose_to_keep:
        return prose_to_keep.rstrip() + "\n\n" + validated.strip()
    return validated.strip()


def _validate_clip_markers_in_text(text, segments, grace_seconds=5.0,
                                   skip_title_anchor=False):
    """Drop ``[CLIP:]`` markers whose timecodes don't anchor to the transcript.

    Small local models occasionally invent timecodes that look plausible
    (``02:31:14`` on a 90-min interview) but fall outside any transcript
    segment, so the resulting clip card scrubs to silence. This pass parses
    the canonical marker form, looks up each ``start`` against the segment
    list, and removes the marker line when start falls more than
    ``grace_seconds`` outside any segment's ``[start, end]`` window.

    The grace window mirrors ``_parse_chunk_response``'s ±5s tolerance — a
    legitimately-anchored clip that ends in trailing silence shouldn't be
    dropped just because the model rounded ``end`` past the last spoken word.

    Returns ``text`` unchanged when ``segments`` is empty or no markers
    are present, so callers can apply this unconditionally.

    ``skip_title_anchor=True`` bypasses ONLY the title-anchor sub-check —
    numeric timecode validation still runs. Chat call sites set it when the
    resolved output language differs from the transcript language: clip
    titles are then translated prose whose words won't literally appear in
    the transcript, so the cross-reference heuristic would drop valid clips.
    """
    import re
    if not text or not segments:
        return text

    # Collect transcript bounds + per-segment ranges. Total bounds let us
    # short-circuit obvious off-the-end timecodes; the per-segment list is
    # used for "is this start time inside any segment" checks. We also keep
    # each segment's text (lowercased) for the title-anchor check below.
    seg_ranges = []
    seg_text_ranges = []  # (start, end, text_lower)
    for s in segments:
        try:
            seg_start = float(s.get('start', 0) or 0)
            seg_end = float(s.get('end', seg_start) or seg_start)
        except (TypeError, ValueError):
            continue
        if seg_end >= seg_start:
            seg_ranges.append((seg_start, seg_end))
            seg_text_ranges.append((seg_start, seg_end, (s.get('text') or '').lower()))
    if not seg_ranges:
        return text
    transcript_start = seg_ranges[0][0]
    transcript_end = max(end for _, end in seg_ranges)
    _full_text_lower = ' '.join(t for _s, _e, t in seg_text_ranges)

    def _start_in_transcript(start_sec):
        # Outside the whole-transcript window? Drop.
        if start_sec < transcript_start - grace_seconds:
            return False
        if start_sec > transcript_end + grace_seconds:
            return False
        # Inside the global window but not anchored to any specific segment?
        # Tolerate small gaps between segments (silences) by accepting any
        # start within `grace_seconds` of a segment edge.
        for seg_start, seg_end in seg_ranges:
            if seg_start - grace_seconds <= start_sec <= seg_end + grace_seconds:
                return True
        return False

    marker_re = re.compile(r'\[CLIP:[^\]]*?start=([^\s\]]+)[^\]]*\]')

    # ── Title-anchor check ────────────────────────────────────────────
    # The model sometimes copies a valid timecode from one segment but
    # titles the clip from a DIFFERENT moment. The timecode check above
    # can't catch that (the start is in-bounds). This conservative pass
    # drops a clip only when its title clearly belongs elsewhere:
    # a distinctive title word is absent from the clip's own window but
    # present somewhere else in the transcript. Purely interpretive
    # titles (no distinctive word, or words that appear nowhere) are
    # kept — we never drop on abstraction, only on demonstrable
    # cross-reference.
    _STOPWORDS = {
        'the', 'and', 'for', 'with', 'that', 'this', 'from', 'about', 'into',
        'what', 'when', 'where', 'which', 'while', 'their', 'there', 'them',
        'they', 'have', 'has', 'his', 'her', 'him', 'she', 'are', 'was', 'were',
        'you', 'your', 'our', 'out', 'not', 'but', 'how', 'why', 'who', 'all',
        'one', 'its', 'his', 'than', 'then', 'over', 'just', 'more', 'most',
        'some', 'such', 'only', 'very', 'will', 'would', 'could', 'should',
        'been', 'being', 'after', 'before', 'because', 'on', 'in', 'of', 'to',
        'a', 'an', 'is', 'it', 'as', 'at', 'by', 'be', 'or', 'so', 'we', 'i',
    }
    title_re = re.compile(r'title\s*=\s*(["\'“‘])(.*?)(["\'”’])', re.I)

    def _distinctive_tokens(title):
        toks = re.findall(r"[A-Za-z][A-Za-z'\-]+", title or '')
        out = []
        for i, t in enumerate(toks):
            low = t.lower().strip("'-")
            if len(low) < 4 or low in _STOPWORDS:
                # Keep mid-title capitalized words (proper nouns) even if short.
                if not (i > 0 and t[:1].isupper() and len(low) >= 3):
                    continue
            out.append(low)
        return out

    def _window_text(start_sec, end_sec):
        lo = start_sec - grace_seconds
        hi = end_sec + grace_seconds if end_sec else start_sec + 60
        parts = [t for (ss, se, t) in seg_text_ranges if se >= lo and ss <= hi]
        return ' '.join(parts)

    def _title_anchored(start_sec, end_sec, title):
        """False => the title clearly describes a different moment (drop)."""
        # Mechanical titles minted by _auto_wrap_timecode_ranges ("Moment
        # at 12:34") carry no editorial claim — their timecode came from
        # the model's own prose and already passed the numeric check.
        # Judging them on the word "moment" deleted the user's sentence
        # whenever the speaker happened to say "moment" elsewhere. The
        # exemption is pinned to the EXACT minted shape so a model-
        # authored "Moment at the funeral" title still gets anchored.
        if re.match(r'^Moment at \d{1,2}:\d{2}(?::\d{2})?\s*$',
                    title or '', re.IGNORECASE):
            return True
        tokens = _distinctive_tokens(title)
        if not tokens:
            return True  # nothing distinctive to judge — keep
        window = _window_text(start_sec, end_sec)
        if any(tok in window for tok in tokens):
            return True  # title relates to its own window — keep
        # No distinctive token in the window. Drop ONLY if at least one
        # appears elsewhere in the transcript (proof it belongs to another
        # moment). If it appears nowhere, the title is interpretive — keep.
        if any(tok in _full_text_lower for tok in tokens):
            return False
        return True

    def _is_valid(match):
        start_str = match.group(1).strip().strip('"\'')
        try:
            start_sec = _tc_to_seconds(start_str)
        except Exception:
            return True  # if we can't parse, leave it for the renderer to sort out
        if not _start_in_transcript(start_sec):
            return False
        # Title-anchor check (best-effort; never raises). Bypassed when the
        # reply language differs from the transcript language — translated
        # titles legitimately share no surface words with the transcript.
        if not skip_title_anchor:
            try:
                full = match.group(0)
                tm = title_re.search(full)
                if tm:
                    em = re.search(r'end=([^\s\]]+)', full)
                    end_sec = _tc_to_seconds(em.group(1).strip().strip('"\'')) if em else start_sec
                    if not _title_anchored(start_sec, end_sec, tm.group(2)):
                        print(f"[clip-validate] dropping misanchored clip: "
                              f"start={start_str} title={tm.group(2)!r}", flush=True)
                        return False
            except Exception:
                pass
        return True

    # Walk lines. A line that is ESSENTIALLY the marker (marker + at most
    # a few characters of glue) is dropped whole so no orphan card-intro
    # survives. But when the marker rides inside a real prose sentence,
    # only the marker is excised — deleting the editor's sentence because
    # its attached card failed validation was destroying legitimate,
    # specific answer content.
    cleaned_lines = []
    for line in text.split('\n'):
        bad_spans = [m.span() for m in marker_re.finditer(line)
                     if not _is_valid(m)]
        if not bad_spans:
            cleaned_lines.append(line)
            continue
        stripped_line = line
        for s, e in reversed(bad_spans):
            stripped_line = stripped_line[:s] + stripped_line[e:]
        # ≥20 chars of surviving prose means the line said something of
        # its own — keep it (minus the bad markers). Below that it was
        # just marker scaffolding; drop the whole line.
        if len(stripped_line.strip()) >= 20:
            cleaned_lines.append(stripped_line.rstrip())
    cleaned = '\n'.join(cleaned_lines)
    # Collapse blank-line runs the dropped markers might have left behind.
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned


def _validate_clip_timecodes(clips, segments, *, text_key=None,
                             grace_seconds=5.0, kind="clip"):
    """Validate (and where safe, repair) the HH:MM:SS ``start``/``end`` of
    structured-analysis clips against real transcript segments — the
    structured-output counterpart to :func:`_validate_clip_markers_in_text`.

    Mirrors that function's anchoring rule: a ``start`` is valid when it lies
    inside the whole-transcript window AND within ``grace_seconds`` of some
    segment's ``[start, end]`` (``_tc_to_seconds`` parses the timecode).
    Decision per clip:

      * **Anchored start** -> KEEP. If ``end`` overran the transcript it is
        clamped back to the last segment's end.
      * **Unanchored start, ``text_key`` given** (soundbites / social clips,
        which carry a verbatim transcript quote) -> REPAIR: snap ``start``/
        ``end`` to the contiguous segment run whose text matches the quote.
        The quote tells us where the clip actually belongs. No confident
        match -> DROP.
      * **Unanchored start, no ``text_key``** (story beats / b-roll, only an
        interpretive label) -> DROP. We never relocate to a guessed segment;
        a confidently-wrong interpretive clip is worse than a missing one.

    Clips with no ``start`` at all are left untouched (a missing timecode is a
    separate concern from a hallucinated one). Each keep-clamp / repair / drop
    is logged to stderr so the rates are observable. Returns the validated
    list; returns ``clips`` unchanged when ``segments`` is empty.
    """
    import re
    import sys

    if not isinstance(clips, list) or not clips or not segments:
        return clips if isinstance(clips, list) else []

    def _tokens(s):
        return re.findall(r'[a-z0-9]+', (s or '').lower())

    seg_ranges = []   # (start, end)
    seg_norm = []     # (start, end, joined_text, token_set)
    for s in segments:
        try:
            ss = float(s.get('start', 0) or 0)
            se = float(s.get('end', ss) or ss)
        except (TypeError, ValueError):
            continue
        if se >= ss:
            toks = _tokens(s.get('text'))
            seg_ranges.append((ss, se))
            seg_norm.append((ss, se, ' '.join(toks), set(toks)))
    if not seg_ranges:
        return clips
    t_start = seg_ranges[0][0]
    t_end = max(e for _, e in seg_ranges)

    def _anchored(sec):
        if sec < t_start - grace_seconds or sec > t_end + grace_seconds:
            return False
        return any(a - grace_seconds <= sec <= b + grace_seconds
                   for a, b in seg_ranges)

    def _text_anchor(quote):
        """Snap to the contiguous segment run best matching the verbatim
        quote. Returns (start_sec, end_sec), or None when no confident match."""
        qtoks = _tokens(quote)
        if len(qtoks) < 3:
            return None
        qjoined = ' '.join(qtoks)
        qset = set(qtoks)
        scores = []
        for (_ss, _se, sj, sset) in seg_norm:
            if len(sset) < 3:
                scores.append(0.0)
            elif sj and sj in qjoined:
                scores.append(1.0)  # segment text appears verbatim in the quote
            else:
                scores.append(len(sset & qset) / len(sset))
        if not scores:
            return None
        best = max(range(len(scores)), key=lambda i: scores[i])
        if scores[best] < 0.6:
            return None
        lo = hi = best
        while lo - 1 >= 0 and scores[lo - 1] >= 0.6:
            lo -= 1
        while hi + 1 < len(scores) and scores[hi + 1] >= 0.6:
            hi += 1
        return seg_norm[lo][0], seg_norm[hi][1]

    out = []
    kept = clamped = repaired = dropped = 0
    for c in clips:
        if not isinstance(c, dict):
            dropped += 1
            print(f"[analysis-validate] {kind} dropped (not an object)", file=sys.stderr)
            continue
        raw_start = c.get('start')
        if not str(raw_start or '').strip():
            out.append(c)  # no timecode to validate — leave as-is
            kept += 1
            continue
        start_sec = _tc_to_seconds(raw_start)
        if _anchored(start_sec):
            nc = dict(c)
            if _tc_to_seconds(c.get('end')) > t_end + grace_seconds:
                nc['end'] = _seconds_to_tc(t_end)
                clamped += 1
                print(f"[analysis-validate] {kind} end clamped "
                      f"{c.get('end')}->{nc['end']} (start={raw_start})",
                      file=sys.stderr)
            else:
                kept += 1
            out.append(nc)
            continue
        # Unanchored start.
        anchor = _text_anchor(c.get(text_key)) if text_key else None
        if anchor:
            nc = dict(c)
            nc['start'] = _seconds_to_tc(anchor[0])
            nc['end'] = _seconds_to_tc(anchor[1])
            repaired += 1
            print(f"[analysis-validate] {kind} repaired start "
                  f"{raw_start}->{nc['start']} (verbatim-text match)",
                  file=sys.stderr)
            out.append(nc)
        else:
            dropped += 1
            reason = ("no verbatim-text match" if text_key
                      else "unanchored, no verbatim text to re-anchor")
            print(f"[analysis-validate] {kind} dropped start={raw_start} ({reason})",
                  file=sys.stderr)
    if clamped or repaired or dropped:
        print(f"[analysis-validate] {kind}: kept={kept} clamped={clamped} "
              f"repaired={repaired} dropped={dropped}", file=sys.stderr)
    return out


def _strip_trailing_repetition(text):
    """Remove degenerate trailing repetition from model output.

    Letter-bearing units trip at 4 repeats; punctuation-only units
    (dividers, dotted lines) are usually formatting, so they only trip
    at a 5× higher count — real degeneration floods far past that.
    After cutting, backs up to the last sentence boundary so the reply
    never ends mid-clause with a dangling colon or comma — skipping
    boundary dots inside numbers ('92.5') so quantities don't get
    corrupted by the retreat.
    """
    if len(text) < 30:
        return text
    import re
    for plen in range(4, 60):
        pat = text[-plen:]
        threshold = 4 if any(c.isalpha() for c in pat) else 20
        count = 0
        pos = len(text) - plen
        while pos >= 0 and text[pos:pos + plen] == pat:
            count += 1
            pos -= plen
        if count >= threshold:
            cut = len(text) - (count * plen)
            head = text[:cut].rstrip()
            # Land on a clean boundary: if the cut left a dangling
            # fragment (no terminal punctuation), retreat to the end of
            # the last complete sentence/marker when one exists nearby.
            # A '.' only counts when followed by whitespace/end AND not
            # sandwiched between digits (decimal points are not sentence
            # ends).
            if head and head[-1] not in '.!?"”\']':
                boundary = -1
                for m in re.finditer(r'(?:(?<!\d)\.(?!\d)|[!?\]])(?=\s|$)',
                                     head):
                    boundary = m.start()
                if boundary > len(head) - 200 and boundary > 0:
                    head = head[:boundary + 1]
            print(f"[chat] trailing repetition stripped (pattern "
                  f"{pat!r} x{count})", flush=True)
            return head.rstrip()
    return text


def _clean_chat_response(text):
    """Strip markdown artifacts and emoji from chat responses."""
    import re
    text = text.strip()
    # Tolerant canonicalization FIRST: markers whose attributes span
    # newlines or whose quoted values contain ']'/parens were HALF-matched
    # by the variant normalizer below (its candidate regex stops at the
    # first ')'/']' even inside quotes), shipping a truncated card plus
    # bracket debris to the UI — the 1.0.30 screenshot bug. The tolerant
    # reader re-serializes every recoverable marker into the canonical
    # single-line grammar before any lossier regex can touch it.
    text = _canonicalize_clip_markers(text)
    # Normalize the remaining variant CLIP markers (paren-bracketed forms
    # etc.) BEFORE markdown stripping so the canonical form survives
    # downstream regexes.
    text = _normalize_clip_markers(text)
    # Wrap stray prose timecode ranges — small models in STORY CONSULTING
    # mode often write "(00:12:34 - 00:13:00)" instead of the [CLIP:]
    # marker. Without this pass those moments render as static timecode
    # pills rather than playable clip cards, which is the regression the
    # user reported after the Gemma 4 upgrade.
    text = _auto_wrap_timecode_ranges(text)
    # Remove markdown headers
    text = re.sub(r'^#{1,4}\s*', '', text, flags=re.MULTILINE)
    # Bold/italic markdown is KEPT: both chat frontends render **bold** and
    # *italic* natively (project.html and collection.js), and the system
    # prompt asks the model to bold beat names. The old stripper also had a
    # nasty interaction: removing ** from "1. **Memory:** text" produced the
    # exact "1. Header: text" shape a later pass then deleted wholesale.
    # Only asterisks WRAPPING a CLIP marker are still unwrapped, so the
    # marker starts its line clean for the frontend card regex.
    text = re.sub(r'\*{1,3}(\[CLIP:[^\]]*\])\*{1,3}', r'\1', text)
    # Remove horizontal rules
    text = re.sub(r'^---+\s*$', '', text, flags=re.MULTILINE)
    # Remove emoji (common unicode ranges)
    text = re.sub(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
        r'\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U0001F900-\U0001F9FF'
        r'\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002600-\U000026FF'
        r'\U0000FE00-\U0000FE0F\U0000200D]+', '', text)
    # Strip empty `[CLIP]` placeholders the model may have emitted instead of
    # filled markers. Done AFTER auto-wrapping so legitimate prose-timecode
    # patterns get a chance to be rescued before we sweep up junk.
    text = _strip_empty_clip_placeholders(text)
    # Strip JSON-array-of-zero-timecodes garbage that small models emit when
    # they try to output markers but can't find any moments. Without this
    # pass the frontend renders the all-zero ranges as adjustable timecode
    # chips, which looks like the chat half-worked.
    text = _strip_placeholder_timecode_garbage(text)
    # Strip raw {"type": "highlight", "content": ..., "context": ...} blobs
    # the model leaks alongside (or instead of) [CLIP:] markers. The
    # frontend has no renderer for those, so they show as raw text between
    # the prose summary and the clip cards.
    text = _strip_raw_json_blobs(text)
    # Strip reasoning-mode wrappers ([Thought Process], [Thoughts], <think>...)
    # that small models leak into responses. None of this is meant for users.
    text = _strip_reasoning_tags(text)
    # Strip "[No specific answer provided...]" placeholder narration that
    # small variants prepend to otherwise-valid clip-marker output.
    text = _strip_no_answer_placeholders(text)
    # Strip Gemma 4's untagged meta-narration: leading "[The user is asking…]"
    # blocks, "Suggested Response:" preamble labels, trailing "(Note: …)"
    # caveats. _strip_reasoning_tags only catches blocks with registered
    # openers; this catches the rest.
    text = _strip_meta_preamble(text)
    # Strip essay scaffolding the prompt forbids but Gemma 4B emits anyway:
    # `> ` blockquoted (and often hallucinated) speaker quotes, numbered
    # section headers like "1. The Aha Moment:", "(hypothetical selection)"
    # admissions of fabrication. Defense in depth — the prompt asks Gemma
    # not to do these; this strips them when it does anyway.
    text = _strip_essay_scaffolding(text)
    # Unwrap "[HH:MM:SS]" single-timecode brackets to a bare timecode.
    # The frontend renders bare HH:MM:SS in prose as a clickable jump link
    # with a +clip button — deleting these (the old behavior) threw away
    # the model's most concrete grounding signal. Range-form
    # "[HH:MM:SS - HH:MM:SS]" is wrapped to a CLIP marker by
    # _auto_wrap_timecode_ranges above; this only handles the leftover
    # single-timecode brackets that don't form a range.
    text = re.sub(
        r'\[(?!\s*[Cc][Ll][Ii][Pp]\b)\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*\]',
        r'\1',
        text,
    )
    # Quality floor + residue sweep AFTER every marker-producing pass
    # above (canonicalize / normalize / auto-wrap) so junk from ANY
    # source is caught: sub-5s sliver cards are dropped (the count
    # top-up refills them with real candidates downstream) and no
    # partial-marker bracket debris can survive to the UI.
    text = _enforce_marker_quality_floor(text)
    text = _scrub_clip_marker_residue(text)
    # If the model fell into a generation loop, truncate at the first repeat.
    # Common with small models when they get confused by the strict format.
    text = _truncate_repetitions(text)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# Variant CLIP-marker shapes Gemma 4 (and other small local models) emit in
# chat: single-quoted titles, curly Unicode quotes, unquoted titles, missing
# colon after CLIP, lowercase "clip", alternate key names (start_time,
# end_time), wrapped in **markdown bold**, or even with different bracket
# styles like (CLIP: ...). Without normalization the frontend regex misses
# every one of these and the user sees raw marker text instead of a
# playable clip card.
_CLIP_CANDIDATE_RE = None
_CLIP_KV_RE = None


def _clip_regexes():
    """Lazily compile the CLIP normalization regexes."""
    import re
    global _CLIP_CANDIDATE_RE, _CLIP_KV_RE
    if _CLIP_CANDIDATE_RE is None:
        # Outer: any [...] or (...) block whose first token contains "clip".
        _CLIP_CANDIDATE_RE = re.compile(
            r'[\[\(]\s*\*{0,3}\s*clip\s*\*{0,3}\s*:?\s*([^\]\)]+?)\s*[\]\)]',
            re.IGNORECASE,
        )
        # Inner: key=value pairs. Values may be "double", 'single', “curly”,
        # ‘curly single’, or unquoted up to the next whitespace.
        _CLIP_KV_RE = re.compile(
            r'(\w+)\s*=\s*'
            r'(?:"([^"]*)"'               # double
            r"|'([^']*)'"                 # single
            r'|\u201c([^\u201d]*)\u201d'  # curly double
            r'|\u2018([^\u2019]*)\u2019'  # curly single
            r'|([^\s\]\)]+))',            # unquoted
        )
    return _CLIP_CANDIDATE_RE, _CLIP_KV_RE


def _normalize_clip_markers(text: str) -> str:
    """Rewrite any CLIP-ish marker in ``text`` to the canonical frontend form.

    The frontend regex only matches the exact
    ``[CLIP: start=HH:MM:SS end=HH:MM:SS title="..."]`` shape. This pass
    accepts the variants small local models actually emit and normalizes
    them so every valid clip the model meant to suggest shows up as a
    playable card in the chat.

    Markers already matching :data:`_CANONICAL_CLIP_MARKER_RE` (the exact
    shape :func:`_canonicalize_clip_markers` serializes) are skipped
    byte-for-byte: the candidate regex below stops at the first ``)`` or
    ``]`` even inside a quoted value, so re-running it over a canonical
    marker whose title/note contains parens would truncate the marker and
    leak the tail as prose — the 1.0.30 bracket-debris screenshot bug.
    """
    candidate_re, kv_re = _clip_regexes()

    def _rewrite(m):
        inside = m.group(1)
        pairs = {}
        for km in kv_re.finditer(inside):
            key = km.group(1).lower()
            val = (km.group(2) or km.group(3) or km.group(4)
                   or km.group(5) or km.group(6) or '').strip()
            if val:
                pairs[key] = val

        start = pairs.get('start') or pairs.get('start_time') or pairs.get('begin') or pairs.get('from')
        end = pairs.get('end') or pairs.get('end_time') or pairs.get('finish') or pairs.get('to')
        title = (
            pairs.get('title') or pairs.get('label')
            or pairs.get('name') or pairs.get('heading') or 'Clip'
        )

        # If we can't extract a real timecode pair, leave the original text
        # alone — better to show the raw words than to invent a broken card.
        if not start or not end:
            return m.group(0)

        # Titles shouldn't break our own double-quote wrapping.
        title = title.replace('"', '').strip() or 'Clip'
        title = _format_clip_title(title)
        # Preserve the collection-only project= field and the editorial
        # note= one-liner when the variant carried them (historically they
        # were dropped here, which is why the collection pipeline stashes
        # them around _clean_chat_response — that stash still works, this
        # just stops the single-project pipeline from losing notes).
        extra = ''
        project = _marker_attr_value(pairs.get('project') or '')
        if project:
            extra += f' project="{project}"'
        note = _marker_attr_value(pairs.get('note') or pairs.get('why') or '')
        marker = f'[CLIP: start={start} end={end}{extra} title="{title}"'
        if note:
            marker += f' note="{note}"'
        return marker + ']'

    # Only rewrite OUTSIDE already-canonical markers. Canonical markers
    # still get their title sentence-cased (a value-only touch that can't
    # truncate the marker), matching what the full rewrite used to do.
    parts = re.split('(' + _CANONICAL_CLIP_MARKER_RE.pattern + ')', text)
    for i in range(0, len(parts), 2):
        parts[i] = candidate_re.sub(_rewrite, parts[i])
    for i in range(1, len(parts), 2):
        parts[i] = re.sub(
            r'(title=")([^"]*)(")',
            lambda m: m.group(1) + _format_clip_title(m.group(2)) + m.group(3),
            parts[i], count=1,
        )
    return ''.join(parts)


# ── Canonical [CLIP:] marker grammar ────────────────────────────────────
#
# Every chat reply is canonicalized server-side so the UI only ever
# receives markers in EXACTLY this shape (one line, this attribute order):
#
#   [CLIP: start=<tc> end=<tc> project="<v>" title="<v>" note="<v>"]
#
# project= appears only on collection markers, note= only when the model
# or pipeline supplied one; start/end/title are always present. <tc> is
# whatever parseable token the source carried (HH:MM:SS / MM:SS / bare
# seconds), verbatim — the collection stash keys on the raw token.
# Values are double-quoted and contain NO double quotes (normalized to
# '), NO square brackets (normalized to parens), and NO newlines
# (collapsed to spaces) — so the marker parses identically under
# _CLIP_MARKER_RE, the naive [^\]]* regexes, and both frontends'
# renderChatReply regexes, and round-trips through every enforcement
# pass. Three passes maintain the grammar inside _clean_chat_response:
#
#   1. _canonicalize_clip_markers — TOLERANT reader: attributes may span
#      newlines, quoted values may contain ']' and newlines, attribute
#      order is free, alt key names accepted; every recoverable marker is
#      re-serialized canonically, unrecoverable [CLIP spans are removed.
#   2. _enforce_marker_quality_floor — drops junk cards (sub-5s slivers,
#      unparseable/backwards timecodes, empty titles) WITH their attached
#      prose; the count top-up then refills from the ranked pool.
#   3. _scrub_clip_marker_residue — sweeps any bracket debris that
#      survived: orphan attr="…"] tails, half-note fragments, lone ].

# Model-emitted (or auto-wrapped) cards shorter than this are junk
# slivers — interviewer-question fragments like "Tell me about that"
# 00:15–00:16 in the 1.0.30 screenshot. Dropped by the quality floor and
# skipped by every deterministic top-up.
_MIN_CLIP_MARKER_SECONDS = 5.0

# The EXACT serialized shape (attribute order fixed, double quotes only,
# no ]/newlines in values). Used to protect canonical markers from the
# lossier variant-normalizer and to verify grammar in tests. Keep every
# group non-capturing — callers wrap the whole pattern in one group for
# re.split.
_CANONICAL_CLIP_MARKER_RE = re.compile(
    r'\[CLIP: start=[^\s"\]]+ end=[^\s"\]]+'
    r'(?: project="[^"\n\]]*")?'
    r' title="[^"\n\]]*"'
    r'(?: note="[^"\n\]]*")?\]'
)

_CANON_CLIP_HEAD_RE = re.compile(
    r'\[\s*\*{0,3}\s*CLIP\b\s*\*{0,3}\s*:?', re.IGNORECASE,
)
# One attribute, tolerantly: quoted values may span newlines and contain
# ']'; bare values run to whitespace/']'. Group 1 = key, groups 2-6 = the
# value alternatives (double, single, curly-double, curly-single, bare).
_CANON_ATTR_RE = re.compile(
    r'[ \t]*(?:\r?\n[ \t]*)?,?[ \t]*(\w+)\s*=\s*'
    r'(?:"([^"]*)"'
    r"|'([^']*)'"
    r'|“([^”]*)”'
    r'|‘([^’]*)’'
    r'|([^\s\]]+))'
)
_CANON_KEY_ALIASES = {
    'start': 'start', 'start_time': 'start', 'begin': 'start',
    'end': 'end', 'end_time': 'end', 'finish': 'end',
    'title': 'title', 'label': 'title', 'name': 'title', 'heading': 'title',
    'note': 'note', 'why': 'note',
    'project': 'project',
}

# Placeholder timecode TEMPLATES ("MM:SS", "HH:MM:SS") — colon-separated
# letter groups. A marker carrying these is someone TALKING ABOUT the
# marker syntax ('I use [CLIP: start=MM:SS end=MM:SS title="..."]
# markers.'), i.e. legitimate prose — never a recoverable card and never
# junk to delete (F2).
_PLACEHOLDER_TC_RE = re.compile(r'^[A-Za-z]{1,2}:[A-Za-z]{2}(?::[A-Za-z]{2})?$')

# A COMPLETE timecode token: HH:MM:SS / MM:SS (two-digit trailing
# fields) or bare seconds, optionally unit-suffixed. Bracketless marker
# recovery requires start/end to match this shape — a stream cut mid-
# token ("end=00:02:3") must be dropped as unrecoverable rather than
# fabricating a card at the wrong end time (F9).
_COMPLETE_TC_RE = re.compile(
    r'^(?:\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?'
    r'|\d+(?:\.\d+)?(?:s|secs?|seconds?)?)$',
    re.IGNORECASE,
)

# Marker-vocabulary attribute keys (attr=), shared by the residue sweeps:
# a fragment is only marker DEBRIS when it carries this vocabulary —
# ordinary prose with brackets must never be swept (F5/F6/F7).
_MARKER_ATTR_KEY_RE = re.compile(
    r'\b(start|end|start_time|end_time|title|label|name|heading|note|why|project)'
    r'\s*=',
    re.IGNORECASE,
)


def _marker_attr_value(val):
    """Normalize an attribute value for the canonical marker grammar:
    whitespace runs (including newlines) collapse to a single space,
    inner double quotes become ', square brackets become parens — so no
    downstream regex can ever truncate mid-marker."""
    val = re.sub(r'\s+', ' ', str(val or '')).strip()
    return val.replace('"', "'").replace('[', '(').replace(']', ')')


def _canonicalize_clip_markers(text):
    """Tolerant [CLIP …] reader → canonical single-line serializer.

    The naive normalizer's candidate regex stops at the first ``)``/``]``
    even inside a quoted value and can't cross newlines, so a marker like

        [CLIP: start=… end=… title="…"
        note="A moment of realization…"]

    or one whose note contains parens was HALF-rewritten: a truncated
    card plus the rest of the attribute leaked to the UI as prose (the
    1.0.30 bracket-debris screenshot). This reader parses attributes
    key=value with quoted values allowed to span newlines and contain
    ']'/parens, in any order, note/project optional, alt key names
    accepted — then re-serializes every recoverable marker into the
    canonical grammar. Unrecoverable ``[CLIP`` spans (no usable
    start+end) are removed outright: bracket junk must never reach the
    UI. Idempotent on canonical markers.
    """
    if not text or '[' not in text:
        return text
    out = []
    pos = 0
    while True:
        m = _CANON_CLIP_HEAD_RE.search(text, pos)
        if not m:
            out.append(text[pos:])
            break
        out.append(text[pos:m.start()])
        cursor = m.end()
        attrs = {}
        while True:
            am = _CANON_ATTR_RE.match(text, cursor)
            if not am:
                break
            key = _CANON_KEY_ALIASES.get(am.group(1).lower())
            val = next(
                (g for g in am.group(2, 3, 4, 5, 6) if g is not None), '')
            # Runaway-quote guards: an unterminated quote can swallow the
            # rest of the reply. A value that ate the NEXT marker or a
            # blank line is garbage — stop the attribute run before it
            # (the residue scrub sweeps the leftover fragment).
            if '[clip' in val.lower() or re.search(r'\n[ \t]*\n', val):
                break
            # Unknown keys are consumed only on the marker's own line —
            # never let attribute scanning cross a newline into prose
            # that happens to contain an '=' ("The pacing = great.").
            if key is None and '\n' in text[cursor:am.end(1)]:
                break
            if key:
                attrs.setdefault(key, val)
            cursor = am.end()
        # Closing bracket: directly after the attributes (possibly on the
        # next line), or past a short run of same-line junk.
        closed = False
        tail = re.match(r'[ \t]*(?:\r?\n[ \t]*)?\]', text[cursor:])
        if tail:
            cursor += tail.end()
            closed = True
        else:
            jm = re.match(r'[^\[\]\n]{0,40}\]', text[cursor:])
            if jm:
                cursor += jm.end()
                closed = True
        head_had_colon = m.group(0).endswith(':')
        start_tok = _marker_attr_value(attrs.get('start', '')).replace(' ', '')
        end_tok = _marker_attr_value(attrs.get('end', '')).replace(' ', '')
        recoverable = bool(start_tok and end_tok)
        if recoverable and _PLACEHOLDER_TC_RE.match(start_tok) \
                and _PLACEHOLDER_TC_RE.match(end_tok):
            # Marker-syntax MENTION (template letters like start=MM:SS):
            # prose about the format, not a marker — pass through (F2).
            out.append(text[m.start():cursor])
        elif recoverable and not closed \
                and not (_COMPLETE_TC_RE.match(start_tok)
                         and _COMPLETE_TC_RE.match(end_tok)):
            # Bracketless recovery of a marker whose trailing timecode was
            # cut mid-token ("end=00:02:3"): fabricating a card at the
            # wrong end time is worse than no card — drop as debris (F9).
            pass
        elif recoverable:
            title = _format_clip_title(_marker_attr_value(attrs.get('title', '')))
            project = _marker_attr_value(attrs.get('project', ''))
            note = _marker_attr_value(attrs.get('note', ''))
            bits = [f'start={start_tok}', f'end={end_tok}']
            if project:
                bits.append(f'project="{project}"')
            bits.append('title="{0}"'.format(title or 'Clip'))
            if note:
                bits.append(f'note="{note}"')
            out.append('[CLIP: ' + ' '.join(bits) + ']')
        elif not head_had_colon and not attrs:
            # Bare bracket REFERENCE ("[clip 3]"): colon-less head with no
            # marker attributes is prose pointing at an earlier card,
            # never marker debris — pass through untouched (F6).
            out.append(text[m.start():cursor])
        # else: unrecoverable — the consumed span is dropped entirely.
        pos = cursor
    return ''.join(out)


def _drop_marker_lines_where(text, is_junk, keep_shared_prose=False):
    """Remove every [CLIP:] marker for which ``is_junk(marker_text)`` is
    true, along with its attached prose: the marker's own line plus the
    note lines that follow it up to the next blank line (mirroring
    ``_strip_trimmed_clip_tail`` so the reply never describes cards that
    no longer exist). Mixed lines keep their healthy markers and lose
    only the junk ones. Shared by the quality floor and the re-emission
    screen.

    ``keep_shared_prose=True`` (the quality floor): a junk marker that
    SHARES its line with narration loses only the marker — the whole
    line (plus attached note lines) is dropped only when the marker
    stands alone on it. Chat must never lose legitimate content: 'Great
    line at [CLIP: …4s sliver…], very short but punchy.' keeps its
    sentence (F2). The re-emission screen keeps the historical
    whole-line behavior (a re-shown card's framing prose is itself
    redundant)."""
    kept = []
    dropping = False  # consuming a dropped marker's attached note lines
    for line in text.splitlines():
        markers = _CLIP_MARKER_RE.findall(line)
        if markers:
            junk = [m for m in markers if is_junk(m)]
            if junk and len(junk) == len(markers):
                if keep_shared_prose:
                    remainder = line
                    for m in junk:
                        remainder = remainder.replace(m, '')
                    if remainder.strip():
                        # Narration shares the marker's line — surgical
                        # marker strip, the prose stays (F2).
                        kept.append(remainder)
                        dropping = False
                        continue
                dropping = True
                continue
            for m in junk:  # mixed line: strip just the junk markers
                line = line.replace(m, '')
            dropping = False
            kept.append(line)
            continue
        if not line.strip():
            dropping = False  # a blank line ends the dropped clip's note
            kept.append(line)
            continue
        if dropping:
            continue
        kept.append(line)
    out = '\n'.join(kept)
    # Collapse the blank-line runs the dropped paragraphs leave behind.
    out = re.sub(r'\n[ \t]*\n(?:[ \t]*\n)+', '\n\n', out)
    return out


def _enforce_marker_quality_floor(text):
    """Drop junk [CLIP:] cards regardless of who emitted them: sub-
    ``_MIN_CLIP_MARKER_SECONDS`` slivers (the 1s "Tell me about that"
    interviewer fragments from the 1.0.30 screenshot), unparseable or
    backwards timecode ranges, and empty titles. Runs inside
    _clean_chat_response AFTER every marker-producing pass and BEFORE
    count/duration enforcement — so the count top-up sees the honest
    count and refills the gap with real candidates from the ranked
    pool. Attached prose goes with the dropped marker
    (``_drop_marker_lines_where``)."""
    if not text or '[CLIP' not in text:
        return text

    def _is_junk(marker_text):
        sm = re.search(r'start=([^\s"\]]+)', marker_text)
        em = re.search(r'end=([^\s"\]]+)', marker_text)
        if not (sm and em):
            return True
        if (_PLACEHOLDER_TC_RE.match(sm.group(1))
                and _PLACEHOLDER_TC_RE.match(em.group(1))):
            # Marker-syntax MENTION ("start=MM:SS end=MM:SS" template
            # letters) — prose about the format, never a junk card (F2).
            return False
        s = _tc_to_seconds(sm.group(1))
        e = _tc_to_seconds(em.group(1))
        if e - s < _MIN_CLIP_MARKER_SECONDS:
            return True  # sliver, backwards, or unparseable (0.0 - 0.0)
        tm = re.search(r'title="([^"]*)"', marker_text)
        if tm is not None and not tm.group(1).strip():
            return True
        return False

    return _drop_marker_lines_where(text, _is_junk, keep_shared_prose=True)


# Attribute fragments outside a well-formed marker — the partial-marker
# shapes the 1.0.30 screenshot leaked as prose (`note="A moment of
# realization…"]`). Key names are restricted to the marker vocabulary,
# the run must actually TERMINATE in ']' , and _sweep_attr_tail_run
# additionally requires a start=+end= pair or ≥2 marker attrs — so
# legitimate prose like 'Set title="My Export" in the dialog' or an
# FCPXML help snippet can never be swept (F5).
_MARKER_ATTR_TAIL_RE = re.compile(
    # The trailing ] stays OPTIONAL in the pattern so every attr run
    # matches exactly once and re.sub scans linearly (requiring it here
    # would make every non-]-terminated run fail and rescan from each
    # attr — quadratic/ReDoS); _sweep_attr_tail_run then enforces the
    # ]-termination. The bare-value alternative is disjoint from the
    # quoted ones ((?!["\'])) so the run parses one way only.
    r'(?:\b(?:start|end|start_time|end_time|title|label|name|heading|note|why|project)'
    r'\s*=\s*(?:"[^"\n]*(?:"|$)|\'[^\'\n]*(?:\'|$)|(?!["\'])[^\s\]]+)[ \t]*)+\]?',
    re.IGNORECASE | re.MULTILINE,
)


def _sweep_attr_tail_run(m):
    """Replacement for :data:`_MARKER_ATTR_TAIL_RE` matches: sweep the
    run only when it is unmistakably marker debris — it actually
    TERMINATES in ``]`` AND carries a start=+end= pair or at least two
    marker-vocabulary attributes (F5). A stray ``title="…"`` in prose
    stays put; single-attr NOTE tails on their own line are handled by
    the line-level sweep below."""
    run = m.group(0)
    if not run.endswith(']'):
        return run
    keys = [k.lower() for k in _MARKER_ATTR_KEY_RE.findall(run)]
    has_start = any(k in ('start', 'start_time') for k in keys)
    has_end = any(k in ('end', 'end_time') for k in keys)
    if (has_start and has_end) or len(keys) >= 2:
        return ''
    return run


def _sweep_clip_fragment(m):
    """Replacement for the unrecoverable-``[CLIP`` fragment sweep: only
    fragments whose head carries the marker COLON ("[CLIP: …") or that
    carry marker-attribute vocabulary are debris. A bare prose reference
    like "[clip 3]" is a pointer at an earlier card — never swept (F6)."""
    frag = m.group(0)
    if re.match(r'\[\s*\*{0,3}\s*CLIP\b\s*\*{0,3}\s*:', frag, re.IGNORECASE) \
            or _MARKER_ATTR_KEY_RE.search(frag):
        return ''
    return frag


def _scrub_clip_marker_residue(text):
    """Sweep bracket debris so no partial-marker fragment can render as
    prose: unrecoverable ``[CLIP:`` fragments, orphan ``attr="…"]``
    tails, half-note lines ending in ``"]``, and lone ``]`` lines.
    Well-formed canonical markers are protected (split on
    ``_CLIP_MARKER_RE``); ordinary prose — including legitimate brackets
    like ``[note]``, "[clip 3]" references, JSON-style lines and
    timecode pills — is untouched."""
    if not text or (']' not in text and '[' not in text):
        return text
    parts = re.split('(' + _CLIP_MARKER_RE.pattern + ')', text)
    for i in range(0, len(parts), 2):
        p = parts[i]
        # Unrecoverable [CLIP fragments: to the closing ] on the same
        # line, or to end-of-line when the bracket never closes. Gated on
        # marker colon/vocabulary so "[clip 3]" prose survives (F6).
        p = re.sub(r'\[\s*\*{0,3}\s*CLIP\b[^\]\n]*\]?', _sweep_clip_fragment,
                   p, flags=re.IGNORECASE)
        # Orphan attribute tails ("start=… end=…]", "title="…" note="…"]")
        # — must end in ']' and carry real marker vocabulary (F5).
        p = _MARKER_ATTR_TAIL_RE.sub(_sweep_attr_tail_run, p)
        parts[i] = p
    text = ''.join(parts)
    cleaned_lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        if stripped in (']', '"]'):
            continue  # stray lone bracket line
        # Half-note fragments: a line with no opening bracket and no
        # surviving marker that ends in "] AND carries marker-attribute
        # vocabulary is the tail of a marker whose head was consumed
        # elsewhere. Without the vocabulary it is indistinguishable from
        # legitimate content (JSON-style answers end lines in "]) and
        # must be kept (F7).
        if (stripped.endswith('"]') and '[' not in line
                and not _CLIP_MARKER_RE.search(line)
                and _MARKER_ATTR_KEY_RE.search(line)):
            continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)


def _strip_markdown_outside_clips(text: str) -> str:
    """Apply the bold/italic stripper only to spans that aren't CLIP markers.

    The raw regex ``\\*{1,3}(.*?)\\*{1,3}`` is greedy-lazy and would happily
    eat across a CLIP marker, corrupting it. Splitting the text on CLIP
    markers first keeps the markers intact while still stripping markdown
    everywhere else.
    """
    import re
    parts = re.split(r'(\[CLIP:[^\]]*\])', text)
    for i, p in enumerate(parts):
        if p.startswith('[CLIP:'):
            continue
        parts[i] = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', p)
    return ''.join(parts)


_TC_RANGE_RE = None


def _tc_range_regex():
    """Lazily compile the prose-timecode-range regex."""
    import re
    global _TC_RANGE_RE
    if _TC_RANGE_RE is None:
        # Match a timecode range with any common separator. Timecodes are
        # MM:SS or HH:MM:SS. Separators: ASCII hyphen, en/em dash, or "to".
        # Optional surrounding parens OR square brackets are consumed so the
        # rewritten marker replaces the whole "(0:30 - 1:00)" or
        # "[00:19:30 - 00:19:40]" span cleanly. Without bracket consumption,
        # Gemma's "[HH:MM:SS - HH:MM:SS]" prose pattern turns into nested
        # "[[CLIP: ...]]" markers and the outer [ ] survive as literal text
        # next to the rendered clip card.
        _TC_RANGE_RE = re.compile(
            r'[\(\[]?\s*'
            r'(?P<start>\d{1,2}:\d{2}(?::\d{2})?)'
            r'\s*(?:to|\u2013|\u2014|-)\s*'
            r'(?P<end>\d{1,2}:\d{2}(?::\d{2})?)'
            r'\s*[\)\]]?',
            re.IGNORECASE,
        )
    return _TC_RANGE_RE


def _auto_wrap_timecode_ranges(text: str) -> str:
    """Convert prose timecode ranges into canonical CLIP markers.

    Small local models in STORY CONSULTING mode often reference specific
    moments with raw ranges like ``(00:12:34 - 00:13:00)`` instead of the
    ``[CLIP: ...]`` marker. Without this pass those references render as
    static timecode pills in chat; with it, every referenced range shows
    up as a playable clip card with Play and Add Clip buttons — the
    experience prior versions delivered. Ranges already inside a CLIP
    marker are skipped so the canonical form isn't double-processed.
    """
    import re
    range_re = _tc_range_regex()

    def _rewrite(m):
        start = m.group('start')
        end = m.group('end')
        # Only wrap ranges the quality floor would keep: backwards pairs
        # ("00:00:52 - 00:00:35") were never clips, and a sub-floor
        # sliver ("00:05:10 - 00:05:14") wrapped here would be junk-
        # dropped downstream TOGETHER with the sentence mentioning it —
        # leaving the prose reference intact is strictly better (F2).
        # _tc_to_seconds is tolerant of both MM:SS and HH:MM:SS forms.
        if (_tc_to_seconds(end) - _tc_to_seconds(start)
                < _MIN_CLIP_MARKER_SECONDS):
            return m.group(0)
        # Generic title — the card shows the exact range + duration, so
        # "Moment at 00:12:34" is enough context. Users rename if they
        # care; any fancier extraction risks garbage from the surrounding
        # prose.
        title = f"Moment at {start}"
        return f'[CLIP: start={start} end={end} title="{title}"]'

    # Skip text that's already inside a CLIP marker so the canonical form
    # isn't rewritten back on top of itself.
    parts = re.split(r'(\[CLIP:[^\]]*\])', text)
    for i, p in enumerate(parts):
        if p.startswith('[CLIP:'):
            continue
        parts[i] = range_re.sub(_rewrite, p)
    return ''.join(parts)


def _build_chat_analysis_index(analysis) -> str:
    """Format pre-computed analysis items as a grounding anchor for the chat model.

    Small local models (Gemma 4 e4b) hallucinate timecodes and fabricate quotes
    when asked "pull the best clip" against a bare transcript. The Story Builder
    pass already produced a vetted list of beats, soundbites, and social clips
    with real timecodes. Surfacing that list in the system prompt gives the
    chat model a short, trusted menu to cite from — it's faster than scanning
    the transcript and it's impossible to get wrong if it sticks to the menu.

    Returns "" when no analysis is available so the prompt stays clean; callers
    can concatenate the result unconditionally.
    """
    if not isinstance(analysis, dict):
        return ""

    def _tc(val) -> str:
        if isinstance(val, (int, float)):
            return _seconds_to_tc(val)
        return str(val or "").strip()

    def _short(text, limit=140) -> str:
        text = str(text or "").strip().replace("\n", " ")
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "…"

    lines = []

    beats = analysis.get('story_beats') if isinstance(analysis.get('story_beats'), list) else []
    if beats:
        lines.append("STORY BEATS:")
        for b in beats:
            if not isinstance(b, dict):
                continue
            start = _tc(b.get('start'))
            end = _tc(b.get('end'))
            if not start or not end:
                continue
            label = _short(b.get('label') or b.get('description'), 60) or 'Story Beat'
            why = _short(b.get('why') or b.get('description'), 120)
            suffix = f" — {why}" if why and why != label else ""
            lines.append(f"  [{start}-{end}] {label}{suffix}")

    soundbites = analysis.get('strongest_soundbites') if isinstance(analysis.get('strongest_soundbites'), list) else []
    if soundbites:
        lines.append("STRONGEST SOUNDBITES:")
        for s in soundbites:
            if not isinstance(s, dict):
                continue
            start = _tc(s.get('start'))
            end = _tc(s.get('end'))
            if not start or not end:
                continue
            text = _short(s.get('text'), 140)
            why = _short(s.get('why'), 100)
            body = f'"{text}"' if text else '(no quote)'
            suffix = f" — {why}" if why else ""
            lines.append(f"  [{start}-{end}] {body}{suffix}")

    clips = analysis.get('social_clips') if isinstance(analysis.get('social_clips'), list) else []
    if clips:
        lines.append("SOCIAL CLIPS:")
        for c in clips:
            if not isinstance(c, dict):
                continue
            start = _tc(c.get('start'))
            end = _tc(c.get('end'))
            if not start or not end:
                continue
            title = _short(c.get('title') or c.get('text'), 60) or 'Social Clip'
            why = _short(c.get('why'), 100)
            suffix = f" — {why}" if why else ""
            lines.append(f"  [{start}-{end}] {title}{suffix}")

    if not lines:
        return ""

    return (
        "\n\nPRE-ANALYZED MOMENTS (real timecodes — when you need a timecode "
        "for a [CLIP:] marker and a question matches one of these, prefer "
        "citing from this list; but the TRANSCRIPT above is the source of "
        "truth for what was actually said — answer from it, not from these "
        "labels):\n" + "\n".join(lines)
    )


# Transcript-chunking threshold. Small local models (Gemma 4 e4b in particular)
# lose attention on very long inputs: they produce 3-4 beats covering only the
# first 5-10 minutes and then stop, even with plenty of output budget. Chunking
# the transcript into ~15-minute slices forces the model to analyze each slice
# independently so we get coverage across the whole interview.
CHUNK_MINUTES = 15
_LONG_INTERVIEW_SECONDS = CHUNK_MINUTES * 60

# Paragraph-grouping threshold for the chat path. Chat fits the full transcript
# into a single prompt (unlike Story Builder, which chunks) because follow-up
# questions need global context. But past ~60 min Gemma 4 can't hold the CLIP
# marker contract across 1,000+ segment lines — on the 1,167-segment Trustees
# project "find me the best clip" returned bare "[133]" instead of clip cards.
# Above this duration the prompt builder switches to same-speaker paragraph
# grouping (~10x fewer lines on a monologue interview) to stay inside the
# model's attention window.
_LONG_CHAT_SECONDS = 60 * 60

# Layer 2 (chunked search) knobs. Only consulted when the transcript is past
# `_LONG_CHAT_SECONDS` AND no keyword matches can be extracted from the user's
# question — i.e. abstract/synthesis queries on multi-hour interviews. Kept as
# module-level constants so they're easy to tune without touching call sites.
_CHAT_CHUNK_TOKENS = 7000
_CHAT_CHUNK_OVERLAP_PARAGRAPHS = 3
_CHAT_CHUNK_CONCURRENCY = 4
_CHAT_TOP_K_CLIPS = 5


# Words → digits, for parsing user requests like "give me one clip".
# Capped at 10 because requesting more than that in one chat is unusual
# and the rest of the prompt machinery isn't tuned for very large outputs.
_NUMBER_WORDS = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'a': 1, 'an': 1, 'single': 1,
}

_BEST_PHRASES = (
    'the best one',
    'the strongest one',
    'the single best',
    'the best clip',
    'the strongest clip',
    'the most powerful one',
)


def _parse_user_clip_count(message):
    """Extract an explicit clip count from a user message.

    Returns an int 1-10 if the user clearly asked for that many clips,
    or None if no count was specified. Used by the long-transcript
    chunked-search path to override the default top-K so that
    "find me 1 clip about X" actually returns 1.

    Examples that return ``1``:
        "find me 1 great clip"
        "give me one clip"
        "the best clip"
        "pull a single clip about resilience"

    Examples that return ``None`` (use default 2-5):
        "what's the emotional arc?"
        "find moments about resilience"
        "show me clips about the bakery"
    """
    if not message:
        return None
    msg = message.lower()

    # Phrases that imply exactly 1 — handle before the digit/word match
    # so "the best clip" doesn't fail to match because "clip" is not
    # preceded by a number.
    for phrase in _BEST_PHRASES:
        if phrase in msg:
            return 1

    import re as _re
    # Digits followed by clip(s)/moment(s)/soundbite(s)/quote(s). The
    # _NOT_TIME_UNIT guard keeps duration phrasings ("2 minute clips") out —
    # those belong to parse_target_duration_seconds.
    m = _re.search(r'\b(\d{1,2})' + _NOT_TIME_UNIT + r'\s+(?:great\s+|strong\s+|best\s+)?(?:clip|moment|soundbite|quote|excerpt)s?\b', msg)
    if m:
        try:
            n = int(m.group(1))
            if 1 <= n <= 10:
                return n
        except ValueError:
            pass

    # Number-words (one, two, three, ..., a, an, single)
    m = _re.search(
        r'\b(one|two|three|four|five|six|seven|eight|nine|ten|a|an|single)\s+'
        r'(?:great\s+|strong\s+|best\s+)?'
        r'(?:clip|moment|soundbite|quote|excerpt)s?\b',
        msg,
    )
    if m:
        return _NUMBER_WORDS.get(m.group(1))

    return None


def _layer2_explicit_count(message):
    """Explicit user clip count for a Layer 2 pick, from EITHER parser,
    or ``None``: "find me 3 clips" hits the Layer-2 parser above, but
    count phrasings without a clip noun ("give me 5 more") only parse via
    the chat-side ``_detect_explicit_clip_count`` — a long-transcript ask
    must honor them the same way Layer 1's trim/top-up does."""
    user_count = _parse_user_clip_count(message)
    if user_count is None:
        user_count = _detect_explicit_clip_count(message)
    return user_count


def _layer2_final_top_k(message):
    """Clip count for a Layer 2 pick: an explicit user count wins
    (``_layer2_explicit_count``); no count → the default top-K."""
    user_count = _layer2_explicit_count(message)
    return user_count if user_count is not None else _CHAT_TOP_K_CLIPS


# Follow-up phrasings that ask for material BEYOND what the conversation
# already surfaced. Deliberately narrow, and anchored to FOLLOW-UP SYNTAX
# rather than bare content words: a fresh re-ask of the same question may
# legitimately re-find the same best moment, and bare 'new'/'different'/
# 'other'/'else' are ordinary content words ("moving to new york",
# "different opinions", "her other siblings", "everything else she lost")
# — matching them silently dropped a fresh ask's best candidates before
# ranking. The words only count when they relate to the ask: modifying a
# clip noun ("other moments", "different clips", "extra options"),
# adjacent to a count ("5 more", "a few more", "another 2"), verb-
# anchored ("show me more", "give me another", "find others"), or in a
# standalone follow-up shape ("what else?", "any others?", "besides
# those"). Precision beats recall here (guiding principle): a missed
# follow-up merely repeats a clip, a false positive withholds the best
# answer with no indication.
_MORE_CLIPS_NOUN = (
    r'(?:clips?|moments?|quotes?|soundbites?|highlights?|excerpts?'
    r'|ones?|options?|selects?|picks?|angles?|takes?)'
)
_MORE_CLIPS_RE = re.compile(
    # "more clips", "another moment", "extra/other/different/new ones"
    r'\b(?:more|another|additional|extra|other|others|different|new|fresh)'
    r'\s+' + _MORE_CLIPS_NOUN + r'\b'
    # "5 more", "one more", "a few more", "some more", "any more"
    r'|\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten'
    r'|few|couple|some|several|any)\s+more\b'
    # "more of those/these/them/the same"
    r'|\bmore\s+of\s+(?:those|these|them|that|the\s+same)\b'
    # verb-anchored bare follow-ups: "show me more", "give me another",
    # "find others", "pull up some more"
    r'|\b(?:show|give|find|pull|get|grab|surface|dig)\s+(?:me\s+|us\s+)?'
    r'(?:up\s+)?(?:some\s+|a\s+few\s+)?(?:more|another|others)\b'
    # "another 2 minutes of selects", "another few"
    r'|\banother\s+(?:\d+|few|couple)\b'
    # standalone follow-up shapes
    r'|\b(?:what|who|anything|something|any)\s+else\b'
    r'|\bany\s+others?\b'
    r'|\bbesides\s+(?:that|those|these|them|what)\b'
)


def _exclude_shown_candidates(candidates, message, history):
    """Drop Layer 2 candidates overlapping clips earlier turns already
    showed — but only on "more"-style follow-ups ("give me 5 more",
    "what other moments are there?"). "5 more" means 5 NEW moments; the
    chunk search itself has no memory, so without this the same top
    candidates come straight back. Non-"more" asks keep the full pool.
    """
    if not candidates or not _MORE_CLIPS_RE.search((message or '').lower()):
        return candidates
    prior = [(s, e) for s, e, _g in _history_clip_spans(history)]
    if not prior:
        return candidates
    return [
        c for c in candidates
        if not _spans_overlap(
            c.get('start_sec', 0), c.get('end_sec', 0), prior)
    ]


# English-only stopwords for Layer 1 keyword extraction. Kept short on purpose:
# the point is to drop boilerplate question words, not to build a full NLP
# stoplist. Anything a user would reasonably *search for* (nouns, proper nouns,
# topical adjectives) must pass through.
# TODO(i18n): expand beyond English when we add multilingual transcript support.
_CHAT_STOPWORDS = frozenset({
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an',
    'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been',
    'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't",
    'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't",
    'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from',
    'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having',
    'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers',
    'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll",
    "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its',
    'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no',
    'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought',
    'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she',
    "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such',
    'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves',
    'then', 'there', "there's", 'these', 'they', "they'd", "they'll",
    "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under',
    'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're",
    "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where',
    "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's",
    'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're",
    "you've", 'your', 'yours', 'yourself', 'yourselves',
    # Apostrophe-free contractions (users skip punctuation in chat).
    'whats', 'hes', 'shes', 'theyre', 'dont', 'doesnt', 'didnt', 'isnt',
    'cant', 'wont', 'wouldnt', 'couldnt', 'shouldnt', 'thats', 'hows',
    # Question/meta words that show up constantly but aren't search terms.
    'find', 'show', 'give', 'tell', 'say', 'said', 'talk', 'talks', 'talked',
    'talking', 'mention', 'mentions', 'mentioned', 'discuss', 'discusses',
    'discussed', 'clip', 'clips', 'moment', 'moments', 'part', 'parts',
    'section', 'sections', 'pull', 'pulled', 'pick', 'best', 'good', 'great',
    'really', 'just', 'also', 'get', 'got', 'make', 'made', 'know', 'think',
    'going', 'want', 'need', 'like', 'something', 'anything', 'everything',
    # Intent/perception verbs that signal the kind of moment a user wants but
    # aren't search terms themselves. Without these, "show me what they felt
    # about the trustees" anchors heavily on every paragraph containing
    # "felt" — which on a 90-min interview is almost every paragraph.
    'feel', 'feels', 'felt', 'feeling', 'look', 'looks', 'looking', 'looked',
    'see', 'sees', 'seeing', 'seen', 'seem', 'seems', 'seemed', 'seeming',
    'sounds', 'sounded',
    # Apostrophe-free contractions (users skip punctuation in chat).
    'whats', 'hes', 'shes', 'theyre', 'dont', 'doesnt', 'didnt', 'isnt',
    'cant', 'wont', 'wouldnt', 'couldnt', 'shouldnt', 'thats', 'hows',
    # Format, genre, platform, and scope words. Synthesis queries like "best
    # social media clip in this whole interview" should produce zero search
    # terms so the model reasons editorially instead of keyword-anchoring on
    # every paragraph containing "social" or "interview".
    'social', 'media', 'tiktok', 'instagram', 'reel', 'reels', 'shorts',
    'youtube', 'podcast', 'interview', 'whole', 'entire', 'highlight',
    'highlights', 'shareable', 'viral', 'quotable', 'strongest', 'powerful',
})


def _extract_query_keywords(message):
    """Extract keyword phrases and individual words from a user question.

    Returns ``(phrases, words)``. ``phrases`` is a list of multi-word runs of
    adjacent non-stopword tokens (lowercased, ≥2 tokens) — these are tried
    first so "Moose Hill" matches as one thing. ``words`` is the flat list of
    individual non-stopword tokens ≥3 chars — the fallback when no phrase
    hits. Both are lowercased; the caller matches case-insensitively.
    """
    import re
    if not message:
        return [], []
    # Split on anything that isn't alphanumeric or apostrophe. Apostrophes stay
    # so "don't" collapses to a single stopword token rather than "don" + "t".
    tokens = re.findall(r"[A-Za-z0-9']+", message.lower())

    phrases = []
    words = []
    run = []
    for tok in tokens:
        if tok in _CHAT_STOPWORDS or len(tok) < 3:
            if len(run) >= 2:
                phrases.append(' '.join(run))
            run = []
            continue
        run.append(tok)
        words.append(tok)
    if len(run) >= 2:
        phrases.append(' '.join(run))

    # Dedupe while preserving order — the user's phrasing is a hint about
    # importance, so earlier occurrences win.
    def _dedupe(items):
        seen = set()
        out = []
        for item in items:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    return _dedupe(phrases), _dedupe(words)


_SYNTHESIS_SIGNALS = frozenset({
    'best', 'strongest', 'powerful', 'highlight', 'highlights', 'top',
    'quotable', 'shareable', 'viral', 'hook', 'hooks',
    'social', 'tiktok', 'instagram', 'reel', 'reels', 'shorts', 'youtube',
    'podcast', 'soundbite', 'soundbites',
})


def _is_synthesis_query(message):
    """Return True when the query asks for editorial judgment, not a lookup.

    Synthesis queries ("best social media clip", "pull a reel", "top 3
    highlights") should skip strict keyword anchoring and let the model
    reason editorially across the full transcript. Lookup queries ("what
    did she say about Moose Hill", "when does he mention his father") need
    keyword anchoring to find the right passage.
    """
    if not message:
        return False
    import re
    tokens = set(re.findall(r"[a-z0-9']+", message.lower()))
    return bool(tokens & _SYNTHESIS_SIGNALS)


_WORD_NUMBERS = {
    'one': 1, 'single': 1,
    'two': 2, 'couple': 2,
    'three': 3, 'few': 3,
    'four': 4, 'five': 5,
}

_BIGRAM_NUMBERS = {
    ('a', 'single'): 1,
    ('a', 'few'): 3,
    ('a', 'couple'): 2,
}

_CLIP_NOUNS = frozenset({
    'clip', 'clips', 'moment', 'moments', 'highlight', 'highlights',
    'soundbite', 'soundbites', 'quote', 'quotes', 'reel', 'reels',
    'short', 'shorts', 'beat', 'beats',
})


def _extract_quantity_hint(message):
    """Extract an explicit clip count from the user's query, or None.

    Returns an int (1-10) when the user specifies a count adjacent to a
    clip noun ("best 1 clip", "top 3 highlights", "a single soundbite").
    Returns None when no count is present, the number isn't near a clip
    noun, or the count is out of range (0, 50, etc.).
    """
    if not message:
        return None
    import re
    tokens = re.findall(r"[a-z0-9']+", message.lower())
    if not tokens:
        return None

    has_clip_noun = bool(set(tokens) & _CLIP_NOUNS)
    if not has_clip_noun:
        return None

    for i in range(len(tokens) - 1):
        bigram = (tokens[i], tokens[i + 1])
        if bigram in _BIGRAM_NUMBERS:
            return _BIGRAM_NUMBERS[bigram]

    for tok in tokens:
        if tok in _WORD_NUMBERS:
            return _WORD_NUMBERS[tok]

    digits = re.findall(r'\b(\d{1,2})\b', message)
    for d in digits:
        n = int(d)
        if 1 <= n <= 10:
            tc_pattern = re.compile(r'(?:\d{1,2}:\d{2}|\d{4})')
            if not tc_pattern.match(d) and not any(
                message[max(0, message.find(d)-1):message.find(d)] == ':'
                for _ in [None]
            ):
                return n
    return None


def _collect_theme_phrases_from_vectors(segment_vectors, message):
    """Return ``theme_tags`` from segment_vectors that overlap the user's query.

    Each segment_vector carries 2-4 short ``theme_tags`` (e.g. ``["loss",
    "decision"]``). When the user's query mentions one of those tag words
    (or contains a tag word as a substring of a multi-word query token),
    we treat the tag itself as an additional search phrase. This rescues
    abstract queries — "show me the resilience moments" finds paragraphs
    flagged with ``resilience`` even when the speaker never literally
    said the word.

    Returns a deduped list of lowercase phrases. Empty when no overlaps,
    so callers can pass through unconditionally.
    """
    if not segment_vectors or not message:
        return []
    import re
    msg_tokens = set(re.findall(r"[a-z0-9']+", message.lower()))
    if not msg_tokens:
        return []
    matches = []
    seen = set()
    for v in segment_vectors:
        if not isinstance(v, dict):
            continue
        for tag in (v.get('theme_tags') or []):
            if not isinstance(tag, str):
                continue
            tag_lc = tag.strip().lower()
            if not tag_lc or tag_lc in seen:
                continue
            tag_tokens = set(re.findall(r"[a-z0-9']+", tag_lc))
            if tag_tokens & msg_tokens:
                seen.add(tag_lc)
                matches.append(tag_lc)
    return matches


def _find_relevant_paragraphs(paragraphs, phrases, words, context=1, theme_phrases=None):
    """Return the subset of ``paragraphs`` matching the user's query, each
    expanded by ``context`` paragraphs of lead-in and lead-out so the model
    sees the moment in context rather than a bare one-liner.

    Matching is case-insensitive substring. Phrase matching is tried first
    (and outranks individual-word matches whenever any phrase hits) so
    "Moose Hill" doesn't balloon into every "hill" mention on the timeline.

    ``theme_phrases`` is an optional list of canonical phrases — typically
    pulled from ``segment_vectors[*].theme_tags`` — that we treat as a
    domain vocabulary. When a user query contains a token that matches a
    theme phrase, paragraphs tagged with that theme get an automatic match
    even if they don't literally contain the user's word. This is what
    rescues abstract queries like "show me the resilience moments" when
    the speaker never literally said "resilience."

    Duplicate/overlapping paragraphs are de-duplicated by index.
    """
    if not paragraphs:
        return []

    lowered = [(p, (p.get('text') or '').lower()) for p in paragraphs]

    hit_indices = set()

    def _scan(needles):
        found = False
        for needle in needles:
            if not needle:
                continue
            for i, (_p, text_lc) in enumerate(lowered):
                if needle in text_lc:
                    hit_indices.add(i)
                    found = True
        return found

    phrase_hit = _scan(phrases)
    if not phrase_hit:
        _scan(words)
    # Theme matching is additive on top of literal hits — even when literal
    # phrase matches landed, theme overlaps deepen the relevance pool with
    # paragraphs that share the topic but used different words.
    _scan(theme_phrases or [])

    if not hit_indices:
        return []

    # Expand each hit with ±`context` neighbors, then dedupe.
    expanded = set()
    for i in hit_indices:
        for j in range(max(0, i - context), min(len(paragraphs), i + context + 1)):
            expanded.add(j)

    return [paragraphs[i] for i in sorted(expanded)]


def _merge_paragraph_lists(*lists):
    """Merge paragraph lists, deduping by ``(start, end)`` and preserving
    chronological order. Used to fold TF-IDF retrieval hits, keyword/theme
    matches, and vector-anchored augmentations into one excerpt pool with
    no duplicates.
    """
    seen = set()
    merged = []
    for lst in lists:
        for p in (lst or []):
            if not isinstance(p, dict):
                continue
            key = (p.get('start'), p.get('end'))
            if key in seen:
                continue
            seen.add(key)
            merged.append(p)
    merged.sort(key=lambda p: float(p.get('start', 0) or 0))
    return merged


def _augment_with_high_score_vectors(matched_paragraphs, segments, segment_vectors, theme_phrases):
    """Add segments overlapping high-score / theme-matching vectors to the relevance pool.

    The literal-keyword match in :func:`_find_relevant_paragraphs` is anchored
    to the transcript text. Vectors carry an *editorial* signal — a segment
    flagged ``narrative_score=high`` plus matching ``theme_tags`` is a
    pre-curated highlight, even when its surface text doesn't contain the
    user's exact words. Folding those into the relevant excerpts means
    abstract queries ("what's the emotional spine?") still surface the
    moments the editor already labeled as the spine.

    Returns a deduped paragraph list that preserves chronological order.
    Falls back to ``matched_paragraphs`` unchanged when vectors are absent
    or have no high-score / theme-matching entries.
    """
    if not segment_vectors:
        return list(matched_paragraphs or [])
    if not segments:
        return list(matched_paragraphs or [])

    theme_set = {t for t in (theme_phrases or [])}

    # Pick vectors that either (a) literally tag-match the query or (b) are
    # high-narrative-score. Cap so we don't flood the prompt — the relevance
    # pool is meant to surface the strongest few, not the whole transcript.
    selected_ranges = []
    for v in segment_vectors:
        if not isinstance(v, dict):
            continue
        score = str(v.get('narrative_score', 'medium')).lower()
        tags = {str(t).strip().lower() for t in (v.get('theme_tags') or []) if isinstance(t, str)}
        is_match = bool(theme_set & tags) or score == 'high'
        if not is_match:
            continue
        try:
            start = _tc_to_seconds(v.get('timecode_in'))
            end = _tc_to_seconds(v.get('timecode_out'))
        except Exception:
            continue
        if end <= start:
            continue
        selected_ranges.append((start, end, score, bool(theme_set & tags)))
        if len(selected_ranges) >= 12:
            break
    if not selected_ranges:
        return list(matched_paragraphs or [])

    # Map vector ranges to overlapping transcript segments.
    augmented = list(matched_paragraphs or [])
    seen_keys = {(s.get('start'), s.get('end')) for s in augmented}
    for vstart, vend, _score, _theme in selected_ranges:
        for seg in segments:
            seg_start = float(seg.get('start', 0) or 0)
            seg_end = float(seg.get('end', seg_start) or seg_start)
            if seg_end <= vstart or seg_start >= vend:
                continue
            key = (seg.get('start'), seg.get('end'))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            augmented.append(seg)
    augmented.sort(key=lambda p: float(p.get('start', 0) or 0))
    return augmented


def _build_relevant_excerpts_block(paragraphs, synthesis=False):
    """Render matched paragraphs as a labeled block suitable for injecting
    between the full transcript and the FINAL REMINDER. Empty when nothing
    matched, so the caller can unconditionally concatenate the result.
    """
    if not paragraphs:
        return ''
    body = _format_paragraphs_as_lines(paragraphs)
    if synthesis:
        header = (
            "POSSIBLY RELEVANT EXCERPTS (these may contain useful moments, "
            "but reason across the full transcript — the best answer may be "
            "elsewhere):"
        )
    else:
        header = (
            "RELEVANT EXCERPTS (auto-selected from the transcript above "
            "based on your question — prefer these timecodes for [CLIP:] "
            "markers, but check the full transcript if none fit):"
        )
    return f"\n\n{header}\n{body}"


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — chunked search for multi-hour interviews.
#
# The attention-bias bug that Layer 1 only partly fixed: Gemma 4 e2b (2B
# active params) can hold the CLIP marker contract on a full 90-min
# paragraph-grouped transcript but struggles to *rank* clips within it. Even
# with RELEVANT EXCERPTS pinned to the prompt tail the model was picking the
# opening line as the "best" moment on every long interview.
#
# Layer 2 trades one big prompt for N parallel small prompts. Each chunk
# covers ~7k tokens (~10-12 min) and returns 0-3 structured candidates with
# scores; we aggregate across chunks and emit the top-K as CLIP markers
# server-side. The model's job per call is narrow (rank within a 10-min
# window) which is well inside its reliable operating range.
# ─────────────────────────────────────────────────────────────────────────────


_CHARS_PER_TOKEN = 3.8  # empirical for English transcript prose, matches our diagnostic


def _chunk_paragraphs(paragraphs, tokens_per_chunk=_CHAT_CHUNK_TOKENS,
                     overlap_paragraphs=_CHAT_CHUNK_OVERLAP_PARAGRAPHS):
    """Split structured paragraphs into overlapping chunks bounded by token
    budget. Overlap preserves context for clips that span chunk boundaries —
    without it, a moment that starts 30s before a cut would lose its lead-in
    and fail to score. Always emits at least one paragraph per chunk even if
    that single paragraph exceeds the budget (avoids infinite loops on
    pathological input).
    """
    if not paragraphs:
        return []
    char_cap = int(tokens_per_chunk * _CHARS_PER_TOKEN)

    chunks = []
    i = 0
    n = len(paragraphs)
    while i < n:
        cur = []
        cur_chars = 0
        j = i
        while j < n:
            text_len = len((paragraphs[j].get('text') or ''))
            if cur and cur_chars + text_len > char_cap:
                break
            cur.append(paragraphs[j])
            cur_chars += text_len
            j += 1
        chunks.append(cur)
        if j >= n:
            break
        next_i = j - overlap_paragraphs
        if next_i <= i:
            next_i = i + 1  # forward-progress guard
        i = next_i
    return chunks


def _build_chunk_search_prompt(chunk, message, phrases, words, chunk_idx,
                               total_chunks, project_name, strict_keyword=False):
    """Build the JSON-extraction prompt for a single chunk.

    Each chunk prompt is self-contained — the model doesn't see other chunks
    and doesn't need to. Its only job is to pick up to 3 moments in THIS
    window that answer the user's question, score them, and return JSON.

    When ``strict_keyword`` is True, the caller has already verified this
    chunk literally contains the user's search term(s); we instruct the
    model to return ONLY moments that mention the term. Without this, small
    models hallucinate abstract matches (e.g. returning "sensory experience
    of the site" for a "moose hill" query).
    """
    chunk_block = _format_paragraphs_as_lines(chunk)

    keyword_hint = ''
    if phrases or words:
        parts = []
        if phrases:
            parts.append('phrases: ' + ', '.join(f'"{p}"' for p in phrases))
        if words:
            parts.append('words: ' + ', '.join(words))
        if strict_keyword:
            keyword_hint = (
                "\n\nThe user specifically asked about: "
                f"{'; '.join(parts)}. "
                "This excerpt contains those terms — find the paragraphs where "
                "they appear and return those as candidates. Pick moments where "
                "the speaker mentions or directly discusses those terms, not "
                "moments that are only tangentially related. Since the excerpt "
                "literally contains the terms, you should return at least one "
                "candidate."
            )
        else:
            keyword_hint = (
                "\n\nThe user's question suggests these search terms — "
                f"{'; '.join(parts)}. "
                "Use them to prioritize, NOT as a filter. Neighboring paragraphs "
                "that set up, lead into, or follow from those terms are equally "
                "valid candidates."
            )

    system_prompt = f"""You are scanning excerpt {chunk_idx + 1} of {total_chunks} from an interview titled "{project_name}".

Your ONLY task: identify up to 3 moments in THIS EXCERPT that best answer the user's question. Return the answer as JSON. Do not write prose outside the JSON. Do not use markdown or code fences.

JSON SCHEMA (copy exactly):
{{
  "candidates": [
    {{
      "title": "2-6 word title, sentence case (start with a capital letter), reads like a card headline not a tag list",
      "start": "HH:MM:SS",
      "end": "HH:MM:SS",
      "score": <integer 1-10>,
      "why": "one sentence, specific and editorial — see guidance below"
    }}
  ]
}}

RULES:
- start and end MUST be copied from the [HH:MM:SS-HH:MM:SS] timecode markers in the excerpt below. Do not invent timecodes.
- A clip can span multiple adjacent paragraphs — set start to the earlier paragraph's start, end to the later paragraph's end.
- Minimum clip length: 5 seconds. Typical: 10-60 seconds.
- score: 10 = directly and powerfully answers the question; 5 = relevant but not a standout; 1 = tangential.
- If nothing in this excerpt is relevant, return {{"candidates": []}} — empty is fine.
- Output JSON ONLY. No preamble, no markdown fences, no explanation outside the JSON object.

WRITING THE "why" FIELD — this is what editors see on each clip card, so be specific:
- Reference WHAT THE SPEAKER ACTUALLY SAYS, not the topic in the abstract.
- Name the person, moment, or concrete detail they describe.
- One sentence, editorial voice. No filler like "this clip shows" or "the interviewee reflects on".

BAD:  "This clip shows the interviewee reflecting on the aesthetic appeal of the landscape."
GOOD: "Chris describes the first time he saw the Crane Estate view from the hilltop and why it changed his approach to the project."

BAD:  "The speaker talks about resilience in their career."
GOOD: "Amanda recalls the phone call that almost made her quit, and the mentor sentence that pulled her back in."

BAD:  "Discussion of the creative process."
GOOD: "She walks through how a single rejected draft became the backbone of the final piece."{keyword_hint}

EXCERPT:
{chunk_block}"""

    user_prompt = f"User's question: {message}\n\nReturn JSON with up to 3 candidate moments from this excerpt."
    return system_prompt, user_prompt


def _call_ai_json(system_prompt, user_prompt, timeout=None, model_override=None):
    """Low-temperature Ollama call optimized for structured output.

    Uses a smaller num_ctx than chat because each chunk fits comfortably
    under 10k tokens. Lower temperature than the main chat (0.1 vs 0.4) to
    keep the JSON shape stable — creativity hurts here.

    ``format: 'json'`` instructs Ollama to constrain decoding to a valid
    JSON token tree. Without it, Gemma 4 e4b wraps the JSON in markdown
    fences or prose ~5–10% of the time on long interviews and
    ``_parse_chunk_response`` falls back to its low-confidence regex
    salvage path. The salvage path tags candidates with score=3, so a few
    bad chunks can poison the cross-chunk aggregation. Format-locked
    decoding eliminates the failure mode at zero parsing cost.

    ``model_override`` lets Layer 2 chunked-search workers run on the
    smaller ``gemma4:e2b`` for 2-3× faster decode while the synthesis
    rerank path keeps using the user's hardware-tier variant. Quality
    drop on per-chunk scoring is mitigated by the global rerank pass
    which still runs on the bigger model.

    Responses are cached by sha1((system_prompt, user_prompt, model, provider))
    so a re-asked query against an unchanged transcript hits the cache on
    every chunk instead of re-billing the LLM. The provider name is part of
    the key so switching from Ollama to Anthropic doesn't return a cached
    Ollama response for the same prompt. Cache busts naturally when chunk
    text changes (re-analysis produces different sha1s).
    """
    from ai_providers import get_active_provider
    # Size the HTTP timeout to the active model when the caller didn't pin one
    # (mirrors _call_ai). The old fixed 180s cut off long-interview chat
    # chunked-search calls on the big local models (gemma4:26b/31b). Explicit
    # callers (e.g. timeout=90) keep their value.
    if timeout is None:
        try:
            from model_config import recommended_analysis_timeout
            timeout = recommended_analysis_timeout()
        except Exception:
            timeout = 600
    system_prompt = inject_storytelling_foundation(system_prompt)
    provider = get_active_provider(model_resolver=_get_ollama_model)
    model_name = model_override or (
        _get_ollama_model() if provider.name == "ollama" else provider.name
    )
    cache_key = (
        hashlib.sha1(system_prompt.encode('utf-8', 'replace')).hexdigest(),
        hashlib.sha1(user_prompt.encode('utf-8', 'replace')).hexdigest(),
        model_name,
        provider.name,
    )
    cached = _chunk_cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        result = provider.generate(
            system_prompt, user_prompt, task_type="analysis",
            timeout=timeout, model_override=model_override,
        )
    except RuntimeError as e:
        # Permanent provider problems (no key, bad key) must abort the
        # whole analysis — silently returning '' on every chunk would
        # leave the user with empty results and no explanation. Transient
        # errors (rate limit, generic API hiccup) drop this single chunk
        # so the chunked-search loop can keep going.
        from ai_providers import ProviderError
        if isinstance(e, ProviderError) and e.code in ('missing_key', 'invalid_key'):
            raise
        return ''
    if result:
        _chunk_cache_put(cache_key, result)
    return result or ''


def _parse_chunk_response(response_text, chunk):
    """Parse a chunk's JSON response into candidate dicts. Permissive:

    1. Strip common wrappers (code fences, ```json blocks, leading prose).
    2. Try strict JSON parse.
    3. If JSON fails (model wrote prose around it, or wrote no JSON), fall
       back to regex-extracting HH:MM:SS ranges from whatever the model
       emitted. Those become low-confidence candidates (score=3) rather
       than dropping the chunk entirely — on long interviews losing even
       one chunk means missing whole topic regions.

    Candidates are clamped to the chunk's own timespan so a hallucinated
    timecode can't poison the aggregation.
    """
    import re
    import json

    if not response_text:
        return []

    # Chunk timespan bounds — clamp candidates to prevent cross-chunk invention.
    if chunk:
        chunk_start = chunk[0].get('start', 0)
        chunk_end = chunk[-1].get('end', chunk[-1].get('start', 0))
    else:
        chunk_start, chunk_end = 0, float('inf')

    def _valid(start_sec, end_sec):
        if end_sec - start_sec < 3:  # absurdly short, likely parse error
            return False
        if end_sec < chunk_start - 5 or start_sec > chunk_end + 5:
            return False
        return True

    candidates = []

    # Strip ```json ... ``` fences if present.
    stripped = response_text.strip()
    fence = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1)

    # Grab outermost {...} — handles "Here is the JSON: {...}" wrappers.
    obj_match = re.search(r'\{.*\}', stripped, re.DOTALL)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group(0))
            raw_cands = parsed.get('candidates') if isinstance(parsed, dict) else None
            if isinstance(raw_cands, list):
                for cand in raw_cands:
                    if not isinstance(cand, dict):
                        continue
                    start = _tc_to_seconds(cand.get('start', ''))
                    end = _tc_to_seconds(cand.get('end', ''))
                    if not _valid(start, end):
                        continue
                    try:
                        score = int(cand.get('score', 5))
                    except (ValueError, TypeError):
                        score = 5
                    candidates.append({
                        'title': (str(cand.get('title') or '').strip())[:80] or 'Moment',
                        'start_sec': max(start, chunk_start),
                        'end_sec': min(end, chunk_end),
                        'score': max(1, min(10, score)),
                        'why': (str(cand.get('why') or '').strip())[:240],
                        'source': 'json',
                    })
                if candidates:
                    return candidates
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: regex-extract timecode ranges. Low-confidence salvage path
    # for chunks where the model ignored the JSON contract but still named
    # real moments in prose.
    range_re = re.findall(
        r'(\d{1,2}:\d{2}:\d{2})\s*(?:-|–|—|to)\s*(\d{1,2}:\d{2}:\d{2})',
        response_text
    )
    for start_tc, end_tc in range_re[:3]:
        start = _tc_to_seconds(start_tc)
        end = _tc_to_seconds(end_tc)
        if not _valid(start, end):
            continue
        candidates.append({
            'title': 'Highlighted moment',
            'start_sec': max(start, chunk_start),
            'end_sec': min(end, chunk_end),
            'score': 3,
            'why': '',
            'source': 'regex_fallback',
        })

    return candidates


def _aggregate_chunk_candidates(candidates, top_k=_CHAT_TOP_K_CLIPS):
    """Dedupe overlapping candidates and keep the top-K by score.

    Two candidates overlap if their [start, end] ranges intersect by more
    than 50% of the shorter clip. In that case keep the higher-scored one.
    This handles the overlap zone between chunks where the same moment is
    scored twice.
    """
    if not candidates:
        return []

    # Highest score first so dedup keeps the stronger pick on ties.
    ordered = sorted(candidates, key=lambda c: (-c.get('score', 0), c.get('start_sec', 0)))

    kept = []
    for cand in ordered:
        cs, ce = cand['start_sec'], cand['end_sec']
        c_len = max(1.0, ce - cs)
        duplicate = False
        for existing in kept:
            es, ee = existing['start_sec'], existing['end_sec']
            overlap = max(0, min(ce, ee) - max(cs, es))
            shorter = min(c_len, max(1.0, ee - es))
            if overlap / shorter > 0.5:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
        if len(kept) >= top_k:
            break

    # Final presentation order: chronological, not score-ordered. Users
    # asked us to pick the best moments — showing them in timeline order
    # is more useful than a ranking list.
    kept.sort(key=lambda c: c.get('start_sec', 0))
    return kept


def _extend_candidates_to_duration(picked, pool, target_seconds):
    """Append ranked pool candidates until ``picked`` covers a duration ask.

    Layer 2's duration guarantee: the rerank pass picks quality, this pass
    guarantees quantity — keep appending the highest-scored unused pool
    candidates (skipping >50% time-overlaps with what's already picked)
    until the total reaches 0.85× the target or the pool runs out.
    Presentation stays chronological, matching _aggregate_chunk_candidates.
    """
    if not target_seconds or target_seconds <= 0:
        return picked
    floor = 0.85 * float(target_seconds)
    ceiling = _DURATION_TARGET_CEILING * float(target_seconds)
    out = list(picked or [])
    taken = [(c.get('start_sec', 0), c.get('end_sec', 0)) for c in out]
    total = sum(max(0.0, e - s) for s, e in taken)
    if total >= floor:
        return out
    ranked = sorted((pool or []), key=lambda c: -c.get('score', 0))
    for cand in ranked:
        if total >= floor:
            break
        s = cand.get('start_sec', 0)
        e = cand.get('end_sec', 0)
        dur = e - s
        if dur <= 0 or _spans_overlap(s, e, taken):
            continue
        if total + dur > ceiling:
            continue
        out.append(cand)
        taken.append((s, e))
        total += dur
    out.sort(key=lambda c: c.get('start_sec', 0))
    return out


def _format_clip_cards_from_candidates(candidates):
    """Emit the final chat reply text with `[CLIP:]` markers server-side.

    Shape matches what the main chat path produces, so the frontend's
    existing marker regex renders identical clip cards. The model's
    editorial "why" sentence rides along as `note="..."` so the card
    renderer can group it visually inside the card (not floating below).
    Candidates salvaged via regex fallback have empty "why" fields and
    just emit a bare marker without the note= attribute.
    """
    if not candidates:
        return ("I searched across the full interview but couldn't find moments that clearly "
                "answer that. Try rephrasing, or ask about a specific topic or theme.")

    def _card(cand, start_sec, end_sec):
        start_tc = _seconds_to_tc(start_sec)
        end_tc = _seconds_to_tc(end_sec)
        raw_title = (cand.get('title') or 'Moment').strip()
        # Strip matching wrapping quotes the model sometimes includes
        # (e.g. "Moment"), then neutralize any remaining internal "/[]
        # so they can't break our marker's own quoting or truncate the
        # naive [^\]]* marker regexes (canonical grammar).
        if len(raw_title) >= 2 and raw_title[0] == raw_title[-1] and raw_title[0] in ('"', "'"):
            raw_title = raw_title[1:-1].strip()
        title = _format_clip_title(_marker_attr_value(raw_title))
        why = _marker_attr_value(cand.get('why') or '')
        if why:
            return f'[CLIP: start={start_tc} end={end_tc} title="{title}" note="{why}"]'
        return f'[CLIP: start={start_tc} end={end_tc} title="{title}"]'

    parts = []
    sub_floor = []  # (duration, cand) — for the all-sub-floor rescue (F4)
    for cand in candidates:
        # No sliver cards from the Layer 2 pick either — same
        # _MIN_CLIP_MARKER_SECONDS floor the canonicalization pass
        # applies to model-emitted markers.
        try:
            start_sec = float(cand.get('start_sec', 0))
            end_sec = float(cand.get('end_sec', 0))
        except (TypeError, ValueError):
            continue
        dur = end_sec - start_sec
        if dur < _MIN_CLIP_MARKER_SECONDS:
            if dur > 0:
                sub_floor.append((dur, cand))
            continue
        parts.append(_card(cand, start_sec, end_sec))
    if not parts and sub_floor:
        # Every candidate sat under the quality floor. Never claim
        # nothing was found when something exists (F4): keep the single
        # LONGEST candidate, snapped up to the floor so the belt-and-
        # braces sliver guards in both frontends still render it.
        dur, best = max(sub_floor, key=lambda t: t[0])
        start_sec = float(best.get('start_sec', 0))
        parts.append(_card(best, start_sec,
                           start_sec + _MIN_CLIP_MARKER_SECONDS))
    if not parts:
        return ("I searched across the full interview but couldn't find moments that clearly "
                "answer that. Try rephrasing, or ask about a specific topic or theme.")
    return '\n'.join(parts)


def _find_keyword_chunk_indices(chunks, phrases, words):
    """Return indices of chunks whose combined text contains a literal match.

    Phrases take precedence: if any chunk contains a multi-word phrase
    ("moose hill"), we return ONLY phrase-matching chunks and ignore
    word-only chunks. This is critical — a "moose hill" query should not
    anchor onto a chunk that merely contains "hill" in an unrelated context
    (e.g. someone describing a Crane Estate hilltop view). Only when no
    phrase matches anywhere do we fall back to individual word matches.

    Layer 2 uses this to narrow the search: when the user asks about a
    specific term, searching every chunk invites abstract false positives
    ("sensory experience of site"). We trust the keyword index over the
    model's ranking for specific-topic queries.
    """
    if not (phrases or words):
        return []
    import re

    phrase_hits = []
    if phrases:
        lowered_phrases = [p.lower() for p in phrases]
        for i, chunk in enumerate(chunks):
            body = ' '.join((p.get('text') or '') for p in chunk).lower()
            if any(phrase in body for phrase in lowered_phrases):
                phrase_hits.append(i)
    if phrase_hits:
        return phrase_hits

    word_hits = []
    if words:
        word_patterns = [re.compile(rf'\b{re.escape(w.lower())}\b') for w in words]
        for i, chunk in enumerate(chunks):
            body = ' '.join((p.get('text') or '') for p in chunk).lower()
            if any(pat.search(body) for pat in word_patterns):
                word_hits.append(i)
    return word_hits


def _find_vector_anchored_chunk_indices(chunks, segment_vectors, score_filter='high',
                                        theme_phrases=None, limit=None):
    """Pick chunks whose paragraphs overlap selected segment_vectors.

    ``score_filter`` controls which vectors qualify:
      - ``'high'``: only ``narrative_score == 'high'`` vectors. Used to cap
        chunk fan-out on multi-hour synthesis queries — the right answer
        nearly always lives in chunks the editor already flagged as
        narratively strong.
      - ``'match'``: vectors whose ``theme_tags`` overlap ``theme_phrases``.
        Used to anchor abstract topical queries onto the chunks the editor
        already labeled with that theme.

    Returns chunk indices in chronological order. ``limit`` (when set)
    truncates the list, with chunks ranked by total qualifying-vector
    overlap so we keep the densest matches.
    """
    if not chunks or not segment_vectors:
        return []

    theme_set = {str(t).strip().lower() for t in (theme_phrases or [])}

    qualifying_ranges = []
    for v in segment_vectors:
        if not isinstance(v, dict):
            continue
        score = str(v.get('narrative_score', 'medium')).lower()
        tags = {str(t).strip().lower() for t in (v.get('theme_tags') or []) if isinstance(t, str)}
        if score_filter == 'high' and score != 'high':
            continue
        if score_filter == 'match' and not (theme_set & tags):
            continue
        try:
            v_start = _tc_to_seconds(v.get('timecode_in'))
            v_end = _tc_to_seconds(v.get('timecode_out'))
        except Exception:
            continue
        if v_end <= v_start:
            continue
        qualifying_ranges.append((v_start, v_end))
    if not qualifying_ranges:
        return []

    # For each chunk, sum overlap seconds with qualifying vectors.
    chunk_scores = []
    for idx, chunk in enumerate(chunks):
        if not chunk:
            continue
        chunk_start = chunk[0].get('start', 0)
        chunk_end = chunk[-1].get('end', chunk[-1].get('start', 0))
        if chunk_end <= chunk_start:
            continue
        total = 0.0
        for vs, ve in qualifying_ranges:
            total += max(0.0, min(chunk_end, ve) - max(chunk_start, vs))
        if total > 0:
            chunk_scores.append((idx, total))

    if not chunk_scores:
        return []

    if limit is not None and len(chunk_scores) > limit:
        chunk_scores.sort(key=lambda x: -x[1])
        chunk_scores = chunk_scores[:limit]

    chunk_scores.sort(key=lambda x: x[0])
    return [idx for idx, _ in chunk_scores]


def _rerank_candidates_globally(candidates, message, top_k=5):
    """Final low-temp rerank pass over Layer 2 candidates.

    Per-chunk scores are local — a chunk full of strong moments produces
    candidates with similar scores to a chunk where everything was
    mediocre. Aggregating by raw score therefore over-weights whichever
    chunks happened to score generously. This pass sends the model a
    short menu of titles + timecodes + ``why`` blurbs (no transcript) and
    asks for the best ``top_k`` for the user's question.

    Falls back to the input list (truncated to ``top_k``) when the rerank
    call fails or returns nothing parseable, so a transient AI error never
    drops Layer 2 to zero clips.
    """
    if not candidates or len(candidates) <= top_k:
        return list(candidates or [])[:top_k]

    menu_lines = []
    indexed = list(enumerate(candidates))
    for i, c in indexed:
        title = (c.get('title') or 'Moment').strip().replace('\n', ' ')
        why = (c.get('why') or '').strip().replace('\n', ' ')
        start = _seconds_to_tc(c.get('start_sec', 0))
        end = _seconds_to_tc(c.get('end_sec', 0))
        menu_lines.append(f"  [{i}] [{start}-{end}] {title} :: {why}")
    menu = '\n'.join(menu_lines)

    system_prompt = (
        "You are an editorial consultant choosing the BEST moments from a "
        "shortlist of pre-scored candidates. The candidates were scored "
        "independently in different transcript chunks, so their scores "
        "aren't directly comparable — your job is to choose globally. "
        "Respond in JSON only, no markdown, no prose."
    )
    user_prompt = (
        f"User's question: {message}\n\n"
        f"Candidates (numbered):\n{menu}\n\n"
        f"Pick the {top_k} best for the user's question. Diversify across "
        "the timeline — don't return five clips from the same chunk if "
        "stronger options exist elsewhere. Return JSON in this shape:\n"
        '{"picks": [<numeric index>, ...]} \n'
        "Only return indices that appear in the menu above."
    )
    try:
        response = _call_ai_json(system_prompt, user_prompt, timeout=90)
    except Exception:
        response = ''
    if not response:
        return _aggregate_chunk_candidates(candidates, top_k=top_k)

    import re
    import json as _json
    parsed = None
    fence = re.search(r'\{.*\}', response, re.DOTALL)
    if fence:
        try:
            parsed = _json.loads(fence.group(0))
        except (ValueError, _json.JSONDecodeError):
            parsed = None
    if not isinstance(parsed, dict):
        return _aggregate_chunk_candidates(candidates, top_k=top_k)

    raw_picks = parsed.get('picks') or parsed.get('best') or []
    if not isinstance(raw_picks, list):
        return _aggregate_chunk_candidates(candidates, top_k=top_k)

    selected = []
    seen = set()
    for entry in raw_picks:
        try:
            idx = int(entry)
        except (TypeError, ValueError):
            continue
        if idx in seen or idx < 0 or idx >= len(candidates):
            continue
        seen.add(idx)
        selected.append(candidates[idx])
        if len(selected) >= top_k:
            break

    if not selected:
        return _aggregate_chunk_candidates(candidates, top_k=top_k)

    selected.sort(key=lambda c: c.get('start_sec', 0))
    return selected


def _chunks_overlapping_paragraphs(chunks, target_paragraphs):
    """Indices of chunks whose timespan overlaps any of ``target_paragraphs``.

    Used to anchor Layer 2 search onto the chunks containing TF-IDF top hits,
    so a multi-hour synthesis query whose answer is in chunk 12 doesn't get
    starved by chunk-fan-out caps that picked chunks 1-6.
    """
    if not chunks or not target_paragraphs:
        return []
    target_ranges = []
    for p in target_paragraphs:
        try:
            t_start = float(p.get('start', 0) or 0)
            t_end = float(p.get('end', t_start) or t_start)
        except (TypeError, ValueError):
            continue
        if t_end > t_start:
            target_ranges.append((t_start, t_end))
    if not target_ranges:
        return []
    indices = []
    for idx, chunk in enumerate(chunks):
        if not chunk:
            continue
        chunk_start = chunk[0].get('start', 0)
        chunk_end = chunk[-1].get('end', chunk[-1].get('start', 0))
        for t_start, t_end in target_ranges:
            if t_end > chunk_start and t_start < chunk_end:
                indices.append(idx)
                break
    return indices


_SPEAKER_DIGEST_MAX_CHARS = 9000  # budget for all speaker voice samples combined
_SPEAKER_DIGEST_SAMPLE_PER_SPEAKER = 22  # max excerpts per speaker


def _build_speaker_digest(segments) -> str:
    """Build a per-speaker voice digest from transcript segments.

    Groups segments by speaker, samples representative excerpts evenly
    distributed across the timeline, and formats them as a compact block
    the LLM can use to discuss what each speaker actually said — tone,
    vocabulary, themes, contradictions, personality.

    Without this, the conversational synthesis path only has the analysis
    summary (which might say "multiple artists discuss…") and the model
    can't tell speakers apart or cite their actual words.

    Returns "" when there's only one speaker or no usable segments.
    """
    if not segments:
        return ''

    # Group segments by speaker
    by_speaker: dict[str, list] = {}
    for seg in segments:
        spk = (seg.get('speaker') or '').strip()
        text = (seg.get('text') or '').strip()
        if not spk or not text or len(text) < 10:
            continue
        by_speaker.setdefault(spk, []).append(seg)

    if len(by_speaker) < 2:
        # Single speaker — the summary already covers them; no need for
        # a voice digest that would just duplicate content.
        return ''

    # Budget per speaker: divide evenly, then cap at sample count
    budget_per = _SPEAKER_DIGEST_MAX_CHARS // len(by_speaker)
    lines = ['SPEAKER VOICE SAMPLES (representative excerpts from each speaker):']

    for spk, segs in by_speaker.items():
        lines.append(f'\n  {spk}:')
        # Sample evenly across timeline — don't cluster at the start
        n = min(len(segs), _SPEAKER_DIGEST_SAMPLE_PER_SPEAKER)
        if n <= 0:
            continue
        step = max(1, len(segs) // n)
        sampled = segs[::step][:n]

        char_used = 0
        for seg in sampled:
            start = seg.get('start', 0)
            tc = _seconds_to_hms_tc(start)
            text = (seg.get('text') or '').strip()
            # Truncate very long segments to keep budget — but allow enough
            # words for the model to recognize voice / theme / vocabulary.
            if len(text) > 320:
                text = text[:320].rstrip() + '…'
            entry = f'    [{tc}] {text}'
            if char_used + len(entry) > budget_per:
                break
            lines.append(entry)
            char_used += len(entry)

    result = '\n'.join(lines)
    if len(result) > _SPEAKER_DIGEST_MAX_CHARS:
        result = result[:_SPEAKER_DIGEST_MAX_CHARS].rsplit('\n', 1)[0]
    return result


def _build_synthesis_context_block(project_name, segments, analysis, labeled_sections, speaker_names=None):
    """Compact context block used by the conversational synthesis path on
    long interviews. The full transcript doesn't fit in 32K context for
    100+ minute interviews, but a curated summary + analysis index + the
    editor's own selections gives the model enough to discuss themes,
    story, and craft without extracting every clip from scratch.

    Layout: PROJECT/DURATION/SPEAKERS header, then optional SUMMARY,
    THEMES, SPEAKER VOICE SAMPLES, the existing PRE-ANALYZED MOMENTS
    block, and the <editor_selections> block when clips have been pulled.
    """
    duration_sec = segments[-1].get('end', 0) if segments else 0
    parts = [
        "Here is the loaded project. The transcript IS loaded — you have "
        "a project summary, per-speaker voice samples (real quotes from "
        "each speaker spread across the timeline), pre-analyzed story "
        "moments with timecodes, and any clips the editor has selected. "
        "When asked about a specific speaker, draw from their voice "
        "samples below — do NOT say their interview isn't loaded. When "
        "asked about themes, story, or structure, synthesize from the "
        "summary, themes, and pre-analyzed moments. Discuss what's here "
        "with confidence and specificity.",
        '',
        f'PROJECT: {project_name}',
        f'DURATION: {_format_duration_seconds(duration_sec)}',
    ]
    speakers = _extract_speaker_names(segments)
    if speakers:
        parts.append(f"SPEAKERS: {', '.join(speakers)}")
    parts.append('')

    if isinstance(analysis, dict):
        summary = (analysis.get('summary') or '').strip()
        if summary:
            parts.append("PROJECT SUMMARY:")
            parts.append(summary)
            parts.append('')
        suggested = (analysis.get('suggested_title') or '').strip()
        if suggested:
            parts.append(f"SUGGESTED TITLE: {suggested}")
            parts.append('')
        themes = analysis.get('themes') or []
        if themes:
            parts.append("THEMES IDENTIFIED IN ANALYSIS:")
            for t in themes:
                parts.append(f"  - {t}")
            parts.append('')

    # Per-speaker voice samples — so the model can discuss individual
    # speakers by name with real quotes, not just the summary's mention
    # of "multiple artists" or "several subjects."
    digest = _build_speaker_digest(segments)
    if digest:
        parts.append(digest)
        parts.append('')

    analysis_block = _build_chat_analysis_index(analysis or {})
    if analysis_block:
        parts.append(analysis_block.strip())
        parts.append('')

    if labeled_sections:
        sel_block = _build_editor_selections_block(labeled_sections, segments, speaker_names)
        if sel_block:
            parts.append(sel_block)

    return '\n'.join(parts)


def _chat_layer2_conversational_synthesis(message, history, project_name, segments,
                                          analysis, profile_id, labeled_sections,
                                          speaker_names=None,
                                          language_directive_text='',
                                          skip_title_anchor=False):
    """Conversational synthesis on long interviews — the divert from
    chunked clip search when the editor's question is discussion-style.

    Builds a compact context (summary + analysis index + selected clips,
    no full transcript), runs it through the standard conversational LLM
    via _build_chat_messages so the orientation paragraph and framing
    apply, and returns the prose response. Skips the clip-salvage
    post-processor so a clean conversational answer doesn't get clips
    bolted on. When the question names something concrete, query-matched
    transcript excerpts ride along so the answer can quote real lines
    instead of paraphrasing the summary.
    """
    context = _build_synthesis_context_block(project_name, segments, analysis, labeled_sections, speaker_names)
    # Query-matched excerpts: the synthesis context is summary-driven by
    # design (a 100-min transcript doesn't fit), which used to mean a
    # question like "what does she say about the dam?" could only be
    # answered from vague summary memory. Pull the real lines the
    # question points at, capped so they can't blow the compact budget.
    try:
        _phrases, _words = _extract_query_keywords(message)
        if _phrases or _words:
            _matched = _find_relevant_paragraphs(
                segments, _phrases, _words, context=1,
            )
            if _matched:
                _excerpts = _build_relevant_excerpts_block(
                    _matched[:12], synthesis=True,
                )
                if len(_excerpts) > 5000:
                    _excerpts = _excerpts[:5000].rsplit('\n', 1)[0]
                if _excerpts.strip():
                    context = context + '\n' + _excerpts.strip() + '\n'
    except Exception as e:
        print(f"[chat] synthesis excerpt retrieval failed: {e}")
    # Content-lookup grounding on long projects: this path skips the
    # final reminder (no marker pressure on discussion answers) which
    # also skipped the CONTENT QUESTION contract — so "what does she say
    # about X" on a 90-min interview got a summary-memory gloss. Append
    # the grounding instruction to the message itself; the query-matched
    # excerpts above give the model real lines to quote.
    if _is_content_lookup_query(message):
        message = (
            f'{message}\n\nCONTENT QUESTION: answer with what was actually '
            f'said — name the speaker and quote their exact words briefly, '
            f'copied verbatim from the excerpts above. If the excerpts '
            f"don't cover it, say so plainly. No thematic summary without "
            f'the actual words.'
        )
    system_message, messages = _build_chat_messages(
        message, history, project_name, segments,
        formatted=context,            # context block in the transcript slot
        analysis_block='',            # already inside the context block above
        relevant_excerpts_block='',
        profile_id=profile_id,
        labeled_sections=labeled_sections,
        speaker_names=speaker_names,
        # This path is only reached on discussion-style questions and skips
        # clip-salvage on purpose — don't nudge the model toward markers.
        include_final_reminder=False,
        language_directive_text=language_directive_text,
    )
    num_ctx = _sticky_chat_num_ctx(project_name, system_message, messages)
    response = _call_ai_chat(system_message, messages, num_ctx=num_ctx)
    response = _strip_trailing_repetition(response)
    cleaned = _clean_chat_response(response)
    cleaned = _validate_clip_markers_in_text(
        cleaned, segments, skip_title_anchor=skip_title_anchor,
    )
    # Deliberately skip _salvage_clips_if_missing — this path is only
    # reached on conversational queries.
    return cleaned


def _chat_layer2_conversational_synthesis_stream(message, history, project_name, segments,
                                                  analysis, profile_id, labeled_sections,
                                                  speaker_names=None,
                                                  language_directive_text='',
                                                  skip_title_anchor=False):
    """Streaming variant of :func:`_chat_layer2_conversational_synthesis`.

    Yields ('progress', label) for the prep step, then ('done', reply) with
    the synthesized prose. Keeps the SSE shape identical to the chunked
    search variant so the frontend doesn't need a separate handler.
    """
    yield ('progress', 'Reading the project context…')
    try:
        reply = _chat_layer2_conversational_synthesis(
            message, history, project_name, segments,
            analysis, profile_id, labeled_sections,
            speaker_names=speaker_names,
            language_directive_text=language_directive_text,
            skip_title_anchor=skip_title_anchor,
        )
    except Exception as e:
        print(f"[chat-stream] conversational synthesis failed: {e}")
        reply = ''
    yield ('done', reply)


def _chat_layer2_chunked_search(paragraphs, message, history, project_name,
                                phrases, words, profile_id, analysis,
                                segment_vectors=None, theme_phrases=None,
                                tfidf_hits=None, speaker_names=None,
                                language_directive_text=''):
    """Orchestrate the Layer 2 path: chunk → concurrent per-chunk search →
    aggregate → render clip cards. No LLM sees the full transcript; the
    model's ranking job is scoped to a single ~10-minute window at a time.

    When the user's query has literal keyword matches in the transcript, we
    restrict the search to keyword-anchored chunks and switch the prompt to
    strict mode. This prevents the small model from scoring abstract matches
    in other chunks above the actual keyword hits.

    When ``segment_vectors`` are provided AND no literal keyword anchors hit,
    we narrow the search to chunks that overlap high-narrative-score / theme-
    matching vectors. On a 3-hour interview that turns 18 chunks × 4 calls
    into ~6 chunks × 4 calls — the right answer is overwhelmingly likely to
    sit in a chunk the editor already flagged as a highlight.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    chunks = _chunk_paragraphs(paragraphs)
    if not chunks:
        return _format_clip_cards_from_candidates([])

    anchored = _find_keyword_chunk_indices(chunks, phrases, words)
    # Theme-tag anchoring: if vectors flagged theme_tags that match the user's
    # query, treat the chunks overlapping those vectors as keyword-anchored
    # too. This is the "abstract query but the editor already labeled the
    # right thread" case ("show me resilience moments" with no literal hit).
    if not anchored and theme_phrases and segment_vectors:
        anchored = _find_vector_anchored_chunk_indices(
            chunks, segment_vectors,
            score_filter='match',
            theme_phrases=theme_phrases,
        )
    # TF-IDF anchoring: if the paragraph index found high-similarity matches
    # for the query, anchor onto chunks containing those paragraphs. This
    # handles abstract synthesis queries with no literal keyword hit AND no
    # theme-tag overlap — the kind of question where neither indexer alone
    # would find anything but cosine similarity would.
    if not anchored and tfidf_hits:
        anchored = sorted(set(_chunks_overlapping_paragraphs(chunks, tfidf_hits)))

    if anchored:
        search_targets = [(i, chunks[i]) for i in anchored]
        strict_keyword = True
    else:
        search_targets = list(enumerate(chunks))
        strict_keyword = False
        # Cap chunk fan-out on multi-hour synthesis queries: the model's job
        # is to pick the BEST 5 across the whole interview, and abstract
        # queries are overwhelmingly answered in high-narrative-score
        # regions. Without this cap a 3-hour project does ~5 sequential
        # rounds of 4 parallel calls; with it we run ~2 rounds.
        if len(search_targets) > 6 and segment_vectors:
            top_indices = _find_vector_anchored_chunk_indices(
                chunks, segment_vectors, score_filter='high', limit=6,
            )
            if top_indices:
                search_targets = [(i, chunks[i]) for i in top_indices]

    all_candidates = []
    fast_model = _get_fast_chunk_model()

    def _run_chunk(idx_chunk):
        idx, chunk = idx_chunk
        system_prompt, user_prompt = _build_chunk_search_prompt(
            chunk, message, phrases, words, idx, len(chunks), project_name,
            strict_keyword=strict_keyword,
        )
        # Output-language directive ('' for English → byte-identical):
        # clip titles/why blurbs come from these per-chunk JSON calls.
        if language_directive_text:
            system_prompt = system_prompt + language_directive_text
        # Per-chunk scoring runs on the smaller variant when it's
        # available (~2-3× faster decode on Apple Silicon). Synthesis
        # rerank below stays on the user's hardware-tier variant so
        # the global pick remains high-quality.
        response = _call_ai_json(system_prompt, user_prompt, model_override=fast_model)
        return _parse_chunk_response(response, chunk)

    with ThreadPoolExecutor(max_workers=_CHAT_CHUNK_CONCURRENCY) as pool:
        futures = [pool.submit(_run_chunk, t) for t in search_targets]
        for fut in as_completed(futures):
            try:
                all_candidates.extend(fut.result() or [])
            except Exception as e:
                print(f"Layer 2 chunk error: {e}")

    # Honor an explicit user-stated clip count ("find me 1", "the best one",
    # "give me 5 more") — it wins even when a duration ALSO parses ("give
    # me 5 clips, 2 minutes of selects": the count is the promise the user
    # made card-by-card, the duration degrades to a soft bound — same H5
    # contract as Layer 1). Only a count-less duration ask derives the
    # clip count from the parsed time budget, and only then does the
    # ranked pool below top the final pick up to that budget. The
    # pre-rerank candidate pool is kept generous so the global rerank
    # still has range to pick from, even when the final output is just
    # one clip.
    target_seconds = parse_target_duration_seconds(message)
    explicit_count = _layer2_explicit_count(message)
    if explicit_count is not None:
        final_top_k = explicit_count
    elif target_seconds is not None:
        final_top_k = _duration_clip_count_hint(target_seconds)
    else:
        final_top_k = _CHAT_TOP_K_CLIPS
    pool_top_k = max(final_top_k * 3, _CHAT_TOP_K_CLIPS * 3)

    # "More" follow-ups must surface NEW footage — drop candidates that
    # overlap clips earlier turns already showed before any ranking runs.
    all_candidates = _exclude_shown_candidates(all_candidates, message, history)

    top = _aggregate_chunk_candidates(all_candidates, top_k=pool_top_k)
    duration_pool = list(top)
    # Cross-chunk synthesis pass: per-chunk scores aren't comparable across
    # chunks (each model call sees only its own window), so a final low-temp
    # rerank decides the global best. Falls back to the local-score top-K
    # if the synthesis call fails — better to ship the original aggregator's
    # answer than to drop everything.
    top = _rerank_candidates_globally(top, message, top_k=final_top_k)
    if target_seconds is not None and explicit_count is None:
        top = _extend_candidates_to_duration(top, duration_pool, target_seconds)
    return _format_clip_cards_from_candidates(top)


def _chat_layer2_chunked_search_stream(paragraphs, message, history, project_name,
                                       phrases, words, profile_id, analysis,
                                       segment_vectors=None, theme_phrases=None,
                                       tfidf_hits=None, speaker_names=None,
                                       language_directive_text=''):
    """Streaming variant of :func:`_chat_layer2_chunked_search`.

    Yields ``('progress', label)`` events as each chunk completes so the
    UI can render forward motion instead of staring at a static spinner
    for 20-60 seconds. The final ``('done', reply)`` event carries the
    same formatted reply the non-streaming variant returns. Logic
    mirrors the non-streaming path 1:1 so future tweaks in either need
    to be mirrored — keeping them in two functions instead of folding
    a flag in keeps each call site flat and easy to read.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    chunks = _chunk_paragraphs(paragraphs)
    if not chunks:
        yield ('done', _format_clip_cards_from_candidates([]))
        return

    anchored = _find_keyword_chunk_indices(chunks, phrases, words)
    if not anchored and theme_phrases and segment_vectors:
        anchored = _find_vector_anchored_chunk_indices(
            chunks, segment_vectors,
            score_filter='match',
            theme_phrases=theme_phrases,
        )
    if not anchored and tfidf_hits:
        anchored = sorted(set(_chunks_overlapping_paragraphs(chunks, tfidf_hits)))

    if anchored:
        search_targets = [(i, chunks[i]) for i in anchored]
        strict_keyword = True
    else:
        search_targets = list(enumerate(chunks))
        strict_keyword = False
        if len(search_targets) > 6 and segment_vectors:
            top_indices = _find_vector_anchored_chunk_indices(
                chunks, segment_vectors, score_filter='high', limit=6,
            )
            if top_indices:
                search_targets = [(i, chunks[i]) for i in top_indices]

    total = len(search_targets)
    yield ('progress', f'Searching {total} chunk{"s" if total != 1 else ""}…')

    all_candidates = []
    fast_model = _get_fast_chunk_model()

    def _run_chunk(idx_chunk):
        idx, chunk = idx_chunk
        system_prompt, user_prompt = _build_chunk_search_prompt(
            chunk, message, phrases, words, idx, len(chunks), project_name,
            strict_keyword=strict_keyword,
        )
        # Mirror of the non-streaming variant — see _chat_layer2_chunked_search.
        if language_directive_text:
            system_prompt = system_prompt + language_directive_text
        response = _call_ai_json(system_prompt, user_prompt, model_override=fast_model)
        return _parse_chunk_response(response, chunk)

    completed = 0
    with ThreadPoolExecutor(max_workers=_CHAT_CHUNK_CONCURRENCY) as pool:
        futures = [pool.submit(_run_chunk, t) for t in search_targets]
        for fut in as_completed(futures):
            completed += 1
            try:
                all_candidates.extend(fut.result() or [])
            except Exception as e:
                print(f"Layer 2 chunk error: {e}")
            yield ('progress', f'{completed}/{total} chunks searched')

    # Mirror of the non-streaming variant: an explicit count (either
    # parser) overrides the default top-K AND a jointly-parsed duration
    # (H5 — the duration degrades to a soft bound); only a count-less
    # duration ask derives the clip count from the parsed time budget and
    # tops the pick up from the ranked pool afterward. "More" follow-ups
    # exclude already-shown clips.
    target_seconds = parse_target_duration_seconds(message)
    explicit_count = _layer2_explicit_count(message)
    if explicit_count is not None:
        final_top_k = explicit_count
    elif target_seconds is not None:
        final_top_k = _duration_clip_count_hint(target_seconds)
    else:
        final_top_k = _CHAT_TOP_K_CLIPS
    pool_top_k = max(final_top_k * 3, _CHAT_TOP_K_CLIPS * 3)

    all_candidates = _exclude_shown_candidates(all_candidates, message, history)

    yield ('progress', 'Picking the best moments…')
    top = _aggregate_chunk_candidates(all_candidates, top_k=pool_top_k)
    duration_pool = list(top)
    top = _rerank_candidates_globally(top, message, top_k=final_top_k)
    if target_seconds is not None and explicit_count is None:
        top = _extend_candidates_to_duration(top, duration_pool, target_seconds)
    yield ('done', _format_clip_cards_from_candidates(top))


def _seconds_to_tc(sec) -> str:
    try:
        sec = int(sec)
    except (TypeError, ValueError):
        return "00:00:00"
    return f"{sec//3600:02d}:{(sec%3600)//60:02d}:{sec%60:02d}"


# Tail chunks shorter than this fold into the previous chunk instead of
# being yielded as a standalone slice. Without this, a 15-minute interview
# can split into "900s + 13s tail", where the tail's analysis is the only
# one some providers keep — the long slice gets dropped on a malformed JSON
# response. Folding the tail in protects the analysis regardless of model.
_MIN_TAIL_CHUNK_SECONDS = 90


def _iter_transcript_chunks(segments, target_minutes=CHUNK_MINUTES):
    """Yield ~N-minute chunks of the transcript in order.

    Each yielded dict has ``start_seconds``, ``end_seconds``, and the slice of
    segments within that window. Segments preserve their original absolute
    timecodes so story beats and social clips come back timeline-absolute
    regardless of which chunk they originated from.

    A trailing chunk shorter than :data:`_MIN_TAIL_CHUNK_SECONDS` is folded
    into the previous chunk before yielding. See the constant's docstring
    for the failure mode this guards against.
    """
    if not segments:
        return
    target_seconds = target_minutes * 60

    # First pass: collect all chunks at the natural N-minute boundaries.
    chunks = []
    cur = []
    cur_start = None
    for seg in segments:
        if not cur:
            cur_start = seg.get('start', 0)
        cur.append(seg)
        elapsed = seg.get('end', seg.get('start', 0)) - cur_start
        if elapsed >= target_seconds:
            chunks.append({
                'start_seconds': cur_start,
                'end_seconds': seg.get('end', cur_start),
                'segments': cur,
            })
            cur = []
            cur_start = None
    if cur:
        chunks.append({
            'start_seconds': cur_start,
            'end_seconds': cur[-1].get('end', cur_start),
            'segments': cur,
        })

    # Second pass: fold a too-short trailing chunk into its predecessor.
    if len(chunks) >= 2:
        last = chunks[-1]
        last_dur = (last.get('end_seconds') or 0) - (last.get('start_seconds') or 0)
        if last_dur < _MIN_TAIL_CHUNK_SECONDS:
            prev = chunks[-2]
            prev['segments'] = prev['segments'] + last['segments']
            prev['end_seconds'] = last['end_seconds']
            chunks.pop()

    for chunk in chunks:
        yield chunk


def _format_segments_for_ai(segments, speaker_names=None) -> str:
    """Format an arbitrary slice of segments for the prompt.

    When ``speaker_names`` is supplied (populated by the Pro diarization
    rename UI) raw labels like ``SPEAKER_00`` are resolved to their custom
    display names so the model sees ``Sarah Chen: …`` instead of the raw
    pyannote token. Single-speaker projects without the map render exactly
    as before.
    """
    lines = []
    for seg in segments:
        start_tc = seg.get('start_formatted', _seconds_to_tc(seg.get('start', 0)))[:8]
        end_s = seg.get('end', seg.get('start', 0))
        end_tc = _seconds_to_tc(end_s)
        raw_speaker = seg.get('speaker', 'Speaker')
        speaker = _display_speaker(raw_speaker, speaker_names)
        text = seg.get('text', '')
        if text.strip():
            lines.append(f"[{start_tc}-{end_tc}] {speaker}: {text}")
    return '\n'.join(lines)


# User-facing copy when a chunk's response can't be repaired into something
# usable. Surfaced via accum['analysis_warnings'] and rendered on the AI
# Analysis tab so the editor knows part of the interview was dropped instead
# of seeing a 3-second story for a 15-minute interview with no explanation.
_CHUNK_DROP_WARNING = (
    'Analysis incomplete: the AI model returned an unexpected response '
    'for part of this interview. Some story beats or clips may be '
    'missing. Try re-running analysis.'
)


def _attempt_chunk_response_repair(value):
    """Salvage a non-conforming AI response into a dict the merge functions
    can pull from.

    :func:`_parse_json_response` already handles markdown fences and trailing
    truncation, but local Gemma in particular can return:
      * the parser's fallback dict (``{'error': ..., 'raw': ...}``) when the
        full response wasn't valid JSON,
      * a bare list ``[{...}, ...]`` when it dropped the wrapping object,
      * prose preambles like "Here's the analysis: { ... }",
      * markdown-fenced JSON nested inside other prose.

    This helper walks those cases. Returns a dict on success, ``None``
    otherwise. Keeps Gemma's outputs out of the warnings log most of the
    time — they're usually repairable.
    """
    import re as _re
    if value is None:
        return None

    # Already a dict — but check whether it's the parser's "I gave up" shape
    # before declaring success. That fallback dict carries the raw text on
    # ``raw``; we can sometimes recover real data from it.
    if isinstance(value, dict):
        is_parser_fallback = (
            'error' in value and 'raw' in value
            and not value.get('story_beats')
            and not value.get('social_clips')
            and not value.get('summary')
            and not value.get('themes')
        )
        if not is_parser_fallback:
            return value
        text = str(value.get('raw') or '')
    elif isinstance(value, list):
        # Bare list — most often it's the social_clips array without the
        # outer wrapper. We hand back a synthetic dict; the merge functions
        # use _first_present_list which tolerates extra keys.
        return {'social_clips': value, 'story_beats': value}
    elif isinstance(value, str):
        text = value
    else:
        return None

    if not text:
        return None

    # Strip markdown fences and any "json" hint.
    stripped = text.strip()
    if stripped.startswith('```'):
        nl = stripped.find('\n')
        if nl != -1:
            stripped = stripped[nl + 1:]
        if stripped.endswith('```'):
            stripped = stripped[:-3]
        stripped = stripped.strip()
        # Some models prefix with the language tag right after the fence.
        if stripped.lower().startswith('json'):
            stripped = stripped[4:].lstrip()

    # Try the cleaned-up text directly.
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {'social_clips': parsed, 'story_beats': parsed}
    except (json.JSONDecodeError, ValueError):
        pass

    # Walk inward from the first '{' to find the largest object that parses.
    # Then the same for '[' as a fallback (bare-list path).
    for opener, closer, key_target in (('{', '}', None), ('[', ']', 'social_clips')):
        start = stripped.find(opener)
        if start == -1:
            continue
        # Try truncating from end; bisect-like scan to find the longest
        # parseable prefix without going O(n²) on large strings.
        end = stripped.rfind(closer)
        while end > start:
            cand = stripped[start:end + 1]
            try:
                parsed = json.loads(cand)
            except (json.JSONDecodeError, ValueError):
                end = stripped.rfind(closer, start, end)
                continue
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list) and key_target:
                return {key_target: parsed, 'story_beats': parsed}
            break

    # Last-ditch: hand the prose to _repair_truncated_json, which closes
    # unmatched braces/brackets and strips trailing commas.
    obj_match = _re.search(r'\{[\s\S]*', stripped)
    if obj_match:
        try:
            repaired = _repair_truncated_json(obj_match.group(0))
            if repaired:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    return parsed
        except Exception:
            # Same never-crash contract as _parse_json_response: a repair
            # bug degrades to the None return, not an exception.
            pass

    return None


def _merge_story_chunk(accum: dict, story_data, is_first_chunk: bool):
    """Merge one chunk's story response into the accumulator.

    Chunk-level summaries/titles are accumulated into lists; the final
    overall summary is synthesized from all of them in
    :func:`_synthesize_overall_summary`. Keeping only the first chunk's
    summary (as an earlier version did) made the sidebar describe only the
    opening of the first interview when multiple interviews were strung
    into one timeline.

    Non-dict / unparseable responses now go through
    :func:`_attempt_chunk_response_repair` first; only when repair fails
    do we append a user-visible warning to ``accum['analysis_warnings']``
    and skip the chunk. The previous behavior was a silent early-return,
    which produced the "3-second story from a 15-minute interview"
    failure mode on Gemma 4b.
    """
    repaired = _attempt_chunk_response_repair(story_data) if not (
        isinstance(story_data, dict)
        and (story_data.get('story_beats') or story_data.get('summary')
             or story_data.get('themes') or story_data.get('strongest_soundbites'))
    ) else story_data
    if not isinstance(repaired, dict):
        if _CHUNK_DROP_WARNING not in accum['analysis_warnings']:
            accum['analysis_warnings'].append(_CHUNK_DROP_WARNING)
        return
    story_data = repaired
    summary = _first_present(story_data, 'summary', 'overview', 'synopsis')
    if summary:
        accum['_chunk_summaries'].append(summary)
    title = _first_present(story_data, 'suggested_title', 'title', 'working_title')
    if title:
        accum['_chunk_titles'].append(title)
    accum['story_beats'].extend(
        _first_present_list(story_data, 'story_beats', 'beats', 'narrative_beats', 'story')
    )
    for t in _first_present_list(story_data, 'themes', 'topics', 'theme_list'):
        if isinstance(t, str) and t not in accum['themes']:
            accum['themes'].append(t)
    accum['strongest_soundbites'].extend(
        _first_present_list(story_data, 'strongest_soundbites', 'soundbites', 'quotes', 'best_quotes')
    )
    accum['broll_suggestions'].extend(
        _first_present_list(story_data, 'broll_suggestions', 'broll', 'bRoll', 'b_roll')
    )


def _merge_social_chunk(accum: dict, social_data):
    """Merge one chunk's social response into the accumulator.

    Same repair-then-warn behavior as :func:`_merge_story_chunk`. Bare lists
    are treated as the social_clips array (Gemma sometimes omits the outer
    wrapper).
    """
    if isinstance(social_data, list):
        accum['social_clips'].extend(social_data)
        return
    repaired = _attempt_chunk_response_repair(social_data) if not (
        isinstance(social_data, dict) and social_data.get('social_clips')
    ) else social_data
    if isinstance(repaired, dict):
        accum['social_clips'].extend(
            _first_present_list(repaired, 'social_clips', 'clips', 'social', 'reels')
        )
        return
    if _CHUNK_DROP_WARNING not in accum['analysis_warnings']:
        accum['analysis_warnings'].append(_CHUNK_DROP_WARNING)


def _synthesize_overall_summary(summaries, titles, project_name, warnings=None,
                                language_directive_text=''):
    """Combine per-chunk summaries + titles into one overall summary/title.

    When multiple interviews are strung into a single timeline the chunked
    path produces one summary per ~15-minute slice. Returning just the
    first chunk's summary makes the sidebar describe only the opening of
    the first interview, which was the user-visible bug. This pass asks
    the model to fold all slice summaries into one coherent overview that
    spans the whole transcript.

    Falls back to joining chunk summaries with blank lines if the model
    call fails — still beats dropping everything after chunk 1.

    ``warnings``: optional list to append a one-line note to whenever we
    fall back. The chunked-analysis caller threads its ``analysis_warnings``
    accumulator through here so a silent fallback shows up on the AI
    Analysis tab instead of just a stderr print.
    """
    summaries = [s for s in summaries if s]
    titles = [t for t in titles if t]
    if not summaries:
        return {'summary': '', 'suggested_title': titles[0] if titles else ''}
    if len(summaries) == 1:
        return {
            'summary': summaries[0],
            'suggested_title': titles[0] if titles else '',
        }

    joined_summaries = '\n'.join(f"- {s}" for s in summaries)
    joined_titles = '\n'.join(f"- {t}" for t in titles) if titles else '(none)'

    system_prompt = (
        "You are a documentary story editor combining per-section summaries of a "
        "long transcript into a single overview. The transcript may contain multiple "
        "distinct interviews strung into one timeline — your overview must cover ALL "
        "sections, not just the first. Respond in valid JSON only. No markdown, no fences."
    )
    if language_directive_text:
        system_prompt = system_prompt + language_directive_text
    prompt = f"""PROJECT: {project_name}

Per-section summaries (in timeline order):
{joined_summaries}

Per-section working titles:
{joined_titles}

Return JSON in this exact shape:
{{
  "summary": "3-4 sentence overall summary covering EVERY section above",
  "suggested_title": "one compelling working title for the whole project"
}}
Return ONLY valid JSON."""

    try:
        parsed = _parse_json_response(_call_ai(prompt, system_prompt))
        if isinstance(parsed, dict):
            summary = _first_present(parsed, 'summary', 'overview', 'synopsis')
            title = _first_present(parsed, 'suggested_title', 'title', 'working_title')
            if summary:
                return {
                    'summary': summary,
                    'suggested_title': title or (titles[0] if titles else ''),
                }
        # Synthesis call returned but didn't yield a usable summary.
        if isinstance(warnings, list):
            warnings.append(
                'Overall-summary synthesis returned no summary; '
                'falling back to concatenated per-chunk summaries.'
            )
    except Exception as e:
        print(f"[analyze] overall summary synthesis failed: {e}")
        if isinstance(warnings, list):
            warnings.append(
                f'Overall-summary synthesis failed ({type(e).__name__}); '
                'falling back to concatenated per-chunk summaries.'
            )

    return {
        'summary': '\n\n'.join(summaries),
        'suggested_title': titles[0] if titles else '',
    }


# Hard upper bound on items returned per analysis category. The user only
# wants the BEST 7 of each kind — more candidates dilute the cards in the UI
# and give the user busywork triaging duplicates.
ANALYSIS_PER_CATEGORY_CAP = 7


def analyze_transcript(transcript, project_name="Interview", analysis_type="all",
                       segment_vectors=None, progress_callback=None,
                       output_language=None):
    """
    Analyze a transcript for story structure and social media clips.

    Short interviews (<15 min) run through a single AI call. Longer interviews
    are chunked into ~15-minute slices and analyzed per-chunk; results are
    concatenated so coverage spans the whole transcript. Without chunking,
    small local models tend to analyze only the first 5-10 minutes of a long
    interview and return nothing for the rest.

    Args:
        transcript: dict with 'segments' list from transcribe.py
        project_name: str
        analysis_type: 'story', 'social', or 'all'
        segment_vectors: optional list of pre-classified segment vectors (from
            a prior /analyze run). When supplied, used to rank candidates by
            ``narrative_score`` during the post-merge cap pass — so the BEST 7
            of each category come back, not the first 7. ``None`` is fine on
            first-ever analysis; ranking falls back to length/recency.
        progress_callback: optional ``callable(step, total, current)`` that
            the analyzer pings at every meaningful milestone (per-chunk LLM
            call, summary synthesis, cap+rerank). The Flask /analyze route
            forwards these to a per-project ``analyze_status.json`` so the
            UI can render an honest progress bar with elapsed/ETA. Pass
            ``None`` and the analyzer runs silently.

    Returns:
        dict canonicalized through :func:`normalize_analysis` so downstream
        renderers see stable keys even when a small model drifted on the
        requested JSON schema.
    """
    import math

    def _emit(step, total, current):
        if progress_callback is None:
            return
        try:
            progress_callback(step=step, total=total, current=current)
        except Exception:
            # Progress writes are advisory — never let a bad callback break
            # the actual analysis.
            pass

    segments = (transcript or {}).get('segments', [])
    total_duration = segments[-1].get('end', 0) if segments else 0
    # Resolved output-language directive — '' for English, so every
    # downstream concat leaves English prompts byte-identical.
    directive = language_directive(output_language)

    # Provider-aware chunking threshold. Cloud providers (Anthropic, OpenAI)
    # have large context windows and a 1-hour interview fits comfortably in
    # a single call — chunking there only adds latency and (per the v0.5.x
    # bug) introduces failure modes around the chunk-merge step. Local
    # Gemma still chunks for very long interviews where its instruction-
    # following degrades on huge inputs, but a 15-25 min interview is
    # better served by a single call (one schema, no merge step). The
    # earlier 900s threshold landed mid-bucket on common interview lengths
    # like 15:17 — we'd chunk into "900s + 17s tail" then fold the tail
    # back into a single 917s chunk and pay the merge complexity for no
    # gain. Bumped to 1500s.
    try:
        from ai_providers import get_active_provider
        active_provider_name = get_active_provider(model_resolver=_get_ollama_model).name
    except Exception:
        active_provider_name = 'ollama'  # safest default — chunk on unknown
    chunk_threshold = (
        3600 if active_provider_name in ('anthropic', 'openai')
        else 1500  # 25 min for local Gemma — single-call up to here
    )

    # Short path: one AI call, original behavior.
    if total_duration < chunk_threshold or not segments:
        formatted = _format_transcript_for_ai(transcript)
        return _analyze_transcript_single(
            formatted, project_name, analysis_type, segment_vectors=segment_vectors,
            progress_emit=_emit, segments=segments,
            output_language=output_language,
        )

    # Chunked path: walk 15-minute slices and merge.
    chunks = list(_iter_transcript_chunks(segments))
    accum = {
        'summary': '',
        'suggested_title': '',
        'story_beats': [],
        'themes': [],
        'strongest_soundbites': [],
        'broll_suggestions': [],
        'social_clips': [],
        # Soft-failure log surfaced on the AI Analysis tab. Per-chunk LLM
        # exceptions, synthesis fallbacks, and cap-pass drops all append a
        # one-line note here instead of failing silently to stderr.
        'analysis_warnings': [],
        '_chunk_summaries': [],
        '_chunk_titles': [],
    }
    # Per-chunk targets sized so the post-merge total lands near the cap with
    # a little margin for the rerank pass to choose from. With 4 chunks and a
    # cap of 7 we ask for 2 per chunk → 8 candidates → trim to 7. Floor at 2
    # so single-chunk-per-category drift doesn't starve the merger.
    chunk_count = max(1, len(chunks))
    per_chunk_target = max(2, math.ceil(ANALYSIS_PER_CATEGORY_CAP / chunk_count))

    # Total step count: each chunk runs (story?, social?) calls, plus the
    # overall summary synthesis pass and the cap+rerank pass. The /analyze
    # route adds another step for paragraph index build outside this scope.
    types_per_chunk = (1 if analysis_type in ('story', 'all') else 0) + \
                      (1 if analysis_type in ('social', 'all') else 0)
    total_steps = chunk_count * types_per_chunk + 2  # +1 summary, +1 cap
    step = 0
    _emit(step, total_steps, "starting")

    # Per-call failure bookkeeping (ported from OSS v3.5.12). Permanently-
    # broken backends abort always; 'unreachable' aborts only before any
    # call has succeeded — mid-run it's usually the bundled-Ollama
    # supervisor relaunching after an OOM (the reconnect bridge covers
    # ≤20s blips; anything longer surfaces here), and throwing away
    # completed chunks over a transient restart is worse than logging one
    # lost chunk. Two consecutive timeouts also abort (one can be a cold
    # model load; two means it's not recovering).
    _FATAL_CODES = ('missing_key', 'invalid_key',
                    'model_missing', 'insufficient_memory')
    chunk_errors = []
    attempted_calls = 0
    consecutive_timeouts = 0
    any_call_succeeded = False

    def _record_chunk_failure(kind, i, e):
        nonlocal consecutive_timeouts
        from ai_providers import ProviderError
        code = ''
        if isinstance(e, ProviderError):
            code = e.code or ''
            if e.code in _FATAL_CODES:
                raise e
            if e.code == 'unreachable' and not any_call_succeeded:
                raise e
            if e.code == 'timeout':
                consecutive_timeouts += 1
                if consecutive_timeouts >= 2:
                    raise e
            else:
                consecutive_timeouts = 0
        else:
            consecutive_timeouts = 0
        detail = str(e)[:200]
        chunk_errors.append((detail, code))
        print(f"[analyze] {kind} chunk {i+1}/{len(chunks)} failed: {e}")
        accum['analysis_warnings'].append(
            f'{kind} analysis failed on chunk {i+1}/{len(chunks)}: {detail}'
        )

    for i, chunk in enumerate(chunks):
        chunk_text = _format_segments_for_ai(chunk['segments'])
        range_label = f"{_seconds_to_tc(chunk['start_seconds'])}-{_seconds_to_tc(chunk['end_seconds'])}"
        chunk_label = f"{project_name} · part {i+1}/{len(chunks)} ({range_label})"
        if analysis_type in ('story', 'all'):
            step += 1
            _emit(step, total_steps, f"chunk {i+1}/{chunk_count}: story beats")
            attempted_calls += 1
            try:
                _merge_story_chunk(
                    accum,
                    _analyze_story(
                        chunk_text, chunk_label,
                        beats_target=per_chunk_target,
                        soundbites_target=per_chunk_target,
                        language_directive_text=directive,
                    ),
                    is_first_chunk=(i == 0),
                )
                consecutive_timeouts = 0
                any_call_succeeded = True
            except Exception as e:
                _record_chunk_failure('Story', i, e)
        if analysis_type in ('social', 'all'):
            step += 1
            _emit(step, total_steps, f"chunk {i+1}/{chunk_count}: social clips")
            attempted_calls += 1
            try:
                _merge_social_chunk(
                    accum,
                    _analyze_social(
                        chunk_text, chunk_label,
                        clips_target=per_chunk_target,
                        language_directive_text=directive,
                    ),
                )
                consecutive_timeouts = 0
                any_call_succeeded = True
            except Exception as e:
                _record_chunk_failure('Social-clip', i, e)

    if attempted_calls and len(chunk_errors) >= attempted_calls:
        # EVERY call failed — this is a backend problem, not a content
        # problem. Promote the most common real error verbatim — WITH its
        # typed code, so the worker persists it and the UI's settings link
        # renders for settings-fixable causes — instead of letting the
        # generic "came back empty" copy bury it.
        from collections import Counter
        from ai_providers import ProviderError
        dominant_msg, dominant_code = Counter(chunk_errors).most_common(1)[0][0]
        raise ProviderError(dominant_msg, code=dominant_code)

    step += 1
    _emit(step, total_steps, "synthesizing summary")
    overall = _synthesize_overall_summary(
        accum.pop('_chunk_summaries', []),
        accum.pop('_chunk_titles', []),
        project_name,
        warnings=accum['analysis_warnings'],
        language_directive_text=directive,
    )
    accum['summary'] = overall['summary']
    accum['suggested_title'] = overall['suggested_title']

    step += 1
    _emit(step, total_steps, "ranking and capping")
    # Normalize first so the cap+rerank sees canonical field names (a model
    # that emitted ``start_time`` shouldn't be filtered out of the cap pass
    # just because the dedup looked for ``start``).
    normalized = normalize_analysis(accum)
    return _cap_and_rank_analysis(
        normalized, segment_vectors=segment_vectors, cap=ANALYSIS_PER_CATEGORY_CAP,
        segments=segments,
    )


def _analyze_transcript_single(formatted_text, project_name, analysis_type,
                                segment_vectors=None, progress_emit=None,
                                segments=None, output_language=None):
    """One-shot analysis path for short interviews."""
    directive = language_directive(output_language)
    types_count = (1 if analysis_type in ('story', 'all') else 0) + \
                  (1 if analysis_type in ('social', 'all') else 0)
    total_steps = types_count + 1  # +1 cap+rerank
    step = 0

    def _emit(current):
        if progress_emit is not None:
            progress_emit(step, total_steps, current)

    _emit("starting")

    result = {'analysis_warnings': []}
    if analysis_type in ('story', 'all'):
        step += 1
        _emit("story beats and soundbites")
        story_data = _analyze_story(
            formatted_text, project_name,
            beats_target=ANALYSIS_PER_CATEGORY_CAP,
            soundbites_target=ANALYSIS_PER_CATEGORY_CAP,
            language_directive_text=directive,
        )
        if isinstance(story_data, dict):
            result['summary'] = _first_present(story_data, 'summary', 'overview', 'synopsis')
            result['suggested_title'] = _first_present(
                story_data, 'suggested_title', 'title', 'working_title'
            )
            result['story_beats'] = _first_present_list(
                story_data, 'story_beats', 'beats', 'narrative_beats', 'story'
            )
            result['themes'] = _first_present_list(story_data, 'themes', 'topics', 'theme_list')
            result['strongest_soundbites'] = _first_present_list(
                story_data, 'strongest_soundbites', 'soundbites', 'quotes', 'best_quotes'
            )
            result['broll_suggestions'] = _first_present_list(
                story_data, 'broll_suggestions', 'broll', 'bRoll', 'b_roll'
            )
            # If the model still produced nothing substantive after both
            # the overview and timecoded sub-calls (each with its own
            # retry), surface a user-visible warning. Without this the
            # AI Analysis tab would silently render only the summary
            # with no story beats / themes / soundbites — exactly the
            # confusing state the user hit on 0.5.4.
            if not result['story_beats'] and not result['strongest_soundbites'] \
                    and not result['broll_suggestions'] and not result['themes']:
                if _CHUNK_DROP_WARNING not in result['analysis_warnings']:
                    result['analysis_warnings'].append(_CHUNK_DROP_WARNING)

    if analysis_type in ('social', 'all'):
        step += 1
        _emit("social clips")
        social_data = _analyze_social(
            formatted_text, project_name,
            clips_target=ANALYSIS_PER_CATEGORY_CAP,
            language_directive_text=directive,
        )
        if isinstance(social_data, dict):
            result['social_clips'] = _first_present_list(
                social_data, 'social_clips', 'clips', 'social', 'reels'
            )
        elif isinstance(social_data, list):
            result['social_clips'] = social_data

    step += 1
    _emit("ranking and capping")
    # Init the warnings list before normalize/cap so the cap pass can log
    # any "missing timecodes" drops back to the caller. Single-file path
    # has no per-chunk failures and no synthesis pass, so this is the only
    # source of warnings here.
    result.setdefault('analysis_warnings', [])
    normalized = normalize_analysis(result)
    return _cap_and_rank_analysis(
        normalized, segment_vectors=segment_vectors, cap=ANALYSIS_PER_CATEGORY_CAP,
        segments=segments,
    )


# ---- Schema-drift tolerance ----
# Small/fast local models (Gemma 4 e2b/e4b especially) don't always match the
# JSON schema we asked for. They rename fields (`beat_description` instead of
# `description`, `start_time` instead of `start`), drop fields (`label`), or
# stick lists under a differently-named top-level key (`clips` vs
# `social_clips`). Rather than fight the model, we canonicalize on the way in
# and on the way out so the templates/JS always see the same shape.
#
# Idempotent — safe to call on already-normalized data.

def _first_present(d, *keys, default=''):
    """Return the first truthy value found at any of the given keys, else default."""
    if not isinstance(d, dict):
        return default
    for k in keys:
        v = d.get(k)
        if v:
            return v
    return default


def _first_present_list(d, *keys):
    """Same as _first_present but defaults to [] and only returns lists."""
    if not isinstance(d, dict):
        return []
    for k in keys:
        v = d.get(k)
        if isinstance(v, list) and v:
            return v
    return []


def _synthesize_label(description: str) -> str:
    """Build a short label from a description when the model didn't provide one.

    Prefers the first sentence when it's short; otherwise the first ~8 words.
    """
    if not description:
        return 'Story Beat'
    first_sentence = description.split('.')[0].strip()
    if 1 <= len(first_sentence.split()) <= 12:
        return first_sentence
    words = description.split()
    return ' '.join(words[:8]) + ('…' if len(words) > 8 else '')


def _normalize_story_beat(beat: dict) -> dict:
    if not isinstance(beat, dict):
        return {}
    description = _first_present(
        beat, 'description', 'beat_description', 'desc', 'text', 'summary', 'why'
    )
    label = _first_present(
        beat, 'label', 'title', 'name', 'heading', 'beat_label', 'beat_title', 'beat'
    )
    if not label:
        label = _synthesize_label(description)
    out = dict(beat)
    out['label'] = label
    out['description'] = description
    out['start'] = _first_present(beat, 'start', 'start_time', 'start_tc', 'begin', 'from')
    out['end'] = _first_present(beat, 'end', 'end_time', 'end_tc', 'to', 'finish')
    out['order'] = beat.get('order', 0)
    return out


def _normalize_social_clip(clip: dict) -> dict:
    if not isinstance(clip, dict):
        return {}
    out = dict(clip)
    title = _first_present(clip, 'title', 'name', 'label', 'heading')
    text = _first_present(clip, 'text', 'description', 'quote', 'content')
    # Small models sometimes return only {start, end, text} and drop the title.
    # Synthesize a readable title from the first few words of the quote so the
    # card isn't blank.
    if not title and text:
        title = _synthesize_label(text)
    out['title'] = title
    out['start'] = _first_present(clip, 'start', 'start_time', 'start_tc', 'begin')
    out['end'] = _first_present(clip, 'end', 'end_time', 'end_tc', 'finish')
    out['text'] = text
    out['platform'] = clip.get('platform', '')
    out['why'] = _first_present(clip, 'why', 'reason', 'rationale', 'note')
    out['hook'] = clip.get('hook', '')
    out['hashtags'] = clip.get('hashtags', []) if isinstance(clip.get('hashtags'), list) else []
    out['rank'] = clip.get('rank', 0)
    out['duration_seconds'] = clip.get('duration_seconds', 0)
    return out


def _normalize_soundbite(sb: dict) -> dict:
    if not isinstance(sb, dict):
        return {}
    out = dict(sb)
    out['text'] = _first_present(sb, 'text', 'quote', 'content', 'soundbite')
    out['start'] = _first_present(sb, 'start', 'start_time', 'start_tc', 'begin')
    out['end'] = _first_present(sb, 'end', 'end_time', 'end_tc')
    out['why'] = _first_present(sb, 'why', 'reason', 'rationale', 'note')
    return out


def _cap_and_rank_analysis(accum, segment_vectors=None, cap=7, segments=None):
    """Trim each analysis list to ``cap`` items, preferring strong candidates.

    Three passes per clip-bearing list:

    1. **Dedupe** items whose ``[start, end]`` ranges overlap by more than
       half of the shorter span — handles the case where a chunk-spanning
       moment is picked up twice across the merge boundary.
    2. **Rank** survivors. ``segment_vectors`` (when present) provides a
       ``narrative_score`` signal: items overlapping a "high" vector
       outrank items overlapping a "medium" or "low" one. Length is the
       tiebreaker — longer well-grounded clips usually mean more substance.
    3. **Cap** at the configured limit and re-sort chronologically so the UI
       reads in timeline order.

    For ``story_beats`` the rank phase also diversifies by ``beat_type``
    (hook / context / pressure / turn / resolution) so we don't keep seven
    hooks and zero resolutions when the model overproduces in one bucket.

    ``themes`` and ``broll_suggestions`` are deduped case-insensitively and
    capped — they don't carry timecodes so the overlap pass is a no-op.

    Idempotent: calling twice with the same input yields the same output.
    """
    if not isinstance(accum, dict):
        return accum or {}

    out = dict(accum)
    # Defensive: caller initializes this, but if we're called on a raw
    # analysis dict (e.g. an old persisted record being re-capped) make
    # sure the slot exists so we can log drop counts.
    warnings = out.get('analysis_warnings')
    if not isinstance(warnings, list):
        warnings = []
        out['analysis_warnings'] = warnings

    # BUG-01: validate every emitted timecode against the real transcript
    # segments BEFORE we dedupe/rank/cap, so hallucinated timecodes are
    # repaired (soundbites/social clips, via verbatim-text match) or dropped
    # (story beats/b-roll, which carry no quote to re-anchor) instead of
    # reaching the editor. Running before the cap means dropped clips don't
    # consume cap slots. Only fires when the caller threaded ``segments`` in.
    if isinstance(segments, list) and segments:
        out['strongest_soundbites'] = _validate_clip_timecodes(
            out.get('strongest_soundbites') or [], segments,
            text_key='text', kind='soundbite')
        out['social_clips'] = _validate_clip_timecodes(
            out.get('social_clips') or [], segments,
            text_key='text', kind='social clip')
        out['story_beats'] = _validate_clip_timecodes(
            out.get('story_beats') or [], segments,
            text_key=None, kind='story beat')
        out['broll_suggestions'] = _validate_clip_timecodes(
            out.get('broll_suggestions') or [], segments,
            text_key=None, kind='b-roll')

    def _ts(val):
        try:
            return _tc_to_seconds(val)
        except Exception:
            return 0.0

    score_weight = {'high': 3, 'medium': 1, 'low': 0}
    vectors = segment_vectors if isinstance(segment_vectors, list) else []

    def _vector_score(start_sec, end_sec):
        if not vectors or end_sec <= start_sec:
            return 0
        best = 0
        for v in vectors:
            if not isinstance(v, dict):
                continue
            v_start = _ts(v.get('timecode_in'))
            v_end = _ts(v.get('timecode_out'))
            if v_end <= v_start:
                continue
            overlap = max(0.0, min(end_sec, v_end) - max(start_sec, v_start))
            if overlap <= 0:
                continue
            w = score_weight.get(str(v.get('narrative_score', 'medium')).lower(), 1)
            if w > best:
                best = w
        return best

    def _item_score(item):
        s = _ts(item.get('start'))
        e = _ts(item.get('end', s))
        if e <= s:
            e = s + 1.0
        return (_vector_score(s, e), e - s)

    def _dedupe_overlap(items, threshold=0.5):
        # Sort highest-scored first so the better candidate wins on overlap.
        ranked = sorted(items, key=lambda x: _item_score(x), reverse=True)
        kept = []
        for cand in ranked:
            cs = _ts(cand.get('start'))
            ce = _ts(cand.get('end', cs))
            # Synthesize an end when the model omits it (common for
            # soundbite schemas that only request {text, start, why}).
            # 20s is a typical soundbite length and is only used for
            # overlap math here — the rendered card uses what the model
            # actually returned.
            if ce <= cs:
                ce = cs + 20.0
            c_len = max(1.0, ce - cs)
            duplicate = False
            for ex in kept:
                es = _ts(ex.get('start'))
                ee = _ts(ex.get('end', es))
                if ee <= es:
                    continue
                overlap = max(0.0, min(ce, ee) - max(cs, es))
                shorter = min(c_len, max(1.0, ee - es))
                if shorter > 0 and overlap / shorter > threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(cand)
        return kept

    def _trim_clip_list(items, prefer_diversity=False, diversity_key='beat_type', label=None):
        if not isinstance(items, list) or not items:
            return [] if isinstance(items, list) else items
        viable = [i for i in items if isinstance(i, dict) and i.get('start')]
        # Log how many items the missing-start filter discarded. Without
        # this the editor sees an unexpectedly empty section with no clue
        # why — typically it's a per-chunk model that emitted beats minus
        # timecodes, which the cap pass can't render.
        dropped = len(items) - len(viable)
        if dropped > 0 and label:
            warnings.append(
                f'{dropped} {label} dropped: missing timecodes from the '
                'model response.'
            )
        deduped = _dedupe_overlap(viable)
        if prefer_diversity and deduped:
            # First pass: pick one of each beat_type bucket (top scorer per bucket).
            # Second pass: fill remaining slots from the leftover pool by score.
            buckets = {}
            for item in deduped:
                key = str(item.get(diversity_key) or item.get('label') or '').strip().lower()
                key = key.split()[0][:12] if key else '__unknown__'
                buckets.setdefault(key, []).append(item)
            for k in buckets:
                buckets[k].sort(key=_item_score, reverse=True)
            spread = []
            leftover = []
            for k in sorted(buckets.keys()):
                bucket = buckets[k]
                if bucket:
                    spread.append(bucket[0])
                    leftover.extend(bucket[1:])
            spread.sort(key=_item_score, reverse=True)
            leftover.sort(key=_item_score, reverse=True)
            picked = (spread + leftover)[:cap]
        else:
            picked = sorted(deduped, key=_item_score, reverse=True)[:cap]
        picked.sort(key=lambda x: _ts(x.get('start')))
        return picked

    if isinstance(out.get('story_beats'), list):
        out['story_beats'] = _trim_clip_list(
            out['story_beats'], prefer_diversity=True, label='story beat(s)',
        )
    if isinstance(out.get('strongest_soundbites'), list):
        out['strongest_soundbites'] = _trim_clip_list(
            out['strongest_soundbites'], label='soundbite(s)',
        )
    if isinstance(out.get('social_clips'), list):
        out['social_clips'] = _trim_clip_list(
            out['social_clips'], label='social clip(s)',
        )

    if isinstance(out.get('themes'), list):
        seen = set()
        deduped = []
        for t in out['themes']:
            if not isinstance(t, str):
                continue
            stripped = t.strip()
            if not stripped:
                continue
            key = stripped.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(stripped)
        out['themes'] = deduped[:cap]

    if isinstance(out.get('broll_suggestions'), list):
        seen = set()
        deduped = []
        for b in out['broll_suggestions']:
            if isinstance(b, dict):
                key = json.dumps({k: b.get(k) for k in sorted(b.keys())}, sort_keys=True)
            else:
                key = str(b).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(b)
        out['broll_suggestions'] = deduped[:cap]

    return out


def normalize_analysis(analysis):
    """Canonicalize analysis field names so small-model drift doesn't break rendering.

    Accepts the variants small models commonly emit (``beat_description`` vs
    ``description``, ``start_time`` vs ``start``, etc.) and returns a dict
    whose renderable fields match what :file:`templates/project.html` expects.
    Idempotent — safe to call on already-normalized data or on the output of
    :func:`analyze_transcript`.
    """
    if not isinstance(analysis, dict):
        return analysis or {}
    out = dict(analysis)
    if isinstance(out.get('story_beats'), list):
        out['story_beats'] = [_normalize_story_beat(b) for b in out['story_beats'] if isinstance(b, dict)]
    if isinstance(out.get('social_clips'), list):
        out['social_clips'] = [_normalize_social_clip(c) for c in out['social_clips'] if isinstance(c, dict)]
    if isinstance(out.get('strongest_soundbites'), list):
        out['strongest_soundbites'] = [_normalize_soundbite(s) for s in out['strongest_soundbites'] if isinstance(s, dict)]
    return out


def _format_transcript_for_ai(transcript, speaker_names=None):
    """Format transcript segments into readable text with timecodes.

    Always sends the full transcript — no truncation.

    When ``speaker_names`` is supplied (Pro diarization rename map) raw
    pyannote labels are resolved to display names before they reach the
    model, so the LLM sees ``Sarah Chen: …`` instead of ``SPEAKER_00: …``.
    Pre-diarization projects render unchanged.
    """
    segments = transcript.get('segments', [])
    if not segments:
        return ''

    # Format all segments with start AND end times so AI can set accurate clip boundaries
    all_lines = []
    for seg in segments:
        start_tc = seg['start_formatted'][:8]
        end_s = seg.get('end', seg.get('start', 0))
        end_tc = f"{int(end_s)//3600:02d}:{(int(end_s)%3600)//60:02d}:{int(end_s)%60:02d}"
        raw_speaker = seg.get('speaker', 'Speaker')
        speaker = _display_speaker(raw_speaker, speaker_names)
        text = seg['text']
        if text.strip():
            all_lines.append(f"[{start_tc}-{end_tc}] {speaker}: {text}")

    return '\n'.join(all_lines)


def _build_paragraphs(transcript, max_paragraph_seconds=60):
    """Merge adjacent same-speaker segments into paragraphs of up to
    ``max_paragraph_seconds``. Returns a list of dicts with keys
    ``speaker``, ``start``, ``end``, ``text``.

    Layer 1 (keyword pre-retrieval) and Layer 2 (chunked search) both
    consume these structured paragraphs, so the grouping logic lives in one
    place and stays consistent with what the main chat prompt sees.
    """
    segments = transcript.get('segments', []) if transcript else []
    paragraphs = []
    cur = None
    for seg in segments:
        text = (seg.get('text') or '').strip()
        if not text:
            continue
        speaker = seg.get('speaker', 'Speaker')
        start = seg.get('start', 0)
        end = seg.get('end', start)
        if (cur
                and cur['speaker'] == speaker
                and (end - cur['start']) <= max_paragraph_seconds):
            cur['end'] = end
            cur['text'] = f"{cur['text']} {text}"
        else:
            if cur:
                paragraphs.append(cur)
            cur = {'speaker': speaker, 'start': start, 'end': end, 'text': text}
    if cur:
        paragraphs.append(cur)
    return paragraphs


def _format_paragraphs_as_lines(paragraphs):
    """Render structured paragraphs as ``[HH:MM:SS-HH:MM:SS] Speaker: text``
    lines. Shared by the main transcript block, the RELEVANT EXCERPTS
    block, and Layer 2 chunk prompts so the grounding rule (copy timecodes
    verbatim) stays valid across all three. Accepts either full paragraph
    dicts or raw segment dicts (both shapes have start/end/text/speaker).
    """
    lines = []
    for p in paragraphs:
        start = p.get('start', 0)
        end = p.get('end', start)
        start_tc = _seconds_to_tc(start)
        end_tc = _seconds_to_tc(end)
        speaker = p.get('speaker', 'Speaker')
        text = (p.get('text') or '').strip()
        lines.append(f"[{start_tc}-{end_tc}] {speaker}: {text}")
    return '\n'.join(lines)


def _format_transcript_paragraphs_for_ai(transcript, max_paragraph_seconds=60):
    """Same as :func:`_format_transcript_for_ai`, but merges adjacent segments
    from the same speaker into paragraphs of up to ``max_paragraph_seconds``.

    Each output line still uses the ``[HH:MM:SS-HH:MM:SS] Speaker: text``
    shape, so the system prompt's grounding rule (copy timecodes verbatim
    from the transcript markers) stays valid — the difference is only that
    one line now covers a paragraph's worth of adjacent segments instead of
    a single sentence. Clip boundaries the model emits will align to
    paragraph edges, which is coarser than per-segment but still well within
    normal clip durations.
    """
    paragraphs = _build_paragraphs(transcript, max_paragraph_seconds=max_paragraph_seconds)
    if not paragraphs:
        return ''
    return _format_paragraphs_as_lines(paragraphs)


def _call_ai(prompt, system_prompt="", task_type="analysis", force_json=True,
             num_predict=None):
    """Single-prompt generation through the active provider.

    ``task_type`` selects the model when the provider tiers (Anthropic uses
    Opus for ``profile_creation``, Sonnet otherwise; Ollama ignores it).
    Editorial DNA's classifier passes ``"analysis"``; My Style synthesis
    passes ``"profile_creation"``.

    ``num_predict`` (Ollama-only) overrides the provider's output-token
    budget. The story-build calls pass _STORY_NUM_PREDICT: a duration-target
    build legitimately returns dozens of clip objects, and the 4096 default
    truncated the JSON mid-array — the whole build then parsed to 0 clips.
    ``None`` (the default) leaves every other caller on the provider's own
    default. Cloud providers ignore it (they size output per task_type).

    ``force_json`` (Ollama-only) toggles the ``format='json'`` decoding
    grammar on the local ``/api/generate`` path. It defaults to True (the
    normal structured-output behavior). Set it False to let the model
    answer in free form — the escape hatch for larger local models
    (e.g. gemma4:26b) that degenerate under the strict JSON grammar and
    return an empty/``{}`` body; in free form they emit usable JSON (often
    wrapped in prose) that the tolerant parser then salvages. Cloud
    providers ignore the flag.

    Raises ``RuntimeError`` on provider error so the caller can surface a
    clear message to the user instead of silently falling back.
    """
    from ai_providers import get_active_provider
    # Skip the storytelling foundation for structured JSON analysis.
    # The foundation is ~28 KB (~7 K tokens) of narrative-editorial
    # guidance. For analysis calls the prompt is a self-contained JSON
    # extraction task: the transcript IS the data and it must survive
    # intact. With num_ctx=12288 and num_predict=4096 the available
    # input budget is ~8192 tokens — the foundation alone would consume
    # ~7094 of those, leaving ~1098 for the entire transcript. Ollama
    # silently truncates the overflow, so the model never sees the real
    # transcript and hallucinates plausible-looking timecodes and generic
    # descriptions. Chat and Story Builder still get the full foundation.
    if task_type != "analysis":
        system_prompt = inject_storytelling_foundation(system_prompt)
    provider = get_active_provider(model_resolver=_get_ollama_model)
    # Size the HTTP timeout to the active model. The old fixed 180s ceiling
    # (in the Ollama provider) cut off the large variants mid-generation —
    # gemma4:26b/31b need minutes to emit a full analysis JSON — leaving the
    # AI Analysis tab and the collection dashboard empty. Local Ollama honors
    # this; cloud providers ignore it.
    try:
        from model_config import recommended_analysis_timeout
        # Sized to the authorized output budget too: the story builds'
        # num_predict=8192 legitimately decodes for 600-850s on the
        # default 8GB profile, but the no-arg call sized the ceiling off
        # the 2000-token representative (504s) — ReadTimeout killed
        # healthy generations mid-decode, bypassing both the JSON repair
        # and the deterministic fallback. None → identical to before.
        timeout = recommended_analysis_timeout(num_predict=num_predict)
    except Exception:
        timeout = 600
    extra = {}
    if num_predict is not None:
        # Only forwarded when set, keeping every default-path provider call
        # byte-identical to before the story-build budget raise.
        extra['num_predict'] = num_predict
    return provider.generate(
        system_prompt, prompt, task_type=task_type, force_json=force_json,
        timeout=timeout, **extra,
    )


def _ollama_is_active():
    """True iff the saved active provider is local Ollama. Cached at the
    call-site level so failures degrade to ``True`` (the historical default)
    rather than blocking the AI flow on a missing config file.
    """
    try:
        from ai_providers import load_provider_config
        return (load_provider_config().get('active_provider') or 'ollama') == 'ollama'
    except Exception:
        return True


def get_effective_ollama_model():
    """Resolve which Ollama model will actually be used for the next chat call.

    Returns ``(effective_model, selected_variant, fallback_used)`` where:
      - ``effective_model`` is the tag Ollama will actually load
      - ``selected_variant`` is the user-selected variant from model_config
      - ``fallback_used`` is True when the selection wasn't installed and we
        had to fall through to something else

    The previous implementation silently swapped to a smaller variant when
    the selection wasn't downloaded, so users picking "large (gemma4:26b)"
    on a laptop with only ``gemma4:e2b`` installed got e2b without any
    indication. The UI now surfaces ``fallback_used`` so the user can
    download what they actually picked.

    Short-circuits to ``('', '', False)`` when the active provider isn't
    Ollama — no point burning an HTTP roundtrip to localhost:11434 when the
    user is running through Anthropic or OpenAI.
    """
    if not _ollama_is_active():
        return '', '', False
    try:
        import model_config
        selected_variant = model_config.get_gemma4_variant()['variant']
    except Exception:
        selected_variant = 'gemma4:e4b'

    # gemma4:latest is what Ollama returns when the user pulls without an
    # explicit tag (`ollama pull gemma4`).  Map it to the medium variant
    # so the substring match below treats them as equivalent.
    _LATEST_ALIASES = {
        'gemma4:latest': 'gemma4:e4b',
    }

    try:
        # 2s timeout (was 5s). /api/ai-model/status calls this synchronously;
        # a longer timeout would race the modal's 12s AbortController on
        # systems where Ollama is mid-boot, hiding the variant list. With
        # 2s we fail fast and the modal renders the hardware-derived
        # variants while marking effective.model = None.
        from ollama_url import ollama_base_url
        response = requests.get(f'{ollama_base_url()}/api/tags', timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            available = [m['name'] for m in models]
            # Exact-prefix match for the selected variant.
            for avail in available:
                if selected_variant in avail:
                    return avail, selected_variant, False
            # Check if any available model is an alias of the selection
            # (e.g. gemma4:latest == gemma4:e4b).
            for avail in available:
                canonical = _LATEST_ALIASES.get(avail)
                if canonical and canonical == selected_variant:
                    return avail, selected_variant, False
            # Fall back through preferred IN-FAMILY variants — but flag
            # it so the UI can warn the user that they're not on their
            # selection. We previously also fell back to gemma3 / llama3 /
            # mistral and to `available[0]`, which produced the 0.7.0
            # "phantom gemma3:4b in Currently-loaded badge" bug. If no
            # gemma4 variant is installed, return None so the UI can
            # render an actionable empty state instead of pretending a
            # different model is loaded.
            fallback = ['gemma4:e4b', 'gemma4:latest', 'gemma4:e2b', 'gemma4:26b', 'gemma4:31b']
            for pref in fallback:
                for avail in available:
                    if pref in avail:
                        print(f"[model] FALLBACK: selected {selected_variant!r} not "
                              f"installed; using {avail!r} (matched on {pref!r}). "
                              "User won't see speed/quality changes from selection.")
                        return avail, selected_variant, True
            # No usable gemma4 variant installed. Don't pretend.
            return None, selected_variant, True
    except Exception as e:
        print(f"[model] could not query Ollama tags: {e}")
    # Ollama unreachable. Don't pretend the selected variant is loaded;
    # return None so the UI surfaces "model not available" rather than
    # quietly showing a phantom "currently loaded" badge.
    return None, selected_variant, True


def _get_ollama_model():
    """Backwards-compat wrapper — returns just the effective model name."""
    effective, _selected, _fallback = get_effective_ollama_model()
    return effective


# Cached availability of the small chunked-search model. None until
# the first probe; True/False thereafter. Probe runs on demand and
# stays cached for the process lifetime — Ollama-installed models
# rarely change mid-session, and re-probing every chunk call would
# add a second's overhead per call.
_FAST_CHUNK_MODEL_CACHE = None


def _get_fast_chunk_model():
    """Return the model name to use for Layer 2 chunked-search workers.

    Off by default — set ``DOZA_FAST_CHUNK_MODEL=1`` to opt in. When
    enabled, uses ``gemma4:e2b`` (2B effective, 2-3× faster decode on
    Apple Silicon) for the per-chunk scoring pass while the synthesis
    paths (``_call_ai_chat``, the rerank pass) keep using the user's
    hardware-tier variant.

    Why opt-in: e2b is reliably faster, but its structured-JSON
    output for the chunk-search prompt is shaky enough to drop a 100-
    minute interview to zero candidates on some queries — the chunk
    parser sees malformed JSON and the salvage path can't recover.
    The hardware-tier variant produces clean JSON every time. When
    the gap closes (bigger small variant, better-tuned prompt, or
    Ollama format=json gets stricter), flip the default to True.

    Falls back to the user's main model if e2b isn't pulled or the
    env var isn't set, so default behavior matches the v3.1.x path.

    Returns ``None`` when the active provider isn't Ollama — API providers
    don't accept per-call model overrides (the model is task_type-driven),
    and skipping the localhost:11434 probe saves a roundtrip per chunk.
    """
    if not _ollama_is_active():
        return None
    global _FAST_CHUNK_MODEL_CACHE
    main_model = _get_ollama_model()
    if not os.environ.get('DOZA_FAST_CHUNK_MODEL'):
        return main_model
    if main_model.startswith('gemma4:e2b'):
        return main_model
    if _FAST_CHUNK_MODEL_CACHE is not None:
        return _FAST_CHUNK_MODEL_CACHE if _FAST_CHUNK_MODEL_CACHE else main_model
    try:
        from ollama_url import ollama_base_url
        response = requests.get(f'{ollama_base_url()}/api/tags', timeout=2)
        if response.status_code == 200:
            available = [m.get('name', '') for m in (response.json() or {}).get('models', [])]
            for name in available:
                if 'gemma4:e2b' in name:
                    _FAST_CHUNK_MODEL_CACHE = name
                    return name
    except Exception:
        pass
    _FAST_CHUNK_MODEL_CACHE = ''
    return main_model


def _estimate_layer1_num_ctx(formatted_transcript, system_prompt_extra=4096):
    """Pick a num_ctx for Layer 1 that fits the formatted transcript plus
    a slack budget for system prompt + storytelling foundation + reply.

    Apple Silicon prompt-eval scales roughly linearly with context size,
    so a 60K-token-budgeted call against a 5-minute interview wastes
    seconds on KV-cache zero-padding. This snaps to the smallest power
    of two that comfortably holds the transcript and leaves room for
    everything else, capped at the original 32K ceiling.

    Token estimate is char-count / 3.8 (matching the heuristic in
    ``_chunk_paragraphs``). Slightly pessimistic on punctuation-heavy
    text, which is what we want — better to over-allocate than truncate.
    """
    text_len = len(formatted_transcript or '')
    # 3.8 chars/token is the same divisor _chunk_paragraphs uses; keep
    # them aligned so a "fits in chunk" estimate matches a "fits in
    # Layer 1" estimate.
    transcript_tokens = int(text_len / 3.8) + 100
    needed = transcript_tokens + system_prompt_extra
    # Round up to a power of two within 8K..32K. Below 8K is a waste of
    # the rounding (Ollama uses pow-of-2 KV blocks anyway); above 32K
    # was the previous hardcoded ceiling and leaving it there keeps the
    # behavior conservative for the longest interviews.
    for ctx in (8192, 12288, 16384, 24576, 32768):
        if needed <= ctx:
            return ctx
    return 32768


def warmup_ollama():
    """Issue a tiny generate call to load the configured model into RAM.

    Called from the Flask app during startup and from project_view as
    a fire-and-forget background warm-up. Combined with keep_alive=30m,
    the user's first chat message after launching the app starts with
    the model already resident — no 4-12s cold-load.

    No-op when the active provider isn't Ollama — there's nothing to warm
    up if the user is routing through Anthropic or OpenAI, and pinging
    localhost:11434 just to fail noisily defeats the purpose.

    Errors are swallowed: the warm-up is best-effort and Ollama may
    not be running yet at app start. Subsequent real chat calls
    re-attempt the load via the normal _call_ai_chat path.
    """
    if not _ollama_is_active():
        return
    try:
        from ollama_url import ollama_base_url
        requests.post(
            f'{ollama_base_url()}/api/generate',
            json={
                'model': _get_ollama_model(),
                'prompt': 'ok',
                'stream': False,
                'keep_alive': _OLLAMA_KEEP_ALIVE,
                'options': {'num_predict': 1, 'num_ctx': 2048},
            },
            timeout=120,
        )
    except Exception:
        pass


def _chunk_cache_get(key):
    with _CHUNK_CACHE_LOCK:
        if key in _CHUNK_CACHE:
            _CHUNK_CACHE.move_to_end(key)
            return _CHUNK_CACHE[key]
    return None


def _chunk_cache_put(key, value):
    if not value:
        return
    with _CHUNK_CACHE_LOCK:
        _CHUNK_CACHE[key] = value
        _CHUNK_CACHE.move_to_end(key)
        while len(_CHUNK_CACHE) > _CHUNK_CACHE_MAX:
            _CHUNK_CACHE.popitem(last=False)


def build_story(transcript, message, project_name="Interview", segment_vectors=None, profile_id=None,
                output_language=None):
    """
    Build a narrative sequence from the transcript based on the user's description.
    Returns a dict with story_title, target_duration, and clips array.

    If segment_vectors is provided, the model is given the pre-classified segments
    instead of having to re-analyze the raw transcript. This makes builds faster and
    much more consistent across runs.

    Duration asks ("build me a 14 minute cut") are parsed in code —
    parse_target_duration_seconds — and enforced deterministically after the
    model's pick (_enforce_duration_budget); the model's arithmetic is never
    trusted. When no duration parses, behavior is unchanged.
    """
    target_seconds = parse_target_duration_seconds(message)
    if segment_vectors:
        return _build_story_from_vectors(
            segment_vectors, message, project_name, profile_id=profile_id,
            output_language=output_language, target_seconds=target_seconds,
        )

    formatted = _format_transcript_for_ai(transcript)

    # Numeric budget line, mirroring the vectors path. No menu exists here,
    # so the clip-count hint assumes the 5-30s per-clip rule below (~20s).
    # A capped ask also swaps the DURATION bullet so the system prompt and
    # the block agree (band vs capped count).
    duration_block = ''
    duration_capped = False
    if target_seconds:
        duration_block = _story_duration_prompt_block(target_seconds, 20.0) + '\n\n'
        duration_capped = (
            _story_min_clip_ask(target_seconds, 20.0) > _STORY_PROMPT_MAX_CLIP_ASK
        )
    duration_bullet = (_STORY_DURATION_BULLET_RAW_CAPPED if duration_capped
                       else _STORY_DURATION_BULLET_RAW)

    system_prompt = """You are a story editor building a narrative sequence from interview transcript footage. The user will describe what kind of story or edit they want. Your job is to select and order clips from the transcript that form a coherent narrative.

Rules:
- Select clips that build a clear narrative arc: hook, rising action, emotional peak, resolution
- MANDATORY ARC: every selected clip fills exactly one role in this five-slot arc — hook, context, pressure, turn, resolution. Tag the role in editorial_note (e.g. "ROLE: turn — ..."). The "order" field reflects the arc, NOT the timecode.
- NON-CHRONOLOGICAL BY DEFAULT: the transcript below is presented in recording order; your output MUST NOT preserve that order unless the user explicitly requested chronological OR a clip's meaning depends on temporal sequence (cause→effect chain).
- ANTI-PATTERN CHECK: if your selected clips' start_time values are monotonically increasing in your chosen order, you have likely defaulted to chronological — re-examine and re-sequence.
- Each clip should be 5-30 seconds long unless the moment requires more breathing room
""" + duration_bullet + """
- For each clip, provide: a short title, start timecode, end timecode, the transcript excerpt, and a one-sentence editorial note explaining why this clip is in this position
- Be selective and opinionated. Don't include filler. Every clip should earn its place
- CRITICAL: Copy the exact HH:MM:SS timecodes from the transcript for start and end times. Use string format like "00:02:45"
- ALWAYS include a "reasoning" field: 2-3 conversational sentences in plain language explaining what this story is really about underneath the surface and why this arc works. Talk like a doc editor, not a corporate brief.
- Respond ONLY in valid JSON with this structure:
{
  "story_title": "suggested title for this sequence",
  "target_duration": "estimated total duration",
  "reasoning": "2-3 conversational sentences on what this story is really about and why this arc works",
  "clips": [
    {
      "order": 1,
      "title": "clip title",
      "start_time": "00:00:00",
      "end_time": "00:00:00",
      "transcript": "the exact words from the transcript",
      "editorial_note": "why this clip is here and what it does for the story"
    }
  ]
}"""

    prompt = f"""Build a narrative sequence from this interview transcript.

PROJECT: {project_name}

USER REQUEST: {message}

{duration_block}TRANSCRIPT (presented in recording order — re-sequence freely for narrative arc):
{formatted}

Return ONLY valid JSON. No markdown, no extra text."""

    system_prompt = inject_my_style(system_prompt, profile_id=profile_id)
    # Output-language directive ('' for English → byte-identical prompt).
    directive = language_directive(output_language)
    if directive:
        system_prompt = system_prompt + directive
    # Raised output budget: a duration-target build returns dozens of clip
    # objects and the provider default truncated the JSON mid-array.
    response = _call_ai(prompt, system_prompt, num_predict=_STORY_NUM_PREDICT)
    result = _parse_json_response(response)
    if target_seconds and isinstance(result, dict) and result.get('clips'):
        # No vector pool on the raw path — the budget pass still measures,
        # trims overshoot, reports honest numbers, and flags shortfalls.
        result['clips'], meta = _enforce_duration_budget(
            result['clips'], target_seconds, segment_vectors=None,
        )
        result['target_duration'] = _format_duration_target(target_seconds)
        result.update(meta)
    return result


_SEGMENT_VECTOR_SYSTEM_PROMPT = """You are a documentary story analyst. You break interview transcripts into discrete narrative segments and classify them with strict, structured metadata. You always respond in valid JSON only — no prose, no markdown, no code fences."""


def _segment_vector_prompt(transcript_text: str, project_name: str) -> str:
    return f"""Analyze this interview transcript and produce a structured set of segment vectors.

PROJECT: {project_name}

TRANSCRIPT:
{transcript_text}

STEP 1 — Identify the distinct threads or topics the speaker discusses. Use the speaker's own words and phrasing for each thread title. Do not invent abstract corporate language. "The day I quit" — yes. "Professional Transition Event" — no.

STEP 2 — Segment the transcript into discrete moments. A segment is one continuous thought, story beat, or topic. Each segment must have an exact start and end timecode copied from the [HH:MM:SS] markers in the transcript.

STEP 3 — Classify each segment with these fields:

  thread_title (string): The thread this segment belongs to, in the speaker's own words.

  memory_type (string): One of:
    - "episodic" — a specific event, sensory detail, "I remember when...", a story with concrete time/place.
    - "semantic" — general knowledge, abstract statement, opinion, "Generally speaking...", reflection without a specific scene.

  narrative_score (string): One of "high", "medium", "low".
    DISTRIBUTION CONSTRAINT — across all segments you produce:
      - roughly 15% should be "high" (top tier — strong emotion, cinematic specificity, the moments an editor would build a film around)
      - roughly 50% should be "medium" (solid, usable for montage or connective tissue)
      - roughly 35% should be "low" (exposition, filler, repetition, throat-clearing)
    Be ruthless. Most segments are NOT "high". If you find yourself marking more than 1 in 6 as "high", downgrade the weakest ones.

  beat_type (string): One of "hook", "context", "pressure", "turn", "resolution".

  theme_tags (array of 2-4 strings): Short keyword tags describing the emotional or thematic content (e.g. ["loss", "decision"]).

Return ONLY valid JSON in this exact shape:
{{
  "segments": [
    {{
      "seg_id": "SEG001",
      "timecode_in": "00:00:00",
      "timecode_out": "00:00:00",
      "thread_title": "...",
      "memory_type": "episodic",
      "narrative_score": "medium",
      "beat_type": "context",
      "theme_tags": ["tag1", "tag2"]
    }}
  ]
}}

Do NOT include the transcript_excerpt field — the application will fill that in.
Use string HH:MM:SS format for timecodes, copied exactly from the transcript markers.
Aim for 12-30 segments depending on transcript length."""


def _extract_segment_list(parsed):
    if isinstance(parsed, dict):
        return parsed.get('segments', []) or []
    if isinstance(parsed, list):
        return parsed
    return []


def _generate_vectors_single_chunk(transcript_text: str, project_name: str):
    """Run one AI call and return the raw list of segment dicts (pre-normalize)."""
    response = _call_ai(
        _segment_vector_prompt(transcript_text, project_name),
        _SEGMENT_VECTOR_SYSTEM_PROMPT,
    )
    return _extract_segment_list(_parse_json_response(response))


def expected_vector_chunks(transcript):
    """Return the number of chunks ``generate_segment_vectors`` will run.

    Mirrors the chunked-vs-single decision in ``generate_segment_vectors``
    so the /analyze route can size the progress total correctly *before*
    actually running vector generation. Returns ``1`` for short transcripts
    (single-call path) or for any case the iterator yields zero chunks.
    """
    segments = (transcript or {}).get('segments', []) if transcript else []
    if not segments:
        return 1
    duration = segments[-1].get('end', 0) if segments else 0
    if duration < _LONG_INTERVIEW_SECONDS:
        return 1
    chunks = list(_iter_transcript_chunks(segments))
    return max(1, len(chunks))


def generate_segment_vectors(transcript, project_name="Interview", progress_callback=None):
    """
    Generate structured segment vectors from a transcript.

    Short transcripts (<15 min) use a single AI call. Longer ones are chunked
    into ~15-minute slices — without chunking, small local models produce 3-4
    segments for the opening and silently give up on the rest of the interview,
    which is the failure mode that was blocking Story Builder on 100-minute
    projects.

    ``progress_callback`` (optional) is invoked as
    ``progress_callback(chunk_idx, total_chunks, label)`` *before* each chunk
    runs. The /analyze route uses this to advance its global progress bar in
    proportion to the actual number of LLM calls (one per chunk) — without
    this signal the bar misreports the segment-vector phase as a single step
    and races ahead of the actual work.

    Returns a list of dicts each shaped like:
      {
        "seg_id": "SEG001",
        "timecode_in": "00:12:34",
        "timecode_out": "00:13:02",
        "thread_title": "The day I quit",
        "memory_type": "episodic" | "semantic",
        "narrative_score": "high" | "medium" | "low",
        "beat_type": "hook" | "context" | "pressure" | "turn" | "resolution",
        "theme_tags": ["loss", "decision"],
        "transcript_excerpt": "First 50 words ...",
        "frozen": true
      }
    """
    def _emit(idx, total, label):
        if progress_callback is None:
            return
        try:
            progress_callback(chunk_idx=idx, total_chunks=total, label=label)
        except Exception:
            pass

    segments = (transcript or {}).get('segments', [])
    if not segments:
        return []
    duration = segments[-1].get('end', 0) if segments else 0

    if duration < _LONG_INTERVIEW_SECONDS:
        _emit(1, 1, "segment vectors")
        raw = _generate_vectors_single_chunk(
            _format_transcript_for_ai(transcript), project_name,
        )
        return _normalize_segment_vectors(raw, transcript)

    # Chunked path: collect raw segments from each slice, then renumber seg_ids
    # globally so the downstream menu and Story Builder hydrator work off
    # unique identifiers.
    all_raw = []
    chunks = list(_iter_transcript_chunks(segments))
    for i, chunk in enumerate(chunks):
        chunk_text = _format_segments_for_ai(chunk['segments'])
        range_label = f"{_seconds_to_tc(chunk['start_seconds'])}-{_seconds_to_tc(chunk['end_seconds'])}"
        chunk_label = f"{project_name} · part {i+1}/{len(chunks)} ({range_label})"
        _emit(i + 1, len(chunks), f"vectors {i+1}/{len(chunks)}")
        try:
            all_raw.extend(_generate_vectors_single_chunk(chunk_text, chunk_label))
        except Exception as e:
            print(f"[vectors] chunk {i+1}/{len(chunks)} failed: {e}")

    # Renumber to guarantee globally-unique IDs — models sometimes restart
    # numbering from SEG001 inside each chunk.
    for i, s in enumerate(all_raw, start=1):
        if isinstance(s, dict):
            s['seg_id'] = f'SEG{i:03d}'

    return _normalize_segment_vectors(all_raw, transcript)


def _normalize_segment_vectors(raw_segments, transcript):
    """Validate, repair, and enrich segment vectors with transcript_excerpt."""
    valid_memory = {'episodic', 'semantic'}
    valid_score = {'high', 'medium', 'low'}
    valid_beat = {'hook', 'context', 'pressure', 'turn', 'resolution'}

    out = []
    for i, seg in enumerate(raw_segments):
        if not isinstance(seg, dict):
            continue
        tc_in = str(seg.get('timecode_in', seg.get('start', '')) or '').strip()
        tc_out = str(seg.get('timecode_out', seg.get('end', '')) or '').strip()
        if not tc_in or not tc_out:
            continue

        memory_type = str(seg.get('memory_type', 'semantic')).strip().lower()
        if memory_type not in valid_memory:
            memory_type = 'semantic'

        narrative_score = str(seg.get('narrative_score', 'medium')).strip().lower()
        if narrative_score not in valid_score:
            narrative_score = 'medium'

        beat_type = str(seg.get('beat_type', 'context')).strip().lower()
        if beat_type not in valid_beat:
            beat_type = 'context'

        tags = seg.get('theme_tags', []) or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',') if t.strip()]
        tags = [str(t).strip() for t in tags if str(t).strip()][:4]

        excerpt = _extract_excerpt_for_range(transcript, tc_in, tc_out, max_words=50)

        out.append({
            'seg_id': seg.get('seg_id') or f'SEG{i + 1:03d}',
            'timecode_in': tc_in,
            'timecode_out': tc_out,
            'thread_title': str(seg.get('thread_title', '') or '').strip()[:120],
            'memory_type': memory_type,
            'narrative_score': narrative_score,
            'beat_type': beat_type,
            'theme_tags': tags,
            'transcript_excerpt': excerpt,
            'frozen': True,
        })

    # Soft-enforce the distribution: if more than ~22% are "high", demote extras to medium.
    if out:
        highs = [s for s in out if s['narrative_score'] == 'high']
        cap = max(1, round(len(out) * 0.22))
        if len(highs) > cap:
            # Keep the first `cap` highs (in order); demote the rest.
            for s in highs[cap:]:
                s['narrative_score'] = 'medium'

    return out


def _tc_to_seconds(tc):
    """Convert HH:MM:SS or MM:SS string to float seconds."""
    if isinstance(tc, (int, float)):
        return float(tc)
    s = str(tc).strip()
    if not s:
        return 0.0
    # Unit-suffixed seconds tokens ("125s", "140 sec") — models emit them
    # in [CLIP:] markers and both frontends already play them (JS
    # parseFloat ignores the suffix), so the server-side enforcement
    # passes must parse them identically instead of scoring them 0.0 and
    # junk-dropping the marker (F3). The lookbehind requires a digit so
    # placeholder tokens like "MM:SS" stay unparseable.
    s = re.sub(r'(?<=[\d.])\s*(?:s|secs?|seconds?)$', '', s,
               flags=re.IGNORECASE)
    if ':' in s:
        parts = s.split(':')
        try:
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _extract_excerpt_for_range(transcript, tc_in, tc_out, max_words=50):
    """Pull the first ~max_words of actual transcript text inside a time range."""
    start = _tc_to_seconds(tc_in)
    end = _tc_to_seconds(tc_out)
    if end <= start:
        end = start + 30

    segments = (transcript or {}).get('segments', []) or []
    collected = []
    for seg in segments:
        seg_start = float(seg.get('start', 0) or 0)
        seg_end = float(seg.get('end', seg_start) or seg_start)
        if seg_end < start or seg_start > end:
            continue
        text = (seg.get('text') or '').strip()
        if text:
            collected.append(text)
            joined = ' '.join(collected)
            if len(joined.split()) >= max_words:
                break

    words = ' '.join(collected).split()
    if len(words) > max_words:
        return ' '.join(words[:max_words]) + '...'
    return ' '.join(words)


# ── Story duration budget ─────────────────────────────────────────────────
#
# The model's clip selection is advisory on duration: small Gemma reliably
# returns ~8-15 clips regardless of the requested runtime (the five-slot
# arc anchors it near one-clip-per-role), and hydration clamps every clip
# to its segment's boundaries — so a 14-minute ask came back ~5.5 minutes,
# every time. The budget is therefore measured and reconciled in code.
_STORY_BUDGET_FLOOR = 0.9      # top-up trigger / trim floor
_STORY_BUDGET_CEILING = 1.15   # trim trigger / top-up ceiling
_STORY_BUDGET_HARD_FLOOR = 0.8  # below this after top-up → shortfall note
_STORY_SCORE_RANK = {'high': 0, 'medium': 1, 'low': 2}
# Shortening floor for the overshoot pass: never tighten a clip below
# this — a sub-15s story beat stops being a usable moment.
_STORY_MIN_CLIP_SECONDS = 15.0
# Cap on the segment count the duration prompt block DEMANDS. A 15-20
# minute ask over a ~30s-average menu computes to 35-40 segments; pushing
# a small local model to emit 40 verbose clip objects blew straight
# through the output-token budget and the JSON truncated mid-array — the
# "AI returned 0 clips" tester failure. 18 objects fit comfortably inside
# the raised story num_predict; the deterministic _enforce_duration_budget
# top-up closes whatever runtime gap the capped ask leaves.
_STORY_PROMPT_MAX_CLIP_ASK = 18
# Output-token budget for the two story-build calls. The provider default
# (4096) fits every other analysis schema, but a duration-target story
# legitimately returns dozens of clip objects with editorial notes — the
# 32B variant writes long ones — and truncation there costs the entire
# build. Ollama-only; cloud providers size output per task_type.
_STORY_NUM_PREDICT = 8192

# System-prompt DURATION bullets, selected per build (one pair for each
# story path — the vectors prompt speaks in menu segments, the raw prompt
# in transcript clips). The band version demands the numeric sum band and
# is correct whenever the prompt block's segment ask can actually reach
# it. When the ask is capped at _STORY_PROMPT_MAX_CLIP_ASK the band is
# unreachable by construction (18 x avg < 0.9 x target), and keeping the
# band bullet alongside the capped block handed the model two
# contradictory instructions — chase the band (the output blowout the cap
# exists to stop) or obey the cap and "fail" the bullet. Capped builds
# get the count-consistent bullet; uncapped builds keep the band wording
# verbatim.
_STORY_DURATION_BULLET_VECTORS = (
    "- DURATION IS CRITICAL: If the user requests a specific duration, the request includes a TARGET TOTAL RUNTIME line with the exact numeric band and the minimum segment count already computed from the menu. Meet that segment count — undershooting by stopping early is the most common failure. Every menu line shows dur=<seconds>; your selected segments' dur values must sum inside the stated band. If your selection runs long, remove the weakest segments."
)
_STORY_DURATION_BULLET_VECTORS_CAPPED = (
    "- DURATION IS CRITICAL: The request includes a TARGET TOTAL RUNTIME line with the segment count to select, already computed from the menu. Meet that count with the strongest segments — the remaining runtime toward the target is completed automatically, so do not pad your selection to chase the total yourself."
)
_STORY_DURATION_BULLET_RAW = (
    "- DURATION IS CRITICAL: If the user requests a specific duration, the request includes a TARGET TOTAL RUNTIME line with the exact numeric band and the minimum clip count already computed. Meet that clip count — undershooting by stopping early is the most common failure. Your selected clips' (end_time - start_time) values must sum inside the stated band."
)
_STORY_DURATION_BULLET_RAW_CAPPED = (
    "- DURATION IS CRITICAL: The request includes a TARGET TOTAL RUNTIME line with the clip count to select, already computed. Meet that count with the strongest clips — the remaining runtime toward the target is completed automatically, so do not pad your selection to chase the total yourself."
)


def _story_min_clip_ask(target_seconds, avg_clip_seconds):
    """ceil(target/avg) — the segment count a duration ask implies. Shared
    by the prompt block and the capped-ask check so the two can't drift."""
    import math
    avg = max(1.0, float(avg_clip_seconds))
    return max(1, math.ceil(int(round(float(target_seconds))) / avg))


def _format_duration_target(seconds):
    """Format parsed seconds for display: 840 → '14:00', 5400 → '1:30:00'."""
    s = max(0, int(round(float(seconds))))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f'{h}:{m:02d}:{sec:02d}' if h else f'{m}:{sec:02d}'


def _story_duration_prompt_block(target_seconds, avg_clip_seconds):
    """Numeric runtime block injected into the story user prompt.

    Every number is computed here — the 0.9-1.1× band, and the segment
    count derived from the ACTUAL average duration on offer. The old
    "3-4 clips per minute" prose heuristic assumed 15-20s clips; vector
    segments average 25-50s, and the model can't do the correction math.

    The demanded count is capped at _STORY_PROMPT_MAX_CLIP_ASK: "select at
    least 40 segments — do not stop early" made the model write past its
    output budget and truncate the JSON mid-array. Past the cap the band
    sentence goes too — 18 x avg cannot reach the 0.9x floor, so demanding
    "must sum to between..." alongside the capped count handed the model
    two contradictory instructions. The capped wording states the total
    the count CAN deliver and leans on the code-side top-up for the rest.
    """
    target = int(round(float(target_seconds)))
    lo = int(round(target * 0.9))
    hi = int(round(target * 1.1))
    avg = max(1.0, float(avg_clip_seconds))
    min_clips = _story_min_clip_ask(target_seconds, avg_clip_seconds)
    shown = min(min_clips, _STORY_PROMPT_MAX_CLIP_ASK)
    if min_clips > shown:
        return (
            f"TARGET TOTAL RUNTIME: {target}s ({_format_duration_target(target)}). "
            f"Your selected clips should total at least {int(round(shown * avg))}s; "
            f"the remaining runtime is completed automatically from the rest "
            f"of the menu. The average segment available runs "
            f"~{int(round(avg))}s, so select at least {shown} segments — the "
            f"strongest ones."
        )
    return (
        f"TARGET TOTAL RUNTIME: {target}s ({_format_duration_target(target)}). "
        f"Your selected clips' durations must sum to between {lo}s and {hi}s. "
        f"The average segment available runs ~{int(round(avg))}s, so "
        f"select at least {shown} segments — more if you choose shorter "
        f"ones. Do not stop early."
    )


def _clip_span_seconds(clip):
    """Duration of a story clip from its (string) timecodes."""
    start = _tc_to_seconds(clip.get('start_time'))
    end = _tc_to_seconds(clip.get('end_time'))
    return max(0.0, end - start)


def _enforce_duration_budget(clips, target_seconds, segment_vectors=None):
    """Deterministic post-hydration duration budget for story builds.

    Measures the hydrated total against the PARSED target (never the
    model's echo). Under 0.9× → top up from unused segment vectors,
    strongest narrative weight first, inserted before the closer so the
    model's arc survives; 'low'-scored segments (pruned from the menu as
    filler) are only drafted when even the 0.8× hard floor is otherwise
    unreachable. Over 1.15× → trim the weakest non-essential clips,
    never the hook or the closer. Returns ``(clips, meta)`` where meta
    carries actual_duration_seconds, duration_enforced, and — when the
    material can't reach 0.8× the target — duration_shortfall_note.
    """
    clips = [c for c in (clips or []) if isinstance(c, dict)]
    target = float(target_seconds)
    total = sum(_clip_span_seconds(c) for c in clips)
    floor = _STORY_BUDGET_FLOOR * target
    hard_floor = _STORY_BUDGET_HARD_FLOOR * target
    ceiling = _STORY_BUDGET_CEILING * target
    changed = False
    shortfall = None

    if total < floor:
        used_ids = {c.get('seg_id') for c in clips if c.get('seg_id')}
        taken = [
            (_tc_to_seconds(c.get('start_time')), _tc_to_seconds(c.get('end_time')))
            for c in clips
        ]
        pool = []
        for seg in (segment_vectors or []):
            if not isinstance(seg, dict) or seg.get('seg_id') in used_ids:
                continue
            s = _tc_to_seconds(seg.get('timecode_in'))
            e = _tc_to_seconds(seg.get('timecode_out'))
            if e <= s:
                continue
            pool.append((seg, s, e))

        def _pref(entry):
            seg, s, _e = entry
            rank = _STORY_SCORE_RANK.get(
                (seg.get('narrative_score') or 'medium').lower(), 1)
            # Nearest picked clip = topical coherence tiebreak.
            dist = min((abs(s - ts) for ts, _te in taken), default=0.0)
            return (rank, dist, seg.get('seg_id') or '')

        for seg, s, e in sorted(pool, key=_pref):
            if total >= floor:
                break
            score = (seg.get('narrative_score') or 'medium').lower()
            if score == 'low' and total >= hard_floor:
                continue  # low-tier filler only closes a hard shortfall
            dur = e - s
            if total + dur > ceiling:
                continue
            if _spans_overlap(s, e, taken):
                continue
            taken.append((s, e))
            total += dur
            changed = True
            insert_at = len(clips) - 1 if len(clips) >= 2 else len(clips)
            clips.insert(insert_at, {
                'order': 0,  # renumbered below
                'seg_id': seg.get('seg_id'),
                'title': seg.get('thread_title') or 'Untitled',
                'start_time': seg.get('timecode_in'),
                'end_time': seg.get('timecode_out'),
                'transcript': seg.get('transcript_excerpt', ''),
                'editorial_note': 'Added to reach the requested runtime.',
                'narrative_score': seg.get('narrative_score'),
                'memory_type': seg.get('memory_type'),
                'beat_type': seg.get('beat_type'),
            })
        if total < hard_floor:
            shortfall = (
                f"The build reaches {_format_duration_target(total)} of the "
                f"{_format_duration_target(target)} requested — there wasn't "
                f"enough usable material to close the gap."
            )
    elif total > ceiling:
        if len(clips) > 2:
            while total > ceiling:
                removable = [
                    (i, c) for i, c in enumerate(clips)
                    if 0 < i < len(clips) - 1
                    and (c.get('beat_type') or '').lower() not in ('hook', 'resolution')
                ]
                if not removable:
                    break
                # Weakest first: lowest narrative weight, later position on ties.
                removable.sort(key=lambda ic: (
                    -_STORY_SCORE_RANK.get(
                        (ic[1].get('narrative_score') or 'medium').lower(), 1),
                    -ic[0],
                ))
                picked = None
                for i, c in removable:
                    if total - _clip_span_seconds(c) >= floor:
                        picked = (i, c)
                        break
                if picked is None:
                    # No removal keeps the floor — drop whichever lands closest
                    # to the target (if that's an improvement), then stop.
                    i, c = min(
                        removable,
                        key=lambda ic: abs((total - _clip_span_seconds(ic[1])) - target),
                    )
                    if abs((total - _clip_span_seconds(c)) - target) < abs(total - target):
                        total -= _clip_span_seconds(c)
                        del clips[i]
                        changed = True
                    break
                i, c = picked
                total -= _clip_span_seconds(c)
                del clips[i]
                changed = True
        # Whole-clip removal can strand a build far over the ceiling: 1-2
        # clip builds never enter the loop at all, and the loop only ever
        # removes interior clips (the hook and the closer always survive)
        # — a "30 second teaser" built from two 40s segments shipped at
        # 80s. Land the residual inside the band by tightening the longest
        # clips' end_time toward the target instead. Only end_time ever
        # moves, and only downward, so each clip stays inside its own
        # segment's boundaries; nothing is shortened below
        # _STORY_MIN_CLIP_SECONDS.
        while total > ceiling:
            shrinkable = [
                c for c in clips
                if _clip_span_seconds(c) > _STORY_MIN_CLIP_SECONDS
            ]
            if not shrinkable:
                break
            c = max(shrinkable, key=_clip_span_seconds)
            span = _clip_span_seconds(c)
            new_span = max(_STORY_MIN_CLIP_SECONDS, span - (total - target))
            if new_span >= span:
                break
            start = _tc_to_seconds(c.get('start_time'))
            c['end_time'] = _seconds_to_tc(start + new_span)
            total = sum(_clip_span_seconds(cl) for cl in clips)
            changed = True

    if changed:
        for i, c in enumerate(clips, start=1):
            c['order'] = i

    meta = {
        'actual_duration_seconds': round(total, 2),
        'duration_enforced': changed,
    }
    if shortfall:
        meta['duration_shortfall_note'] = shortfall
    return clips, meta


def _fallback_entry_rank(entry):
    """Sort weight for a fallback pool entry: narrative tier first."""
    score = (entry[2].get('narrative_score') or 'medium').lower()
    return _STORY_SCORE_RANK.get(score, 1)


def _stride_pick(entries, want):
    """Pick ``want`` entries spread evenly across a chronological list.

    One pick per contiguous window, preferring the higher narrative tier
    inside each window (earliest within a tier). Windows are disjoint and
    ordered, so the picks stay chronological. Used by the fallback build
    so an oversupplied timeline is SAMPLED end-to-end instead of consumed
    greedily from the front.
    """
    if want >= len(entries):
        return list(entries)
    if want <= 0:
        return []
    picked = []
    for k in range(want):
        i0 = (k * len(entries)) // want
        i1 = max(i0 + 1, ((k + 1) * len(entries)) // want)
        window = entries[i0:i1]
        picked.append(min(window, key=lambda e: (_fallback_entry_rank(e), e[0])))
    return picked


def _fallback_story_clips(segment_vectors, target_seconds=None):
    """Deterministic last-resort clip selection from the vector menu.

    Runs when the model's story response was unusable end-to-end — nothing
    parsed (even after truncation repair) or nothing survived hydration.
    Returning clips=[] there surfaced "The AI returned 0 clips" to a user
    whose project already has a fully classified segment menu; a
    deterministic straight-line assembly is strictly better than an error.

    Selection: chronological order, high/medium narrative weight only —
    'low' filler is drafted only when the stronger tiers can't fill the
    ask. Sized to the duration target when one parsed (the caller's
    _enforce_duration_budget pass then lands the total inside the band and
    handles top-up/trim/shortfall), else ~10 segments (8 minimum). When
    the strong pool oversupplies the ask, picks are stride-sampled across
    the WHOLE timeline — greedy-from-the-front turned a 100-minute
    interview into a draft of its first 27 minutes, with the arbitrary
    cut point tagged 'resolution' (a slot the budget pass protects) — and
    the closer is reserved for the final third so the draft ends on
    actual ending material. The first clip is tagged hook and the last
    resolution, with the middle carrying context. No narrative
    re-sequencing is claimed: this is a watchable draft, not the model's
    arc.
    """
    pool = []
    for seg in (segment_vectors or []):
        if not isinstance(seg, dict) or not seg.get('seg_id'):
            continue
        s = _tc_to_seconds(seg.get('timecode_in'))
        e = _tc_to_seconds(seg.get('timecode_out'))
        if e <= s:
            continue
        pool.append((s, e - s, seg))
    pool.sort(key=lambda entry: entry[0])
    strong = [p for p in pool
              if (p[2].get('narrative_score') or 'medium').lower() != 'low']
    weak = [p for p in pool
            if (p[2].get('narrative_score') or 'medium').lower() == 'low']

    if target_seconds:
        budget = float(target_seconds)
        strong_total = sum(entry[1] for entry in strong)
        if strong and strong_total > budget:
            # More strong material than the ask: sample the timeline.
            import math
            avg = strong_total / len(strong)
            want = max(1, min(len(strong), math.ceil(budget / avg)))
            # Reserve the closer: the strongest segment in the final
            # third of the timeline (latest within its tier).
            span_lo, span_hi = strong[0][0], strong[-1][0]
            cutoff = span_lo + (span_hi - span_lo) * (2.0 / 3.0)
            tail_pool = [e for e in strong if e[0] >= cutoff] or [strong[-1]]
            closer = min(tail_pool,
                         key=lambda e: (_fallback_entry_rank(e), -e[0]))
            head = [e for e in strong if e[0] < closer[0]]
            picked = _stride_pick(head, want - 1) + [closer]
        else:
            picked, total = [], 0.0
            for entry in strong:
                if total >= budget:
                    break
                picked.append(entry)
                total += entry[1]
            if total < budget:
                # Strong tiers exhausted short of the ask — draft filler.
                for entry in weak:
                    if total >= budget:
                        break
                    picked.append(entry)
                    total += entry[1]
                picked.sort(key=lambda entry: entry[0])  # back into chronology
    else:
        # No parsed target: a ~10-segment draft (8 minimum when 'low'
        # filler has to make up the numbers), sampled across the timeline.
        picked = _stride_pick(strong, 10)
        if len(picked) < 8 and weak:
            picked.extend(weak[:8 - len(picked)])
            picked.sort(key=lambda entry: entry[0])

    clips = []
    last = len(picked) - 1
    for i, (_start, _dur, seg) in enumerate(picked):
        role = 'hook' if i == 0 else ('resolution' if i == last else 'context')
        clips.append({
            'order': i + 1,
            'seg_id': seg.get('seg_id'),
            'title': seg.get('thread_title') or 'Untitled',
            'start_time': seg.get('timecode_in'),
            'end_time': seg.get('timecode_out'),
            'transcript': seg.get('transcript_excerpt', ''),
            'editorial_note': f'ROLE: {role} — deterministic fallback pick.',
            'narrative_score': seg.get('narrative_score'),
            'memory_type': seg.get('memory_type'),
            'beat_type': role,
        })
    return clips


def _build_story_from_vectors(segment_vectors, message, project_name, profile_id=None,
                              output_language=None, target_seconds=None):
    """Build a narrative using pre-classified segment vectors as the menu of clips.

    Prioritizes "high" narrative scores; uses "episodic" segments for key moments
    and "semantic" segments for context/transitions. The model only chooses and
    orders — it does not invent timecodes — which is why this path is more reliable.
    """
    # Compact menu of available segments for the prompt. We deliberately drop
    # segments that were classified "low" (explicitly labeled as exposition/
    # filler by the vector pass) because the model won't pick them for a story
    # anyway — they just pad the prompt. This shrinks the input enough to keep
    # the clip-selection pass inside the ollama timeout on long interviews
    # (182-segment Trustees menu → ~115 after pruning; ~35% smaller prompt).
    def _tc_to_sec(tc):
        parts = tc.split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return 0

    candidates = [
        s for s in segment_vectors
        if (s.get('narrative_score') or 'medium') != 'low'
    ]
    # Fallback: if pruning wiped the menu (every segment was "low"), restore
    # the full list so the user still gets a build attempt.
    if not candidates:
        candidates = list(segment_vectors)

    # Re-order the menu by NARRATIVE WEIGHT before showing it to the model:
    # high-score segments first, then within each score tier sort by beat in
    # arc-natural order (hook → context → pressure → turn → resolution). The
    # default chronological-by-transcript layout was anchoring the model's
    # output to recording order — small local models pick clips top-to-bottom
    # and the build came out chronological even though the prompt asked for
    # narrative ordering. Hydration is seg_id-keyed, so menu order has no
    # downstream effect.
    _SCORE_RANK = {'high': 0, 'medium': 1, 'low': 2}
    _BEAT_RANK = {'hook': 0, 'context': 1, 'pressure': 2, 'turn': 3, 'resolution': 4}

    def _menu_sort_key(s):
        return (
            _SCORE_RANK.get((s.get('narrative_score') or 'medium').lower(), 1),
            _BEAT_RANK.get((s.get('beat_type') or 'context').lower(), 1),
            s.get('seg_id', ''),
        )

    ordered = sorted(candidates, key=_menu_sort_key)

    menu_lines = [
        "# Segments below are listed BY NARRATIVE WEIGHT (high score first, "
        "then hook/context/pressure/turn/resolution within each tier), "
        "NOT by recording time. Choose and order purely on story logic.",
        "",
    ]
    menu_durs = []
    for s in ordered:
        dur = _tc_to_sec(s.get('timecode_out', '0:0:0')) - _tc_to_sec(s.get('timecode_in', '0:0:0'))
        if dur > 0:
            menu_durs.append(dur)
        menu_lines.append(
            f"- {s.get('seg_id', '?')} [{s.get('timecode_in', '')}-{s.get('timecode_out', '')}] "
            f"dur={int(dur)}s "
            f"score={s.get('narrative_score', 'medium')} memory={s.get('memory_type', 'semantic')} "
            f"beat={s.get('beat_type', 'context')} thread=\"{s.get('thread_title', '')}\" "
            f"tags={','.join(s.get('theme_tags', []))} "
            f":: {(s.get('transcript_excerpt') or '')[:140]}"
        )
    menu = '\n'.join(menu_lines)

    # Numeric budget line — computed in code from the parsed target and the
    # ACTUAL menu durations. '' when no duration was asked for, keeping the
    # prompt byte-identical to pre-feature behavior. A capped ask also
    # swaps the system prompt's DURATION bullet below, so the model gets
    # ONE consistent instruction (band vs capped count).
    duration_block = ''
    duration_capped = False
    if target_seconds:
        avg_menu = (sum(menu_durs) / len(menu_durs)) if menu_durs else 30.0
        duration_block = _story_duration_prompt_block(target_seconds, avg_menu) + '\n\n'
        duration_capped = (
            _story_min_clip_ask(target_seconds, avg_menu) > _STORY_PROMPT_MAX_CLIP_ASK
        )
    duration_bullet = (_STORY_DURATION_BULLET_VECTORS_CAPPED if duration_capped
                       else _STORY_DURATION_BULLET_VECTORS)

    system_prompt = """You are a documentary story editor. You are given a menu of pre-classified interview segments and a user's brief. You select and order segments from the menu to form a coherent narrative arc.

Rules:
- ONLY select segments that appear in the menu. Do not invent new ones. Use their seg_id.
- Prioritize segments with narrative_score "high" — those are the spine.
- Use "episodic" segments (specific events, sensory) for key emotional moments.
- Use "semantic" segments (general reflection) for context and transitions between episodic beats.
- MANDATORY ARC: every selected clip fills exactly one role in this five-slot arc — hook, context, pressure, turn, resolution. Tag each clip's role in its editorial_note (e.g. "ROLE: turn — ..."). The "order" field reflects the arc, NOT the timecode.
- NON-CHRONOLOGICAL BY DEFAULT: the menu is listed by narrative weight, not by recording time. Your output ordering is independent of timecode_in. Re-sequence ruthlessly. Use chronological order only when the user's brief explicitly asks for it OR when a clip's meaning depends on a prior clip's information (cause→effect chain).
- ANTI-PATTERN CHECK: before you finalize, scan your selected clips' timecode_in values in your chosen order. If they are monotonically increasing (each clip's timecode_in is later than the previous), you have likely failed to reorder for narrative arc — re-examine and re-sequence unless the story genuinely requires the temporal sequence.
""" + duration_bullet + """
- ALWAYS include a "reasoning" field: 2-3 conversational sentences in plain language explaining what this story is really about underneath the surface, why this arc works, and what the emotional spine is. Talk like a doc editor, not a corporate brief. No bullet points.
- Always respond in valid JSON only. No markdown, no prose outside the JSON."""

    prompt = f"""PROJECT: {project_name}

USER REQUEST: {message}

{duration_block}AVAILABLE SEGMENTS (pre-classified):
{menu}

Return ONLY valid JSON in this shape:
{{
  "story_title": "suggested title",
  "target_duration": "estimated total duration",
  "reasoning": "2-3 conversational sentences on what this story is really about and why this arc works",
  "clips": [
    {{
      "order": 1,
      "seg_id": "SEG001",
      "title": "short clip title",
      "start_time": "00:00:00",
      "end_time": "00:00:00",
      "editorial_note": "why this clip is in this position"
    }}
  ]
}}"""

    system_prompt = inject_my_style(system_prompt, profile_id=profile_id)
    # Output-language directive ('' for English → byte-identical prompt).
    directive = language_directive(output_language)
    if directive:
        system_prompt = system_prompt + directive
    # Raised output budget: a duration-target build returns dozens of clip
    # objects and the provider default truncated the JSON mid-array.
    from ai_providers import ProviderError
    provider_error = None
    try:
        response = _call_ai(prompt, system_prompt, num_predict=_STORY_NUM_PREDICT)
    except ProviderError as e:
        # A mid-generation death (typically the HTTP read timeout on slow
        # hardware — the raised budget decodes for minutes) must still
        # produce a build when a classified menu exists: treat it like an
        # unusable response and let the deterministic fallback below own
        # the draft. Re-raised further down only if the fallback comes up
        # empty, so the endpoint keeps its typed provider message then.
        provider_error = e
        parsed = {'clips': []}
    else:
        parsed = _parse_json_response(response)
    if not isinstance(parsed, dict):
        parsed = {'clips': []}

    # Hydrate each clip from the segment vector by seg_id so timecodes/excerpts
    # are guaranteed to be correct, regardless of what the model echoed back.
    by_id = {s['seg_id']: s for s in segment_vectors}
    hydrated = []
    for i, clip in enumerate(parsed.get('clips', []) or []):
        if not isinstance(clip, dict):
            continue
        sid = clip.get('seg_id')
        seg = by_id.get(sid)
        if not seg:
            # Fall back to whatever the model returned, if it's plausible
            if clip.get('start_time') and clip.get('end_time'):
                hydrated.append({
                    'order': clip.get('order', i + 1),
                    'title': clip.get('title', 'Untitled'),
                    'start_time': clip.get('start_time'),
                    'end_time': clip.get('end_time'),
                    'transcript': clip.get('transcript', ''),
                    'editorial_note': clip.get('editorial_note', ''),
                })
            continue
        hydrated.append({
            'order': clip.get('order', i + 1),
            'seg_id': seg['seg_id'],
            'title': clip.get('title') or seg.get('thread_title') or 'Untitled',
            'start_time': seg['timecode_in'],
            'end_time': seg['timecode_out'],
            'transcript': seg.get('transcript_excerpt', ''),
            'editorial_note': clip.get('editorial_note', ''),
            'narrative_score': seg.get('narrative_score'),
            'memory_type': seg.get('memory_type'),
            'beat_type': seg.get('beat_type'),
        })

    result = {
        'story_title': parsed.get('story_title', 'Untitled'),
        'target_duration': parsed.get('target_duration', ''),
        'reasoning': parsed.get('reasoning', ''),
        'clips': hydrated,
    }
    if not hydrated and segment_vectors:
        # The model's response was unusable end-to-end (unparseable even
        # after repair, or every clip failed hydration). Build a
        # deterministic draft from the classified menu instead of handing
        # the endpoint a 0-clip result — that path shows the user an error
        # even though a perfectly good segment menu exists.
        result['clips'] = _fallback_story_clips(segment_vectors, target_seconds)
        if result['clips']:
            result['fallback_build'] = True
            result['reasoning'] = (
                'The AI response was unusable; built deterministically from '
                'the segment menu — strong moments sampled across the full '
                'timeline, in running order.'
            )
            if not parsed.get('story_title'):
                result['story_title'] = f'{project_name} — draft assembly'
    if provider_error is not None and not result['clips']:
        # Nothing to fall back on (degenerate menu) — surface the typed
        # provider message rather than a bare 0-clip result.
        raise provider_error
    if target_seconds and result['clips']:
        # Code-side budget: measure the hydrated total, top up from unused
        # vectors, trim overshoot. target_duration becomes the PARSED ask —
        # the model's echoed string was never validated against anything.
        result['clips'], meta = _enforce_duration_budget(
            result['clips'], target_seconds, segment_vectors,
        )
        result['target_duration'] = _format_duration_target(target_seconds)
        result.update(meta)
    return result


def _analyze_story(transcript_text, project_name, beats_target=7, soundbites_target=7,
                   language_directive_text=''):
    """Analyze transcript for documentary story structure.

    Three-pass split: local Gemma 4b can't reliably populate multiple
    timecoded arrays in a single call — the model runs out of output
    budget (num_predict) or gets confused by overlapping instructions
    and returns partial results. Dedicating one call per concern keeps
    each schema small enough to fill completely:

      1. **Soundbites** — the most valuable editorial asset; done first
         so it always gets the model's freshest attention.
      2. **Story beats + b-roll** — narrative structure and visual ideas
         share the same arc-reasoning pass.
      3. **Overview** — summary, title, themes. No timecodes needed;
         easiest for the model and fine to run last.

    ``beats_target`` and ``soundbites_target`` set the upper bound each
    pass is asked to return. Post-merge ranking trims further if the
    model overshoots.
    """
    beats_target = max(1, int(beats_target))
    soundbites_target = max(1, int(soundbites_target))

    # Pass 1 — soundbites (highest editorial value, runs first)
    soundbites_result = _analyze_story_soundbites(
        transcript_text, project_name, soundbites_target,
        language_directive_text=language_directive_text,
    )
    # Pass 2 — story beats + b-roll suggestions
    beats_result = _analyze_story_beats(
        transcript_text, project_name, beats_target,
        language_directive_text=language_directive_text,
    )
    # Pass 3 — overview (summary, title, themes — no timecodes)
    overview = _analyze_story_overview(
        transcript_text, project_name,
        language_directive_text=language_directive_text,
    )

    return {
        'summary': _first_present(overview or {}, 'summary', 'overview', 'synopsis'),
        'suggested_title': _first_present(
            overview or {}, 'suggested_title', 'title', 'working_title'
        ),
        'themes': _first_present_list(overview or {}, 'themes', 'topics', 'theme_list'),
        'story_beats': _first_present_list(
            beats_result or {}, 'story_beats', 'beats', 'narrative_beats', 'story'
        ),
        'strongest_soundbites': _first_present_list(
            soundbites_result or {}, 'strongest_soundbites', 'soundbites',
            'quotes', 'best_quotes'
        ),
        'broll_suggestions': _first_present_list(
            beats_result or {}, 'broll_suggestions', 'broll', 'bRoll', 'b_roll'
        ),
    }


def _analyze_story_soundbites(transcript_text, project_name, soundbites_target,
                              language_directive_text=''):
    """Pass 1: strongest soundbites only.

    Dedicated call so the model focuses entirely on finding the best
    quotes. This was the most common casualty when soundbites shared a
    call with story beats and b-roll — Gemma 4b would populate the first
    two arrays and truncate before reaching soundbites, or produce
    shallow generic entries when attention was split three ways.
    """
    system_prompt = (
        "You are an expert documentary film editor. Output JSON only. "
        "No markdown, no fences, no commentary, no <think> tags. "
        "Copy HH:MM:SS timecodes exactly from the transcript — "
        "do not invent or round timecodes."
    )
    if language_directive_text:
        system_prompt = system_prompt + language_directive_text
    prompt = f"""PROJECT: {project_name}

TRANSCRIPT:
{transcript_text}

Return ONLY this JSON object:
{{
  "strongest_soundbites": [
    {{"text": "the actual verbatim quote from the transcript", "start": "00:02:00", "end": "00:02:18", "why": "why this is editorially powerful"}}
  ]
}}

Find the {soundbites_target} BEST soundbites. A great soundbite is a self-contained moment that works pulled out of context: emotional, surprising, quotable, or carrying the story's thesis in a single breath.
- "text" MUST be the speaker's actual words copied from the transcript — not a paraphrase.
- "start" and "end" MUST be HH:MM:SS timecodes copied from the transcript's timecodes for that passage.
- "why" is a short phrase explaining editorial value (emotional peak, thesis statement, surprising admission, etc.).
Be ruthless — return fewer if the transcript only has fewer genuine standouts.
Return ONLY valid JSON, nothing else."""
    parsed = _parse_json_response(_call_ai(prompt, system_prompt))
    if isinstance(parsed, dict) and (
        parsed.get('strongest_soundbites') or parsed.get('soundbites')
        or parsed.get('quotes') or parsed.get('best_quotes')
    ):
        return parsed
    retry = _parse_json_response(_call_ai(
        prompt + '\n\nNO MARKDOWN. JSON ONLY. Fill the strongest_soundbites array with real quotes from the transcript.',
        system_prompt, force_json=False,
    ))
    return retry if isinstance(retry, dict) else (parsed if isinstance(parsed, dict) else {})


def _analyze_story_beats(transcript_text, project_name, beats_target,
                         language_directive_text=''):
    """Pass 2: story beats + b-roll suggestions.

    These share a pass because they reason about the same narrative arc —
    b-roll ideas are naturally anchored to the same moments the beats
    identify. Two small arrays in one call is within Gemma 4b's reliable
    output budget.
    """
    system_prompt = (
        "You are an expert documentary film editor. Output JSON only. "
        "No markdown, no fences, no commentary, no <think> tags. "
        "Copy HH:MM:SS timecodes exactly from the transcript — "
        "do not invent or round timecodes."
    )
    if language_directive_text:
        system_prompt = system_prompt + language_directive_text
    prompt = f"""PROJECT: {project_name}

TRANSCRIPT:
{transcript_text}

Return ONLY this JSON object — fill both lists:
{{
  "story_beats": [
    {{"order": 1, "label": "Opening Hook", "description": "why this moment works editorially", "start": "00:00:45", "end": "00:01:02"}}
  ],
  "broll_suggestions": [
    {{"description": "concrete visual to cut to here — be specific", "start": "00:03:10", "end": "00:03:25"}}
  ]
}}

Pick the {beats_target} BEST story beats following a documentary arc: hook, context, rising action, emotional peak, resolution, closing. Diversify across beat types — don't stack three hooks.
Suggest 3-7 b-roll moments. Each one should be a CONCRETE visual idea pinned to the timecode where it would land — describe what you'd specifically want to see, not generic filler like "nature shots" or "stock footage".
CRITICAL: Copy the exact HH:MM:SS timecodes from the transcript for start and end. Use string format like "00:02:45".
Return ONLY valid JSON, nothing else."""
    parsed = _parse_json_response(_call_ai(prompt, system_prompt))
    if isinstance(parsed, dict) and (
        parsed.get('story_beats') or parsed.get('beats')
        or parsed.get('broll_suggestions') or parsed.get('broll')
    ):
        return parsed
    retry = _parse_json_response(_call_ai(
        prompt + '\n\nNO MARKDOWN. JSON ONLY. FILL BOTH LISTS — story_beats and broll_suggestions.',
        system_prompt, force_json=False,
    ))
    return retry if isinstance(retry, dict) else (parsed if isinstance(parsed, dict) else {})


def _analyze_story_overview(transcript_text, project_name,
                            language_directive_text=''):
    """Pass 3: summary + suggested_title + themes only.

    Small schema, no timecodes — easy for Gemma 4b to fill reliably.
    Runs last because the timecoded passes carry higher editorial value.
    """
    system_prompt = (
        "You are an expert documentary film editor. Output JSON only. "
        "No markdown, no fences, no commentary, no <think> tags."
    )
    if language_directive_text:
        system_prompt = system_prompt + language_directive_text
    prompt = f"""PROJECT: {project_name}

TRANSCRIPT:
{transcript_text}

Return ONLY this JSON object:
{{
  "summary": "2-3 sentence overview of the story",
  "suggested_title": "A compelling working title",
  "themes": ["short 2-5 word phrase", "another recurring theme"]
}}

Pick 3-7 themes — short noun phrases for the recurring topics. An empty list is fine if there aren't real recurring patterns.
Return ONLY valid JSON, nothing else."""
    parsed = _parse_json_response(_call_ai(prompt, system_prompt))
    if isinstance(parsed, dict) and (
        parsed.get('summary') or parsed.get('themes')
        or parsed.get('overview') or parsed.get('synopsis')
    ):
        return parsed
    # Retry in free-form mode — see _analyze_story_soundbites for why the
    # grammar-free path rescues larger local models that blank out under
    # format='json'.
    retry = _parse_json_response(_call_ai(
        prompt + '\n\nNO MARKDOWN. NO PROSE. JSON ONLY.', system_prompt,
        force_json=False,
    ))
    return retry if isinstance(retry, dict) else (parsed if isinstance(parsed, dict) else {})


def _looks_usable_story(value):
    """Return True if a story-analyze response carries enough substance
    that we trust it without retrying.

    The previous bar — "any of the schema fields is truthy" — was wrong
    for Gemma 4b: the model often returned ``{"summary": "..."}`` with
    every list empty (a partial generation hidden behind format='json'
    closing the JSON early at the num_predict cap). That passed this
    check, skipped the retry, and produced an analysis tab with only
    a summary line and no story beats. We now require at least one of
    the substantive lists — the things the editor actually sees as
    cards — to be non-empty before declaring the response usable."""
    if not isinstance(value, dict):
        return False
    if 'error' in value and 'raw' in value:
        return False
    if _first_present_list(value, 'story_beats', 'beats', 'narrative_beats', 'story'):
        return True
    if _first_present_list(value, 'themes', 'topics', 'theme_list'):
        return True
    if _first_present_list(value, 'strongest_soundbites', 'soundbites', 'quotes', 'best_quotes'):
        return True
    if _first_present_list(value, 'broll_suggestions', 'broll', 'bRoll', 'b_roll'):
        return True
    return False


def _looks_usable_social(value):
    """Same predicate for social-analyze responses."""
    if isinstance(value, list):
        return len(value) > 0
    if not isinstance(value, dict):
        return False
    if 'error' in value and 'raw' in value:
        return False
    for key in ('social_clips', 'clips', 'social', 'reels'):
        v = value.get(key)
        if isinstance(v, list) and v:
            return True
    return False


def _analyze_social(transcript_text, project_name, clips_target=7,
                    language_directive_text=''):
    """Find social media clip opportunities in the transcript.

    ``clips_target`` sets the upper bound the model is asked to return.
    The chunked-analysis path passes a smaller per-chunk value so the
    merged total lands near the global cap; post-merge ranking trims
    further if needed.
    """
    clips_target = max(1, int(clips_target))
    system_prompt = """You are a social media content strategist who specializes in
repurposing long-form documentary interview content into viral short-form clips.
You know what performs well on Instagram Reels, TikTok, LinkedIn, and YouTube Shorts.
Always respond in valid JSON format only. No other text."""
    if language_directive_text:
        system_prompt = system_prompt + language_directive_text

    prompt = f"""Analyze this interview transcript and identify the best social media clip opportunities.

PROJECT: {project_name}

TRANSCRIPT:
{transcript_text}

Return a JSON object with this exact structure:
{{
  "social_clips": [
    {{
      "rank": 1,
      "title": "Short punchy title for the clip",
      "start": "00:00:45",
      "end": "00:01:12",
      "duration_seconds": 27,
      "text": "The key quote or moment in this clip",
      "platform": "instagram_reels",
      "why": "Why this would perform well",
      "hook": "Suggested text overlay or caption hook for the first 3 seconds",
      "hashtags": ["relevant", "hashtags"]
    }}
  ]
}}

Rules:
- Each clip should be 15-60 seconds and work as a standalone moment
- Look for: emotional peaks, surprising statements, humor, strong opinions, quotable moments
- Platform suggestions: instagram_reels, tiktok, linkedin, youtube_shorts
- Pick the {clips_target} BEST clips. Return fewer if the transcript only has fewer standouts — quality over volume.
- Rank by predicted engagement (1 = highest)

CRITICAL: The "start" and "end" values MUST be copied exactly from the [HH:MM:SS] timecodes in the transcript.
Use the HH:MM:SS format as a string, like "00:02:45". Do NOT convert to decimal numbers.
Return ONLY valid JSON, no markdown formatting."""

    response = _call_ai(prompt, system_prompt)
    parsed = _parse_json_response(response)
    if _looks_usable_social(parsed):
        return parsed
    # Retry once with a stricter prompt — same rationale as
    # :func:`_analyze_story`. We pin Gemma to the bare social_clips array
    # (no wrapper, no prose) since that's the shape the merge function
    # most reliably accepts.
    retry_system = (
        'You output JSON only. No markdown, no prose, no <think> tags. '
        'Output a single JSON array of clip objects matching the schema below.'
    )
    if language_directive_text:
        retry_system = retry_system + language_directive_text
    retry_prompt = (
        'Re-analyze the interview below for short-form social media clips. '
        'Respond ONLY with this JSON array:\n'
        '[{"rank":1,"title":"...","start":"00:00:00","end":"00:00:00",'
        '"duration_seconds":30,"text":"...","platform":"instagram_reels",'
        '"why":"...","hook":"...","hashtags":["..."]}]\n\n'
        f'PROJECT: {project_name}\n\nTRANSCRIPT:\n{transcript_text}\n\n'
        f'Pick the {clips_target} BEST clips, 15-60 seconds each, '
        'ranked by predicted engagement. Return ONLY the JSON array, '
        'nothing else.'
    )
    retry = _parse_json_response(_call_ai(retry_prompt, retry_system))
    if _looks_usable_social(retry):
        return retry
    return retry if (isinstance(retry, dict) or isinstance(retry, list)) and retry else parsed


def _parse_json_response(response_text):
    """Parse JSON from AI response, handling common formatting issues and truncation."""
    text = response_text.strip()

    # Remove markdown code fences
    if text.startswith('```'):
        text = text.split('\n', 1)[-1]
    if text.endswith('```'):
        text = text.rsplit('```', 1)[0]
    if text.startswith('json'):
        text = text[4:]

    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the response
    start = text.find('{')
    if start != -1:
        text = text[start:]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Handle truncated JSON — try closing open braces/brackets. The
        # repair is best-effort text surgery: a bug in it must degrade to
        # the tolerant error dict below, never crash the parse contract
        # (a repair IndexError once escaped to all 12 call sites).
        try:
            repaired = _repair_truncated_json(text)
        except Exception:
            repaired = None
        if repaired:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    return {
        'error': 'Failed to parse AI response',
        'raw': response_text[:500],
        'story_beats': [],
        'social_clips': [],
    }


def _repair_truncated_json(text):
    """Repair truncated JSON by salvaging the complete leading elements.

    The old version only closed open braces/brackets around whatever the
    cut left behind. On the common failure — a story/clips array truncated
    mid-object when the model runs out of output tokens — that produced
    things like ``{"clips": [{...}, {"order": 3, "title"]}``: still invalid,
    so the whole response (39 perfectly good clips included) parsed to
    nothing and the story endpoint reported "0 clips".

    Now: one scan tracks the container stack and in-string state, and each
    open container remembers where its last COMPLETE child ended (a child
    container closing, a string closing — as an array element or an object
    pair's value — or a comma terminating a primitive). On a truncated
    tail the text is cut back to the deepest open ARRAY's last element
    boundary, dropping the partial trailing element entirely, and the
    still-open structures above it are closed. The result is the valid
    leading prefix of what the model managed to emit.
    """
    text = text.rstrip()
    if not text:
        return text

    # Frames: [opening char, open index, last complete child end, after-':']
    stack = []
    in_str = False
    esc = False
    top_close = 0  # end of the last balanced TOP-LEVEL structure
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if in_str:
            if ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
                if stack:
                    top = stack[-1]
                    # In an array a closed string IS a complete element; in
                    # an object it only completes a pair as the VALUE half
                    # (cutting after a bare key would dangle `"key"`).
                    if top[0] == '[' or top[3]:
                        top[2] = i + 1
            continue
        if ch == '"':
            in_str = True
        elif ch in '{[':
            stack.append([ch, i, -1, False])
        elif ch in '}]':
            if stack:
                stack.pop()
                if stack:
                    stack[-1][2] = i + 1
                    stack[-1][3] = False
                else:
                    top_close = i + 1
        elif ch == ':':
            if stack and stack[-1][0] == '{':
                stack[-1][3] = True
        elif ch == ',':
            if stack:
                # A comma also seals primitive elements/values (numbers,
                # true/false/null) that no quote or bracket event recorded.
                stack[-1][2] = i
                stack[-1][3] = False

    if not stack and not in_str:
        # Structurally complete — only trailing junk can be wrong.
        while text and text[-1] in (',', ':', ' ', '\n', '\t'):
            text = text[:-1]
        return text

    if not stack:
        # Stack empty but in_str still True: the JSON itself is balanced
        # and the odd quote lives in trailing PROSE after it (the model
        # finished the object, then its commentary got cut mid-quoted-
        # sentence). Nothing is open to close — cut back to the last
        # balanced top-level point so the leading JSON parses. Indexing
        # the empty stack below raised IndexError and crashed every
        # _parse_json_response caller.
        return text[:top_close] if top_close else text

    # Truncated mid-structure. Prefer cutting inside the deepest open ARRAY
    # (dropping the partial tail element whole — a clip missing half its
    # fields helps nobody); with only objects open, keep the innermost
    # complete key/value pairs instead.
    cut_frame = None
    for idx in range(len(stack) - 1, -1, -1):
        if stack[idx][0] == '[':
            cut_frame = idx
            break
    if cut_frame is None:
        for idx in range(len(stack) - 1, -1, -1):
            if stack[idx][2] > 0:
                cut_frame = idx
                break
        if cut_frame is None:
            cut_frame = 0  # nothing completed anywhere → empty shell
    frame = stack[cut_frame]
    cut_at = frame[2] if frame[2] > frame[1] else frame[1] + 1
    text = text[:cut_at].rstrip()
    while text and text[-1] in (',', ':', ' ', '\n', '\t'):
        text = text[:-1]
    # Close the cut container and everything still open above it.
    closers = ''.join(
        ']' if stack[idx][0] == '[' else '}'
        for idx in range(cut_frame, -1, -1)
    )
    return text + closers
