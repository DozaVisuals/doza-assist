"""
AI Analysis for Doza Assist.
Uses Ollama (local) or Claude API for story structure and social clip suggestions.
"""

import hashlib
import os
import json
import threading
from collections import OrderedDict

import requests

from editorial_dna.injector import get_active_style_block, inject_my_style
from editorial_dna.storytelling import inject_storytelling_foundation

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
)
_NO_CLIP_SIGNALS = (
    'no clip', 'without clip', 'no markers', 'without markers',
    'just talk', 'just tell me', 'just tell me in general', 'in general',
    "don't pull", "don't find", "don't list", "don't return",
    "don't surface", 'do not pull', 'do not find', 'do not return',
    'no need for clips', 'skip the clips', 'skip clips',
)


def _is_conversational_query(message: str, segments=None) -> bool:
    """Return True when the editor's message looks like discussion (themes,
    story, character, craft, opinion, chitchat) rather than clip extraction.

    Heuristic order:
      1. Empty / whitespace → conversational (let model handle gracefully)
      2. Explicit "no clips" instruction → conversational, hard signal
      3. Mentions a known speaker name → extractive (route to chunk search
         which actually scans the transcript for that speaker's content)
      4. Starts with an extractive verb → extractive
      5. Default → conversational (matches the orientation: default to talk)

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
            import re
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
    return True


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
            "1. <storytelling_foundation> describes how this editor builds stories based on their past work\n"
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
    'a marker only when a specific moment directly anchors your point.'
)


def _build_chat_messages(message, history, project_name, segments,
                        formatted, analysis_block, relevant_excerpts_block,
                        profile_id, labeled_sections=None, speaker_names=None,
                        include_final_reminder=True):
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
            messages.append({'role': role, 'content': content})

    if include_final_reminder:
        # Recency-correct contract restatement: after the transcript AND the
        # history, immediately before generation. Only the FINAL turn carries
        # it — replayed history turns come from chat_history, which stores the
        # raw message, so the KV prefix stays stable across turns.
        messages.append({'role': 'user', 'content': f'{message}\n\n{_FINAL_REMINDER}'})
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
                          speaker_names=None):
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
    phrases, words = _extract_query_keywords(message)
    theme_phrases = _collect_theme_phrases_from_vectors(segment_vectors, message)
    tfidf_hits = []
    if paragraph_index is not None:
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
            )
        paragraphs = _build_paragraphs(transcript)
        return _chat_layer2_chunked_search(
            paragraphs, message, history, project_name,
            phrases, words, profile_id, analysis,
            segment_vectors=segment_vectors, theme_phrases=theme_phrases,
            tfidf_hits=tfidf_hits, speaker_names=speaker_names,
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
    )
    num_ctx = _estimate_layer1_num_ctx(formatted)
    response = _call_ai_chat(system_message, messages, num_ctx=num_ctx)
    response = _strip_trailing_repetition(response)
    cleaned = _clean_chat_response(response)
    cleaned = _validate_clip_markers_in_text(cleaned, segments)
    # Skip clip salvage when the editor's question is conversational
    # (themes, story, craft, chitchat, or explicit "no clips"). Forcing
    # markers into a discussion answer breaks the orientation contract.
    if not _is_conversational_query(message, segments=segments):
        cleaned = _salvage_clips_if_missing(
            cleaned, formatted, segments, num_ctx=num_ctx,
            matched_paragraphs=matched, user_message=message,
        )
    # Enforce explicit clip count from the user message. Gemma 4B
    # routinely ignores "1 clip" / "one more" / "another" and emits 2-3.
    # Trim server-side so the user sees what they asked for.
    cleaned = _enforce_clip_count(cleaned, _detect_explicit_clip_count(message))
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
                                 speaker_names=None):
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
    if paragraph_index is not None:
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
    )

    pieces = []
    num_ctx = _estimate_layer1_num_ctx(formatted)
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
            count = 0
            pos = len(tail) - plen
            while pos >= 0 and tail[pos:pos + plen] == pat:
                count += 1
                pos -= plen
            if count >= _REP_THRESHOLD:
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

    # Strip any trailing repetition the model produced before we cut it off.
    full = ''.join(pieces)
    full = _strip_trailing_repetition(full)
    cleaned = _clean_chat_response(full)
    cleaned = _validate_clip_markers_in_text(cleaned, segments)
    # Salvage pass mirrors the non-streaming path. Adds latency only when
    # the model's first attempt produced zero markers — most calls return
    # immediately. Streamed clients see a brief pause after the prose
    # finishes, then the marker block lands as part of the final message.
    # Skipped on conversational queries — see _is_conversational_query.
    if not _is_conversational_query(message, segments=segments):
        cleaned = _salvage_clips_if_missing(
            cleaned, formatted, segments, num_ctx=num_ctx,
            matched_paragraphs=matched, user_message=message,
        )
    # Enforce explicit clip count from the user message — same defense
    # the non-streaming path applies. See _enforce_clip_count.
    cleaned = _enforce_clip_count(cleaned, _detect_explicit_clip_count(message))
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


def _detect_explicit_clip_count(message):
    """Parse an explicit clip count from a user message, or return ``None``.

    Pattern coverage matches the prompt's CLIP COUNT rules:
    - "1 clip", "2 clips", "3 clips" (digits)
    - "one clip", "two clips" through "five clips" (words)
    - "give me one", "the best one", "just one" → 1
    - "one more", "another (clip|one)" → 1
    - "a few more", "some more" → 3 (upper bound of "a few")

    Returns ``None`` if no explicit count is detectable — the model uses
    its judgment in that case (1-4 typical per the prompt).
    """
    if not message:
        return None
    import re
    msg = message.lower().strip()

    # "1 clip" / "2 clips" / etc.
    m = re.search(r'\b(\d+)\s+clips?\b', msg)
    if m:
        return max(1, int(m.group(1)))

    # "1 more" / "2 more" / "3 more" — digit + "more" (with optional "clip"
    # after). The user means N additional clips. Same parse as "1 clip"
    # but with "more" as the noun.
    m = re.search(r'\b(\d+)\s+more(?:\s+clips?)?\b', msg)
    if m:
        return max(1, int(m.group(1)))

    # "find me 3" / "give me 2" / "pull 4" — bare digit after a request verb.
    m = re.search(
        r'\b(?:find|give|pull|show|get|make)\s+(?:me\s+)?(\d+)\b',
        msg,
    )
    if m:
        return max(1, int(m.group(1)))

    word_to_num = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}
    m = re.search(r'\b(one|two|three|four|five)\s+clips?\b', msg)
    if m:
        return word_to_num[m.group(1)]

    # Word-form + "more": "two more", "three more clips", etc.
    m = re.search(r'\b(one|two|three|four|five)\s+more(?:\s+clips?)?\b', msg)
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
        r'|the\s+(?:best|strongest)\s+(?:one|clip|moment|single))\b',
        msg,
    ):
        return 1

    # "A few more" / "some more" — cap at 3.
    if re.search(r'\b(?:a\s+few(?:\s+more)?|some\s+more|several)\b', msg):
        return 3

    return None


def _enforce_clip_count(text, target):
    """Trim ``text`` so it contains at most ``target`` [CLIP:] markers.

    Strips excess markers and any prose that immediately precedes them
    (single-line "preamble" prose introducing each excess clip), but
    preserves the through-line opener and the first ``target`` clips.
    Used to defend against Gemma 4 ignoring the explicit count rule in
    the prompt — the prompt asks for "EXACTLY 1" but the model emits
    2-3 anyway, so we enforce server-side.
    """
    if target is None or target < 1 or not text:
        return text
    import re
    clips = list(re.finditer(r'\[CLIP:[^\]]*\]', text))
    if len(clips) <= target:
        return text
    # Cut at the end of the target-th marker. Anything after gets the
    # CLIP markers stripped (so any salvageable prose stays, but no
    # excess clip cards render). In practice the tail is almost always
    # just more markers + their preamble lines, so the trim is clean.
    cut = clips[target - 1].end()
    head = text[:cut]
    tail = text[cut:]
    tail = re.sub(r'\[CLIP:[^\]]*\]', '', tail)
    # Drop any leftover "Here's another:" / "Plus:" connector lines that
    # introduced the excess clips. A connector is a short line ending in
    # a colon with nothing meaningful on either side.
    tail = re.sub(r'\n\s*[A-Z][^.\n]{0,40}:\s*\n', '\n', tail)
    result = (head + tail).rstrip()
    return result


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
    """Drop essay scaffolding small models emit despite the prompt
    forbidding it: ``> blockquoted`` (and often hallucinated) speaker
    quotes, numbered list items with section headers like ``1. The Aha
    Moment: explanation``, and ``(hypothetical / fabricated / illustrative
    / paraphrased ...)`` confessions of fabrication.

    Defense-in-depth — the prompt forbids all three. This catches what
    Gemma 4B emits when it ignores the prompt anyway.
    """
    if not text:
        return text
    import re

    # 1. Markdown blockquote lines starting with "> ". Drop the whole
    # line; the clip card already shows the speaker text.
    text = re.sub(r'^[ \t]*>\s+.*$\n?', '', text, flags=re.MULTILINE)

    # 2. Numbered list items that double as section headers — the pattern
    # is "1. Capitalized Phrase: explanation". The phrase before the colon
    # can contain quoted titles like 'The "Aha!" Moment', so the char
    # class is lenient (anything that's not the colon or newline).
    text = re.sub(
        r'^[ \t]*\d+\.\s+[A-Z][^:\n]{0,80}:\s*[^\n]*\n?',
        '',
        text,
        flags=re.MULTILINE,
    )

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
    return pattern.sub('', text)


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

    # 4. Follow-up offers — "If you'd like, I can pull…" / "Let me know if…"
    # / "Would you like me to…" / "I can also…" / "Want me to…". Gemma
    # signals availability for further work; the user just wants the answer.
    # Line-anchored so a whole follow-up paragraph drops cleanly. In the
    # rare case the trigger shares a line with real content we lose the
    # content too — acceptable, the user came for the answer not the offer.
    text = re.sub(
        r'^[ \t]*(?:If you(?:[’\']?d| would| want)? like'
        r'|If you want'
        r'|Let me know if'
        r'|Would you like(?: me)?'
        r'|Want me to'
        r'|I can (?:also |further |)?(?:pull|provide|find|offer|elaborate|'
        r'expand|dig|share|extract|analyze))[^\n]*\n?',
        '',
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # 5. Section sub-headers between the opening sentence and the first
    # [CLIP:] marker. Pattern: a paragraph that is one short Capitalized
    # phrase followed by ": " and a sentence — e.g. "Natural Resilience:
    # This is seen in…". The new prompt forbids these but Gemma still
    # leaks them; strip the leading label-and-prose pair, keep nothing.
    # Only applied to the prose ABOVE the first marker so legitimate
    # post-marker prose (rare, but possible) is left alone.
    first_clip = re.search(r'\[CLIP\s*:', text)
    if first_clip:
        head = text[:first_clip.start()]
        tail = text[first_clip.start():]
        # Match a line that starts with 1-5 Capitalized words then ":"
        # then more text on the same line. Conservative width to avoid
        # hitting natural prose like "But here's the thing: …".
        sub_header_re = re.compile(
            r'^[ \t]*[A-Z][A-Za-z]+(?:[ /\-][A-Z][A-Za-z]+){0,4}\s*:\s+[A-Z][^\n]*\n?',
            re.MULTILINE,
        )
        head = sub_header_re.sub('', head)
        text = head + tail

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
    for opener, _closer in _REASONING_TAG_PAIRS:
        opener_re = re.escape(opener)
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


_CLIP_COUNT_PATTERNS = [
    (r'\b(?:find|give|pull|get|show|grab)\s+(?:me\s+)?(\d+)\s+clips?\b', None),
    (r'\b(\d+)\s+clips?\b', None),
    (r'\bthe\s+(?:single\s+)?best\s+(?:one|clip)\b', 1),
    (r'\bjust\s+one\b|\bgive\s+me\s+one\b|\bonly\s+one\b', 1),
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
                                matched_paragraphs=None, user_message=None):
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
        validated = _validate_clip_markers_in_text(normalized, segments)

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


def _validate_clip_markers_in_text(text, segments, grace_seconds=5.0):
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
        # Title-anchor check (best-effort; never raises).
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

    # Walk lines so we strip both the marker AND the editorial sentence that
    # rides along with it. A "moment doesn't exist" line with explanatory
    # prose underneath would just confuse the user.
    cleaned_lines = []
    for line in text.split('\n'):
        bad_marker = False
        for m in marker_re.finditer(line):
            if not _is_valid(m):
                bad_marker = True
                break
        if not bad_marker:
            cleaned_lines.append(line)
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
    """Remove degenerate trailing repetition from model output."""
    if len(text) < 30:
        return text
    for plen in range(4, 60):
        pat = text[-plen:]
        count = 0
        pos = len(text) - plen
        while pos >= 0 and text[pos:pos + plen] == pat:
            count += 1
            pos -= plen
        if count >= 4:
            cut = len(text) - (count * plen)
            return text[:cut].rstrip()
    return text


def _clean_chat_response(text):
    """Strip markdown artifacts and emoji from chat responses."""
    import re
    text = text.strip()
    # Normalize variant CLIP markers BEFORE markdown stripping so the
    # canonical form survives downstream regexes.
    text = _normalize_clip_markers(text)
    # Wrap stray prose timecode ranges — small models in STORY CONSULTING
    # mode often write "(00:12:34 - 00:13:00)" instead of the [CLIP:]
    # marker. Without this pass those moments render as static timecode
    # pills rather than playable clip cards, which is the regression the
    # user reported after the Gemma 4 upgrade.
    text = _auto_wrap_timecode_ranges(text)
    # Remove markdown headers
    text = re.sub(r'^#{1,4}\s*', '', text, flags=re.MULTILINE)
    # Remove bold/italic markdown (but never inside a CLIP marker — the title
    # may legitimately contain asterisks, and stripping them could also chew
    # up the marker itself if the model wrapped it in **bold**).
    text = _strip_markdown_outside_clips(text)
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
    # Strip "[HH:MM:SS]" single-timecode brackets the frontend would
    # otherwise render as standalone chips above the clip cards. The model
    # emits these as a "table of contents" preamble — they aren't useful,
    # the clip cards already show their own timecodes. Range-form
    # "[HH:MM:SS - HH:MM:SS]" is wrapped to a CLIP marker by
    # _auto_wrap_timecode_ranges above; this only handles the leftover
    # single-timecode brackets that don't form a range.
    text = re.sub(
        r'\[(?!\s*[Cc][Ll][Ii][Pp]\b)\s*\d{1,2}:\d{2}(?::\d{2})?\s*\]',
        '',
        text,
    )
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
        return f'[CLIP: start={start} end={end} title="{title}"]'

    return candidate_re.sub(_rewrite, text)


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
        # Only wrap if end is strictly after start — avoids turning an
        # unrelated pair like "00:00:52 - 00:00:35" (backwards) into a
        # broken clip card. _tc_to_seconds is tolerant of both MM:SS and
        # HH:MM:SS forms.
        if _tc_to_seconds(end) <= _tc_to_seconds(start):
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
        "\n\nPRE-ANALYZED MOMENTS (real timecodes — prefer citing from this list "
        "when a question matches):\n" + "\n".join(lines)
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
    # Digits followed by clip(s)/moment(s)/soundbite(s)/quote(s)
    m = _re.search(r'\b(\d{1,2})\s+(?:great\s+|strong\s+|best\s+)?(?:clip|moment|soundbite|quote|excerpt)s?\b', msg)
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

    parts = []
    for cand in candidates:
        start_tc = _seconds_to_tc(cand['start_sec'])
        end_tc = _seconds_to_tc(cand['end_sec'])
        raw_title = (cand.get('title') or 'Moment').strip()
        # Strip matching wrapping quotes the model sometimes includes
        # (e.g. "Moment"), then neutralize any remaining internal " so
        # it can't break our marker's own quoting.
        if len(raw_title) >= 2 and raw_title[0] == raw_title[-1] and raw_title[0] in ('"', "'"):
            raw_title = raw_title[1:-1].strip()
        title = _format_clip_title(raw_title.replace('"', "'"))
        why = (cand.get('why') or '').strip().replace('"', "'")
        if why:
            parts.append(f'[CLIP: start={start_tc} end={end_tc} title="{title}" note="{why}"]')
        else:
            parts.append(f'[CLIP: start={start_tc} end={end_tc} title="{title}"]')
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
                                          speaker_names=None):
    """Conversational synthesis on long interviews — the divert from
    chunked clip search when the editor's question is discussion-style.

    Builds a compact context (summary + analysis index + selected clips,
    no transcript), runs it through the standard conversational LLM via
    _build_chat_messages so the orientation paragraph and framing apply,
    and returns the prose response. Skips the clip-salvage post-processor
    so a clean conversational answer doesn't get clips bolted on.
    """
    context = _build_synthesis_context_block(project_name, segments, analysis, labeled_sections, speaker_names)
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
    )
    num_ctx = max(8192, _estimate_layer1_num_ctx(context))
    response = _call_ai_chat(system_message, messages, num_ctx=num_ctx)
    response = _strip_trailing_repetition(response)
    cleaned = _clean_chat_response(response)
    cleaned = _validate_clip_markers_in_text(cleaned, segments)
    # Deliberately skip _salvage_clips_if_missing — this path is only
    # reached on conversational queries.
    return cleaned


def _chat_layer2_conversational_synthesis_stream(message, history, project_name, segments,
                                                  analysis, profile_id, labeled_sections,
                                                  speaker_names=None):
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
        )
    except Exception as e:
        print(f"[chat-stream] conversational synthesis failed: {e}")
        reply = ''
    yield ('done', reply)


def _chat_layer2_chunked_search(paragraphs, message, history, project_name,
                                phrases, words, profile_id, analysis,
                                segment_vectors=None, theme_phrases=None,
                                tfidf_hits=None, speaker_names=None):
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
    # "give me 3"). Falls back to _CHAT_TOP_K_CLIPS when the user didn't
    # specify a number. The pre-rerank candidate pool is kept generous so
    # the global rerank still has range to pick from, even when the final
    # output is just one clip.
    user_count = _parse_user_clip_count(message)
    final_top_k = user_count if user_count is not None else _CHAT_TOP_K_CLIPS
    pool_top_k = max(final_top_k * 3, _CHAT_TOP_K_CLIPS * 3)

    top = _aggregate_chunk_candidates(all_candidates, top_k=pool_top_k)
    # Cross-chunk synthesis pass: per-chunk scores aren't comparable across
    # chunks (each model call sees only its own window), so a final low-temp
    # rerank decides the global best. Falls back to the local-score top-K
    # if the synthesis call fails — better to ship the original aggregator's
    # answer than to drop everything.
    top = _rerank_candidates_globally(top, message, top_k=final_top_k)
    return _format_clip_cards_from_candidates(top)


def _chat_layer2_chunked_search_stream(paragraphs, message, history, project_name,
                                       phrases, words, profile_id, analysis,
                                       segment_vectors=None, theme_phrases=None,
                                       tfidf_hits=None, speaker_names=None):
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

    user_count = _parse_user_clip_count(message)
    final_top_k = user_count if user_count is not None else _CHAT_TOP_K_CLIPS
    pool_top_k = max(final_top_k * 3, _CHAT_TOP_K_CLIPS * 3)

    yield ('progress', 'Picking the best moments…')
    top = _aggregate_chunk_candidates(all_candidates, top_k=pool_top_k)
    top = _rerank_candidates_globally(top, message, top_k=final_top_k)
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
        except (json.JSONDecodeError, ValueError):
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


def _synthesize_overall_summary(summaries, titles, project_name, warnings=None):
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
                       segment_vectors=None, progress_callback=None):
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

    for i, chunk in enumerate(chunks):
        chunk_text = _format_segments_for_ai(chunk['segments'])
        range_label = f"{_seconds_to_tc(chunk['start_seconds'])}-{_seconds_to_tc(chunk['end_seconds'])}"
        chunk_label = f"{project_name} · part {i+1}/{len(chunks)} ({range_label})"
        if analysis_type in ('story', 'all'):
            step += 1
            _emit(step, total_steps, f"chunk {i+1}/{chunk_count}: story beats")
            try:
                _merge_story_chunk(
                    accum,
                    _analyze_story(
                        chunk_text, chunk_label,
                        beats_target=per_chunk_target,
                        soundbites_target=per_chunk_target,
                    ),
                    is_first_chunk=(i == 0),
                )
            except Exception as e:
                # Missing/invalid API key is permanent — re-raise so the
                # whole analysis aborts with the clear message instead of
                # silently producing empty results across every chunk.
                from ai_providers import ProviderError
                if isinstance(e, ProviderError) and e.code in ('missing_key', 'invalid_key'):
                    raise
                print(f"[analyze] story chunk {i+1}/{len(chunks)} failed: {e}")
                accum['analysis_warnings'].append(
                    f'Story analysis failed on chunk {i+1}/{len(chunks)} '
                    f'({type(e).__name__}). Some beats / themes / soundbites '
                    'may be missing for that section.'
                )
        if analysis_type in ('social', 'all'):
            step += 1
            _emit(step, total_steps, f"chunk {i+1}/{chunk_count}: social clips")
            try:
                _merge_social_chunk(
                    accum,
                    _analyze_social(
                        chunk_text, chunk_label,
                        clips_target=per_chunk_target,
                    ),
                )
            except Exception as e:
                from ai_providers import ProviderError
                if isinstance(e, ProviderError) and e.code in ('missing_key', 'invalid_key'):
                    raise
                print(f"[analyze] social chunk {i+1}/{len(chunks)} failed: {e}")
                accum['analysis_warnings'].append(
                    f'Social-clip analysis failed on chunk {i+1}/{len(chunks)} '
                    f'({type(e).__name__}). Some clip suggestions may be '
                    'missing for that section.'
                )

    step += 1
    _emit(step, total_steps, "synthesizing summary")
    overall = _synthesize_overall_summary(
        accum.pop('_chunk_summaries', []),
        accum.pop('_chunk_titles', []),
        project_name,
        warnings=accum['analysis_warnings'],
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
                                segments=None):
    """One-shot analysis path for short interviews."""
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


def _call_ai(prompt, system_prompt="", task_type="analysis", force_json=True):
    """Single-prompt generation through the active provider.

    ``task_type`` selects the model when the provider tiers (Anthropic uses
    Opus for ``profile_creation``, Sonnet otherwise; Ollama ignores it).
    Editorial DNA's classifier passes ``"analysis"``; My Style synthesis
    passes ``"profile_creation"``.

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
        timeout = recommended_analysis_timeout()
    except Exception:
        timeout = 600
    return provider.generate(
        system_prompt, prompt, task_type=task_type, force_json=force_json,
        timeout=timeout,
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


def build_story(transcript, message, project_name="Interview", segment_vectors=None, profile_id=None):
    """
    Build a narrative sequence from the transcript based on the user's description.
    Returns a dict with story_title, target_duration, and clips array.

    If segment_vectors is provided, the model is given the pre-classified segments
    instead of having to re-analyze the raw transcript. This makes builds faster and
    much more consistent across runs.
    """
    if segment_vectors:
        return _build_story_from_vectors(segment_vectors, message, project_name, profile_id=profile_id)

    formatted = _format_transcript_for_ai(transcript)

    system_prompt = """You are a story editor building a narrative sequence from interview transcript footage. The user will describe what kind of story or edit they want. Your job is to select and order clips from the transcript that form a coherent narrative.

Rules:
- Select clips that build a clear narrative arc: hook, rising action, emotional peak, resolution
- MANDATORY ARC: every selected clip fills exactly one role in this five-slot arc — hook, context, pressure, turn, resolution. Tag the role in editorial_note (e.g. "ROLE: turn — ..."). The "order" field reflects the arc, NOT the timecode.
- NON-CHRONOLOGICAL BY DEFAULT: the transcript below is presented in recording order; your output MUST NOT preserve that order unless the user explicitly requested chronological OR a clip's meaning depends on temporal sequence (cause→effect chain).
- ANTI-PATTERN CHECK: if your selected clips' start_time values are monotonically increasing in your chosen order, you have likely defaulted to chronological — re-examine and re-sequence.
- Each clip should be 5-30 seconds long unless the moment requires more breathing room
- DURATION IS CRITICAL: If the user requests a specific duration (e.g. "4 minute story"), you MUST hit that target. Calculate the total duration of all clips you select by adding up (end_time - start_time) for each clip. Aim for roughly 3-4 clips per minute. For a 4-minute story, that means 12-16 clips totaling approximately 3:30-4:30 of content. If your first selection is too short, add more clips.
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

TRANSCRIPT (presented in recording order — re-sequence freely for narrative arc):
{formatted}

Return ONLY valid JSON. No markdown, no extra text."""

    system_prompt = inject_my_style(system_prompt, profile_id=profile_id)
    response = _call_ai(prompt, system_prompt)
    return _parse_json_response(response)


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


def _build_story_from_vectors(segment_vectors, message, project_name, profile_id=None):
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
    for s in ordered:
        dur = _tc_to_sec(s.get('timecode_out', '0:0:0')) - _tc_to_sec(s.get('timecode_in', '0:0:0'))
        menu_lines.append(
            f"- {s.get('seg_id', '?')} [{s.get('timecode_in', '')}-{s.get('timecode_out', '')}] "
            f"dur={int(dur)}s "
            f"score={s.get('narrative_score', 'medium')} memory={s.get('memory_type', 'semantic')} "
            f"beat={s.get('beat_type', 'context')} thread=\"{s.get('thread_title', '')}\" "
            f"tags={','.join(s.get('theme_tags', []))} "
            f":: {(s.get('transcript_excerpt') or '')[:140]}"
        )
    menu = '\n'.join(menu_lines)

    system_prompt = """You are a documentary story editor. You are given a menu of pre-classified interview segments and a user's brief. You select and order segments from the menu to form a coherent narrative arc.

Rules:
- ONLY select segments that appear in the menu. Do not invent new ones. Use their seg_id.
- Prioritize segments with narrative_score "high" — those are the spine.
- Use "episodic" segments (specific events, sensory) for key emotional moments.
- Use "semantic" segments (general reflection) for context and transitions between episodic beats.
- MANDATORY ARC: every selected clip fills exactly one role in this five-slot arc — hook, context, pressure, turn, resolution. Tag each clip's role in its editorial_note (e.g. "ROLE: turn — ..."). The "order" field reflects the arc, NOT the timecode.
- NON-CHRONOLOGICAL BY DEFAULT: the menu is listed by narrative weight, not by recording time. Your output ordering is independent of timecode_in. Re-sequence ruthlessly. Use chronological order only when the user's brief explicitly asks for it OR when a clip's meaning depends on a prior clip's information (cause→effect chain).
- ANTI-PATTERN CHECK: before you finalize, scan your selected clips' timecode_in values in your chosen order. If they are monotonically increasing (each clip's timecode_in is later than the previous), you have likely failed to reorder for narrative arc — re-examine and re-sequence unless the story genuinely requires the temporal sequence.
- DURATION IS CRITICAL: If the user requests a specific duration (e.g. "4 minute story"), you MUST hit that target. Calculate the total duration of all clips you select by adding up (end_time - start_time) for each clip. Each segment in the menu shows its timecode range — use that to calculate duration. Aim for roughly 3-4 clips per minute of requested duration. For a 4-minute story, select enough clips to total approximately 3:30-4:30 of content. If your first selection is too short, add more clips. If too long, trim or remove clips.
- ALWAYS include a "reasoning" field: 2-3 conversational sentences in plain language explaining what this story is really about underneath the surface, why this arc works, and what the emotional spine is. Talk like a doc editor, not a corporate brief. No bullet points.
- Always respond in valid JSON only. No markdown, no prose outside the JSON."""

    prompt = f"""PROJECT: {project_name}

USER REQUEST: {message}

AVAILABLE SEGMENTS (pre-classified):
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
    response = _call_ai(prompt, system_prompt)
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

    return {
        'story_title': parsed.get('story_title', 'Untitled'),
        'target_duration': parsed.get('target_duration', ''),
        'reasoning': parsed.get('reasoning', ''),
        'clips': hydrated,
    }


def _analyze_story(transcript_text, project_name, beats_target=7, soundbites_target=7):
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
    )
    # Pass 2 — story beats + b-roll suggestions
    beats_result = _analyze_story_beats(
        transcript_text, project_name, beats_target,
    )
    # Pass 3 — overview (summary, title, themes — no timecodes)
    overview = _analyze_story_overview(transcript_text, project_name)

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


def _analyze_story_soundbites(transcript_text, project_name, soundbites_target):
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


def _analyze_story_beats(transcript_text, project_name, beats_target):
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


def _analyze_story_overview(transcript_text, project_name):
    """Pass 3: summary + suggested_title + themes only.

    Small schema, no timecodes — easy for Gemma 4b to fill reliably.
    Runs last because the timecoded passes carry higher editorial value.
    """
    system_prompt = (
        "You are an expert documentary film editor. Output JSON only. "
        "No markdown, no fences, no commentary, no <think> tags."
    )
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


def _analyze_social(transcript_text, project_name, clips_target=7):
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

        # Handle truncated JSON — try closing open braces/brackets
        repaired = _repair_truncated_json(text)
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
    """Attempt to repair truncated JSON by closing open structures."""
    # Strip trailing whitespace
    text = text.rstrip()

    # If we're mid-string, close it: find if we have unmatched quote
    in_str = False
    esc = False
    last_quote = -1
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if ch == '\\' and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            last_quote = i

    # If we're inside an open string, truncate to last clean point before it
    if in_str and last_quote > 0:
        # Close the string and trim any trailing partial value
        text = text[:last_quote + 1]
        # We may now have something like  "key": "value  — close quote
        if not text.endswith('"'):
            text += '"'

    # Remove trailing commas, colons, or partial tokens
    text = text.rstrip()
    while text and text[-1] in (',', ':', ' ', '\n', '\t'):
        text = text[:-1]

    # Count open braces/brackets and close them
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape_next = False

    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            open_braces += 1
        elif ch == '}':
            open_braces -= 1
        elif ch == '[':
            open_brackets += 1
        elif ch == ']':
            open_brackets -= 1

    # Close any remaining open structures
    text += ']' * max(0, open_brackets)
    text += '}' * max(0, open_braces)

    return text
