"""
AI Analysis for Doza Assist.
Uses Ollama (local) or Claude API for story structure and social clip suggestions.
"""

import os
import json
import requests

from editorial_dna.injector import inject_my_style


def chat_about_transcript(transcript, message, history=None, project_name="Interview",
                          analysis=None, profile_id=None, segment_vectors=None,
                          paragraph_index=None):
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
        paragraphs = _build_paragraphs(transcript)
        return _chat_layer2_chunked_search(
            paragraphs, message, history, project_name,
            phrases, words, profile_id, analysis,
            segment_vectors=segment_vectors, theme_phrases=theme_phrases,
            tfidf_hits=tfidf_hits,
        )

    formatted = _format_transcript_for_ai(transcript)
    relevant_excerpts_block = ''
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
        relevant_excerpts_block = _build_relevant_excerpts_block(matched)
    analysis_block = _build_chat_analysis_index(analysis)

    system_prompt = f"""You are an expert editorial consultant embedded in a documentary and interview editing tool called Doza Assist. You have the full transcript of the project "{project_name}" loaded in context.

═══════════════════════════════════════════════════════════════
HARD OUTPUT CONTRACT — THIS IS THE ONLY FORMAT THAT WORKS:

Every response MUST cite 2-5 specific moments from the transcript using this EXACT marker:

  [CLIP: start=HH:MM:SS end=HH:MM:SS title="short descriptive title"]

The frontend renders these markers as playable clip cards with Play and Add Clip buttons. If you don't emit them, the user sees nothing but plain text and the task has failed.

GROUNDING RULE — NO EXCEPTIONS:
- Every timecode in a [CLIP:] marker MUST come from the TRANSCRIPT segment markers below (lines like [00:05:12-00:05:28] Speaker: text) or from the PRE-ANALYZED MOMENTS list (if present).
- NEVER invent timecodes. NEVER fabricate quotes. If you can't find a relevant moment in the data, say "I don't see that in the transcript" — do not make something up.
- Do NOT include "> " blockquotes, quoted speaker text, or paraphrased lines in your reply. The clip card already shows the exact words; duplicating them in prose is wasted output.

HOW TO RESPOND:
1. Pick 2-5 moments from the data that best answer the user's question.
2. For each, emit a [CLIP:] marker with start, end, and a 2-6 word title.
3. Add ONE short sentence before or after each marker saying WHY it answers the question.
4. That's the whole reply. No preamble, no "here are some suggestions", no numbered headers.

This applies to every question — "what's the emotional arc", "pull the best clip", "find moments about X", anything. If the user asks about the arc, pick 3-4 clips that trace that arc. If they ask for the best clip, pick 1-3 [CLIP:] markers.
═══════════════════════════════════════════════════════════════

Personality: seasoned doc editor leaning over the desk, warm, opinionated, short. Use contractions. No filler, no hedging.

Modes (which kinds of clips to prioritize):
1. CLIP DISCOVERY — user asks for moments, clips, soundbites, quotes. Pick the strongest standalone moments.
2. STORY CONSULTING — user asks about arc, theme, structure, character. Pick moments that anchor each beat you describe.
3. SOCIAL — user asks about reels/shorts/social. Pick 15-60s moments with strong hooks.

Clip constraints:
- Minimum 5 seconds, typical 10-60 seconds.
- Span multiple transcript segments if needed to capture a complete thought.
- Title under 8 words, no quotes around it.

Other rules:
- Keep prose between markers to 1 sentence max. Never explain things without a marker.
- No emojis, markdown headers, bullet lists, or blockquotes.
- If you truly can't ground an answer, say so in one sentence — don't pad.

TRANSCRIPT:
{formatted}{analysis_block}{relevant_excerpts_block}

═══════════════════════════════════════════════════════════════
FINAL REMINDER — DO NOT SKIP:

Your reply MUST contain 2-5 [CLIP: start=HH:MM:SS end=HH:MM:SS title="..."] markers using real timecodes copied from the transcript segment markers above (and from the PRE-ANALYZED MOMENTS list if present). Without these markers the user sees nothing but plain text and the task has failed. Do NOT write a prose summary in place of clip markers — cite specific moments with [CLIP:] markers. This applies to every question, including synthesis questions like "what's the most revealing thing" or "what did they say about X" — answer those with 2-5 cited moments, not a paragraph.
═══════════════════════════════════════════════════════════════"""

    # Flatten conversation history (Ollama generate takes single prompt)
    conversation = ""
    if history:
        for msg in history[-6:]:  # Last 6 for speed
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'user':
                conversation += f"\nUser: {content}"
            else:
                conversation += f"\nAssistant: {content}"

    prompt = f"{conversation}\nUser: {message}\nAssistant:"

    system_prompt = inject_my_style(system_prompt, profile_id=profile_id)
    response = _call_ai_chat(prompt, system_prompt)
    cleaned = _clean_chat_response(response)
    return _validate_clip_markers_in_text(cleaned, segments)


def _call_ai_chat_stream(prompt, system_prompt=""):
    """Streaming variant of :func:`_call_ai_chat` that yields token chunks.

    Yields ``str`` pieces as they arrive from Ollama. On Claude fallback the
    HTTP API returns the whole reply at once (we don't use the SSE variant
    on that path), so the entire reply is yielded as a single chunk — the
    UX win there is bounded but still real (no buffer waiting for the full
    parse). Yields nothing on hard backend failure; the caller should treat
    that the same as a non-streaming empty reply.
    """
    try:
        with requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': _get_ollama_model(),
                'prompt': prompt,
                'system': system_prompt,
                'stream': True,
                'options': {
                    'temperature': 0.4,
                    'num_predict': 4096,
                    'num_ctx': 32768,
                }
            },
            timeout=300,
            stream=True,
        ) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except (ValueError, json.JSONDecodeError):
                        continue
                    piece = chunk.get('response', '')
                    if piece:
                        yield piece
                    if chunk.get('done'):
                        break
                return
    except requests.exceptions.ConnectionError:
        pass
    except Exception as e:
        print(f"Ollama streaming error: {e}")

    # Claude fallback — non-streaming HTTP, but we yield the result as a
    # single chunk so the SSE client logic still flows.
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not api_key:
        return
    try:
        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={
                'Content-Type': 'application/json',
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01',
            },
            json={
                'model': 'claude-sonnet-4-20250514',
                'max_tokens': 2048,
                'system': system_prompt,
                'messages': [{'role': 'user', 'content': prompt}],
            },
            timeout=60,
        )
        if response.status_code == 200:
            data = response.json()
            yield data['content'][0]['text']
    except Exception as e:
        print(f"Claude API streaming-fallback error: {e}")


def chat_about_transcript_stream(transcript, message, history=None, project_name="Interview",
                                 analysis=None, profile_id=None, segment_vectors=None,
                                 paragraph_index=None):
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
    segments = (transcript or {}).get('segments', [])
    duration = segments[-1].get('end', 0) if segments else 0
    phrases, words = _extract_query_keywords(message)
    theme_phrases = _collect_theme_phrases_from_vectors(segment_vectors, message)
    tfidf_hits = []
    if paragraph_index is not None:
        try:
            tfidf_hits = paragraph_index.query_paragraphs(message, k=8) or []
        except Exception:
            tfidf_hits = []

    if duration > _LONG_CHAT_SECONDS:
        # Layer 2 is parallel by chunk; no streaming token feed is meaningful.
        # Emit a progress hint, then deliver the final reply as one chunk.
        yield ('progress', 'Searching across the full interview…')
        paragraphs = _build_paragraphs(transcript)
        reply = _chat_layer2_chunked_search(
            paragraphs, message, history, project_name,
            phrases, words, profile_id, analysis,
            segment_vectors=segment_vectors, theme_phrases=theme_phrases,
            tfidf_hits=tfidf_hits,
        )
        yield ('done', reply)
        return

    # Layer 1: build the same prompt the non-streaming path builds, then
    # stream the model's tokens. Mirrors `chat_about_transcript` so prompt
    # construction stays in one place — a shared `_build_layer1_prompt`
    # helper would be cleaner, but the inline block keeps the streaming
    # path self-contained for now.
    formatted = _format_transcript_for_ai(transcript)
    relevant_excerpts_block = ''
    if phrases or words or theme_phrases or tfidf_hits:
        matched = _find_relevant_paragraphs(
            segments, phrases, words, context=2, theme_phrases=theme_phrases,
        )
        matched = _merge_paragraph_lists(matched, tfidf_hits)
        matched = _augment_with_high_score_vectors(matched, segments, segment_vectors, theme_phrases)
        relevant_excerpts_block = _build_relevant_excerpts_block(matched)
    analysis_block = _build_chat_analysis_index(analysis)

    system_prompt = f"""You are an expert editorial consultant embedded in a documentary and interview editing tool called Doza Assist. You have the full transcript of the project "{project_name}" loaded in context.

═══════════════════════════════════════════════════════════════
HARD OUTPUT CONTRACT — THIS IS THE ONLY FORMAT THAT WORKS:

Every response MUST cite 2-5 specific moments from the transcript using this EXACT marker:

  [CLIP: start=HH:MM:SS end=HH:MM:SS title="short descriptive title"]

The frontend renders these markers as playable clip cards. Pure prose without markers fails the user.

GROUNDING RULE: every timecode MUST come from the transcript segment markers below. Never invent timecodes.

Personality: seasoned doc editor leaning over the desk, warm, opinionated, short. No filler.

TRANSCRIPT:
{formatted}{analysis_block}{relevant_excerpts_block}

═══════════════════════════════════════════════════════════════
FINAL REMINDER: Your reply MUST contain 2-5 [CLIP: ...] markers using real timecodes from the transcript above.
═══════════════════════════════════════════════════════════════"""

    conversation = ""
    if history:
        for msg in history[-6:]:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'user':
                conversation += f"\nUser: {content}"
            else:
                conversation += f"\nAssistant: {content}"
    prompt = f"{conversation}\nUser: {message}\nAssistant:"

    system_prompt = inject_my_style(system_prompt, profile_id=profile_id)

    pieces = []
    for piece in _call_ai_chat_stream(prompt, system_prompt):
        pieces.append(piece)
        yield ('token', piece)

    # Run the same post-processing the non-streaming path runs so the final
    # text is identical regardless of which endpoint the client used.
    full = ''.join(pieces)
    cleaned = _clean_chat_response(full)
    cleaned = _validate_clip_markers_in_text(cleaned, segments)
    yield ('done', cleaned)


def _call_ai_chat(prompt, system_prompt=""):
    """Call AI for chat — uses lower token limit for faster responses."""
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': _get_ollama_model(),
                'prompt': prompt,
                'system': system_prompt,
                'stream': False,
                'options': {
                    'temperature': 0.4,
                    'num_predict': 4096,
                    'num_ctx': 32768,
                }
            },
            timeout=300
        )
        if response.status_code == 200:
            return response.json().get('response', '')
    except requests.exceptions.ConnectionError:
        pass

    # Try Claude API
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if api_key:
        try:
            response = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers={
                    'Content-Type': 'application/json',
                    'x-api-key': api_key,
                    'anthropic-version': '2023-06-01',
                },
                json={
                    'model': 'claude-sonnet-4-20250514',
                    'max_tokens': 2048,
                    'system': system_prompt,
                    'messages': [{'role': 'user', 'content': prompt}],
                },
                timeout=60,
            )
            if response.status_code == 200:
                data = response.json()
                return data['content'][0]['text']
        except Exception as e:
            print(f"Claude API error: {e}")

    raise RuntimeError("No AI backend available.")


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
    # used for "is this start time inside any segment" checks.
    seg_ranges = []
    for s in segments:
        try:
            seg_start = float(s.get('start', 0) or 0)
            seg_end = float(s.get('end', seg_start) or seg_start)
        except (TypeError, ValueError):
            continue
        if seg_end >= seg_start:
            seg_ranges.append((seg_start, seg_end))
    if not seg_ranges:
        return text
    transcript_start = seg_ranges[0][0]
    transcript_end = max(end for _, end in seg_ranges)

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

    def _is_valid(match):
        start_str = match.group(1).strip().strip('"\'')
        try:
            return _start_in_transcript(_tc_to_seconds(start_str))
        except Exception:
            return True  # if we can't parse, leave it for the renderer to sort out

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
        # Optional surrounding parens are consumed so the rewritten marker
        # replaces the whole "(0:30 - 1:00)" span cleanly.
        _TC_RANGE_RE = re.compile(
            r'\(?\s*'
            r'(?P<start>\d{1,2}:\d{2}(?::\d{2})?)'
            r'\s*(?:to|\u2013|\u2014|-)\s*'
            r'(?P<end>\d{1,2}:\d{2}(?::\d{2})?)'
            r'\s*\)?',
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


def _build_relevant_excerpts_block(paragraphs):
    """Render matched paragraphs as a labeled block suitable for injecting
    between the full transcript and the FINAL REMINDER. Empty when nothing
    matched, so the caller can unconditionally concatenate the result.
    """
    if not paragraphs:
        return ''
    body = _format_paragraphs_as_lines(paragraphs)
    return (
        "\n\n"
        "RELEVANT EXCERPTS (auto-selected from the transcript above based on "
        "your question — use these timecodes for [CLIP:] markers):\n"
        f"{body}"
    )


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
      "title": "2-6 word title, no quotes",
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


def _call_ai_json(system_prompt, user_prompt, timeout=180):
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
    """
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': _get_ollama_model(),
                'prompt': user_prompt,
                'system': system_prompt,
                'stream': False,
                'format': 'json',
                'options': {
                    'temperature': 0.1,
                    'num_predict': 768,
                    'num_ctx': 12288,
                }
            },
            timeout=timeout,
        )
        if response.status_code == 200:
            return response.json().get('response', '')
    except requests.exceptions.ConnectionError:
        pass
    except requests.exceptions.Timeout:
        return ''

    # Claude fallback mirrors _call_ai_chat — same env var, same shape.
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if api_key:
        try:
            response = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers={
                    'Content-Type': 'application/json',
                    'x-api-key': api_key,
                    'anthropic-version': '2023-06-01',
                },
                json={
                    'model': 'claude-sonnet-4-20250514',
                    'max_tokens': 1024,
                    'system': system_prompt,
                    'messages': [{'role': 'user', 'content': user_prompt}],
                },
                timeout=60,
            )
            if response.status_code == 200:
                data = response.json()
                return data['content'][0]['text']
        except Exception:
            pass
    return ''


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
        title = raw_title.replace('"', "'")
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


def _chat_layer2_chunked_search(paragraphs, message, history, project_name,
                                phrases, words, profile_id, analysis,
                                segment_vectors=None, theme_phrases=None,
                                tfidf_hits=None):
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

    def _run_chunk(idx_chunk):
        idx, chunk = idx_chunk
        system_prompt, user_prompt = _build_chunk_search_prompt(
            chunk, message, phrases, words, idx, len(chunks), project_name,
            strict_keyword=strict_keyword,
        )
        response = _call_ai_json(system_prompt, user_prompt)
        return _parse_chunk_response(response, chunk)

    with ThreadPoolExecutor(max_workers=_CHAT_CHUNK_CONCURRENCY) as pool:
        futures = [pool.submit(_run_chunk, t) for t in search_targets]
        for fut in as_completed(futures):
            try:
                all_candidates.extend(fut.result() or [])
            except Exception as e:
                print(f"Layer 2 chunk error: {e}")

    top = _aggregate_chunk_candidates(all_candidates, top_k=_CHAT_TOP_K_CLIPS * 3)
    # Cross-chunk synthesis pass: per-chunk scores aren't comparable across
    # chunks (each model call sees only its own window), so a final low-temp
    # rerank decides the global best. Falls back to the local-score top-K
    # if the synthesis call fails — better to ship the original aggregator's
    # answer than to drop everything.
    top = _rerank_candidates_globally(top, message, top_k=_CHAT_TOP_K_CLIPS)
    return _format_clip_cards_from_candidates(top)


def _seconds_to_tc(sec) -> str:
    try:
        sec = int(sec)
    except (TypeError, ValueError):
        return "00:00:00"
    return f"{sec//3600:02d}:{(sec%3600)//60:02d}:{sec%60:02d}"


def _iter_transcript_chunks(segments, target_minutes=CHUNK_MINUTES):
    """Yield ~N-minute chunks of the transcript in order.

    Each yielded dict has ``start_seconds``, ``end_seconds``, and the slice of
    segments within that window. Segments preserve their original absolute
    timecodes so story beats and social clips come back timeline-absolute
    regardless of which chunk they originated from.
    """
    if not segments:
        return
    target_seconds = target_minutes * 60
    cur = []
    cur_start = None
    for seg in segments:
        if not cur:
            cur_start = seg.get('start', 0)
        cur.append(seg)
        elapsed = seg.get('end', seg.get('start', 0)) - cur_start
        if elapsed >= target_seconds:
            yield {
                'start_seconds': cur_start,
                'end_seconds': seg.get('end', cur_start),
                'segments': cur,
            }
            cur = []
            cur_start = None
    if cur:
        yield {
            'start_seconds': cur_start,
            'end_seconds': cur[-1].get('end', cur_start),
            'segments': cur,
        }


def _format_segments_for_ai(segments) -> str:
    """Format an arbitrary slice of segments for the prompt."""
    lines = []
    for seg in segments:
        start_tc = seg.get('start_formatted', _seconds_to_tc(seg.get('start', 0)))[:8]
        end_s = seg.get('end', seg.get('start', 0))
        end_tc = _seconds_to_tc(end_s)
        speaker = seg.get('speaker', 'Speaker')
        text = seg.get('text', '')
        if text.strip():
            lines.append(f"[{start_tc}-{end_tc}] {speaker}: {text}")
    return '\n'.join(lines)


def _merge_story_chunk(accum: dict, story_data: dict, is_first_chunk: bool):
    """Merge one chunk's story response into the accumulator.

    Chunk-level summaries/titles are accumulated into lists; the final
    overall summary is synthesized from all of them in
    :func:`_synthesize_overall_summary`. Keeping only the first chunk's
    summary (as an earlier version did) made the sidebar describe only the
    opening of the first interview when multiple interviews were strung
    into one timeline.
    """
    if not isinstance(story_data, dict):
        return
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
    """Merge one chunk's social response into the accumulator."""
    if isinstance(social_data, dict):
        accum['social_clips'].extend(
            _first_present_list(social_data, 'social_clips', 'clips', 'social', 'reels')
        )
    elif isinstance(social_data, list):
        accum['social_clips'].extend(social_data)


def _synthesize_overall_summary(summaries, titles, project_name):
    """Combine per-chunk summaries + titles into one overall summary/title.

    When multiple interviews are strung into a single timeline the chunked
    path produces one summary per ~15-minute slice. Returning just the
    first chunk's summary makes the sidebar describe only the opening of
    the first interview, which was the user-visible bug. This pass asks
    the model to fold all slice summaries into one coherent overview that
    spans the whole transcript.

    Falls back to joining chunk summaries with blank lines if the model
    call fails — still beats dropping everything after chunk 1.
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
    except Exception as e:
        print(f"[analyze] overall summary synthesis failed: {e}")

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

    # Short path: one AI call, original behavior.
    if total_duration < _LONG_INTERVIEW_SECONDS or not segments:
        formatted = _format_transcript_for_ai(transcript)
        return _analyze_transcript_single(
            formatted, project_name, analysis_type, segment_vectors=segment_vectors,
            progress_emit=_emit,
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
                print(f"[analyze] story chunk {i+1}/{len(chunks)} failed: {e}")
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
                print(f"[analyze] social chunk {i+1}/{len(chunks)} failed: {e}")

    step += 1
    _emit(step, total_steps, "synthesizing summary")
    overall = _synthesize_overall_summary(
        accum.pop('_chunk_summaries', []),
        accum.pop('_chunk_titles', []),
        project_name,
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
    )


def _analyze_transcript_single(formatted_text, project_name, analysis_type,
                                segment_vectors=None, progress_emit=None):
    """One-shot analysis path for short interviews."""
    types_count = (1 if analysis_type in ('story', 'all') else 0) + \
                  (1 if analysis_type in ('social', 'all') else 0)
    total_steps = types_count + 1  # +1 cap+rerank
    step = 0

    def _emit(current):
        if progress_emit is not None:
            progress_emit(step, total_steps, current)

    _emit("starting")

    result = {}
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
    normalized = normalize_analysis(result)
    return _cap_and_rank_analysis(
        normalized, segment_vectors=segment_vectors, cap=ANALYSIS_PER_CATEGORY_CAP,
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


def _cap_and_rank_analysis(accum, segment_vectors=None, cap=7):
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

    def _trim_clip_list(items, prefer_diversity=False, diversity_key='beat_type'):
        if not isinstance(items, list) or not items:
            return [] if isinstance(items, list) else items
        viable = [i for i in items if isinstance(i, dict) and i.get('start')]
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
        out['story_beats'] = _trim_clip_list(out['story_beats'], prefer_diversity=True)
    if isinstance(out.get('strongest_soundbites'), list):
        out['strongest_soundbites'] = _trim_clip_list(out['strongest_soundbites'])
    if isinstance(out.get('social_clips'), list):
        out['social_clips'] = _trim_clip_list(out['social_clips'])

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


def _format_transcript_for_ai(transcript):
    """Format transcript segments into readable text with timecodes.

    Always sends the full transcript — no truncation.
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
        speaker = seg.get('speaker', 'Speaker')
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


def _call_ai(prompt, system_prompt=""):
    """
    Call an AI model. Tries Ollama first (local), then Claude API.
    Returns the response text.
    """
    # Try Ollama first (local, free)
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': _get_ollama_model(),
                'prompt': prompt,
                'system': system_prompt,
                'stream': False,
                'options': {
                    'temperature': 0.3,
                    'num_predict': 16384,
                    'num_ctx': 32768,
                }
            },
            # 15 min — Story Builder on long transcripts with an 8B model
            # (gemma4:latest resolves to 8B on several setups) can push past
            # the old 10-min cap while still succeeding on the second try.
            timeout=900,
        )
        if response.status_code == 200:
            return response.json().get('response', '')
    except requests.exceptions.ConnectionError:
        pass

    # Try Claude API
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if api_key:
        try:
            response = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers={
                    'Content-Type': 'application/json',
                    'x-api-key': api_key,
                    'anthropic-version': '2023-06-01',
                },
                json={
                    'model': 'claude-sonnet-4-20250514',
                    'max_tokens': 4096,
                    'system': system_prompt,
                    'messages': [{'role': 'user', 'content': prompt}],
                },
                timeout=120,
            )
            if response.status_code == 200:
                data = response.json()
                return data['content'][0]['text']
        except Exception as e:
            print(f"Claude API error: {e}")

    raise RuntimeError(
        "No AI backend available. Either:\n"
        "  1. Start Ollama: ollama serve (then pull a model like gemma4:e4b)\n"
        "  2. Set ANTHROPIC_API_KEY environment variable for Claude API"
    )


def _get_ollama_model():
    """Get the Ollama model to use, honoring hardware-tier config."""
    try:
        import model_config
        selected_variant = model_config.get_gemma4_variant()['variant']
    except Exception:
        selected_variant = 'gemma4:e4b'

    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            available = [m['name'] for m in models]
            # Try the hardware-selected Gemma 4 variant first
            for avail in available:
                if selected_variant in avail:
                    return avail
            # Fall back through other Gemma 4 variants, then other models
            fallback = [
                'gemma4:e4b', 'gemma4:latest', 'gemma4:e2b', 'gemma4:26b', 'gemma4:31b',
                'llama3:8b', 'mistral', 'llama3:70b',
            ]
            for pref in fallback:
                for avail in available:
                    if pref in avail:
                        return avail
            if available:
                return available[0]
    except Exception:
        pass
    return selected_variant


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
- Order them for maximum story impact, not chronological order unless that serves the story
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

TRANSCRIPT:
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

    menu_lines = []
    for s in candidates:
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
- Build a clear arc: hook, context, pressure, turn, resolution.
- DURATION IS CRITICAL: If the user requests a specific duration (e.g. "4 minute story"), you MUST hit that target. Calculate the total duration of all clips you select by adding up (end_time - start_time) for each clip. Each segment in the menu shows its timecode range — use that to calculate duration. Aim for roughly 3-4 clips per minute of requested duration. For a 4-minute story, select enough clips to total approximately 3:30-4:30 of content. If your first selection is too short, add more clips. If too long, trim or remove clips.
- Ordered for story impact (not necessarily chronological).
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

    ``beats_target`` and ``soundbites_target`` set the upper bound the model
    is asked to return. The chunked-analysis path passes smaller per-chunk
    values so the merged total lands near the global cap; the single-chunk
    path uses the global default. Post-merge ranking trims further if the
    model overshoots.
    """
    beats_target = max(1, int(beats_target))
    soundbites_target = max(1, int(soundbites_target))
    system_prompt = """You are an expert documentary film editor.
Analyze transcripts for story structure and narrative beats.
Always respond in valid JSON only. No markdown fences, no extra text."""

    prompt = f"""Analyze this interview for documentary story structure.

PROJECT: {project_name}

TRANSCRIPT:
{transcript_text}

Return a JSON object with:
{{
  "summary": "2-3 sentence overview of the story",
  "suggested_title": "A compelling working title",
  "story_beats": [
    {{
      "order": 1,
      "label": "Opening Hook",
      "description": "Why this moment works",
      "start": "00:00:45",
      "end": "00:01:02"
    }}
  ],
  "strongest_soundbites": [
    {{
      "text": "The actual quote",
      "start": "00:02:00",
      "end": "00:02:18",
      "why": "Why this is powerful"
    }}
  ]
}}

Pick the {beats_target} BEST story beats following a documentary arc: hook, context, rising action, emotional peak, resolution, closing. Diversify across beat types — don't stack three hooks.
Pick the {soundbites_target} BEST soundbites. Be ruthless; return fewer if the transcript only has fewer standouts.
CRITICAL: Copy the exact HH:MM:SS timecodes from the transcript for start and end. Use string format like "00:02:45".
Return ONLY valid JSON."""

    response = _call_ai(prompt, system_prompt)
    return _parse_json_response(response)


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
    return _parse_json_response(response)


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
