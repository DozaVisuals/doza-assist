"""
AI Analysis for Doza Assist.
Uses Ollama (local) or Claude API for story structure and social clip suggestions.
"""

import os
import json
import requests


def chat_about_transcript(transcript, message, history=None, project_name="Interview", analysis=None):
    """
    Chat with AI about the transcript. Supports follow-up questions.
    Returns the AI reply as a string (may contain embedded clip suggestions).
    """
    formatted = _format_transcript_for_ai(transcript, max_chars=8000)

    system_prompt = f"""You are an expert editorial consultant embedded in a documentary and interview editing tool called Doza Assist. You have the full transcript of the project "{project_name}" loaded in context.

Your personality: You're the best story editor alive. You speak briefly, naturally, and with confidence. No filler, no hedging, no "here are some suggestions you might consider." Just direct, sharp editorial insight like a seasoned doc editor sitting next to the user in the edit suite.

You have three modes depending on what the user asks:

1. CLIP DISCOVERY: When the user asks for moments, clips, soundbites, or quotes, return specific timecoded clips from the transcript. Format each clip with a title, start and end timecode, and the transcript excerpt. Be selective. Don't dump everything. Pick the strongest moments and explain briefly why each one works.

2. STORY CONSULTING: When the user asks about structure, narrative, themes, character arcs, or editorial direction, give expert story advice grounded in what's actually in the transcript. Reference specific moments to support your recommendations. Think like a documentary editor with 20 years of experience. Talk about what the story is really about underneath the surface. Identify the emotional spine. Point out what's missing or what could be stronger.

3. SOCIAL MEDIA AND CONTENT STRATEGY: When the user asks about social clips, content strategy, or platform-specific advice, be an expert in short-form content. Know what works on Instagram Reels, TikTok, YouTube Shorts, LinkedIn, and X. Recommend specific clips from the transcript with reasoning about why they'd perform on each platform. Consider hook strength (first 2 seconds), emotional payoff, shareability, and trending formats. Suggest captions, hashtags, and posting strategies when asked.

General rules:
- Keep responses short and direct. 2-3 sentences for simple answers. A few short paragraphs max for complex editorial questions.
- Never repeat the entire transcript back. Reference specific moments by timecode.
- When suggesting clips, always include timecodes so the app can render them as playable cards. Use this exact format:
  [CLIP: start=45.2 end=62.8 title="Short descriptive title"]
- Keep clip titles under 8 words.
- Use natural conversational language. No bullet point lists unless the user specifically asks for a list.
- No emojis. No markdown headers (#). No bullet points with asterisks. Use plain text with line breaks.
- Have opinions. Don't present five equal options. Say which one is the strongest and why.
- If the user asks something that can't be answered from the transcript, say so directly.
- When giving story advice, think in terms of: What's the hook? What's the tension? What's the emotional turning point? What's the resolution? What's the takeaway?
- For social clips, default to recommending moments under 60 seconds. Flag if something would work better as a 15-second cut vs. a 3-minute piece.

TRANSCRIPT:
{formatted}"""

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

    response = _call_ai_chat(prompt, system_prompt)
    return _clean_chat_response(response)


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
                    'num_predict': 2048,
                }
            },
            timeout=120
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


def _clean_chat_response(text):
    """Strip markdown artifacts and emoji from chat responses."""
    import re
    text = text.strip()
    # Remove markdown headers
    text = re.sub(r'^#{1,4}\s*', '', text, flags=re.MULTILINE)
    # Remove bold/italic markdown
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)
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


def analyze_transcript(transcript, project_name="Interview", analysis_type="all"):
    """
    Analyze a transcript for story structure and social media clips.

    Args:
        transcript: dict with 'segments' list from transcribe.py
        project_name: str
        analysis_type: 'story', 'social', or 'all'

    Returns:
        dict with 'story_beats' and/or 'social_clips'
    """
    formatted = _format_transcript_for_ai(transcript)
    result = {}

    if analysis_type in ('story', 'all'):
        story_data = _analyze_story(formatted, project_name)
        # Flatten: the AI returns {summary, suggested_title, story_beats: [...], ...}
        # We want all keys at the top level of result
        if isinstance(story_data, dict):
            result['summary'] = story_data.get('summary', '')
            result['suggested_title'] = story_data.get('suggested_title', '')
            result['story_beats'] = story_data.get('story_beats', [])
            result['themes'] = story_data.get('themes', [])
            result['strongest_soundbites'] = story_data.get('strongest_soundbites', [])
            result['broll_suggestions'] = story_data.get('broll_suggestions', [])

    if analysis_type in ('social', 'all'):
        social_data = _analyze_social(formatted, project_name)
        # Flatten: AI returns {social_clips: [...]}
        if isinstance(social_data, dict):
            result['social_clips'] = social_data.get('social_clips', [])
        elif isinstance(social_data, list):
            result['social_clips'] = social_data

    return result


def _format_transcript_for_ai(transcript, max_chars=12000):
    """Format transcript segments into readable text with timecodes.

    Limits output to max_chars to avoid overwhelming the model.
    Samples evenly from beginning, middle, and end for full coverage.
    """
    segments = transcript.get('segments', [])
    if not segments:
        return ''

    # Format all segments
    all_lines = []
    for seg in segments:
        tc = seg['start_formatted'][:8]
        speaker = seg.get('speaker', 'Speaker')
        text = seg['text']
        if text.strip():
            all_lines.append(f"[{tc}] {speaker}: {text}")

    full = '\n'.join(all_lines)
    if len(full) <= max_chars:
        return full

    # If too long, take beginning + middle + end portions
    n = len(all_lines)
    third = n // 3
    sampled = (
        all_lines[:third] +
        ['', '--- [middle of interview] ---', ''] +
        all_lines[third:2*third] +
        ['', '--- [final portion] ---', ''] +
        all_lines[2*third:]
    )
    result = '\n'.join(sampled)

    # If still too long, hard truncate
    if len(result) > max_chars:
        result = result[:max_chars] + '\n[... transcript truncated for length ...]'

    return result


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
        "  1. Start Ollama: ollama serve (then pull a model like gemma3:27b)\n"
        "  2. Set ANTHROPIC_API_KEY environment variable for Claude API"
    )


def _get_ollama_model():
    """Detect which Ollama model is available. Prefers fast models for responsiveness."""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            # Prefer smaller/faster models for snappy UX — quality is good enough
            preferred = [
                'gemma4:latest', 'gemma3:12b', 'llama3:8b', 'mistral',
                'gemma4:31b', 'gemma3:27b', 'llama3:70b',
            ]
            available = [m['name'] for m in models]
            for pref in preferred:
                for avail in available:
                    if pref in avail:
                        return avail
            if available:
                return available[0]
    except:
        pass
    return 'gemma4:latest'


def build_story(transcript, message, project_name="Interview", segment_vectors=None):
    """
    Build a narrative sequence from the transcript based on the user's description.
    Returns a dict with story_title, target_duration, and clips array.

    If segment_vectors is provided, the model is given the pre-classified segments
    instead of having to re-analyze the raw transcript. This makes builds faster and
    much more consistent across runs.
    """
    if segment_vectors:
        return _build_story_from_vectors(segment_vectors, message, project_name)

    formatted = _format_transcript_for_ai(transcript, max_chars=12000)

    system_prompt = """You are a story editor building a narrative sequence from interview transcript footage. The user will describe what kind of story or edit they want. Your job is to select and order clips from the transcript that form a coherent narrative.

Rules:
- Select clips that build a clear narrative arc: hook, rising action, emotional peak, resolution
- Order them for maximum story impact, not chronological order unless that serves the story
- Each clip should be 5-30 seconds long unless the moment requires more breathing room
- Include 5-15 clips depending on the requested duration (roughly 3-4 clips per minute of final edit)
- For each clip, provide: a short title, start timecode, end timecode, the transcript excerpt, and a one-sentence editorial note explaining why this clip is in this position
- Be selective and opinionated. Don't include filler. Every clip should earn its place
- If the user asks for a specific duration, respect it. Calculate total clip time and stay within range
- CRITICAL: Copy the exact HH:MM:SS timecodes from the transcript for start and end times. Use string format like "00:02:45"
- Respond ONLY in valid JSON with this structure:
{
  "story_title": "suggested title for this sequence",
  "target_duration": "estimated total duration",
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

    response = _call_ai(prompt, system_prompt)
    return _parse_json_response(response)


def generate_segment_vectors(transcript, project_name="Interview"):
    """
    Generate structured segment vectors from a transcript.

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
    formatted = _format_transcript_for_ai(transcript, max_chars=12000)

    system_prompt = """You are a documentary story analyst. You break interview transcripts into discrete narrative segments and classify them with strict, structured metadata. You always respond in valid JSON only — no prose, no markdown, no code fences."""

    prompt = f"""Analyze this interview transcript and produce a structured set of segment vectors.

PROJECT: {project_name}

TRANSCRIPT:
{formatted}

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

    response = _call_ai(prompt, system_prompt)
    parsed = _parse_json_response(response)

    raw_segments = []
    if isinstance(parsed, dict):
        raw_segments = parsed.get('segments', []) or []
    elif isinstance(parsed, list):
        raw_segments = parsed

    return _normalize_segment_vectors(raw_segments, transcript)


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


def _build_story_from_vectors(segment_vectors, message, project_name):
    """Build a narrative using pre-classified segment vectors as the menu of clips.

    Prioritizes "high" narrative scores; uses "episodic" segments for key moments
    and "semantic" segments for context/transitions. The model only chooses and
    orders — it does not invent timecodes — which is why this path is more reliable.
    """
    # Compact menu of available segments for the prompt
    menu_lines = []
    for s in segment_vectors:
        menu_lines.append(
            f"- {s.get('seg_id', '?')} [{s.get('timecode_in', '')}-{s.get('timecode_out', '')}] "
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
- 5-15 clips total, ordered for story impact (not necessarily chronological).
- Always respond in valid JSON only. No markdown, no prose."""

    prompt = f"""PROJECT: {project_name}

USER REQUEST: {message}

AVAILABLE SEGMENTS (pre-classified):
{menu}

Return ONLY valid JSON in this shape:
{{
  "story_title": "suggested title",
  "target_duration": "estimated total duration",
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
        'clips': hydrated,
    }


def _analyze_story(transcript_text, project_name):
    """Analyze transcript for documentary story structure."""
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
      "why": "Why this is powerful"
    }}
  ]
}}

Find 4-6 story beats following a documentary arc: hook, context, rising action, emotional peak, resolution, closing.
Find 3-5 strongest soundbites.
CRITICAL: Copy the exact HH:MM:SS timecodes from the transcript for start and end. Use string format like "00:02:45".
Return ONLY valid JSON."""

    response = _call_ai(prompt, system_prompt)
    return _parse_json_response(response)


def _analyze_social(transcript_text, project_name):
    """Find social media clip opportunities in the transcript."""
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
- Find at least 5 clips if the transcript is long enough
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
