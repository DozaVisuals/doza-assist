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

    system_prompt = f"""You are a film editor's assistant. You have the transcript for "{project_name}".

Rules:
- Be concise and direct. Short paragraphs, no filler.
- No emojis. No markdown headers (#). No bullet points with asterisks.
- Use plain text with line breaks.
- When suggesting clips, use this exact format:
  [CLIP: start=45.2 end=62.8 title="Short descriptive title"]
- Always include exact timecodes from the transcript.
- Keep clip titles under 8 words.

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
