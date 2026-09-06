"""Generated clip titles for the clip library (1.0.47).

A clip painted on the transcript, pulled from a Story Brief moment or added
from an AI Analysis soundbite used to carry the first words of its own
transcript as its title ("And how do we help as daybreak? Yeah, great
question..."). Chat clips were the exception: the model writes a short
headline for every [CLIP:] card. This module gives every other clip the same
kind of title.

Data shape on ``labeled_sections`` entries (meta.json):

    text        unchanged: the transcript fragment (brush, Story Brief,
                soundbites) or the headline (Chat, story clips) the add wrote
    title       the display title (new, optional)
    title_auto  True when the model wrote ``title``; absent or False when the
                title was carried over from ``text`` or typed by a person

Pure helpers here (transcript slicing, fragment detection, prompt, reply
parsing, the batch plan) take plain dicts so they run in tests without a
model. ``title_clips`` is the orchestrator the route calls; the model call is
injected so the route can pass ai_analysis._call_ai and tests can pass a stub.
"""
from __future__ import annotations

import re
from typing import Callable, Iterable

# How many clips one model call titles. Small enough that a slow local
# model answers in seconds and a truncated reply loses little.
BATCH_SIZE = 12

# Excerpts longer than this are shown to the model as head ... tail.
EXCERPT_HEAD = 520
EXCERPT_TAIL = 200

# A title the model returns is cut here (word boundary) if it runs on.
TITLE_MAX_CHARS = 80

# Same tolerance the page uses when it decides two clips are one moment.
MATCH_TOLERANCE = 0.5

SYSTEM_PROMPT = (
    'You title interview clips for a documentary editor. Reply with JSON only.'
)


def to_seconds(val) -> float:
    if isinstance(val, bool):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val or '').strip()
    if ':' in s:
        parts = [float(p) for p in s.split(':')]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
    try:
        return float(s or 0)
    except ValueError:
        return 0.0


def transcript_for_range(segments: Iterable[dict], start: float, end: float) -> str:
    """The words spoken between start and end, from word timings when the
    segments carry them, else whole overlapping segments."""
    if end <= start:
        return ''
    bits = []
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        try:
            s0 = float(seg.get('start', 0) or 0)
            e0 = float(seg.get('end', s0) or s0)
        except (TypeError, ValueError):
            continue
        if e0 <= start or s0 >= end:
            continue
        words = seg.get('words') or []
        if words:
            for w in words:
                if not isinstance(w, dict):
                    continue
                try:
                    ws = float(w.get('start', 0) or 0)
                except (TypeError, ValueError):
                    continue
                if start <= ws < end:
                    bits.append((w.get('word') or w.get('text') or '').strip())
        else:
            bits.append((seg.get('text') or '').strip())
    return re.sub(r'\s+', ' ', ' '.join(b for b in bits if b)).strip()


def speaker_for_range(segments: Iterable[dict], start: float, end: float,
                      names: dict | None = None) -> str:
    """The first speaker heard in the range, mapped through speaker_names."""
    names = names or {}
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        try:
            s0 = float(seg.get('start', 0) or 0)
            e0 = float(seg.get('end', s0) or s0)
        except (TypeError, ValueError):
            continue
        if s0 < end and e0 > start:
            raw = (seg.get('speaker') or '').strip()
            if not raw:
                return ''
            resolved = names.get(raw, raw)
            return resolved.strip() if isinstance(resolved, str) else raw
    return ''


_PUNCT = re.compile(r"[^\w\s']+", re.UNICODE)


def _normalize(text: str) -> str:
    t = (text or '').lower().replace('’', "'")
    t = re.sub(r'(\.\.\.|…)\s*$', '', t.strip())
    t = _PUNCT.sub(' ', t)
    return re.sub(r'\s+', ' ', t).strip()


# The transcript a clip's text is compared against reaches this far outside
# the clip: an AI select stores the whole overlapping segments, so its text
# can begin a sentence before the clip's start.
FRAGMENT_CONTEXT_BEFORE = 30.0
FRAGMENT_CONTEXT_AFTER = 10.0


def _bigrams(words: list[str]) -> set:
    return {(a, b) for a, b in zip(words, words[1:])}


def is_transcript_fragment(text: str, transcript: str, context: str = '') -> bool:
    """True when ``text`` is just the transcript talking, not a title.

    The brush stores the first 200 characters of the painted words, Story
    Brief stores the quote, AI Analysis stores the first 40 characters of a
    soundbite, an AI select stores its overlapping segments: all of these
    are the transcript itself. A Chat headline or a story clip title is not
    found verbatim in or around the range.

    ``context`` is the transcript of a wider window around the clip (see
    FRAGMENT_CONTEXT_BEFORE / AFTER); the text is looked up in the clip's
    own transcript first, then in the window. A long text (8+ words) whose
    word pairs almost all occur in the window counts too, so a fragment
    with one transcription difference is still recognized.

    Empty text always needs a title. With no transcript to compare against
    nothing can be generated either, so a non-empty text is kept as the
    title.
    """
    nt = _normalize(text)
    if not nt:
        return True
    ntr = _normalize(transcript)
    nctx = _normalize(context) or ntr
    if not ntr and not nctx:
        return False
    if ntr and (ntr.startswith(nt) or nt in ntr):
        return True
    if nctx and nt in nctx:
        return True
    words = nt.split()
    if len(words) >= 8 and nctx:
        pairs = _bigrams(words)
        have = _bigrams(nctx.split())
        if pairs and len(pairs & have) / len(pairs) >= 0.8:
            return True
    return False


# A lead shorter than this keeps taking sentences (a clip that starts on the
# tail of a sentence would otherwise read "daybreak?" and nothing else).
LEAD_MIN_CHARS = 36


def first_line(text: str, max_chars: int = 140) -> str:
    """The opening of a transcript as one line: whole sentences while they
    fit (at least LEAD_MIN_CHARS of them), else a word-boundary cut with an
    ellipsis."""
    t = re.sub(r'\s+', ' ', (text or '')).strip()
    if not t:
        return ''
    if len(t) <= max_chars:
        return t
    end = 0
    for m in re.finditer(r'[.!?]["\')\]]*(\s|$)', t):
        if m.end() > max_chars:
            break
        end = m.end()
        if end >= LEAD_MIN_CHARS:
            break
    if end >= LEAD_MIN_CHARS:
        return t[:end].strip()
    cut = t.rfind(' ', 0, max_chars)
    if cut < max_chars // 2:
        cut = max_chars
    return t[:cut].rstrip(' ,;:') + '…'


def _excerpt(transcript: str) -> str:
    t = transcript.strip()
    if len(t) <= EXCERPT_HEAD + EXCERPT_TAIL + 20:
        return t
    return t[:EXCERPT_HEAD].rstrip() + ' ... ' + t[-EXCERPT_TAIL:].lstrip()


def _fmt_len(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f'{seconds}s'
    return f'{seconds // 60}:{seconds % 60:02d}'


def build_prompt(items: list[dict], language_directive: str = '') -> str:
    """The user prompt for one batch. Each item: transcript, speaker,
    duration. Numbered from 1 so the reply keys line up with the batch."""
    lines = [
        'Write one short title for each transcript excerpt below.',
        'A title is a 3 to 7 word headline a film editor would put on a select:',
        'specific and concrete, built from the speaker\'s own key words where',
        'possible, never a whole sentence copied from the excerpt, no quotation',
        'marks, no trailing period, no numbering, no speaker name.',
        'Good titles read like: "Turnover Costs Twice the Salary",',
        '"Nobody Understands Their Deductible", "Start Before You Feel Ready".',
        '',
        'Return exactly this JSON and nothing else:',
        '{"titles": {"1": "title for excerpt 1", "2": "title for excerpt 2"}}',
        '',
    ]
    for i, it in enumerate(items, 1):
        who = (it.get('speaker') or '').strip()
        head = f'Excerpt {i}'
        meta = []
        if who:
            meta.append(who)
        dur = it.get('duration')
        if isinstance(dur, (int, float)) and dur > 0:
            meta.append(_fmt_len(dur) + ' long')
        if meta:
            head += ' (' + ', '.join(meta) + ')'
        lines.append(head + ':')
        lines.append('"' + _excerpt(it.get('transcript') or '') + '"')
        lines.append('')
    prompt = '\n'.join(lines).rstrip()
    if language_directive:
        prompt += language_directive
    return prompt


def clean_title(raw) -> str:
    """Normalize one model title; '' when unusable."""
    if raw is None:
        return ''
    if isinstance(raw, dict):
        raw = raw.get('title') or raw.get('text') or ''
    t = re.sub(r'\s+', ' ', str(raw)).strip()
    t = re.sub(r'^\s*(\d+[.):]|[-*•])\s*', '', t)
    t = t.strip(' "\'“”‘’')
    t = t.rstrip('.').strip()
    if t.lower().startswith('title:'):
        t = t[6:].strip()
    if not t:
        return ''
    if len(t) > TITLE_MAX_CHARS:
        cut = t.rfind(' ', 0, TITLE_MAX_CHARS)
        if cut < TITLE_MAX_CHARS // 2:
            cut = TITLE_MAX_CHARS
        t = t[:cut].rstrip(' ,;:')
    return t


def parse_titles(parsed, count: int) -> dict[int, str]:
    """Map 1-based excerpt numbers to cleaned titles from a parsed reply.

    Accepts ``{"titles": {"1": ...}}``, ``{"titles": [...]}``, a bare list,
    or a bare dict of numbers. Anything unreadable simply yields fewer
    titles; the caller marks the rest failed.
    """
    if parsed is None:
        return {}
    titles = parsed
    if isinstance(parsed, dict):
        if 'titles' in parsed:
            titles = parsed['titles']
        elif 'error' in parsed and len(parsed) <= 2:
            return {}
    out: dict[int, str] = {}
    if isinstance(titles, list):
        for i, raw in enumerate(titles, 1):
            if i > count:
                break
            t = clean_title(raw)
            if t:
                out[i] = t
        return out
    if isinstance(titles, dict):
        for k, raw in titles.items():
            m = re.search(r'\d+', str(k))
            if not m:
                continue
            i = int(m.group(0))
            if 1 <= i <= count:
                t = clean_title(raw)
                if t:
                    out[i] = t
    return out


def _same_range(sec: dict, start: float, end: float, tol: float = MATCH_TOLERANCE) -> bool:
    try:
        return (abs(to_seconds(sec.get('start', 0)) - start) < tol
                and abs(to_seconds(sec.get('end', 0)) - end) < tol)
    except (TypeError, ValueError):
        return False


def _find_section(sections: list[dict], start: float, end: float) -> int:
    for i, sec in enumerate(sections):
        if isinstance(sec, dict) and _same_range(sec, start, end):
            return i
    return -1


def title_clips(project: dict, requested: list[dict] | None,
                call_model: Callable[[str, str], object],
                language_directive: str = '',
                include_transcript: bool = False,
                batch_size: int = BATCH_SIZE) -> dict:
    """Title the requested clips (or every stored clip without a title).

    ``requested``: list of {start, end, text?}. Clips not yet in the stored
    array are titled too (the page saves labels on a debounce, so a request
    can arrive first); their titles are returned but only stored clips are
    written. ``None`` means every stored clip that lacks a title.

    ``call_model(prompt, system_prompt)`` returns the parsed JSON reply (a
    dict) or raises. One failing batch marks its clips failed and moves on;
    the first error message is reported so the page can show it.

    Returns {'items': [...], 'sections': [...], 'changed': bool,
             'generated': n, 'carried': n, 'kept': n, 'failed': n,
             'error': str | None}.
    Each item: {start, end, title, title_auto, status, lead[, transcript]}
    with status one of kept, carried, generated, failed.
    """
    sections = [dict(s) if isinstance(s, dict) else s for s in (project.get('labeled_sections') or [])]
    segments = (project.get('transcript') or {}).get('segments') or []
    names = project.get('speaker_names') or {}

    work: list[dict] = []
    if requested is None:
        for i, sec in enumerate(sections):
            if not isinstance(sec, dict):
                continue
            if (sec.get('title') or '').strip():
                continue
            work.append({'index': i, 'start': to_seconds(sec.get('start', 0)),
                         'end': to_seconds(sec.get('end', 0)), 'text': sec.get('text') or ''})
    else:
        for req in requested:
            if not isinstance(req, dict):
                continue
            try:
                start = to_seconds(req.get('start', 0))
                end = to_seconds(req.get('end', 0))
            except (TypeError, ValueError):
                continue
            idx = _find_section(sections, start, end)
            text = req.get('text')
            if text is None and idx >= 0:
                text = sections[idx].get('text') or ''
            work.append({'index': idx, 'start': start, 'end': end, 'text': text or ''})

    items: list[dict] = []
    pending: list[dict] = []
    counts = {'generated': 0, 'carried': 0, 'kept': 0, 'failed': 0}
    changed = False

    for w in work:
        idx = w['index']
        sec = sections[idx] if idx >= 0 else None
        transcript = transcript_for_range(segments, w['start'], w['end'])
        context = transcript_for_range(segments, max(0.0, w['start'] - FRAGMENT_CONTEXT_BEFORE),
                                       w['end'] + FRAGMENT_CONTEXT_AFTER)
        item = {
            'start': w['start'], 'end': w['end'],
            'lead': first_line(transcript) or first_line(w['text']),
        }
        if include_transcript:
            item['transcript'] = transcript
        if sec is not None and (sec.get('title') or '').strip():
            item.update(title=sec['title'], title_auto=bool(sec.get('title_auto')), status='kept')
            counts['kept'] += 1
            items.append(item)
            continue
        text = (w['text'] or '').strip()
        if text and not is_transcript_fragment(text, transcript, context):
            item.update(title=text[:200], title_auto=False, status='carried')
            counts['carried'] += 1
            if sec is not None:
                sec['title'] = text[:200]
                sec.pop('title_auto', None)
                changed = True
            items.append(item)
            continue
        if not transcript and not text:
            item.update(title='', title_auto=False, status='failed')
            counts['failed'] += 1
            items.append(item)
            continue
        item.update(title='', title_auto=True, status='failed')
        pending.append({
            'item': item, 'section': sec,
            'transcript': transcript or text,
            'speaker': (sec or {}).get('speaker') or speaker_for_range(segments, w['start'], w['end'], names),
            'duration': w['end'] - w['start'],
        })
        items.append(item)

    error = None
    for b in range(0, len(pending), max(1, batch_size)):
        batch = pending[b:b + batch_size]
        prompt = build_prompt([{'transcript': p['transcript'], 'speaker': p['speaker'],
                                'duration': p['duration']} for p in batch], language_directive)
        try:
            parsed = call_model(prompt, SYSTEM_PROMPT)
        except Exception as exc:  # provider errors surface as a message, never a 500
            error = error or (str(exc) or exc.__class__.__name__)
            parsed = None
        got = parse_titles(parsed, len(batch))
        for i, p in enumerate(batch, 1):
            title = got.get(i, '')
            if not title:
                continue
            p['item'].update(title=title, status='generated')
            counts['generated'] += 1
            if p['section'] is not None:
                p['section']['title'] = title
                p['section']['title_auto'] = True
                changed = True
    counts['failed'] += sum(1 for p in pending if p['item']['status'] != 'generated')

    return {'items': items, 'sections': sections, 'changed': changed,
            'error': error, **counts}
