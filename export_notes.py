"""The Notes text on an exported select clip (1.1).

Line 1 is the short note the export already carried (a label title, an AI
"why", a beat description) plus " — Speaker" when the speaker is known.
After a blank line comes the verbatim transcript for the select's time range,
one paragraph per speaker turn, prefixed with the speaker's name when the
project has speakers. The verbatim block is capped at VERBATIM_CAP
characters, cut at a sentence or word boundary with an ellipsis.

Both writers use this: the raw-media exporter puts the text in a ``<note>``
as the first child of each asset-clip / clip, and the round-trip writer
passes it to ``_set_clip_note`` (which keeps the DTD position). Callers
escape the finished text for XML themselves; nothing here is escaped, and
nothing is sliced after escaping.
"""
from __future__ import annotations

import re
from typing import Iterable

VERBATIM_CAP = 1000
ELLIPSIS = '…'
SPEAKER_SEP = ' — '   # the existing "note — Speaker" convention


def _to_seconds(val) -> float:
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


def _display_speaker(raw: str, names: dict | None) -> str:
    raw = (raw or '').strip()
    if not raw:
        return ''
    mapped = (names or {}).get(raw, raw)
    return mapped.strip() if isinstance(mapped, str) else raw


def verbatim_for_range(segments: Iterable[dict], start, end, speaker_names: dict | None = None,
                       cap: int = VERBATIM_CAP) -> str:
    """The words spoken between start and end, as speaker-labelled paragraphs.

    Word timings are used when a segment carries them (only the words inside
    the range), otherwise the whole overlapping segment. Consecutive turns by
    the same speaker fold into one paragraph. Speaker prefixes appear only
    when at least one segment names a speaker.
    """
    try:
        s0, e0 = _to_seconds(start), _to_seconds(end)
    except (TypeError, ValueError):
        return ''
    if e0 <= s0:
        return ''
    turns: list[tuple[str, list[str]]] = []
    for seg in segments or []:
        if not isinstance(seg, dict):
            continue
        try:
            ss = float(seg.get('start', 0) or 0)
            se = float(seg.get('end', ss) or ss)
        except (TypeError, ValueError):
            continue
        if se <= s0 or ss >= e0:
            continue
        words = seg.get('words') or []
        bits: list[str] = []
        if words:
            for w in words:
                if not isinstance(w, dict):
                    continue
                try:
                    ws = float(w.get('start', 0) or 0)
                except (TypeError, ValueError):
                    continue
                if s0 <= ws < e0:
                    t = (w.get('word') or w.get('text') or '').strip()
                    if t:
                        bits.append(t)
        else:
            t = (seg.get('text') or '').strip()
            if t:
                bits.append(t)
        if not bits:
            continue
        speaker = _display_speaker(seg.get('speaker') or '', speaker_names)
        if turns and turns[-1][0] == speaker:
            turns[-1][1].extend(bits)
        else:
            turns.append((speaker, bits))
    if not turns:
        return ''
    labelled = any(sp for sp, _ in turns)
    paragraphs = []
    for speaker, bits in turns:
        text = re.sub(r'\s+', ' ', ' '.join(bits)).strip()
        if not text:
            continue
        paragraphs.append(f'{speaker}: {text}' if labelled and speaker else text)
    return cap_text('\n'.join(paragraphs), cap)


def cap_text(text: str, cap: int = VERBATIM_CAP) -> str:
    """Cut ``text`` to ``cap`` characters at a sentence end when one falls in
    the second half, else at a word boundary, and mark the cut with an
    ellipsis. Text within the cap is returned unchanged."""
    text = text or ''
    if len(text) <= cap:
        return text
    if cap <= 1:
        return ELLIPSIS
    window = text[:cap - 1]
    best = -1
    for m in re.finditer(r'[.!?]["\')\]]*(?=\s)', window):
        best = m.end()
    if best >= (cap - 1) // 2:
        return window[:best].rstrip() + ELLIPSIS
    cut = window.rfind(' ')
    if cut < (cap - 1) // 2:
        cut = cap - 1
    return window[:cut].rstrip(' ,;:') + ELLIPSIS


def compose_clip_note(short_note: str, speaker: str = '', verbatim: str = '') -> str:
    """Line 1 (note, plus " — Speaker" when known), blank line, verbatim."""
    note = re.sub(r'\s+', ' ', str(short_note or '')).strip()
    speaker = re.sub(r'\s+', ' ', str(speaker or '')).strip()
    if speaker:
        line1 = f'{note}{SPEAKER_SEP}{speaker}' if note else speaker
    else:
        line1 = note
    verbatim = (verbatim or '').strip()
    if not verbatim:
        return line1
    return f'{line1}\n\n{verbatim}' if line1 else verbatim
