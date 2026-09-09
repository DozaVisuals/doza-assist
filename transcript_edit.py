"""Inline transcript correction.

Text correction only. Media timing never changes: no operation here alters
an existing word's ``start`` or ``end``. Every mutation runs inside
``app.update_project`` (under the project lock, via its ``mutate`` hook),
marks ``derived_stale`` so the next Analyze re-runs, stamps
``transcript_edited_at``, and rebuilds ``paragraph_index.json`` so chat
retrieval sees the corrected wording immediately.

Routes (all under ``/project/<project_id>/transcript/``):

- ``GET  paragraph-html?start=<sec>[&end=<sec>]`` re-rendered paragraph
  partial for the paragraphs overlapping that range (same markup as the page).
- ``POST edit-word``   ``{seg, w, expected, text}``; ``w`` omitted or -1 edits
  a segment with no ``words[]`` (``expected`` is then the segment text).
- ``POST delete-word`` ``{seg, w, expected}``; the inverse of insert-word.
- ``POST split-segment`` ``{seg, at_w, new_speaker?}``; ``words[at_w]`` starts
  a new segment. Both halves take start/end from their first/last word.
- ``POST merge-segment`` ``{seg}``; joins ``seg`` and ``seg + 1`` (same raw
  speaker only). The inverse of split-segment.
- ``POST reassign-speaker`` ``{seg, speaker}``; a raw label or ``__new__``.
  Marks the segment ``speaker_manual`` so diarization leaves it alone.
- ``GET  speakers`` raw labels in use with their display names, for pickers.

Payload conventions: ``seg`` indexes ``transcript.segments``, ``w`` indexes
``segments[seg].words``; ``expected`` is the text the client believes is
there and is compared with leading/trailing whitespace ignored. A mismatch
answers 409 with the current text so the page can refresh and retry.

The Flask ``app`` module registers ``transcript_edit_bp`` itself; helpers are
imported lazily to avoid the circular import at module load.
"""
from __future__ import annotations

import re
from datetime import datetime

from flask import Blueprint, jsonify, render_template, request

transcript_edit_bp = Blueprint('transcript_edit', __name__)


class EditError(Exception):
    """A rejected edit: ``status`` is the HTTP code, ``payload`` the JSON body."""

    def __init__(self, status, payload):
        super().__init__(payload.get('error', 'edit rejected'))
        self.status = status
        self.payload = payload


def _core():
    """The running app module (lazy: app.py imports this module)."""
    import app as _app
    return _app


def _segments(project):
    transcript = (project or {}).get('transcript') or {}
    segs = transcript.get('segments')
    return segs if isinstance(segs, list) else []


def _paragraph_for_time(paragraphs, t):
    """The paragraph whose range contains ``t`` (its start <= t < next start).
    Falls back to the last paragraph starting at or before ``t``."""
    chosen = None
    for para in paragraphs:
        if para['start'] <= t + 1e-6:
            chosen = para
        else:
            break
    return chosen


def _paragraph_start_for_segment(segments, seg_index):
    """Start time of the paragraph that contains segment ``seg_index``."""
    core = _core()
    for para in core.group_into_paragraphs(segments):
        first = para.get('first_index', 0)
        if first <= seg_index < first + len(para['segments']):
            return para['start']
    return segments[seg_index]['start'] if 0 <= seg_index < len(segments) else 0.0


def _rebuild_text(words):
    """Segment text from its words. Engines store one leading space per
    word (Parakeet) or bare tokens (Whisper); both normalise to single
    spaces between stripped tokens, which is what ``seg.text`` held."""
    return ' '.join(w.get('word', '').strip() for w in words if w.get('word', '').strip())


def _keep_space_convention(reference, text):
    """Store ``text`` the way its neighbours are stored: with the leading
    space Parakeet words carry (parakeet_worker.py) or bare."""
    clean = text.strip()
    return (' ' + clean) if str(reference or '').startswith(' ') else clean


def _norm(text):
    return str(text if text is not None else '').strip()


def _format_timestamp(seconds):
    from transcribe import format_timestamp
    return format_timestamp(seconds)


def _set_bounds_from_words(seg):
    """Segment start/end (and the formatted twins) from its first/last
    word. Word times themselves are never touched."""
    words = seg.get('words') or []
    if not words:
        return
    seg['start'] = words[0]['start']
    seg['end'] = words[-1]['end']
    seg['start_formatted'] = _format_timestamp(seg['start'])
    seg['end_formatted'] = _format_timestamp(seg['end'])


_LABEL_RE = re.compile(r'^SPEAKER_(\d+)$')
NEW_SPEAKER = '__new__'


def _mint_speaker_label(project, segments):
    """Lowest unused ``SPEAKER_NN`` across the segments and the diarization
    speaker list. Added to ``meta.diarization.speakers`` when that list
    exists; ``speaker_names`` is left unmapped so the raw label shows until
    the user renames it through the existing rename path."""
    used = set()
    for seg in segments:
        m = _LABEL_RE.match(str(seg.get('speaker', '')))
        if m:
            used.add(int(m.group(1)))
    diar = project.get('diarization') if isinstance(project.get('diarization'), dict) else None
    diar_list = diar.get('speakers') if diar and isinstance(diar.get('speakers'), list) else None
    for label in diar_list or []:
        m = _LABEL_RE.match(str(label))
        if m:
            used.add(int(m.group(1)))
    n = 0
    while n in used:
        n += 1
    label = f'SPEAKER_{n:02d}'
    if diar_list is not None:
        diar_list.append(label)
    return label


def _apply_speaker(project, segments, seg, speaker):
    """Reassign rules shared by split-segment and reassign-speaker:
    ``speaker`` is a raw label or ``__new__`` (mint one). Marks the segment
    ``speaker_manual`` so a later diarization pass leaves it alone."""
    label = _norm(speaker)
    if not label:
        raise EditError(400, {'error': 'speaker is required'})
    if label == NEW_SPEAKER:
        label = _mint_speaker_label(project, segments)
    seg['speaker'] = label
    seg['speaker_manual'] = True
    return label


def _speaker_list(project, segments):
    names = project.get('speaker_names') if isinstance(project.get('speaker_names'), dict) else {}
    seen = []
    for seg in segments:
        raw = seg.get('speaker')
        if raw and raw not in seen:
            seen.append(raw)
    diar = project.get('diarization') if isinstance(project.get('diarization'), dict) else {}
    for raw in (diar.get('speakers') or []):
        if raw and raw not in seen and any(s.get('speaker') == raw for s in segments):
            seen.append(raw)
    return [{'raw': raw, 'display': names.get(raw) or raw} for raw in seen]


def _int_field(payload, key, default=None, required=False):
    value = payload.get(key, default)
    if value is None:
        if required:
            raise EditError(400, {'error': f'{key} is required'})
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        raise EditError(400, {'error': f'{key} must be an integer'})


def _seg_at(segments, seg_index):
    if not (0 <= seg_index < len(segments)):
        raise EditError(404, {'error': f'No segment {seg_index}'})
    seg = segments[seg_index]
    if not isinstance(seg, dict):
        raise EditError(500, {'error': f'Segment {seg_index} is malformed'})
    return seg


def _words_of(seg):
    words = seg.get('words')
    return words if isinstance(words, list) and words else None


def _check_expected(expected, current, what):
    if _norm(expected) != _norm(current):
        raise EditError(409, {
            'error': f'{what} changed since the page loaded',
            'current': current,
        })


def _apply(project_id, op):
    """Run ``op(project, segments) -> dict`` inside update_project's locked
    read-modify-write, then rebuild the paragraph index. Returns
    ``(status, body)``."""
    core = _core()
    result = {}

    def mutate(current):
        segments = _segments(current)
        if not segments:
            raise EditError(400, {'error': 'No transcript available'})
        result.update(op(current, segments) or {})
        current['derived_stale'] = True
        current['transcript_edited_at'] = datetime.now().isoformat()

    try:
        merged = core.update_project(project_id, {}, mutate=mutate)
    except EditError as e:
        return e.status, e.payload
    if merged is None:
        return 404, {'error': 'Project not found'}

    # Chat retrieval reads this index, so rebuild it now (as /transcribe and
    # /analyze do) rather than leaving pre-edit wording in the ranker.
    try:
        from doza_assist.retrieval import build_paragraph_index, save_index
        save_index(build_paragraph_index(merged['transcript']),
                   core._paragraph_index_path(project_id))
    except Exception as ie:  # pragma: no cover - index rebuild is best effort
        print(f"[transcript-edit] paragraph index rebuild failed: {ie}", flush=True)

    result.setdefault('status', 'ok')
    result['derived_stale'] = True
    result['transcript_edited_at'] = merged.get('transcript_edited_at')
    result['has_analysis'] = bool(merged.get('analysis'))
    return 200, result


def _respond(project_id, op):
    core = _core()
    if core.get_project(project_id) is None:
        return jsonify({'error': 'Project not found'}), 404
    status, body = _apply(project_id, op)
    return jsonify(body), status


def _payload():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _paragraph_bounds(segments, seg_index, last_index=None):
    """(start of the paragraph holding ``seg_index``, end of the paragraph
    holding ``last_index``) so the page knows which blocks to re-render."""
    core = _core()
    if last_index is None:
        last_index = seg_index
    start = end = None
    for para in core.group_into_paragraphs(segments):
        first = para.get('first_index', 0)
        last = first + len(para['segments']) - 1
        if start is None and first <= seg_index <= last:
            start = para['start']
        if first <= last_index <= last:
            end = para['segments'][-1].get('end', para['start'])
    if start is None:
        start = segments[seg_index]['start']
    if end is None:
        end = segments[min(last_index, len(segments) - 1)].get('end', start)
    return start, end


def _segment_result(segments, seg_index, last_index=None, **extra):
    start, end = _paragraph_bounds(segments, seg_index, last_index)
    out = {
        'seg': seg_index,
        'segment': segments[seg_index],
        'paragraph_start': start,
        'paragraph_end': end,
    }
    out.update(extra)
    return out


# ── read ────────────────────────────────────────────────────────────────

@transcript_edit_bp.route('/project/<project_id>/transcript/paragraph-html', methods=['GET'])
def paragraph_html(project_id):
    """Re-render the paragraphs overlapping ``[start, end]`` (``end``
    defaults to ``start``) so the page can swap them in place after an edit.

    ``multi=1`` plus ``color=<name>`` reproduce the multi-project badge the
    page render adds. Returns ``{html, start, end, first_index,
    segment_count, paragraph_count}`` where ``start``/``end`` bound the
    returned paragraphs so the client knows which blocks to replace.
    """
    core = _core()
    project = core.get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404
    try:
        t0 = float(request.args.get('start', ''))
        t1 = float(request.args.get('end', t0))
    except (TypeError, ValueError):
        return jsonify({'error': 'start (seconds) is required'}), 400
    if t1 < t0:
        t0, t1 = t1, t0

    segments = _segments(project)
    if not segments:
        return jsonify({'error': 'No transcript available'}), 400

    paragraphs = core.group_into_paragraphs(segments)
    first = _paragraph_for_time(paragraphs, t0)
    if first is None:
        return jsonify({'error': 'No paragraph at that time'}), 404
    chosen = []
    for para in paragraphs:
        if para is first or (para['start'] > first['start'] and para['start'] <= t1 + 1e-6):
            chosen.append(para)

    is_multi = request.args.get('multi') == '1'
    if is_multi:
        color = request.args.get('color') or 'accent'
        if not color.replace('-', '').isalnum():
            color = 'accent'
        for para in chosen:
            para['project_id'] = project_id
            para['project_name'] = project.get('name', 'Untitled')
            para['project_color'] = color

    html = render_template(
        '_transcript_paragraphs.html',
        paragraphs=chosen,
        project=project,
        is_multi=is_multi,
    )
    last = chosen[-1]
    return jsonify({
        'html': html,
        'start': first['start'],
        'end': last['segments'][-1].get('end', last['start']),
        'first_index': first.get('first_index', -1),
        'segment_count': sum(len(p['segments']) for p in chosen),
        'paragraph_count': len(chosen),
    })


# ── edit-word / delete-word ─────────────────────────────────────────────

@transcript_edit_bp.route('/project/<project_id>/transcript/edit-word', methods=['POST'])
def edit_word(project_id):
    """Replace one word's text. ``{seg, w, expected, text}``; with ``w``
    absent or -1 the segment's own text is edited (segments without
    ``words[]``). Timing fields are never touched."""
    data = _payload()
    seg_index = _int_field(data, 'seg', required=True)
    w_index = _int_field(data, 'w', default=-1)
    expected = data.get('expected')
    text = _norm(data.get('text'))
    if not text:
        return jsonify({'error': 'text must not be empty'}), 400

    def op(project, segments):
        seg = _seg_at(segments, seg_index)
        words = _words_of(seg)
        if w_index is not None and w_index >= 0:
            if not words or w_index >= len(words):
                raise EditError(404, {'error': f'No word {w_index} in segment {seg_index}'})
            word = words[w_index]
            _check_expected(expected, word.get('word', ''), 'That word')
            word['word'] = _keep_space_convention(word.get('word', ''), text)
            seg['text'] = _rebuild_text(words)
            return _segment_result(segments, seg_index, w=w_index, word=word)
        _check_expected(expected, seg.get('text', ''), 'That segment')
        if words:
            raise EditError(400, {'error': 'Segment has words; edit them individually'})
        seg['text'] = text
        return _segment_result(segments, seg_index, w=-1)

    return _respond(project_id, op)


@transcript_edit_bp.route('/project/<project_id>/transcript/delete-word', methods=['POST'])
def delete_word(project_id):
    """Remove one word (the inverse of insert-word). ``{seg, w, expected}``.
    Refuses to empty a segment. Neighbouring word times and the segment's
    own start/end are left exactly as they were."""
    data = _payload()
    seg_index = _int_field(data, 'seg', required=True)
    w_index = _int_field(data, 'w', required=True)
    expected = data.get('expected')

    def op(project, segments):
        seg = _seg_at(segments, seg_index)
        words = _words_of(seg)
        if not words or not (0 <= w_index < len(words)):
            raise EditError(404, {'error': f'No word {w_index} in segment {seg_index}'})
        if len(words) == 1:
            raise EditError(400, {'error': 'Cannot delete the only word in a segment'})
        _check_expected(expected, words[w_index].get('word', ''), 'That word')
        removed = words.pop(w_index)
        seg['text'] = _rebuild_text(words)
        return _segment_result(segments, seg_index, w=w_index, removed=removed)

    return _respond(project_id, op)


# ── split-segment / merge-segment ───────────────────────────────────────

@transcript_edit_bp.route('/project/<project_id>/transcript/split-segment', methods=['POST'])
def split_segment(project_id):
    """Split so ``words[at_w]`` begins a new segment. ``{seg, at_w,
    new_speaker?}``. Both halves take start/end from their first/last word;
    the new segment inherits the speaker unless ``new_speaker`` (a raw label
    or ``__new__``) is given, in which case it is marked ``speaker_manual``.
    ``at_w`` must be at least 1 and inside the segment."""
    data = _payload()
    seg_index = _int_field(data, 'seg', required=True)
    at_w = _int_field(data, 'at_w', required=True)
    new_speaker = data.get('new_speaker')

    def op(project, segments):
        seg = _seg_at(segments, seg_index)
        words = _words_of(seg)
        if not words:
            raise EditError(400, {'error': 'Segment has no words to split on'})
        if at_w <= 0 or at_w >= len(words):
            raise EditError(400, {'error': f'at_w must be between 1 and {len(words) - 1}'})
        right = {k: v for k, v in seg.items() if k != 'words'}
        right['words'] = words[at_w:]
        seg['words'] = words[:at_w]
        for half in (seg, right):
            half['text'] = _rebuild_text(half['words'])
            _set_bounds_from_words(half)
        segments.insert(seg_index + 1, right)
        label = None
        if new_speaker is not None and _norm(new_speaker):
            label = _apply_speaker(project, segments, right, new_speaker)
        return _segment_result(segments, seg_index, last_index=seg_index + 1,
                               new_seg=seg_index + 1, new_segment=right,
                               speaker=right.get('speaker'), speaker_changed=label is not None,
                               speakers=_speaker_list(project, segments))

    return _respond(project_id, op)


@transcript_edit_bp.route('/project/<project_id>/transcript/merge-segment', methods=['POST'])
def merge_segment(project_id):
    """Join ``seg`` and ``seg + 1`` (the inverse of split-segment). ``{seg}``.
    Only when both share a raw speaker; the merged segment keeps
    ``speaker_manual`` if either half had it. Word times are untouched."""
    data = _payload()
    seg_index = _int_field(data, 'seg', required=True)

    def op(project, segments):
        left = _seg_at(segments, seg_index)
        if seg_index + 1 >= len(segments):
            raise EditError(400, {'error': 'No following segment to merge with'})
        right = _seg_at(segments, seg_index + 1)
        if _norm(left.get('speaker')) != _norm(right.get('speaker')):
            raise EditError(409, {
                'error': 'Segments have different speakers; reassign one first',
                'current': [left.get('speaker'), right.get('speaker')],
            })
        lw, rw = _words_of(left), _words_of(right)
        if (lw is None) != (rw is None):
            raise EditError(400, {'error': 'Cannot merge a worded segment with a segment-level one'})
        if lw:
            left['words'] = lw + rw
            left['text'] = _rebuild_text(left['words'])
            _set_bounds_from_words(left)
        else:
            left['text'] = _norm(left.get('text', '')) + ' ' + _norm(right.get('text', ''))
            left['end'] = right.get('end', left.get('end'))
            left['end_formatted'] = _format_timestamp(left['end'])
        if left.get('speaker_manual') or right.get('speaker_manual'):
            left['speaker_manual'] = True
        segments.pop(seg_index + 1)
        return _segment_result(segments, seg_index, merged_from=seg_index + 1)

    return _respond(project_id, op)


@transcript_edit_bp.route('/project/<project_id>/transcript/speakers', methods=['GET'])
def speakers(project_id):
    """Raw speaker labels in use, each with its display name."""
    core = _core()
    project = core.get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404
    return jsonify({'speakers': _speaker_list(project, _segments(project)),
                    'new_speaker': NEW_SPEAKER})


# ── reassign-speaker ────────────────────────────────────────────────────

@transcript_edit_bp.route('/project/<project_id>/transcript/reassign-speaker', methods=['POST'])
def reassign_speaker(project_id):
    """Give one segment a different speaker. ``{seg, speaker}`` where
    ``speaker`` is a raw label (``SPEAKER_NN``) or ``__new__`` (mint the
    lowest unused one). Sets ``speaker_manual`` so a later diarization pass
    keeps it (``speaker_manual: false`` in the body clears the flag; undo
    uses that to restore an untouched segment).

    Deliberately not gated on diarization state: the per-segment
    ``update-speakers`` routes refuse diarized projects because the rename
    map owns naming there; this route writes raw labels, which the map still
    resolves for display."""
    data = _payload()
    seg_index = _int_field(data, 'seg', required=True)
    speaker = data.get('speaker')
    manual = data.get('speaker_manual', True)

    def op(project, segments):
        seg = _seg_at(segments, seg_index)
        pre_start, pre_end = _paragraph_bounds(segments, seg_index)
        old = seg.get('speaker')
        old_manual = bool(seg.get('speaker_manual'))
        label = _apply_speaker(project, segments, seg, speaker)
        if manual is False:
            seg.pop('speaker_manual', None)
        result = _segment_result(segments, seg_index, speaker=label, previous_speaker=old,
                                 previous_manual=old_manual, speaker_changed=label != old,
                                 speakers=_speaker_list(project, segments))
        # A speaker change can re-paragraph its neighbours: re-render the
        # union of the paragraph it was in and the one it is in now.
        result['paragraph_start'] = min(pre_start, result['paragraph_start'])
        result['paragraph_end'] = max(pre_end, result['paragraph_end'])
        return result

    return _respond(project_id, op)
