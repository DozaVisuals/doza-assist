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

Payload conventions: ``seg`` indexes ``transcript.segments``, ``w`` indexes
``segments[seg].words``; ``expected`` is the text the client believes is
there and is compared with leading/trailing whitespace ignored. A mismatch
answers 409 with the current text so the page can refresh and retry.

The Flask ``app`` module registers ``transcript_edit_bp`` itself; helpers are
imported lazily to avoid the circular import at module load.
"""
from __future__ import annotations

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


def _segment_result(segments, seg_index, **extra):
    out = {
        'seg': seg_index,
        'segment': segments[seg_index],
        'paragraph_start': _paragraph_start_for_segment(segments, seg_index),
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
