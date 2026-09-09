"""Inline transcript correction.

Text correction only. Media timing never changes: no operation here alters
an existing word's ``start`` or ``end``. Every mutation goes through
``app.update_project`` under the project lock, marks ``derived_stale`` so the
next Analyze re-runs, and rebuilds ``paragraph_index.json`` so chat retrieval
sees the corrected wording immediately.

Routes (all under ``/project/<project_id>/transcript/``):

- ``GET  paragraph-html?start=<sec>`` re-rendered paragraph partial for the
  paragraph containing that time (same markup as the page render).

The Flask ``app`` module registers ``transcript_edit_bp`` itself; helpers are
imported lazily to avoid the circular import at module load.
"""
from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

transcript_edit_bp = Blueprint('transcript_edit', __name__)


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


@transcript_edit_bp.route('/project/<project_id>/transcript/paragraph-html', methods=['GET'])
def paragraph_html(project_id):
    """Re-render one paragraph so the page can swap it in place after an edit.

    Query: ``start`` (seconds, required) picks the paragraph containing that
    time. ``multi=1`` plus ``color=<name>`` reproduce the multi-project badge
    the page render adds. Returns ``{html, start, first_index, segment_count}``.
    """
    core = _core()
    project = core.get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404
    try:
        t = float(request.args.get('start', ''))
    except (TypeError, ValueError):
        return jsonify({'error': 'start (seconds) is required'}), 400

    segments = _segments(project)
    if not segments:
        return jsonify({'error': 'No transcript available'}), 400

    paragraphs = core.group_into_paragraphs(segments)
    para = _paragraph_for_time(paragraphs, t)
    if para is None:
        return jsonify({'error': 'No paragraph at that time'}), 404

    is_multi = request.args.get('multi') == '1'
    if is_multi:
        para['project_id'] = project_id
        para['project_name'] = project.get('name', 'Untitled')
        color = request.args.get('color') or 'accent'
        para['project_color'] = color if color.replace('-', '').isalnum() else 'accent'

    html = render_template(
        '_transcript_paragraphs.html',
        paragraphs=[para],
        project=project,
        is_multi=is_multi,
    )
    return jsonify({
        'html': html,
        'start': para['start'],
        'first_index': para.get('first_index', -1),
        'segment_count': len(para['segments']),
    })
