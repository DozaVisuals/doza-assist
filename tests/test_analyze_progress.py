"""Tests for the analyze progress writer + status endpoint.

The endpoint is what powers the live progress bar — verify the contract:
- /analyze/status returns {idle: true} when no run is active
- During a run, returns {step, total, current, started_at, updated_at, done}
- Progress callback receives kwargs (step, total, current) — same contract
  the analyzer enforces internally
"""

import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


@pytest.fixture
def project_dir(client):
    pid = "progress-test"
    d = Path(app_module.app.config['PROJECTS_DIR']) / pid
    d.mkdir(parents=True, exist_ok=True)
    return pid, d


def test_status_returns_idle_when_no_run(client, project_dir):
    pid, _ = project_dir
    resp = client.get(f'/project/{pid}/analyze/status')
    assert resp.status_code == 200
    assert resp.get_json() == {'idle': True}


def test_progress_writer_persists_step(client, project_dir):
    pid, d = project_dir
    write = app_module._make_progress_writer(pid)
    write(step=2, total=10, current="chunk 1/3: story beats")
    status = json.loads((d / 'analyze_status.json').read_text())
    assert status['step'] == 2
    assert status['total'] == 10
    assert status['current'] == "chunk 1/3: story beats"
    assert status['done'] is False
    assert 'started_at' in status and 'updated_at' in status


def test_done_flag_set_when_step_reaches_total(client, project_dir):
    pid, _ = project_dir
    write = app_module._make_progress_writer(pid)
    write(step=10, total=10, current="finishing")
    resp = client.get(f'/project/{pid}/analyze/status')
    body = resp.get_json()
    assert body['done'] is True
    assert body['step'] == 10
    assert body['total'] == 10


def test_status_endpoint_returns_latest_snapshot(client, project_dir):
    pid, _ = project_dir
    write = app_module._make_progress_writer(pid)
    write(step=1, total=5, current="starting")
    write(step=3, total=5, current="halfway")
    resp = client.get(f'/project/{pid}/analyze/status')
    body = resp.get_json()
    assert body['step'] == 3
    assert body['current'] == "halfway"


def test_clear_status_removes_file(client, project_dir):
    pid, d = project_dir
    write = app_module._make_progress_writer(pid)
    write(step=1, total=5, current="starting")
    assert (d / 'analyze_status.json').exists()
    app_module._clear_analyze_status(pid)
    assert not (d / 'analyze_status.json').exists()
    # Idempotent — clearing twice doesn't raise.
    app_module._clear_analyze_status(pid)


def test_expected_vector_chunks_short_transcript():
    from ai_analysis import expected_vector_chunks
    short = {'segments': [{'start': 0, 'end': 60, 'text': 'x'}]}
    assert expected_vector_chunks(short) == 1


def test_expected_vector_chunks_long_transcript():
    from ai_analysis import expected_vector_chunks
    # 60-min transcript → ~4 chunks at the 15-min boundary.
    segs = [
        {'start': i * 5, 'end': (i + 1) * 5, 'text': f's{i}'}
        for i in range(60 * 12)  # 60 min @ 5s segments
    ]
    n = expected_vector_chunks({'segments': segs})
    assert n >= 3 and n <= 6  # exact count depends on chunk boundary math


def test_vector_progress_callback_fires_per_chunk(monkeypatch):
    """The vector phase has to bump the bar per Ollama call. Otherwise the
    bar stalls at ~94% while 7 chunk-classification calls grind for minutes —
    which was the regression the user just reported.
    """
    import ai_analysis

    chunk_calls = []

    def _stub_single_chunk(text, name):
        chunk_calls.append(name)
        return [{'seg_id': f'SEG{len(chunk_calls):03d}', 'timecode_in': '0:0:0', 'timecode_out': '0:0:5'}]

    monkeypatch.setattr(ai_analysis, '_generate_vectors_single_chunk', _stub_single_chunk)

    # 90-min transcript triggers chunked path.
    segs = [
        {'start': i * 10, 'end': (i + 1) * 10, 'text': f's{i}', 'speaker': 'A'}
        for i in range(60 * 9)  # 90 min @ 10s segments
    ]

    progress_events = []
    def _track(chunk_idx, total_chunks, label):
        progress_events.append((chunk_idx, total_chunks, label))

    ai_analysis.generate_segment_vectors(
        {'segments': segs}, project_name='P', progress_callback=_track,
    )
    # One progress event per chunk, monotonically increasing.
    assert len(progress_events) == len(chunk_calls)
    assert [e[0] for e in progress_events] == list(range(1, len(chunk_calls) + 1))
    assert all(e[1] == len(chunk_calls) for e in progress_events)


def test_progress_callback_invoked_during_analyze(client, project_dir, monkeypatch):
    """End-to-end check: when /analyze runs, the progress writer fires at
    every milestone the frontend will display."""
    pid, _ = project_dir

    # Make this look like a project with a transcript.
    meta_path = Path(app_module.app.config['PROJECTS_DIR']) / pid / 'meta.json'
    meta_path.write_text(json.dumps({
        'id': pid, 'name': 'P',
        'transcript': {'segments': [
            {'start': 0, 'end': 5, 'text': 'a', 'speaker': 'X'},
            {'start': 5, 'end': 10, 'text': 'b', 'speaker': 'X'},
        ]},
    }))

    captured = []

    def _spy_analyze(*a, **k):
        # Echo a few progress events through the callback so we can verify it
        # was wired through.
        cb = k.get('progress_callback')
        if cb:
            cb(step=0, total=3, current="starting")
            cb(step=1, total=3, current="story beats")
            cb(step=2, total=3, current="ranking")
        captured.append(k)
        return {'summary': 's', 'story_beats': []}

    monkeypatch.setattr('ai_analysis.analyze_transcript', _spy_analyze)
    monkeypatch.setattr('ai_analysis.generate_segment_vectors', lambda *a, **k: [])

    resp = client.post(f'/project/{pid}/analyze', json={'type': 'all'})
    assert resp.status_code == 200
    # The route augments the analyzer's total with 2 finalization steps.
    # After completion, _clear_analyze_status removes the file.
    status_resp = client.get(f'/project/{pid}/analyze/status')
    assert status_resp.get_json() == {'idle': True}
    # And the progress_callback was wired through to analyze_transcript.
    assert 'progress_callback' in captured[0]
    assert callable(captured[0]['progress_callback'])
