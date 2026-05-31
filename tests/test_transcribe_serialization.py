"""Tests for the process-global transcription gate.

Transcription engines share one in-process model singleton and one Metal
GPU. Two transcriptions running at once means two threads driving the same
MLX model — which deadlocks, the symptom users hit when adding a folder of
clips (every clip's background thread launches at once and they all freeze
at the "load_model" phase). ``_run_transcribe_job`` must serialize so a
second job parks at phase="queued" until the first finishes.
"""

import os
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module  # noqa: E402
import transcribe as transcribe_module  # noqa: E402


@pytest.fixture
def configured(tmp_path, monkeypatch):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    # Reset the gate + job table between tests so a leaked lock from a
    # previous test can't wedge this one.
    app_module._transcribe_run_lock = threading.Lock()
    with app_module._transcribe_jobs_lock:
        app_module._transcribe_jobs.clear()
    # Don't let the worker's downstream steps (project save, diarization,
    # paragraph index) run — we only care about the transcribe gate.
    monkeypatch.setattr(app_module, 'get_project', lambda pid: {'name': pid})
    monkeypatch.setattr(app_module, 'save_project', lambda pid, p: None)
    monkeypatch.setattr(app_module, 'log_activity', lambda *a, **k: None)
    return app_module


def test_only_one_transcription_runs_at_a_time(configured, monkeypatch):
    """Two jobs launched together must not execute transcribe_file
    concurrently; peak concurrency stays at 1."""
    active = {'n': 0, 'peak': 0}
    lock = threading.Lock()

    def fake_transcribe_file(*args, **kwargs):
        with lock:
            active['n'] += 1
            active['peak'] = max(active['peak'], active['n'])
        time.sleep(0.2)  # hold the "engine" so overlap would be observable
        with lock:
            active['n'] -= 1
        return {'segments': [], 'language': 'en', 'duration': 0}

    monkeypatch.setattr(transcribe_module, 'transcribe_file', fake_transcribe_file)

    threads = [
        threading.Thread(
            target=configured._run_transcribe_job,
            args=(f'proj-{i}', f'/tmp/src-{i}.mov', 2, 'en', 'Interviewer', 'Subject'),
        )
        for i in range(3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert active['peak'] == 1, "transcriptions ran concurrently — the gate failed"


def test_contended_job_reports_queued_phase(configured, monkeypatch):
    """A job that has to wait for the gate surfaces phase='queued' so the
    frontend poll shows 'Queued…' instead of a frozen 'Load_model…'."""
    seen_queued = threading.Event()
    release_first = threading.Event()

    def fake_transcribe_file(*args, **kwargs):
        # First job holds the gate until we observe the second go to queued.
        release_first.wait(timeout=5)
        return {'segments': [], 'language': 'en', 'duration': 0}

    monkeypatch.setattr(transcribe_module, 'transcribe_file', fake_transcribe_file)

    t1 = threading.Thread(
        target=configured._run_transcribe_job,
        args=('first', '/tmp/a.mov', 2, 'en', 'I', 'S'),
    )
    t1.start()
    # Give job 1 time to grab the gate.
    time.sleep(0.1)

    def watch_second():
        # Poll the in-memory job table for the queued phase.
        for _ in range(50):
            snap = configured._transcribe_jobs.get('second') or {}
            if snap.get('phase') == 'queued':
                seen_queued.set()
                return
            time.sleep(0.02)

    watcher = threading.Thread(target=watch_second)
    watcher.start()
    t2 = threading.Thread(
        target=configured._run_transcribe_job,
        args=('second', '/tmp/b.mov', 2, 'en', 'I', 'S'),
    )
    t2.start()

    watcher.join(timeout=5)
    assert seen_queued.is_set(), "second job never reported phase='queued' while waiting"

    release_first.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
