"""Regression tests for the on-demand Whisper installer flow (issue #36).

The original v3.5.8 fix surfaced an install button, but a fresh user still
dead-ended because:
  * picking "Auto-detect" ('auto') bypassed the guard and crashed downstream
    with a bare 500 and no installer; and
  * the guard persisted status='error', which hid the only card that renders
    the install button, so the affordance vanished after one attempt / reload.

These tests pin the corrected contract: every non-English request without the
engine returns a structured needs_whisper_install response AND leaves the
project renderable (status='uploaded'), while English is unaffected.
"""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as A  # noqa: E402
import transcribe as T  # noqa: E402


@pytest.fixture
def src_file():
    f = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    f.write(b'\x00' * 1024)
    f.close()
    yield f.name
    os.unlink(f.name)


@pytest.fixture
def client(monkeypatch, src_file):
    """A test client with whisper mocked MISSING and the project store stubbed.
    Captures the persisted project so tests can assert on saved status."""
    saved = {}
    state = {'project': None}

    monkeypatch.setattr(A, '_engine_available', lambda name: name == 'parakeet_mlx')
    monkeypatch.setattr(A, 'get_project', lambda pid: state['project'])
    monkeypatch.setattr(A, 'save_project', lambda pid, proj: saved.update(proj))

    c = A.app.test_client()
    c._saved = saved
    c._src = src_file
    c._state = state
    return c


def _project(client, language):
    client._state['project'] = {
        'id': 'p', 'name': 'n', 'status': 'uploaded', 'language': language,
        'num_speakers': 2, 'source_path': client._src, 'filepath': client._src,
        'interviewer_name': 'I', 'subject_name': 'S',
    }


@pytest.mark.parametrize('language', ['de', 'auto', 'fr', 'es'])
def test_non_english_without_engine_offers_installer(client, language):
    """Any non-English language (including 'auto') returns the structured
    installer response instead of dead-ending."""
    _project(client, language)
    resp = client.post('/project/p/transcribe')
    body = resp.get_json()
    assert resp.status_code == 400
    assert body['needs_whisper_install'] is True
    assert body['requested_language'] == language


@pytest.mark.parametrize('language', ['de', 'auto'])
def test_guard_keeps_project_renderable(client, language):
    """The guard must NOT persist status='error' — that hid the install card on
    reload (the core #36 dead-end). Status stays 'uploaded'; error is cleared."""
    _project(client, language)
    client.post('/project/p/transcribe')
    assert client._saved.get('status') == 'uploaded'
    assert client._saved.get('error') in (None, '')


def test_english_is_unaffected_by_guard(client, monkeypatch):
    """English never needs Whisper; the guard must let it through to the normal
    transcription path (mocked here so no model downloads)."""
    _project(client, 'en')
    monkeypatch.setattr(
        T, 'transcribe_file',
        lambda *a, **k: {'segments': [], 'language': 'en', 'engine': 'parakeet'})
    resp = client.post('/project/p/transcribe')
    body = resp.get_json()
    assert resp.status_code == 200
    assert not body.get('needs_whisper_install')


def test_downstream_engine_loss_still_offers_installer(client, monkeypatch):
    """Defense in depth: if a non-English job slips past the guard and fails in
    transcribe_file because the engine is missing, the route still returns the
    installer (not a raw 500) and keeps the project renderable."""
    _project(client, 'de')
    # Guard sees whisper present, so it passes; transcribe_file then fails as if
    # the engine vanished (TOCTOU). Engine probe reports missing in the except.
    calls = {'n': 0}

    def flaky_available(name):
        if name == 'parakeet_mlx':
            return True
        # whisper: present for the guard's first call, missing afterwards.
        calls['n'] += 1
        return calls['n'] == 1

    monkeypatch.setattr(A, '_engine_available', flaky_available)
    monkeypatch.setattr(T, 'transcribe_file', lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError('No transcription engine found.')))
    resp = client.post('/project/p/transcribe')
    body = resp.get_json()
    assert resp.status_code == 400
    assert body['needs_whisper_install'] is True
    assert client._saved.get('status') == 'uploaded'


def test_whisper_ready_requires_package_and_model(monkeypatch):
    """_whisper_ready gates on BOTH the package and the cached model so the UI
    doesn't report 'done' while the ~1.5GB model is still missing."""
    monkeypatch.setattr(A, '_engine_available', lambda name: True)
    monkeypatch.setattr(A, '_whisper_model_cached', lambda: False)
    assert A._whisper_ready() is False
    monkeypatch.setattr(A, '_whisper_model_cached', lambda: True)
    assert A._whisper_ready() is True


def test_stream_subprocess_captures_progress_and_returncode():
    """The installer streams live progress: tqdm-style \\r frames and \\n lines
    are both surfaced, and the child's exit code is returned."""
    details = []
    prog = (
        'import sys\n'
        'for p in (10, 55, 100):\n'
        '    sys.stdout.write(f"\\r {p}/100")\n'
        '    sys.stdout.flush()\n'
        'sys.stdout.write("\\nfinished\\n")\n'
    )
    rc, tail = A._stream_subprocess([sys.executable, '-c', prog], details.append)
    assert rc == 0
    assert details, 'expected streamed progress frames'
    assert 'finished' in tail
    assert '100/100' in tail

    rc2, _ = A._stream_subprocess([sys.executable, '-c', 'import sys; sys.exit(7)'],
                                  details.append)
    assert rc2 == 7
