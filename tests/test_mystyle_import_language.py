"""My Style (Editorial DNA) import must auto-detect language, not force English.

The profile-import path used to call transcribe_file with no language arg,
defaulting to 'en' → English-only Parakeet, which garbled non-English finished
cuts before a profile was built. The import now passes language='auto' so the
engine auto-detects (and routes non-English onto WhisperX). This guards that
the import path invokes transcription with auto-detect, not hardcoded 'en'.
"""

import io
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('DOZA_DATA_DIR', str(tmp_path))
    import importlib
    import app as app_module
    importlib.reload(app_module)
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


def test_my_style_import_uses_auto_language_detection(client, monkeypatch):
    """The import's transcription call must receive language='auto'."""
    import transcribe as transcribe_module

    captured = {}

    # No ffmpeg on the fake bytes — return a stand-in audio path.
    monkeypatch.setattr(transcribe_module, 'extract_audio',
                        lambda path, project_dir=None, **kw: path)

    def _capture(filepath, project_dir=None, **kwargs):
        captured['language'] = kwargs.get('language', '__missing__')
        # Short-circuit the heavy downstream (analyze/classify/summarize);
        # the route's per-file try/except records this file as an error and
        # the stream still completes, but the kwarg is already captured.
        raise RuntimeError('stop after capture')

    monkeypatch.setattr(transcribe_module, 'transcribe_file', _capture)

    data = {'files': (io.BytesIO(b'\x00' * 64), 'finished_cut.mp4')}
    resp = client.post('/my-style/import', data=data,
                       content_type='multipart/form-data')
    # Consume the streaming body so the generator runs.
    _ = resp.get_data()

    assert captured.get('language') == 'auto', (
        f"My Style import must transcribe with auto-detect, got "
        f"{captured.get('language')!r}")


def test_project_worker_transcription_default_unchanged(monkeypatch):
    """Fix is scoped to the import site: transcribe_file's own default stays
    'en' for every other caller (no model-routing change elsewhere)."""
    import inspect
    import transcribe as transcribe_module
    sig = inspect.signature(transcribe_module.transcribe_file)
    assert sig.parameters['language'].default == 'en'
