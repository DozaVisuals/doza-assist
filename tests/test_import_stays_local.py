"""FCPXML import cannot land a project on a cloud provider (Rule 1), so the
leave-local dialog needs no import wiring. Ingest is faked; the point is the
meta the creation funnel writes for an imported timeline."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402
import ai_providers  # noqa: E402


@pytest.fixture
def projects(tmp_path):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    app_module.app.config['EXPORTS_DIR'] = str(tmp_path / 'exports')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    Path(app_module.app.config['EXPORTS_DIR']).mkdir(parents=True, exist_ok=True)
    return Path(app_module.app.config['PROJECTS_DIR'])


def _read_meta(projects, pid):
    return json.loads((projects / pid / 'meta.json').read_text())


def test_fcpxml_import_starts_local_and_private(projects, tmp_path, monkeypatch):
    # The app-wide file says Anthropic, as an older build might have left it.
    monkeypatch.setattr(ai_providers, 'load_provider_config', lambda: {
        'active_provider': 'anthropic', 'ollama': {}, 'anthropic': {'api_key': 'k'}, 'openai': {}})
    monkeypatch.setattr(app_module, 'get_media_duration', lambda p: None)
    monkeypatch.setattr(app_module, 'get_video_framerate', lambda p: None)
    monkeypatch.setattr(app_module, 'get_video_resolution', lambda p: (1920, 1080))
    monkeypatch.setattr(app_module, 'get_video_start_timecode_info', lambda p, fps: (0, None))

    def fake_ingest(fcpxml_path, project_dir, event_clip_index=None):
        audio = Path(project_dir) / 'timeline_audio.wav'
        audio.write_bytes(b'RIFF' + b'\0' * 32)
        return {'audio_path': str(audio),
                'fcpxml_source': {'container_type': 'sync-clip', 'project_name': 'Imported cut',
                                  'stored_fcpxml_path': fcpxml_path, 'parse_warnings': []}}
    monkeypatch.setattr(app_module, '_ingest_fcpxml', fake_ingest)

    src = tmp_path / 'cut.fcpxml'
    src.write_text('<fcpxml version="1.14"/>')
    pid = app_module.create_project_from_path(str(src))
    meta = _read_meta(projects, pid)
    assert meta['fcpxml_source']['project_name'] == 'Imported cut'
    assert meta['name'] == 'Imported cut'
    assert meta['ai_provider'] == 'ollama'
    assert 'ai_provider_confirmed' not in meta
    assert 'studio' not in meta
    assert ai_providers.provider_for_project(meta) == 'ollama'
