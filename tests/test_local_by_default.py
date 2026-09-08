"""Local by default (feature/settings-menu-local-guard).

Rule 1: a new project starts on local Ollama whatever any other project or
app-wide file says, and every AI call path resolves the provider per project.
Rule 2: assistant access is off on a new project and nothing inherits it.
The analysis job test spawns the real background worker for a project on
local while provider_config.json says Anthropic, and asserts the Anthropic
provider is never instantiated.
"""

import contextlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402
import ai_analysis  # noqa: E402
import ai_providers  # noqa: E402
from ai_providers import anthropic_provider  # noqa: E402


@pytest.fixture
def client(tmp_path):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    app_module.app.config['EXPORTS_DIR'] = str(tmp_path / 'exports')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    Path(app_module.app.config['EXPORTS_DIR']).mkdir(parents=True, exist_ok=True)
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


def _config_says(monkeypatch, active, keys=True):
    """provider_config.json as an older build would leave it: active_provider
    set to a cloud provider, keys present. Nothing here may read active_provider."""
    cfg = {
        'active_provider': active,
        'ollama': {'model': '', 'base_url': ''},
        'anthropic': {'api_key': 'sk-ant-test' if keys else ''},
        'openai': {'api_key': 'sk-test' if keys else ''},
    }
    monkeypatch.setattr(ai_providers, 'load_provider_config', lambda: dict(cfg))
    monkeypatch.setattr(ai_providers, 'has_api_key', lambda name: bool(cfg.get(name, {}).get('api_key')))


def _make_project(pid, **extra):
    pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
    pdir.mkdir(parents=True)
    meta = {'id': pid, 'name': 'P', 'status': 'complete',
            'transcript': {'segments': [{'start': 0, 'end': 1, 'text': 'hello there'}], 'duration': 1.0}}
    meta.update(extra)
    (pdir / 'meta.json').write_text(json.dumps(meta))
    return pdir


def _read_meta(pid):
    return json.loads((Path(app_module.app.config['PROJECTS_DIR']) / pid / 'meta.json').read_text())


def _quiet_probes(monkeypatch):
    monkeypatch.setattr(app_module, 'get_media_duration', lambda p: None)
    monkeypatch.setattr(app_module, 'get_video_framerate', lambda p: None)
    monkeypatch.setattr(app_module, 'get_video_resolution', lambda p: (1920, 1080))
    monkeypatch.setattr(app_module, 'get_video_start_timecode_info', lambda p, fps: (0, None))


# ── Rule 1 and Rule 2 at creation ──────────────────────────────────────────

def test_new_project_starts_local_and_private_even_when_config_says_anthropic(client, tmp_path, monkeypatch):
    _config_says(monkeypatch, 'anthropic')
    _quiet_probes(monkeypatch)
    media = tmp_path / 'interview.wav'
    media.write_bytes(b'RIFF' + b'\0' * 64)
    pid = app_module.create_project_from_path(str(media), project_name='Fresh')
    meta = _read_meta(pid)
    assert meta['ai_provider'] == 'ollama'                # Rule 1
    assert 'ai_provider_confirmed' not in meta
    assert 'studio' not in meta                           # Rule 2: no access flag at all
    assert ai_providers.provider_for_project(meta) == 'ollama'


def test_provider_for_project_defaults_and_never_guesses():
    assert ai_providers.provider_for_project({}) == 'ollama'
    assert ai_providers.provider_for_project(None) == 'ollama'
    assert ai_providers.provider_for_project({'ai_provider': 'bogus'}) == 'ollama'
    assert ai_providers.provider_for_project({'ai_provider': 'anthropic'}) == 'anthropic'
    assert ai_providers.provider_for_project({'ai_provider': 'OpenAI '}) == 'openai'


# ── the resolver ignores the app-wide field ────────────────────────────────

def test_get_active_provider_ignores_config_active_provider(monkeypatch):
    _config_says(monkeypatch, 'anthropic')
    assert ai_providers.current_provider_name() == 'ollama'
    provider = ai_providers.get_active_provider()
    assert type(provider).__name__ == 'OllamaProvider'
    with ai_providers.using_provider('anthropic'):
        assert ai_providers.current_provider_name() == 'anthropic'
        assert type(ai_providers.get_active_provider()).__name__ == 'AnthropicProvider'
    assert ai_providers.current_provider_name() == 'ollama'
    with ai_providers.using_project_provider({'ai_provider': 'openai'}):
        assert type(ai_providers.get_active_provider()).__name__ == 'OpenAIProvider'


# ── the background analysis job ────────────────────────────────────────────

def _stub_analysis_job(monkeypatch, seen):
    def fake_analyze(transcript, **kwargs):
        seen.append(ai_providers.get_active_provider().name)
        return {'story_beats': [{'label': 'Beat', 'start': '00:00:00', 'end': '00:00:01'}],
                'social_clips': [], 'strongest_soundbites': []}

    monkeypatch.setattr(ai_analysis, 'analyze_transcript', fake_analyze)
    monkeypatch.setattr(ai_analysis, 'generate_segment_vectors', lambda *a, **k: [])
    monkeypatch.setattr(ai_analysis, 'expected_vector_chunks', lambda *a, **k: 0)
    import doza_assist.retrieval as retrieval
    monkeypatch.setattr(retrieval, 'build_paragraph_index', lambda *a, **k: {'paragraphs': []})
    monkeypatch.setattr(retrieval, 'save_index', lambda *a, **k: None)
    monkeypatch.setattr(app_module, '_rewarm_chat_after_heavy_call', lambda pid: None)
    monkeypatch.setattr(app_module, '_memory_heavy_stage', lambda *a, **k: contextlib.nullcontext())


def test_analysis_job_for_a_local_project_never_instantiates_anthropic(client, monkeypatch):
    _config_says(monkeypatch, 'anthropic')

    def boom(self, *a, **k):
        raise AssertionError('AnthropicProvider was instantiated for a local project')
    monkeypatch.setattr(anthropic_provider.AnthropicProvider, '__init__', boom)

    seen = []
    _stub_analysis_job(monkeypatch, seen)
    _make_project('local1', ai_provider='ollama')
    app_module._run_analysis_worker('local1', 'story', 'hash1', {})
    assert seen == ['ollama']
    assert _read_meta('local1')['analysis']['story_beats'][0]['label'] == 'Beat'
    # the worker's context does not leak into the caller
    assert ai_providers.current_provider_name() == 'ollama'


def test_analysis_job_reads_the_project_provider_from_meta(client, monkeypatch):
    _config_says(monkeypatch, 'ollama')  # the app-wide field says local; the project says Anthropic
    seen = []
    _stub_analysis_job(monkeypatch, seen)
    _make_project('cloud1', ai_provider='anthropic')
    app_module._run_analysis_worker('cloud1', 'story', 'hash2', {})
    assert seen == ['anthropic']


# ── request scope ──────────────────────────────────────────────────────────

def test_requests_run_with_the_named_projects_provider(client, monkeypatch):
    _config_says(monkeypatch, 'openai')
    _make_project('r1', ai_provider='anthropic')
    _make_project('r2')  # no field: local
    seen = {}
    real_label = app_module._ai_model_label

    def spy(project):
        seen[project['id']] = ai_providers.current_provider_name()
        return real_label(project)
    monkeypatch.setattr(app_module, '_ai_model_label', spy)
    assert client.get('/project/r1/settings-summary').status_code == 200
    assert client.get('/project/r2/settings-summary').status_code == 200
    assert seen == {'r1': 'anthropic', 'r2': 'ollama'}
    # and nothing lingers after the request
    assert ai_providers.current_provider_name() == 'ollama'


def test_project_ai_provider_route(client, monkeypatch):
    _config_says(monkeypatch, 'ollama', keys=False)
    _make_project('a1')
    d = client.get('/project/a1/ai-provider').get_json()
    assert d['provider'] == 'ollama' and d['confirmed'] is False
    assert d['has_anthropic_key'] is False
    # cloud without a saved key is refused; the project stays local
    res = client.put('/project/a1/ai-provider', json={'provider': 'anthropic'})
    assert res.status_code == 400 and 'ai_provider' not in _read_meta('a1')
    assert client.put('/project/a1/ai-provider', json={'provider': 'bogus'}).status_code == 400
    _config_says(monkeypatch, 'ollama', keys=True)
    d = client.put('/project/a1/ai-provider', json={'provider': 'anthropic', 'confirmed': True}).get_json()
    assert d['provider'] == 'anthropic' and d['confirmed'] is True and d['ai_model_label'] == 'Anthropic'
    meta = _read_meta('a1')
    assert meta['ai_provider'] == 'anthropic' and meta['ai_provider_confirmed'] is True
    # back to local keeps the once-confirmed flag (the dialog fires once per project)
    d = client.put('/project/a1/ai-provider', json={'provider': 'ollama'}).get_json()
    assert d['provider'] == 'ollama' and d['confirmed'] is True
    assert client.get('/project/nope/ai-provider').status_code == 404


def test_setting_one_project_leaves_every_other_project_local(client, monkeypatch):
    _config_says(monkeypatch, 'ollama', keys=True)
    _make_project('one')
    _make_project('two')
    client.put('/project/one/ai-provider', json={'provider': 'openai', 'confirmed': True})
    assert _read_meta('two').get('ai_provider') is None
    assert client.get('/project/two/ai-provider').get_json()['provider'] == 'ollama'
