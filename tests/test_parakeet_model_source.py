"""Parakeet loads from the local cache without asking huggingface.co
whether the model is current (a network request on every transcription)."""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.abspath(os.path.join(HERE, '..'))
if CORE not in sys.path:
    sys.path.insert(0, CORE)

import parakeet_worker  # noqa: E402


def _stub_hub(monkeypatch, cached, tmp_path):
    calls = []
    hub = types.ModuleType('huggingface_hub')

    def hf_hub_download(repo, filename, local_files_only=False, **kw):
        calls.append((repo, filename, local_files_only))
        if not local_files_only:
            raise AssertionError('must not go online for a cached model')
        if not cached:
            raise FileNotFoundError(filename)
        d = tmp_path / 'snapshots' / 'abc'
        d.mkdir(parents=True, exist_ok=True)
        f = d / filename
        f.write_text('{}')
        return str(f)
    hub.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, 'huggingface_hub', hub)
    return calls


def test_cached_model_loads_from_the_snapshot_directory(monkeypatch, tmp_path):
    calls = _stub_hub(monkeypatch, True, tmp_path)
    src = parakeet_worker.parakeet_model_source('mlx-community/parakeet-tdt-0.6b-v2')
    assert src == str(tmp_path / 'snapshots' / 'abc')
    assert all(local for _, _, local in calls)


def test_missing_model_falls_back_to_the_repo_id(monkeypatch, tmp_path):
    _stub_hub(monkeypatch, False, tmp_path)
    assert parakeet_worker.parakeet_model_source('mlx-community/parakeet-tdt-0.6b-v2') == 'mlx-community/parakeet-tdt-0.6b-v2'


def test_transcribe_uses_the_resolver():
    src = open(os.path.join(CORE, 'parakeet_worker.py')).read()
    assert "from_pretrained(parakeet_model_source(PARAKEET_REPO))" in src
