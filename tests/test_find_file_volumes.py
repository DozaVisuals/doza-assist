"""Regression: /find-file must search /Volumes (NAS) again, not deprioritize it.

1.0.17 (commit 1cc6504) moved /Volumes to the END of the search roots under a
15s wall-clock cap, so broadcast/newsroom masters on a mounted NAS (incl. .mxf)
could be cut off before being found — the .mxf-not-showing regression. The fix
restores /Volumes to the FRONT and raises the cap so the NAS walk isn't cut
off, while keeping early-exit-on-first-match.

These tests simulate the search-root order via a fake os.walk (no real NAS):
the target lives only under /Volumes, home dirs yield nothing.
"""

import inspect
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def app_module(tmp_path, monkeypatch):
    monkeypatch.setenv('DOZA_DATA_DIR', str(tmp_path))
    import importlib
    import app as app_mod
    importlib.reload(app_mod)
    app_mod.app.config['TESTING'] = True
    return app_mod


def _install_fake_fs(monkeypatch, app_mod, volumes_tree):
    """Fake os.walk so only '/Volumes' yields the target; record walk order."""
    walked = []

    def fake_walk(root, followlinks=False):
        walked.append(root)
        for entry in volumes_tree.get(root, []):
            yield entry  # (dirpath, dirnames, filenames)

    class _Stat:
        st_size = 1000
        st_mtime = 0

    monkeypatch.setattr(app_mod.os, 'walk', fake_walk)
    monkeypatch.setattr(app_mod.os, 'stat', lambda p: _Stat())
    monkeypatch.setattr(app_mod.os.path, 'realpath', lambda p: p)
    return walked


def test_mxf_resolvable_only_under_volumes_is_found(app_module, monkeypatch):
    tree = {'/Volumes': [('/Volumes/NAS/Masters', [], ['interview_master.mxf'])]}
    _install_fake_fs(monkeypatch, app_module, tree)
    client = app_module.app.test_client()
    r = client.post('/find-file',
                    json={'filename': 'interview_master.mxf', 'size': 1000})
    assert r.status_code == 200
    body = r.get_json()
    assert body.get('status') == 'found', body
    assert body.get('path') == '/Volumes/NAS/Masters/interview_master.mxf'


def test_volumes_is_searched_first(app_module, monkeypatch):
    """The match lives under /Volumes; with /Volumes first + early-exit, it is
    the only root walked (home dirs never reached). Pre-revert, /Volumes was
    last, so home dirs would be walked first."""
    tree = {'/Volumes': [('/Volumes/NAS', [], ['master.mxf'])]}
    walked = _install_fake_fs(monkeypatch, app_module, tree)
    client = app_module.app.test_client()
    r = client.post('/find-file', json={'filename': 'master.mxf', 'size': 1000})
    assert r.get_json().get('status') == 'found'
    assert walked[0] == '/Volumes', f"/Volumes must be searched first, got {walked}"


def test_volumes_budget_was_raised_above_15s(app_module):
    """The 15s cap that cut off the NAS walk is gone; the ceiling is generous
    so a slow NAS walk reaches the masters."""
    src = inspect.getsource(app_module.find_file)
    assert 'time.monotonic() + 120.0' in src
    assert 'time.monotonic() + 15.0' not in src
