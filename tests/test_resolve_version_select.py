"""Resolve version selection, connect-before-launch, and the bounded import.

Covers the 1.0.19 Resolve fixes (tester: "Test Again launches Resolve 20 not
21" + "still slow and hangs / folds back to one-time setup"):

  - _rank_app_paths picks the NEWEST install within a location tier, so a
    user with both Resolve 20 and 21 in /Applications gets 21 on a cold
    launch (location still dominates version).
  - CONNECT-BEFORE-LAUNCH: when a Resolve is already running, the resolve
    app-path resolver attaches to THAT instance and never version-ranks or
    relaunches; import_timeline never launches a second copy.
  - The ENTIRE import handshake (not just the two import calls) runs on a
    bounded worker, so a wizard-blocked GetProjectManager() can't hang the
    request — it returns import_timeout within budget.
  - /export/send-to-nle forwards source_media_path/project_name/timeline_name
    (it previously dropped them, so Clips/Send-to-NLE imports landed offline).
"""

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module
from exporters import resolve_import


# ── Gate 12: version tiebreak ────────────────────────────────────────────────

class TestRankAppPathsVersion:
    def _versions(self, monkeypatch, mapping):
        monkeypatch.setattr(app_module, '_app_short_version',
                            lambda p: mapping.get(p, ()))

    def test_newest_version_wins_same_location(self, monkeypatch):
        v21 = '/Applications/DaVinci Resolve/DaVinci Resolve.app'
        v20 = '/Applications/DaVinci Resolve 20/DaVinci Resolve.app'
        self._versions(monkeypatch, {v21: (21, 0), v20: (20, 0, 1)})
        # Pass v20 first to prove ordering is by version, not input order.
        assert app_module._rank_app_paths([v20, v21])[0] == v21

    def test_location_tier_dominates_version(self, monkeypatch):
        sys_v20 = '/Applications/DaVinci Resolve/DaVinci Resolve.app'
        home_v21 = str(Path.home() / 'Applications/DaVinci Resolve/DaVinci Resolve.app')
        self._versions(monkeypatch, {sys_v20: (20, 0), home_v21: (21, 0)})
        # /Applications beats ~/Applications even though the home copy is newer.
        assert app_module._rank_app_paths([home_v21, sys_v20])[0] == sys_v20

    def test_unknown_version_sorts_after_known(self, monkeypatch):
        known = '/Applications/DaVinci Resolve/DaVinci Resolve.app'
        unknown = '/Applications/DaVinci Resolve 20/DaVinci Resolve.app'
        self._versions(monkeypatch, {known: (20, 0), unknown: ()})
        assert app_module._rank_app_paths([unknown, known])[0] == known


# ── Gate 13: connect-before-launch short-circuit ─────────────────────────────

class TestConnectBeforeLaunch:
    def test_running_resolve_skips_rank_and_mdfind(self, monkeypatch):
        running = '/Applications/DaVinci Resolve 20/DaVinci Resolve.app'
        monkeypatch.setattr(resolve_import, 'running_app_path', lambda: running)
        called = {'rank': False, 'mdfind': False}
        monkeypatch.setattr(app_module, '_rank_app_paths',
                            lambda paths: called.__setitem__('rank', True) or paths)
        monkeypatch.setattr(app_module, '_mdfind_app_by_bundle_id',
                            lambda bid: called.__setitem__('mdfind', True) or [])
        app_module._nle_path_cache.pop('resolve', None)
        got = app_module._find_nle_app_path('resolve')
        assert got == running
        assert called['rank'] is False, "version-rank must not run when attached"
        assert called['mdfind'] is False, "discovery must not run when attached"

    def test_import_timeline_does_not_launch_when_running(self, monkeypatch):
        launched = {'n': 0}
        monkeypatch.setattr(resolve_import, '_launch_resolve',
                            lambda app_path=None: launched.__setitem__('n', launched['n'] + 1))
        monkeypatch.setattr(resolve_import, '_is_resolve_running', lambda: True)
        monkeypatch.setattr(resolve_import, '_try_import_module',
                            lambda: SimpleNamespace(scriptapp=lambda name: _FakeHandle()))
        res = resolve_import.import_timeline(
            '/tmp/t.fcpxml', source_media_path=None,
            project_name='P', timeline_name='T')
        assert res.ok is True
        assert launched['n'] == 0, "must attach to the running Resolve, not relaunch"


# ── Gate 14: the ENTIRE handshake is bounded ─────────────────────────────────

class TestBoundedHandshake:
    def test_handshake_block_returns_timeout_within_budget(self, monkeypatch):
        # Shrink the overall cap so the test is fast; the block is in the
        # HANDSHAKE (GetProjectManager), which the old 1.0.16 fix left
        # unguarded — only the two import calls were wrapped.
        monkeypatch.setattr(resolve_import, '_IMPORT_TIMEOUT', 0.3)
        monkeypatch.setattr(resolve_import, '_is_resolve_running', lambda: True)

        class _BlockingHandle:
            def GetProjectManager(self):
                time.sleep(5)  # wizard-blocked daemon
                return None

        monkeypatch.setattr(resolve_import, '_try_import_module',
                            lambda: SimpleNamespace(scriptapp=lambda name: _BlockingHandle()))
        t0 = time.monotonic()
        res = resolve_import.import_timeline(
            '/tmp/t.fcpxml', source_media_path=None,
            project_name='P', timeline_name='T')
        elapsed = time.monotonic() - t0
        assert res.ok is False
        assert res.reason == 'import_timeout'
        assert elapsed < 2.0, f"handshake not bounded — took {elapsed:.1f}s"


# ── Gate 15: send-to-nle forwards the three args ─────────────────────────────

@pytest.fixture
def client(tmp_path):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


def _make_project(pid, **extra):
    pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
    pdir.mkdir(parents=True)
    meta = {'id': pid, 'name': 'P', 'status': 'uploaded'}
    meta.update(extra)
    (pdir / 'meta.json').write_text(json.dumps(meta))
    return pdir


class TestSendToNleForwardsArgs:
    def test_forwards_source_project_timeline(self, client, monkeypatch):
        _make_project('pX', source_path='/media/src.mxf', name='My Project')
        monkeypatch.setattr(app_module, '_find_nle_app_path',
                            lambda nle: '/Applications/Fake.app')
        captured = {}

        def _capture(file_path, nle, **kwargs):
            captured['file_path'] = file_path
            captured.update(kwargs)
            return ('scripted', {})

        monkeypatch.setattr(app_module, '_hand_file_to_nle', _capture)
        stub = SimpleNamespace(file_path='/tmp/out.fcpxml',
                               filename='out.fcpxml', format_name='FCPXML')
        monkeypatch.setattr(app_module, '_build_nle_export',
                            lambda project, body, force_platform=None: (stub, None))

        res = client.post('/export/send-to-nle',
                          json={'project_id': 'pX', 'nle': 'resolve'})
        assert res.status_code == 200
        assert captured.get('source_media_path') == '/media/src.mxf'
        assert captured.get('project_name') == 'My Project'
        assert captured.get('timeline_name') == 'out'


# ── Fakes ────────────────────────────────────────────────────────────────────

class _FakeMediaPool:
    def ImportTimelineFromFile(self, path, opts):
        return object()  # truthy timeline


class _FakeProject:
    def GetMediaPool(self):
        return _FakeMediaPool()

    def SetCurrentTimeline(self, tl):
        return True


class _FakePM:
    def GetCurrentProject(self):
        return _FakeProject()

    def CreateProject(self, name):
        return _FakeProject()


class _FakeStorage:
    def AddItemListToMediaPool(self, items):
        return True


class _FakeHandle:
    def GetProjectManager(self):
        return _FakePM()

    def GetMediaStorage(self):
        return _FakeStorage()
