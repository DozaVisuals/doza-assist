"""Resolve STUDIO scripted auto-import (exporters/resolve_import.py).

The Studio path imports the FCPXML via the scripting API; the Free path (no
API) falls through to the manual reveal-in-Finder export. These mock the
DaVinciResolveScript module so no real Resolve install is needed, and pin the
behaviours that matter for correctness:

  - source media is added to the Media Pool BEFORE ImportTimelineFromFile
    (order is the fix for "timeline imports but has no clips"),
  - an already-running Resolve is reused, never cold-launched,
  - a None or empty (zero-clip) timeline is treated as a FAILURE so the caller
    falls back to the manual path instead of reporting false success,
  - the Free path (scripting import unavailable) still returns cleanly,
  - the scripting env vars get set so the import can resolve.
"""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from exporters import resolve_import


# ── Fakes for the Resolve scripting object graph ─────────────────────────────

class _FakeTimeline:
    def __init__(self, clips):
        self._clips = clips

    def GetTrackCount(self, track_type):
        return 1 if track_type == 'video' else 0

    def GetItemListInTrack(self, track_type, index):
        return self._clips if (track_type == 'video' and index == 1) else []


class _FakeMediaPool:
    def __init__(self, timeline, log):
        self._timeline = timeline
        self._log = log

    def ImportTimelineFromFile(self, path, opts):
        self._log.append(('import', path, opts))
        return self._timeline


class _FakeMediaStorage:
    def __init__(self, log):
        self._log = log

    def AddItemListToMediaPool(self, items):
        self._log.append(('add', items))
        return ['clip']  # non-empty -> media landed


class _FakeProject:
    def __init__(self, media_pool):
        self._mp = media_pool
        self.set_timeline = None

    def GetMediaPool(self):
        return self._mp

    def SetCurrentTimeline(self, tl):
        self.set_timeline = tl
        return True


class _FakePM:
    def __init__(self, project):
        self._project = project

    def GetCurrentProject(self):
        return self._project

    def CreateProject(self, name):
        return self._project


class _FakeHandle:
    def __init__(self, pm, storage):
        self._pm = pm
        self._storage = storage

    def GetProjectManager(self):
        return self._pm

    def GetMediaStorage(self):
        return self._storage


def _wire(timeline, log):
    """A fake handle whose media pool returns ``timeline`` and that records
    media-add / import calls into ``log``."""
    return _FakeHandle(_FakePM(_FakeProject(_FakeMediaPool(timeline, log))),
                       _FakeMediaStorage(log))


def _studio(monkeypatch, handle):
    """Pretend a scriptable Studio is running and hands back ``handle``."""
    monkeypatch.setattr(resolve_import, '_is_resolve_running', lambda: True)
    monkeypatch.setattr(resolve_import, '_try_import_module',
                        lambda: SimpleNamespace(scriptapp=lambda name: handle))


# ── Order: media added before the timeline import ────────────────────────────

def test_media_added_to_pool_before_timeline_import(monkeypatch, tmp_path):
    log = []
    _studio(monkeypatch, _wire(_FakeTimeline(['c1', 'c2']), log))
    src = tmp_path / 'master.mov'
    src.write_bytes(b'\x00')  # must exist for the media-add branch
    res = resolve_import.import_timeline(
        '/tmp/t.fcpxml', source_media_path=str(src),
        project_name='P', timeline_name='T')
    assert res.ok is True
    kinds = [e[0] for e in log]
    assert kinds == ['add', 'import'], f"media must be added before import, got {kinds}"
    # the import got the README-verified options
    import_opts = next(e[2] for e in log if e[0] == 'import')
    assert import_opts.get('timelineName') == 'T'
    assert import_opts.get('importSourceClips') is True


# ── Warm reuse: never cold-launch when one is already running ────────────────

def test_running_resolve_is_reused_not_launched(monkeypatch):
    launched = {'n': 0}
    monkeypatch.setattr(resolve_import, '_launch_resolve',
                        lambda app_path=None: launched.__setitem__('n', launched['n'] + 1))
    _studio(monkeypatch, _wire(_FakeTimeline(['c1']), []))
    res = resolve_import.import_timeline(
        '/tmp/t.fcpxml', source_media_path=None, project_name='P', timeline_name='T')
    assert res.ok is True
    assert launched['n'] == 0, "a warm Resolve must be reused, not relaunched"


# ── None / empty timeline -> failure -> caller's manual fallback ─────────────

def test_none_timeline_is_failure(monkeypatch):
    _studio(monkeypatch, _wire(None, []))  # ImportTimelineFromFile -> None
    res = resolve_import.import_timeline(
        '/tmp/t.fcpxml', source_media_path=None, project_name='P', timeline_name='T')
    assert res.ok is False and res.reason == 'import_failed'


def test_empty_timeline_is_failure(monkeypatch):
    _studio(monkeypatch, _wire(_FakeTimeline([]), []))  # zero clips
    res = resolve_import.import_timeline(
        '/tmp/t.fcpxml', source_media_path=None, project_name='P', timeline_name='T')
    assert res.ok is False and res.reason == 'import_empty', \
        "a timeline with no clips must NOT report success"


# ── Free path (no scripting) is unaffected ───────────────────────────────────

def test_free_path_when_module_unavailable(monkeypatch):
    # Import of DaVinciResolveScript fails (Free / not installed).
    monkeypatch.setattr(resolve_import, '_try_import_module', lambda: None)
    res = resolve_import.import_timeline(
        '/tmp/t.fcpxml', source_media_path=None, project_name='P', timeline_name='T')
    assert res.ok is False and res.reason == 'module_missing'


def test_free_path_when_scriptapp_returns_none(monkeypatch):
    # Module loads but scriptapp never connects, and the install is Free.
    monkeypatch.setattr(resolve_import, '_is_resolve_running', lambda: True)
    monkeypatch.setattr(resolve_import, '_try_import_module',
                        lambda: SimpleNamespace(scriptapp=lambda name: None))
    monkeypatch.setattr(resolve_import, '_get_resolve_handle',
                        lambda dvr, timeout_seconds=0: (None, None))
    monkeypatch.setattr(resolve_import, 'edition', lambda app_path=None: 'free')
    res = resolve_import.import_timeline(
        '/tmp/t.fcpxml', source_media_path=None, project_name='P', timeline_name='T')
    assert res.ok is False and res.reason == 'requires_studio'


# ── Env vars set so the import can resolve fusionscript.so ────────────────────

def test_env_vars_set_when_unset(monkeypatch):
    monkeypatch.delenv('RESOLVE_SCRIPT_API', raising=False)
    monkeypatch.delenv('RESOLVE_SCRIPT_LIB', raising=False)
    monkeypatch.delenv('PYTHONPATH', raising=False)
    resolve_import._ensure_resolve_env()
    assert os.environ['RESOLVE_SCRIPT_API'].endswith('Developer/Scripting')
    assert os.environ['RESOLVE_SCRIPT_LIB'].endswith('fusionscript.so')
    assert 'Modules' in os.environ.get('PYTHONPATH', '')


def test_env_vars_do_not_override_user_values(monkeypatch):
    monkeypatch.setenv('RESOLVE_SCRIPT_API', '/custom/scripting')
    monkeypatch.setenv('RESOLVE_SCRIPT_LIB', '/custom/fusionscript.so')
    resolve_import._ensure_resolve_env()
    assert os.environ['RESOLVE_SCRIPT_API'] == '/custom/scripting'
    assert os.environ['RESOLVE_SCRIPT_LIB'] == '/custom/fusionscript.so'
