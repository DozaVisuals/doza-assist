"""Export-endpoint hardening regressions (adversarial review wave E).

Pins the server-side fixes for the confirmed export/delivery defects:

  - C16/C33/C40: user/AI story titles with '/' or ':' are sanitized into the
    export filename instead of crashing the multicam round-trip with a raw
    HTML 500 (_safe_filename + composed-filename sanitization).
  - C24/C56: Premiere delivery is XML-file-only — never launches or requires
    Premiere Pro; not-installed is NOT an error (write + reveal + note).
  - C28: round-trip (FCPXML-imported, non-single-asset) projects may only be
    SENT to Final Cut Pro — server-side 400 on resolve/premiere targets
    (the UI gate alone was bypassable over raw HTTP).
  - C32: the FCP `open` handoff checks the returncode (a trashed/moved app
    used to report status:ok) and drops the stale _nle_path_cache entry.
  - C34: only genuine setup failures (module_missing / scripting_disabled /
    requires_studio) tag setup_required; operational Resolve failures carry
    import_fallback + hint instead of the External-Scripting walkthrough.
  - C35: the Resolve import budget hierarchy stays coherent with the UI
    fetch timeout (TOTAL_IMPORT_BUDGET).
  - C39: /export/send-to-nle multicam payload carries skipped/skipped_count
    (parity with /project/<id>/export/fcpxml-multicam).
  - C44: a labeled section stored with "text": null no longer TypeErrors
    every direct export for the project.
  - C45: one malformed 'HH:MM:SS' timecode skips that item with a payload
    warning instead of 500ing the whole export.
  - C46: multicam exports write via temp file + os.replace (atomic swap), so
    a rapid second export can't truncate the file FCP is still reading.
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module
from exporters import resolve_import


@pytest.fixture
def client(tmp_path):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    app_module.app.config['EXPORTS_DIR'] = str(tmp_path / 'exports')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    Path(app_module.app.config['EXPORTS_DIR']).mkdir(parents=True, exist_ok=True)
    app_module.app.config['TESTING'] = True
    return app_module.app.test_client()


def _make_project(pid, **extra):
    pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
    pdir.mkdir(parents=True)
    meta = {'id': pid, 'name': 'P', 'status': 'uploaded'}
    meta.update(extra)
    (pdir / 'meta.json').write_text(json.dumps(meta))
    return pdir


def _make_roundtrip_project(pid, tmp_path, container_type='sync-clip',
                            timeline_audio_rendered=True, **extra):
    """A project ingested from an FCPXML (fcpxml_source block present)."""
    stored = tmp_path / f'{pid}-stored.fcpxml'
    stored.write_text('<fcpxml version="1.14"/>')
    return _make_project(pid, fcpxml_source={
        'container_type': container_type,
        'timeline_audio_rendered': timeline_audio_rendered,
        'stored_fcpxml_path': str(stored),
    }, **extra)


class _StubExporter:
    """Captures markers; returns an exporter-shaped result without disk IO."""
    file_extension = '.fcpxml'

    def __init__(self):
        self.markers = None

    def _result(self, name):
        return SimpleNamespace(
            file_path=f'/tmp/{name}', filename=name, format_name='FCPXML',
            platform_name='Final Cut Pro', warnings=[])

    def export_markers(self, markers, **kwargs):
        self.markers = markers
        return self._result('out.fcpxml')

    def export_story(self, markers, **kwargs):
        self.markers = markers
        return self._result('story.fcpxml')


def _quiet_probes(monkeypatch):
    """Stub the media probes _build_nle_export/_build_nle_story_export run."""
    monkeypatch.setattr(app_module, 'get_media_duration', lambda p: None)
    monkeypatch.setattr(app_module, 'get_video_framerate', lambda p: None)
    monkeypatch.setattr(app_module, 'get_video_resolution', lambda p: (1920, 1080))
    monkeypatch.setattr(app_module, 'get_video_start_timecode_info',
                        lambda p, fps: (0, None))


def _stub_multicam_writer(monkeypatch):
    """Bypass the real FCPXML parse/write for route-level multicam tests."""
    monkeypatch.setattr(app_module, 'parse_fcpxml', lambda p: object())

    def _writer(parsed, selects, preserve_order=False, skipped_out=None):
        return b'<fcpxml version="1.14"/>'

    monkeypatch.setattr(app_module, 'write_selects_as_new_project', _writer)
    monkeypatch.setattr(app_module, 'write_markers_on_timeline',
                        lambda parsed, selects, skipped_out=None: b'<fcpxml/>')


# ── C16/C33/C40: filename sanitization ──────────────────────────────────────

class TestSafeFilename:
    def test_slashes_and_colons_become_dashes(self):
        out = app_module._safe_filename('24/7 — The: Grind')
        assert '/' not in out and ':' not in out
        assert out.startswith('24-7')

    def test_control_chars_and_nuls_stripped(self):
        assert app_module._safe_filename('a\x00b\nc\td') == 'a b c d'

    def test_whitespace_collapsed(self):
        assert app_module._safe_filename('  Day   1  ') == 'Day 1'

    def test_nothing_left_falls_back(self):
        assert app_module._safe_filename('\x00\x01 ') == 'Export'
        assert app_module._safe_filename(None) == 'Export'
        assert app_module._safe_filename('', fallback='X') == 'X'


class TestMulticamStoryTitleSanitized:
    def _export(self, client, title):
        return client.post('/project/rt1/export/fcpxml-multicam', json={
            'mode': 'selects_project',
            'sources': ['story_build'],
            'preserve_order': True,
            'story_title': title,
            'story_build_clips': [
                {'start_time': 1.0, 'end_time': 2.0, 'title': 'A', 'order': 0},
            ],
            'deliver_to': 'file',
        })

    def test_slash_title_returns_json_200(self, client, tmp_path, monkeypatch):
        _make_roundtrip_project('rt1', tmp_path, name='Proj')
        _stub_multicam_writer(monkeypatch)
        monkeypatch.setattr(app_module, '_reveal_in_finder', lambda p: None)

        res = self._export(client, 'Intro/Outro: 24/7 assembly')
        assert res.status_code == 200, res.data
        data = res.get_json()
        assert '/' not in data['filename'] and ':' not in data['filename']
        assert os.path.isfile(data['file'])

    def test_write_is_atomic_temp_then_replace(self, client, tmp_path, monkeypatch):
        _make_roundtrip_project('rt1', tmp_path, name='Proj')
        _stub_multicam_writer(monkeypatch)
        monkeypatch.setattr(app_module, '_reveal_in_finder', lambda p: None)

        replaced = []
        real_replace = os.replace

        def _spy(src, dst):
            replaced.append((src, dst))
            return real_replace(src, dst)

        monkeypatch.setattr(app_module.os, 'replace', _spy)
        res = self._export(client, 'Plain Title')
        assert res.status_code == 200, res.data
        final = res.get_json()['file']
        # The export itself went through a unique temp name + atomic swap.
        export_swaps = [(s, d) for s, d in replaced if d == final]
        assert len(export_swaps) == 1
        assert export_swaps[0][0].endswith('.part')
        # No temp droppings left behind.
        leftovers = [p for p in os.listdir(app_module.app.config['EXPORTS_DIR'])
                     if p.endswith('.part')]
        assert leftovers == []


# ── C28: server-side FCP-only gate for round-trip projects ──────────────────

class TestRoundTripFcpOnlyGate:
    MSG = 'FCPXML round-trips can only be sent back to Final Cut Pro'

    @pytest.mark.parametrize('nle', ['resolve', 'premiere'])
    def test_send_to_nle_rejects_non_fcp(self, client, tmp_path, nle):
        _make_roundtrip_project('rt2', tmp_path)
        res = client.post('/export/send-to-nle',
                          json={'project_id': 'rt2', 'nle': nle})
        assert res.status_code == 400
        assert self.MSG in res.get_json()['error']

    def test_export_fcpxml_rejects_nle_delivery_to_resolve(self, client, tmp_path):
        _make_roundtrip_project('rt3', tmp_path)
        res = client.post('/project/rt3/export/fcpxml',
                          json={'deliver_to': 'nle', 'nle': 'resolve',
                                'types': ['labels']})
        assert res.status_code == 400
        assert self.MSG in res.get_json()['error']

    def test_story_export_rejects_nle_delivery_to_premiere(self, client, tmp_path):
        _make_roundtrip_project('rt4', tmp_path)
        res = client.post('/project/rt4/story/export', json={
            'clips': [{'start_time': 1, 'end_time': 2, 'title': 'A'}],
            'deliver_to': 'nle', 'nle': 'premiere',
        })
        assert res.status_code == 400
        assert self.MSG in res.get_json()['error']

    def test_multisource_assetclip_is_gated(self, client, tmp_path):
        # asset-clip container BUT timeline_audio_rendered → flat export
        # would reference the app-internal WAV; must be FCP-only too.
        _make_roundtrip_project('rt5', tmp_path, container_type='asset-clip',
                                timeline_audio_rendered=True)
        res = client.post('/export/send-to-nle',
                          json={'project_id': 'rt5', 'nle': 'resolve'})
        assert res.status_code == 400
        assert self.MSG in res.get_json()['error']

    def test_single_source_assetclip_not_gated(self, client, tmp_path, monkeypatch):
        # Single-source asset-clip imports are deliberately EXEMPT from the
        # gate (their source_path is the real camera media, so a flat export
        # is safe on any platform) — prove the request gets PAST the gate by
        # reaching the app-path lookup.
        _make_roundtrip_project('rt6', tmp_path, container_type='asset-clip',
                                timeline_audio_rendered=False)
        monkeypatch.setattr(app_module, '_find_nle_app_path', lambda nle: None)
        res = client.post('/export/send-to-nle',
                          json={'project_id': 'rt6', 'nle': 'resolve'})
        assert res.status_code == 404
        assert 'not found on this Mac' in res.get_json()['error']

    def test_fcp_target_passes_gate(self, client, tmp_path, monkeypatch):
        _make_roundtrip_project('rt7', tmp_path,
                                labeled_sections=[{'start': 1.0, 'end': 2.0,
                                                   'color': 'green', 'text': 'x'}])
        _stub_multicam_writer(monkeypatch)
        monkeypatch.setattr(app_module, '_find_nle_app_path',
                            lambda nle: '/Applications/Fake.app')
        monkeypatch.setattr(app_module, '_hand_file_to_nle',
                            lambda *a, **k: ('app', {}))
        res = client.post('/export/send-to-nle',
                          json={'project_id': 'rt7', 'export_type': 'multicam'})
        assert res.status_code == 200, res.data

    def test_file_delivery_gated_too(self, client, tmp_path, monkeypatch):
        # A flat export of a round-trip project references only the audio
        # angle's file (or the app-internal timeline WAV) — broken output
        # for EVERY delivery target, so 'file' delivery is gated as well
        # (2026-08-31: the same defect class shipped visibly through the
        # collection exporter as audio-only 'visuals never reconnect'
        # files). The Round-Trip export is the steer.
        _quiet_probes(monkeypatch)
        stub = _StubExporter()
        monkeypatch.setattr(app_module, 'get_exporter', lambda p: stub)
        monkeypatch.setattr(app_module, '_reveal_in_finder', lambda p: None)
        _make_roundtrip_project('rt8', tmp_path, source_path='',
                                labeled_sections=[{'start': 1.0, 'end': 2.0,
                                                   'color': 'green', 'text': 'x'}])
        res = client.post('/project/rt8/export/fcpxml',
                          json={'deliver_to': 'file', 'types': ['labels']})
        assert res.status_code == 400, res.data
        assert 'Round-Trip' in (res.get_json() or {}).get('error', '')


# ── C24/C56: Premiere delivery is XML-file-only ─────────────────────────────

class TestPremiereFileOnlyDelivery:
    def test_hand_file_not_installed_still_succeeds(self, monkeypatch, tmp_path):
        monkeypatch.setattr(app_module, '_find_nle_app_path', lambda nle: None)
        opens = []

        def _fake_open(args, timeout=5.0):
            opens.append(list(args))
            return True, ''

        monkeypatch.setattr(app_module, '_run_open', _fake_open)
        f = tmp_path / 'out.xml'
        f.write_text('<xmeml/>')
        opened_in, info = app_module._hand_file_to_nle(str(f), 'premiere')
        assert opened_in == 'finder'
        assert info['premiere_manual_import'] is True
        assert info['delivery'] == 'file'
        assert 'not detected' in info['note']
        # Only a Finder reveal ran — Premiere itself is never launched.
        assert opens == [['-R', str(f)]]

    def test_hand_file_installed_is_still_file_delivery(self, monkeypatch):
        monkeypatch.setattr(app_module, '_find_nle_app_path',
                            lambda nle: '/Applications/Adobe Premiere Pro.app')
        monkeypatch.setattr(app_module, '_run_open', lambda a, timeout=5.0: (True, ''))
        opened_in, info = app_module._hand_file_to_nle('/tmp/o.xml', 'premiere')
        assert opened_in == 'finder'
        assert info['premiere_manual_import'] is True
        assert info['delivery'] == 'file'
        assert 'File → Import' in info['note']
        assert 'not detected' not in info['note']

    def test_send_to_nle_premiere_without_install_returns_ok(self, client, monkeypatch):
        _make_project('pp1', source_path='/media/src.mov')
        monkeypatch.setattr(app_module, '_find_nle_app_path', lambda nle: None)
        monkeypatch.setattr(app_module, '_run_open', lambda a, timeout=5.0: (True, ''))
        stub = SimpleNamespace(file_path='/tmp/out.xml', filename='out.xml',
                               format_name='Premiere XML')
        monkeypatch.setattr(
            app_module, '_build_nle_export',
            lambda project, body, force_platform=None, **kw: (stub, None))
        res = client.post('/export/send-to-nle',
                          json={'project_id': 'pp1', 'nle': 'premiere'})
        assert res.status_code == 200, res.data
        data = res.get_json()
        assert data['status'] == 'ok'
        assert data['premiere_manual_import'] is True
        assert data['delivery'] == 'file'
        assert data['opened_in'] == 'finder'
        assert data['note']
        assert 'skipped' not in data  # non-multicam payloads stay clean

    def test_export_fcpxml_premiere_delivery_without_install(self, client, monkeypatch):
        _make_project('pp2', source_path='/media/src.mov')
        monkeypatch.setattr(app_module, '_find_nle_app_path', lambda nle: None)
        monkeypatch.setattr(app_module, '_run_open', lambda a, timeout=5.0: (True, ''))
        stub = SimpleNamespace(file_path='/tmp/out.xml', filename='out.xml',
                               format_name='Premiere XML')
        monkeypatch.setattr(
            app_module, '_build_nle_export',
            lambda project, body, force_platform=None, **kw: (stub, None))
        res = client.post('/project/pp2/export/fcpxml',
                          json={'deliver_to': 'nle', 'nle': 'premiere',
                                'types': ['labels']})
        assert res.status_code == 200, res.data
        data = res.get_json()
        assert data['premiere_manual_import'] is True
        assert data['delivery'] == 'file'  # info overrides the 'nle' default
        assert data['note']


# ── C32: `open` returncode is checked ───────────────────────────────────────

class TestOpenReturncodeChecked:
    def test_run_open_nonzero_returncode_fails(self, monkeypatch):
        proc = SimpleNamespace(returncode=1, stderr='Unable to find application',
                               stdout='')
        monkeypatch.setattr(app_module.subprocess, 'run', lambda *a, **k: proc)
        ok, err = app_module._run_open(['-b', 'com.apple.FinalCut', '/tmp/x'])
        assert not ok
        assert 'Unable to find application' in err

    def test_run_open_success(self, monkeypatch):
        proc = SimpleNamespace(returncode=0, stderr='', stdout='')
        monkeypatch.setattr(app_module.subprocess, 'run', lambda *a, **k: proc)
        ok, err = app_module._run_open(['-R', '/tmp/x'])
        assert ok and err == ''

    def test_failed_fcp_open_errors_and_drops_stale_cache(self, monkeypatch):
        fake_path = '/Applications/Final Cut Pro.app'
        monkeypatch.setattr(app_module, '_find_nle_app_path', lambda nle: fake_path)
        monkeypatch.setitem(app_module._nle_path_cache, 'fcp', fake_path)
        monkeypatch.setattr(app_module, '_run_open',
                            lambda a, timeout=5.0: (False, 'Unable to find application'))
        opened_in, info = app_module._hand_file_to_nle('/tmp/o.fcpxml', 'fcp')
        assert opened_in is None
        assert 'Final Cut Pro' in info['error']
        assert 'Unable to find application' in info['error']
        # Stale cache entry dropped so the next attempt re-runs mdfind.
        assert 'fcp' not in app_module._nle_path_cache

    def test_successful_fcp_open_reports_app(self, monkeypatch):
        monkeypatch.setattr(app_module, '_find_nle_app_path',
                            lambda nle: '/Applications/Final Cut Pro.app')
        monkeypatch.setattr(app_module, '_run_open', lambda a, timeout=5.0: (True, ''))
        opened_in, info = app_module._hand_file_to_nle('/tmp/o.fcpxml', 'fcp')
        assert opened_in == 'app'
        assert info == {}


# ── C34: setup_required only for genuine setup failures ─────────────────────

class TestResolveFallbackTagging:
    def _hand(self, monkeypatch, reason):
        monkeypatch.setattr(
            resolve_import, 'import_timeline',
            lambda *a, **k: resolve_import.ImportResult(ok=False, reason=reason,
                                                        hint='H'))
        # Keep the fallback from actually opening Finder/Resolve.
        monkeypatch.setattr(app_module.subprocess, 'Popen',
                            lambda *a, **k: None)
        return app_module._hand_file_to_resolve(
            '/tmp/o.fcpxml', app_path='/Applications/Fake.app',
            source_media_path=None, project_name='P', timeline_name='T')

    @pytest.mark.parametrize('reason', ['import_failed', 'import_empty',
                                        'import_timeout', 'no_project',
                                        'unexpected_error'])
    def test_operational_failures_are_not_setup(self, monkeypatch, reason):
        opened_in, info = self._hand(monkeypatch, reason)
        assert opened_in == 'finder+app'
        assert 'setup_required' not in info
        assert info['import_fallback'] is True
        assert info['reason'] == reason
        assert info['hint'] == 'H'

    @pytest.mark.parametrize('reason', ['scripting_disabled', 'module_missing',
                                        'requires_studio'])
    def test_setup_failures_keep_the_modal(self, monkeypatch, reason):
        opened_in, info = self._hand(monkeypatch, reason)
        assert opened_in == 'finder+app'
        assert info['setup_required'] is True
        assert 'import_fallback' not in info
        assert info['reason'] == reason


# ── C35: timeout budget coherence ───────────────────────────────────────────

class TestResolveTimeoutBudget:
    def test_budget_hierarchy_is_coherent(self):
        total = resolve_import.TOTAL_IMPORT_BUDGET
        assert total == (resolve_import._COLD_HANDLE_TIMEOUT
                         + resolve_import._IMPORT_TIMEOUT)
        # The documented client fetch cap (240s, project.html) must exceed
        # the server worst case with headroom for export build + probes.
        assert total + 25 <= 240


# ── C39: send-to-nle multicam payload carries skipped ───────────────────────

class TestSendToNleMulticamSkipped:
    def test_payload_carries_skipped_and_count(self, client, tmp_path, monkeypatch):
        _make_roundtrip_project('rt9', tmp_path)
        monkeypatch.setattr(app_module, '_find_nle_app_path',
                            lambda nle: '/Applications/Fake.app')
        monkeypatch.setattr(app_module, '_hand_file_to_nle',
                            lambda *a, **k: ('app', {}))
        monkeypatch.setattr(
            app_module, '_build_nle_multicam_export',
            lambda project, body: ('/tmp/x.fcpxml', 'x.fcpxml',
                                   'selects_project', ['Clip A', 'Clip B'],
                                   ['clip "X" is retimed']))
        res = client.post('/export/send-to-nle',
                          json={'project_id': 'rt9', 'export_type': 'multicam'})
        assert res.status_code == 200, res.data
        data = res.get_json()
        assert data['skipped'] == ['Clip A', 'Clip B']
        assert data['skipped_count'] == 2
        # R5: the stored FCPXML's parse warnings ride the payload's warnings
        # list so the UI can explain misaligned selects.
        assert data['warnings'] == ['clip "X" is retimed']


# ── C44: labeled section with "text": null ──────────────────────────────────

class TestNullLabelText:
    def test_null_text_exports_with_empty_note(self, monkeypatch):
        _quiet_probes(monkeypatch)
        stub = _StubExporter()
        monkeypatch.setattr(app_module, 'get_exporter', lambda p: stub)
        project = {'name': 'P', 'source_path': '',
                   'color_labels': {'green': 'Best'},
                   'labeled_sections': [
                       {'start': 1.0, 'end': 2.0, 'color': 'green', 'text': None},
                   ]}
        result, _ = app_module._build_nle_export(project, {'types': ['labels']})
        assert result.filename == 'out.fcpxml'
        assert len(stub.markers) == 1
        assert stub.markers[0]['note'] == ''


# ── C45: malformed 'HH:MM:SS' skips the item, not the export ────────────────

class TestMalformedTimecodeSkips:
    def test_direct_export_skips_bad_label_with_warning(self, monkeypatch):
        _quiet_probes(monkeypatch)
        stub = _StubExporter()
        monkeypatch.setattr(app_module, 'get_exporter', lambda p: stub)
        project = {'name': 'P', 'source_path': '',
                   'color_labels': {},
                   'labeled_sections': [
                       {'start': '00:0X:12', 'end': '00:01:00', 'color': 'green',
                        'text': 'bad'},
                       {'start': 5.0, 'end': 9.0, 'color': 'blue', 'text': 'good'},
                   ]}
        warnings = []
        app_module._build_nle_export(project, {'types': ['labels']},
                                     warnings_out=warnings)
        assert len(stub.markers) == 1
        assert stub.markers[0]['note'] == 'good'
        assert len(warnings) == 1
        assert 'unreadable timecode' in warnings[0]

    def test_story_export_skips_bad_clip_and_payload_warns(self, client, monkeypatch):
        _quiet_probes(monkeypatch)
        stub = _StubExporter()
        monkeypatch.setattr(app_module, 'get_exporter', lambda p: stub)
        monkeypatch.setattr(app_module, '_reveal_in_finder', lambda p: None)
        _make_project('st1', source_path='')
        res = client.post('/project/st1/story/export', json={
            'clips': [
                {'start_time': '00:00:10', 'end_time': '00:02:4S', 'title': 'Bad'},
                {'start_time': 5, 'end_time': 9, 'title': 'Good'},
            ],
            'deliver_to': 'file',
        })
        assert res.status_code == 200, res.data
        data = res.get_json()
        assert any('Bad' in w and 'unreadable timecode' in w
                   for w in data.get('warnings', []))
        assert len(stub.markers) == 1
        assert stub.markers[0]['text'] == 'Good'


# ── R8: X-Export-Warnings header must never carry non-latin-1 text ──────────

class TestExporterResponseHeaderAscii:
    """werkzeug (and http.server behind the packaged app) encode response
    headers latin-1 STRICT: one em-dash in any exporter warning raised
    UnicodeEncodeError mid-header and closed the legacy attachment response
    (deliver_to omitted) with no body and no error JSON. _exporter_response
    now ASCII-folds every X-Export-* header value at the choke point."""

    def _respond(self, tmp_path, warnings):
        f = tmp_path / 'out.edl'
        f.write_text('TITLE: x\n')
        result = SimpleNamespace(
            file_path=str(f), filename='out.edl', format_name='EDL',
            platform_name='DaVinci Resolve', warnings=warnings)
        exporter = SimpleNamespace(file_extension='.edl')
        with app_module.app.test_request_context():
            resp = app_module._exporter_response(result, {}, exporter)
        resp.close()
        return resp

    def test_non_ascii_warning_is_folded_not_fatal(self, tmp_path):
        resp = self._respond(tmp_path, [
            'EDL contains 1005 events — beyond the CMX 3600 limit',
            'Håkon needs review',
        ])
        value = resp.headers['X-Export-Warnings']
        value.encode('ascii')   # raises = the header would kill the response
        value.encode('latin-1')
        assert '1005 events' in value and ' | ' in value

    def test_ascii_warnings_pass_through_unchanged(self, tmp_path):
        resp = self._respond(tmp_path, ['plain warning'])
        assert resp.headers['X-Export-Warnings'] == 'plain warning'
        assert resp.headers['X-Export-Format'] == 'EDL'
        assert resp.headers['X-Export-Platform'] == 'DaVinci Resolve'
        assert resp.headers['X-Export-Extension'] == '.edl'

    def test_shipped_edl_over_cap_warning_survives_the_header(self, tmp_path):
        """End-to-end pin: the real EDL >999-events warning text riding the
        real header path encodes clean (the shipped string is ASCII)."""
        from exporters.edl import EDLExporter
        markers = [{"start": float(i), "end": float(i + 1), "text": f"S{i}"}
                   for i in range(1005)]
        result = EDLExporter().export_markers(
            markers, project_name='P', source_path='/tmp/interview.mov',
            media_duration=2000.0, framerate=25.0, width=1920, height=1080,
            export_type='all', exports_dir=str(tmp_path))
        exporter = SimpleNamespace(file_extension='.edl')
        with app_module.app.test_request_context():
            resp = app_module._exporter_response(result, {}, exporter)
        resp.close()
        value = resp.headers['X-Export-Warnings']
        value.encode('latin-1')
        assert 'CMX 3600' in value and '?' not in value  # nothing was folded


# ── R15: all-items-skipped 400 still carries the skip warnings ──────────────

class _EmptyGuardExporter(_StubExporter):
    """Mirrors the real exporters' _guard_exportable: no markers → ValueError."""

    def export_markers(self, markers, **kwargs):
        if not markers:
            raise ValueError('No clips to export for the selected types.')
        return super().export_markers(markers, **kwargs)

    def export_story(self, markers, **kwargs):
        if not markers:
            raise ValueError('No clips to export for the selected types.')
        return super().export_story(markers, **kwargs)


class TestAllSkipped400CarriesWarnings:
    """When EVERY item is skipped for a malformed timecode the export fails
    400 ("No clips to export…") — but the per-item skip diagnostics collected
    before the failure must ride the error payload, or the user retries
    forever with no clue which items were unreadable (R15)."""

    _BAD_SOCIAL = [
        {'start': '00:02:4S', 'end': '00:03:00', 'title': 'Bad One',
         'platform': 'TikTok'},
        {'start': '00:0X:12', 'end': '00:01:00', 'title': 'Bad Two',
         'platform': 'Reels'},
    ]

    def _stub(self, monkeypatch):
        _quiet_probes(monkeypatch)
        monkeypatch.setattr(app_module, 'get_exporter',
                            lambda p: _EmptyGuardExporter())

    def test_export_fcpxml_400_names_the_bad_clips(self, client, monkeypatch):
        self._stub(monkeypatch)
        _make_project('as1', source_path='',
                      analysis={'social_clips': self._BAD_SOCIAL})
        res = client.post('/project/as1/export/fcpxml',
                          json={'types': ['social']})
        assert res.status_code == 400, res.data
        data = res.get_json()
        assert 'Export failed' in data['error']
        assert len(data['warnings']) == 2
        assert all('unreadable timecode' in w for w in data['warnings'])
        assert any('Bad One' in w for w in data['warnings'])
        assert any('Bad Two' in w for w in data['warnings'])

    def test_send_to_nle_400_names_the_bad_clips(self, client, monkeypatch):
        self._stub(monkeypatch)
        _make_project('as2', source_path='',
                      analysis={'social_clips': self._BAD_SOCIAL})
        res = client.post('/export/send-to-nle',
                          json={'project_id': 'as2', 'nle': 'premiere',
                                'types': ['social']})
        assert res.status_code == 400, res.data
        data = res.get_json()
        assert 'Export failed' in data['error']
        assert any('Bad One' in w and 'unreadable timecode' in w
                   for w in data['warnings'])

    def test_story_export_400_names_the_bad_clips(self, client, monkeypatch):
        self._stub(monkeypatch)
        _make_project('as3', source_path='')
        res = client.post('/project/as3/story/export', json={
            'clips': [
                {'start_time': '00:00:10', 'end_time': '00:02:4S',
                 'title': 'Corrupt A'},
                {'start_time': '00:0X:00', 'end_time': '00:01:00',
                 'title': 'Corrupt B'},
            ],
        })
        assert res.status_code == 400, res.data
        data = res.get_json()
        assert 'Export failed' in data['error']
        assert len(data['warnings']) == 2
        assert all('unreadable timecode' in w for w in data['warnings'])
        assert any('Corrupt A' in w for w in data['warnings'])


# ── 1.0.45: round-trip delivery follows the Edit-in platform ────────────────

class TestRoundTripPlatformDelivery:
    """The 1.0.44 field bug: with Edit-in = DaVinci Resolve, the round-trip
    export force-launched Final Cut Pro. Delivery now follows the ``nle``
    body field (_hand_round_trip_to_nle): fcp auto-imports; resolve gets
    save + reveal + focus + hint; premiere gets save + reveal + hint. The
    FILE stays an FCP container by construction — scripted Resolve import is
    deliberately NOT attempted (1.0.24: mc-clip timelines import EMPTY)."""

    def _project(self, pid, tmp_path):
        return _make_roundtrip_project(
            pid, tmp_path,
            labeled_sections=[{'start': 1.0, 'end': 2.0,
                               'color': 'green', 'text': 'x'}])

    def _no_scripted_resolve(self, monkeypatch):
        def _boom(*a, **k):
            raise AssertionError('scripted Resolve import must not run for '
                                 'round-trip files (imports empty, 1.0.24)')
        monkeypatch.setattr(app_module, '_hand_file_to_resolve', _boom)

    def test_multicam_route_resolve_reveals_instead_of_launching_fcp(
            self, client, tmp_path, monkeypatch):
        self._project('rp1', tmp_path)
        _stub_multicam_writer(monkeypatch)
        self._no_scripted_resolve(monkeypatch)
        fcp_handoffs = []
        monkeypatch.setattr(
            app_module, '_hand_file_to_nle',
            lambda *a, **k: fcp_handoffs.append((a, k)) or ('app', {}))
        opens = []

        def _fake_open(args, timeout=5.0):
            opens.append(list(args))
            return True, ''

        monkeypatch.setattr(app_module, '_run_open', _fake_open)
        monkeypatch.setattr(app_module, '_find_nle_app_path',
                            lambda nle: '/Applications/DaVinci Resolve.app')
        res = client.post('/project/rp1/export/fcpxml-multicam',
                          json={'deliver_to': 'nle', 'nle': 'resolve',
                                'sources': ['client_selects']})
        assert res.status_code == 200, res.data
        data = res.get_json()
        assert data['nle'] == 'resolve'
        assert data['nle_name'] == 'DaVinci Resolve'
        assert data['opened_in'] == 'finder+app'
        assert data['import_fallback'] is True
        assert 'Import' in data['hint'] and 'Timeline' in data['hint']
        # The old force-FCP handoff must NOT have run…
        assert fcp_handoffs == []
        # …and the file was revealed + Resolve focused instead.
        assert any(a and a[0] == '-R' for a in opens)
        assert any(a and a[0] == '-a' for a in opens)

    def test_multicam_route_absent_nle_keeps_fcp_contract(
            self, client, tmp_path, monkeypatch):
        # Raw-HTTP callers that never sent `nle` relied on the guaranteed
        # FCP launch — absent must still mean fcp.
        self._project('rp2', tmp_path)
        _stub_multicam_writer(monkeypatch)
        handoffs = []

        def _hand(path, nle, **k):
            handoffs.append((nle, k))
            return 'app', {}

        monkeypatch.setattr(app_module, '_hand_file_to_nle', _hand)
        res = client.post('/project/rp2/export/fcpxml-multicam',
                          json={'deliver_to': 'nle',
                                'sources': ['client_selects']})
        assert res.status_code == 200, res.data
        data = res.get_json()
        assert data['nle'] == 'fcp'
        assert data['opened_in'] == 'app'
        assert [h[0] for h in handoffs] == ['fcp']

    def test_multicam_route_fcp_passes_media_context(
            self, client, tmp_path, monkeypatch):
        # The multicam FCP handoff used to drop source_media_path /
        # project_name / timeline_name (unlike every other export route).
        self._project('rp3', tmp_path)
        _stub_multicam_writer(monkeypatch)
        handoffs = []

        def _hand(path, nle, **k):
            handoffs.append(k)
            return 'app', {}

        monkeypatch.setattr(app_module, '_hand_file_to_nle', _hand)
        res = client.post('/project/rp3/export/fcpxml-multicam',
                          json={'deliver_to': 'nle', 'nle': 'fcp',
                                'sources': ['client_selects']})
        assert res.status_code == 200, res.data
        (kwargs,) = handoffs
        assert 'source_media_path' in kwargs
        assert kwargs['timeline_name']  # splitext of the export filename

    def test_multicam_route_premiere_saves_with_honest_hint(
            self, client, tmp_path, monkeypatch):
        self._project('rp4', tmp_path)
        _stub_multicam_writer(monkeypatch)
        monkeypatch.setattr(app_module, '_run_open',
                            lambda args, timeout=5.0: (True, ''))
        res = client.post('/project/rp4/export/fcpxml-multicam',
                          json={'deliver_to': 'nle', 'nle': 'premiere',
                                'sources': ['client_selects']})
        assert res.status_code == 200, res.data
        data = res.get_json()
        assert data['nle'] == 'premiere'
        assert data['opened_in'] == 'finder'
        assert data['import_fallback'] is True
        assert data['premiere_manual_import'] is True
        assert 'cannot read FCPXML' in data['hint']

    def test_multicam_route_unknown_nle_rejected(self, client, tmp_path):
        self._project('rp5', tmp_path)
        res = client.post('/project/rp5/export/fcpxml-multicam',
                          json={'deliver_to': 'nle', 'nle': 'avid',
                                'sources': ['client_selects']})
        assert res.status_code == 400
        assert 'Unknown NLE' in res.get_json()['error']

    def test_multicam_route_resolve_missing_app_still_succeeds(
            self, client, tmp_path, monkeypatch):
        # Resolve delivery is save + reveal — a missing Resolve must not
        # fail the export (same rule Premiere has always had).
        self._project('rp6', tmp_path)
        _stub_multicam_writer(monkeypatch)
        self._no_scripted_resolve(monkeypatch)
        monkeypatch.setattr(app_module, '_run_open',
                            lambda args, timeout=5.0: (True, ''))
        monkeypatch.setattr(app_module, '_find_nle_app_path', lambda nle: None)
        res = client.post('/project/rp6/export/fcpxml-multicam',
                          json={'deliver_to': 'nle', 'nle': 'resolve',
                                'sources': ['client_selects']})
        assert res.status_code == 200, res.data
        data = res.get_json()
        assert data['opened_in'] == 'finder'
        assert data['import_fallback'] is True

    def test_send_to_nle_multicam_resolve_no_longer_forced_to_fcp(
            self, client, tmp_path, monkeypatch):
        # /export/send-to-nle used to hard-force nle='fcp' for multicam.
        self._project('rp7', tmp_path)
        _stub_multicam_writer(monkeypatch)
        self._no_scripted_resolve(monkeypatch)
        fcp_handoffs = []
        monkeypatch.setattr(
            app_module, '_hand_file_to_nle',
            lambda *a, **k: fcp_handoffs.append((a, k)) or ('app', {}))
        monkeypatch.setattr(app_module, '_run_open',
                            lambda args, timeout=5.0: (True, ''))
        monkeypatch.setattr(app_module, '_find_nle_app_path',
                            lambda nle: '/Applications/DaVinci Resolve.app')
        res = client.post('/export/send-to-nle',
                          json={'project_id': 'rp7', 'export_type': 'multicam',
                                'nle': 'resolve'})
        assert res.status_code == 200, res.data
        data = res.get_json()
        assert data['nle'] == 'resolve'
        assert data['import_fallback'] is True
        assert fcp_handoffs == []

    def test_send_to_nle_flat_types_still_gated(self, client, tmp_path):
        # The round-trip exemption is multicam-only: flat exports of a
        # round-trip-shaped project still 400 for non-FCP targets (their
        # source_path is the app-internal timeline_audio.wav).
        self._project('rp8', tmp_path)
        res = client.post('/export/send-to-nle',
                          json={'project_id': 'rp8', 'nle': 'resolve'})
        assert res.status_code == 400
        assert 'Final Cut Pro' in res.get_json()['error']
