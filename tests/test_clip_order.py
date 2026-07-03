"""Clips-tab manual drag-to-reorder (clip_order_mode) regressions.

The Clips tab lets the editor drag clips into a manual order. The design
(mirroring the Story Builder drag): the persisted ``labeled_sections`` ARRAY
order is the canonical manual order, gated by one persisted flag
``clip_order_mode`` ('time' default | 'manual', set by the first drag).

Pins the server-side contract:

  - save_labels persists a VALID clip_order_mode from the body; an omitted or
    invalid value preserves the stored mode (older clients and the 1.0.27
    subset-export _persistLabelSections calls omit the key — they must not
    reset a manual ordering), and the stored array order round-trips verbatim.
  - Cross-origin (share-portal) callers can never flip the mode, and a
    cross-origin reorder of labeled_sections stays REJECTED by the
    append-only gate (403) — reorder is a same-origin editor capability.
  - Direct /export/fcpxml path: the labels->markers loop tags '_order' only
    when the project is in manual mode AND the export is clips-only; the
    FCPXML generator then emits the spine in manual order, and stays
    chronological for every untagged (pre-existing) caller.
  - The OTHER direct exporters honor the same tags: PremiereXMLExporter
    (also serving 'resolve-xml' via ResolveXMLExporter) and EDLExporter
    ('resolve-edl') sort by '_order' when present — the labels loop tags
    markers platform-agnostically, so an untagged-aware FCP path plus
    chronological-only Premiere/EDL paths would silently discard the manual
    arrangement on those platforms. EDL record TC must stay monotonic
    regardless (it accumulates durations in event order).
  - Round-trip /export/fcpxml-multicam: the body's preserve_order flag
    reaches write_selects_as_new_project and the selects arrive in stored
    (manual) array order.
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module
from fcpxml_export import generate_fcpxml
from exporters import edl as edl_module
from exporters import premiere_xml as premiere_module
from exporters.edl import EDLExporter
from exporters.premiere_xml import PremiereXMLExporter


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


def _read_meta(pid):
    pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
    return json.loads((pdir / 'meta.json').read_text())


def _make_roundtrip_project(pid, tmp_path, **extra):
    """A project ingested from an FCPXML (fcpxml_source block present)."""
    stored = tmp_path / f'{pid}-stored.fcpxml'
    stored.write_text('<fcpxml version="1.14"/>')
    return _make_project(pid, fcpxml_source={
        'container_type': 'sync-clip',
        'timeline_audio_rendered': True,
        'stored_fcpxml_path': str(stored),
    }, **extra)


# Three sections deliberately NOT in chronological order: the array order
# (manual) is C(20s), A(2s), B(10s); chronological is A, B, C.
MANUAL_SECTIONS = [
    {'start': 20.0, 'end': 25.0, 'color': 'green', 'text': 'CLIP_CHARLIE'},
    {'start': 2.0, 'end': 5.0, 'color': 'blue', 'text': 'CLIP_ALPHA'},
    {'start': 10.0, 'end': 14.0, 'color': 'red', 'text': 'CLIP_BRAVO'},
]

PORTAL_ORIGIN = {'Origin': 'https://share.doza.ai'}


class _StubExporter:
    """Captures markers; returns an exporter-shaped result without disk IO."""
    file_extension = '.fcpxml'

    def __init__(self):
        self.markers = None

    def export_markers(self, markers, **kwargs):
        self.markers = markers
        return SimpleNamespace(
            file_path='/tmp/out.fcpxml', filename='out.fcpxml',
            format_name='FCPXML', platform_name='Final Cut Pro', warnings=[])


def _quiet_probes(monkeypatch):
    monkeypatch.setattr(app_module, 'get_media_duration', lambda p: None)
    monkeypatch.setattr(app_module, 'get_video_framerate', lambda p: None)
    monkeypatch.setattr(app_module, 'get_video_resolution', lambda p: (1920, 1080))
    monkeypatch.setattr(app_module, 'get_video_start_timecode_info',
                        lambda p, fps: (0, None))


# ── save_labels: clip_order_mode persistence ────────────────────────────────

class TestSaveLabelsClipOrderMode:
    def test_manual_mode_persisted(self, client):
        _make_project('p1')
        res = client.post('/project/p1/labels', json={
            'color_labels': {},
            'labeled_sections': MANUAL_SECTIONS,
            'clip_order_mode': 'manual',
        })
        assert res.status_code == 200
        assert _read_meta('p1')['clip_order_mode'] == 'manual'

    def test_time_mode_persisted_back(self, client):
        _make_project('p2', clip_order_mode='manual')
        res = client.post('/project/p2/labels', json={
            'color_labels': {},
            'labeled_sections': MANUAL_SECTIONS,
            'clip_order_mode': 'time',
        })
        assert res.status_code == 200
        assert _read_meta('p2')['clip_order_mode'] == 'time'

    def test_omitted_key_preserves_stored_mode(self, client):
        # The 1.0.27 subset-export flow (_persistLabelSections) and older
        # clients POST without the key — that must NOT reset manual ordering.
        _make_project('p3', clip_order_mode='manual')
        res = client.post('/project/p3/labels', json={
            'color_labels': {},
            'labeled_sections': MANUAL_SECTIONS[:1],
        })
        assert res.status_code == 200
        assert _read_meta('p3')['clip_order_mode'] == 'manual'

    def test_invalid_value_ignored(self, client):
        _make_project('p4', clip_order_mode='manual')
        res = client.post('/project/p4/labels', json={
            'color_labels': {},
            'labeled_sections': MANUAL_SECTIONS,
            'clip_order_mode': 'banana',
        })
        assert res.status_code == 200
        assert _read_meta('p4')['clip_order_mode'] == 'manual'

    def test_array_order_round_trips_verbatim(self, client):
        # The manual order IS the array order — save_labels must store the
        # list exactly as submitted (this is also what the 1.0.27 subset
        # filter/persist/restore cycle relies on).
        _make_project('p5')
        res = client.post('/project/p5/labels', json={
            'color_labels': {},
            'labeled_sections': MANUAL_SECTIONS,
            'clip_order_mode': 'manual',
        })
        assert res.status_code == 200
        stored = _read_meta('p5')['labeled_sections']
        assert [s['text'] for s in stored] == [
            'CLIP_CHARLIE', 'CLIP_ALPHA', 'CLIP_BRAVO']


# ── portal (cross-origin) guardrails ────────────────────────────────────────

class TestPortalOrderGuardrails:
    def test_cross_origin_reorder_rejected(self, client):
        # The share-portal append-only gate must keep REJECTING reorders:
        # same items, different order = not a pure addition -> 403.
        _make_project('p6', labeled_sections=MANUAL_SECTIONS, color_labels={})
        reordered = [MANUAL_SECTIONS[1], MANUAL_SECTIONS[0], MANUAL_SECTIONS[2]]
        res = client.post('/project/p6/labels', headers=PORTAL_ORIGIN, json={
            'color_labels': {},
            'labeled_sections': reordered,
        })
        assert res.status_code == 403
        assert 'reorder' in res.get_json()['error']
        # Nothing was written.
        stored = _read_meta('p6')['labeled_sections']
        assert [s['text'] for s in stored] == [
            'CLIP_CHARLIE', 'CLIP_ALPHA', 'CLIP_BRAVO']

    def test_cross_origin_pure_addition_cannot_flip_mode(self, client):
        # A portal client may append a clip, but clip_order_mode is a
        # same-origin editor concern — the appended write must not flip it.
        _make_project('p7', labeled_sections=MANUAL_SECTIONS, color_labels={},
                      clip_order_mode='time')
        added = MANUAL_SECTIONS + [
            {'start': 30.0, 'end': 31.0, 'color': 'blue', 'text': 'NEW'}]
        res = client.post('/project/p7/labels', headers=PORTAL_ORIGIN, json={
            'color_labels': {},
            'labeled_sections': added,
            'clip_order_mode': 'manual',
        })
        assert res.status_code == 200
        meta = _read_meta('p7')
        assert meta['clip_order_mode'] == 'time'
        assert len(meta['labeled_sections']) == 4  # the append itself worked


# ── direct export path: '_order' tagging + generator ordering ───────────────

class TestDirectExportManualOrder:
    def _project(self, mode):
        p = {'name': 'P', 'source_path': '', 'color_labels': {},
             'labeled_sections': MANUAL_SECTIONS}
        if mode is not None:
            p['clip_order_mode'] = mode
        return p

    def _run(self, monkeypatch, project, body):
        _quiet_probes(monkeypatch)
        stub = _StubExporter()
        monkeypatch.setattr(app_module, 'get_exporter', lambda p: stub)
        app_module._build_nle_export(project, body)
        return stub.markers

    def test_manual_mode_tags_order_in_array_sequence(self, monkeypatch):
        markers = self._run(monkeypatch, self._project('manual'),
                            {'type': 'labels'})
        assert [m['_order'] for m in markers] == [0, 1, 2]
        assert [m['start'] for m in markers] == [20.0, 2.0, 10.0]

    def test_time_mode_and_legacy_projects_untagged(self, monkeypatch):
        for mode in ('time', None):
            markers = self._run(monkeypatch, self._project(mode),
                                {'type': 'labels'})
            assert all('_order' not in m for m in markers)

    def test_mixed_category_export_stays_untagged(self, monkeypatch):
        # Manual interleave across AI buckets is undefined — only a
        # clips-only export honors the manual order.
        project = self._project('manual')
        project['analysis'] = {'social_clips': [
            {'start': 1.0, 'end': 2.0, 'title': 'S'}]}
        markers = self._run(monkeypatch, project,
                            {'types': ['labels', 'social']})
        assert all('_order' not in m for m in markers)

    def test_skipped_bad_timecode_keeps_contiguous_order(self, monkeypatch):
        # A malformed section is skipped; the '_order' tags of the survivors
        # must stay contiguous with their marker positions.
        project = self._project('manual')
        project['labeled_sections'] = [
            MANUAL_SECTIONS[0],
            {'start': '00:0X:12', 'end': '00:01:00', 'color': 'blue',
             'text': 'BAD'},
            MANUAL_SECTIONS[2],
        ]
        markers = self._run(monkeypatch, project, {'type': 'labels'})
        assert [m['_order'] for m in markers] == [0, 1]
        assert [m['note'] for m in markers] == ['CLIP_CHARLIE', 'CLIP_BRAVO']


class TestGeneratorOrderIdiom:
    MARKERS_TAGGED = [
        {'start': 20.0, 'end': 25.0, 'text': 'MK_CHARLIE', 'category': 'c',
         '_order': 0},
        {'start': 2.0, 'end': 5.0, 'text': 'MK_ALPHA', 'category': 'c',
         '_order': 1},
        {'start': 10.0, 'end': 14.0, 'text': 'MK_BRAVO', 'category': 'c',
         '_order': 2},
    ]

    def _source(self, tmp_path):
        p = tmp_path / 'src.wav'
        p.write_bytes(b'x')  # generator only checks existence
        return str(p)

    def test_order_tags_win_over_chronology(self, tmp_path):
        xml = generate_fcpxml(self.MARKERS_TAGGED, source_path=self._source(tmp_path),
                              media_duration=60.0)
        assert xml.index('MK_CHARLIE') < xml.index('MK_ALPHA') < xml.index('MK_BRAVO')

    def test_untagged_markers_stay_chronological(self, tmp_path):
        untagged = [{k: v for k, v in m.items() if k != '_order'}
                    for m in self.MARKERS_TAGGED]
        xml = generate_fcpxml(untagged, source_path=self._source(tmp_path),
                              media_duration=60.0)
        assert xml.index('MK_ALPHA') < xml.index('MK_BRAVO') < xml.index('MK_CHARLIE')


# ── Premiere XML + EDL direct exporters honor '_order' (F7) ─────────────────
#
# _build_nle_export tags '_order' platform-agnostically, so every direct
# exporter must honor it: 'premiere' (PremiereXMLExporter), 'resolve-xml'
# (ResolveXMLExporter — inherits this export_markers), and 'resolve-edl'
# (EDLExporter). Before the fix these sorted chronologically, silently
# discarding the manual arrangement that the FCP path preserved.

class TestPremiereMarkersManualOrder:
    MARKERS_TAGGED = [
        {'start': 20.0, 'end': 25.0, 'text': 'MK_CHARLIE', 'note': '',
         'color': 'green', 'category': 'c', '_order': 0},
        {'start': 2.0, 'end': 5.0, 'text': 'MK_ALPHA', 'note': '',
         'color': 'blue', 'category': 'c', '_order': 1},
        {'start': 10.0, 'end': 14.0, 'text': 'MK_BRAVO', 'note': '',
         'color': 'red', 'category': 'c', '_order': 2},
    ]

    def _export(self, tmp_path, monkeypatch, markers):
        # Deterministic audio wiring — no ffprobe on the fake source.
        monkeypatch.setattr(premiere_module, 'get_audio_layout', lambda p: (1, 1))
        result = PremiereXMLExporter().export_markers(
            markers,
            project_name='P',
            source_path='/tmp/fake.mov',
            media_duration=60.0,
            framerate=25.0,
            width=1920, height=1080,
            export_type='labels',
            exports_dir=str(tmp_path),
        )
        return Path(result.file_path).read_text()

    def test_order_tags_win_over_chronology(self, tmp_path, monkeypatch):
        xml = self._export(tmp_path, monkeypatch, self.MARKERS_TAGGED)
        assert xml.index('MK_CHARLIE') < xml.index('MK_ALPHA') < xml.index('MK_BRAVO')

    def test_untagged_markers_stay_chronological(self, tmp_path, monkeypatch):
        untagged = [{k: v for k, v in m.items() if k != '_order'}
                    for m in self.MARKERS_TAGGED]
        xml = self._export(tmp_path, monkeypatch, untagged)
        assert xml.index('MK_ALPHA') < xml.index('MK_BRAVO') < xml.index('MK_CHARLIE')

    def test_zero_length_markers_still_dropped(self, tmp_path, monkeypatch):
        markers = self.MARKERS_TAGGED + [
            {'start': 30.0, 'end': 30.0, 'text': 'MK_EMPTY', 'note': '',
             'color': 'blue', 'category': 'c', '_order': 3}]
        xml = self._export(tmp_path, monkeypatch, markers)
        assert 'MK_EMPTY' not in xml


class TestEDLMarkersManualOrder:
    MARKERS_TAGGED = TestPremiereMarkersManualOrder.MARKERS_TAGGED

    def _export(self, tmp_path, monkeypatch, markers):
        # Deterministic AA/V channel — no ffprobe on the fake source.
        monkeypatch.setattr(edl_module, 'has_video_stream', lambda p: True)
        result = EDLExporter().export_markers(
            markers,
            project_name='P',
            source_path='/tmp/fake.mov',
            media_duration=60.0,
            framerate=25.0,
            width=1920, height=1080,
            export_type='labels',
            exports_dir=str(tmp_path),
        )
        return Path(result.file_path).read_text()

    @staticmethod
    def _edit_lines(text):
        return [ln for ln in text.splitlines()
                if ln[:3].isdigit() and ' C        ' in ln]

    def test_order_tags_win_over_chronology(self, tmp_path, monkeypatch):
        text = self._export(tmp_path, monkeypatch, self.MARKERS_TAGGED)
        assert text.index('MK_CHARLIE') < text.index('MK_ALPHA') < text.index('MK_BRAVO')
        # Source-in of the FIRST event is the manually-first clip (20s @25fps).
        first_src_in = self._edit_lines(text)[0].split()[-4]
        assert first_src_in == '00:00:20:00'

    def test_record_tc_stays_monotonic_with_manual_order(self, tmp_path, monkeypatch):
        # sequential_record accumulates durations in event order, so record
        # TC must stay monotonic even when source TC jumps around.
        text = self._export(tmp_path, monkeypatch, self.MARKERS_TAGGED)
        edits = self._edit_lines(text)
        rec_ins = [ln.split()[-2] for ln in edits]
        rec_outs = [ln.split()[-1] for ln in edits]
        assert rec_ins[0] == '01:00:00:00'
        assert rec_ins == sorted(rec_ins)
        assert all(ri < ro for ri, ro in zip(rec_ins, rec_outs))
        # Consecutive events butt together: durations 5s + 3s + 4s @25fps.
        assert rec_outs[:2] == rec_ins[1:]
        assert rec_outs[-1] == '01:00:12:00'

    def test_untagged_markers_stay_chronological(self, tmp_path, monkeypatch):
        untagged = [{k: v for k, v in m.items() if k != '_order'}
                    for m in self.MARKERS_TAGGED]
        text = self._export(tmp_path, monkeypatch, untagged)
        assert text.index('MK_ALPHA') < text.index('MK_BRAVO') < text.index('MK_CHARLIE')


# ── round-trip export path: preserve_order body flag ────────────────────────

class TestRoundTripManualOrder:
    def _stub_writer(self, monkeypatch, captured):
        monkeypatch.setattr(app_module, 'parse_fcpxml', lambda p: object())

        def _writer(parsed, selects, preserve_order=False, skipped_out=None):
            captured['preserve_order'] = preserve_order
            captured['starts'] = [s.start_seconds for s in selects]
            return b'<fcpxml version="1.14"/>'

        monkeypatch.setattr(app_module, 'write_selects_as_new_project', _writer)

    def _post(self, client, extra=None):
        body = {'mode': 'selects_project', 'sources': ['client_selects'],
                'deliver_to': 'file'}
        body.update(extra or {})
        return client.post('/project/rt1/export/fcpxml-multicam', json=body)

    def test_preserve_order_flag_keeps_manual_array_order(
            self, client, tmp_path, monkeypatch):
        _make_roundtrip_project('rt1', tmp_path,
                                labeled_sections=MANUAL_SECTIONS,
                                clip_order_mode='manual')
        captured = {}
        self._stub_writer(monkeypatch, captured)
        monkeypatch.setattr(app_module, '_reveal_in_finder', lambda p: None)

        res = self._post(client, {'preserve_order': True})
        assert res.status_code == 200, res.data
        assert captured['preserve_order'] is True
        # Selects reach the writer in the stored (manual) array order; with
        # preserve_order the writer's _iter_selects keeps it (writer-side
        # behavior pinned in test_fcpxml_writer.py — Story Builder path).
        assert captured['starts'] == [20.0, 2.0, 10.0]

    def test_without_flag_writer_gets_default_sorting(
            self, client, tmp_path, monkeypatch):
        _make_roundtrip_project('rt1', tmp_path,
                                labeled_sections=MANUAL_SECTIONS)
        captured = {}
        self._stub_writer(monkeypatch, captured)
        monkeypatch.setattr(app_module, '_reveal_in_finder', lambda p: None)

        res = self._post(client)
        assert res.status_code == 200, res.data
        assert captured['preserve_order'] is False
