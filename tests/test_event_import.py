"""Event-level FCPXML import: browser exports with no scratch sequence.

Covers the ``event_import`` module (enumeration + wrapper synthesis), the
``/fcpxml/inspect`` pre-flight, the ``/create`` ``event_clip_index`` path, and
the guarantee that the synthesized wrapper is indistinguishable — numerically
and structurally — from the hand-made scratch sequence it replaces.

Two fixture families:
  * a synthetic two-multicam event export whose angle media are real WAVs on
    disk (drives the Flask end-to-end tests), and
  * an event-shaped derivation of ``multiseg_angle.fcpxml`` (the anonymized
    copy of the 2026-08-21 field file: 4 angles, 43 stop-start camera files,
    720000-timescale assets) proving wrapper-parse parity part-for-part
    against the original spine parse.
"""

import json
import os
import re
import sys
import textwrap
from pathlib import Path

import pytest
from lxml import etree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module  # noqa: E402
from doza_assist.fcpxml import parse_fcpxml  # noqa: E402
from doza_assist.fcpxml.event_import import (  # noqa: E402
    ParseError,
    enumerate_event_clips,
    synthesize_wrapper,
)

MULTISEG_FIXTURE = Path(__file__).parent / "fixtures" / "multiseg_angle.fcpxml"

APPLE_DTD = Path(
    "/Applications/Final Cut Pro.app/Contents/Frameworks/"
    "Interchange.framework/Versions/A/Resources/FCPXMLv1_14.dtd"
)


def _make_wav(path, seconds=0.1):
    import wave as _wave
    with _wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * int(16000 * seconds))


# Two multicams at event level plus one sync-clip (which V1 skips).
# Multicam B has a non-zero tcStart and its mc-clip omits ``start`` — the
# wrapper must pin start to tcStart or the audio window misses every angle
# file (the 1.0.43 single-file-fallback field bug shape).
_EVENT_EXPORT = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>

    <fcpxml version="1.14">
        <resources>
            <format id="r1" name="FF30" frameDuration="1/30s" width="1920" height="1080"/>
            <asset id="r2" name="wavA" start="0s" duration="100s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                <media-rep kind="original-media" src="file://{wav_a}"/>
            </asset>
            <asset id="r3" name="wavB" start="0s" duration="100s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                <media-rep kind="original-media" src="file://{wav_b}"/>
            </asset>
            <media id="r10" name="Interview A">
                <multicam format="r1" tcStart="0s" tcFormat="NDF">
                    <mc-angle name="A1" angleID="A1">
                        <asset-clip ref="r2" offset="0s" name="wavA" start="0s" duration="10s" audioRole="dialogue"/>
                    </mc-angle>
                </multicam>
            </media>
            <media id="r11" name="Interview B">
                <multicam format="r1" tcStart="3600s" tcFormat="NDF">
                    <mc-angle name="B1" angleID="B1">
                        <asset-clip ref="r3" offset="0s" name="wavB" start="0s" duration="8s" audioRole="dialogue"/>
                    </mc-angle>
                </multicam>
            </media>
        </resources>
        <event name="Shoot Day 1">
            <mc-clip ref="r10" name="Interview A" offset="0s" start="0s" duration="10s">
                <mc-source angleID="A1" srcEnable="all"/>
            </mc-clip>
            <mc-clip ref="r11" name="Interview B" duration="8s">
                <mc-source angleID="B1" srcEnable="all"/>
            </mc-clip>
            <sync-clip name="Not Multicam" offset="0s" start="0s" duration="5s" format="r1">
                <asset-clip ref="r2" offset="0s" name="wavA" start="0s" duration="5s" audioRole="dialogue"/>
            </sync-clip>
        </event>
    </fcpxml>
""")


def _event_doc(tmp_path) -> bytes:
    wav_a = tmp_path / "wav_a.wav"
    wav_b = tmp_path / "wav_b.wav"
    _make_wav(wav_a)
    _make_wav(wav_b)
    return _EVENT_EXPORT.format(wav_a=wav_a, wav_b=wav_b).encode()


def _multiseg_event_doc() -> bytes:
    """multiseg_angle.fcpxml re-housed as an event export (no project/sequence)."""
    src = MULTISEG_FIXTURE.read_bytes()
    mc = re.search(rb"<mc-clip .*?</mc-clip>", src, re.S).group(0)
    res = re.search(rb"<resources>.*</resources>", src, re.S).group(0)
    return (b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n\n'
            b'<fcpxml version="1.14">\n    ' + res +
            b'\n    <event name="Interviews">\n        ' + mc +
            b'\n    </event>\n</fcpxml>\n')


# ---------- enumeration ------------------------------------------------------

class TestEnumerate:
    def test_event_shape(self, tmp_path):
        clips, skipped = enumerate_event_clips(_event_doc(tmp_path))
        assert [(c.index, c.name, c.angle_count, c.event_name) for c in clips] == [
            (0, "Interview A", 1, "Shoot Day 1"),
            (1, "Interview B", 1, "Shoot Day 1"),
        ]
        assert skipped == 1  # the sync-clip
        assert clips[0].duration_seconds == pytest.approx(10.0)

    def test_bare_fcpxml_shape(self, tmp_path):
        doc = _event_doc(tmp_path)
        doc = doc.replace(b'<event name="Shoot Day 1">', b"").replace(b"</event>", b"")
        clips, _ = enumerate_event_clips(doc)
        assert len(clips) == 2
        assert clips[0].event_name is None

    def test_sequence_shaped_file_refuses(self):
        # A document WITH project/sequence belongs to the normal import path.
        clips, skipped = enumerate_event_clips(MULTISEG_FIXTURE.read_bytes())
        assert clips == [] and skipped == 0

    def test_unsupported_version_refuses(self, tmp_path):
        doc = _event_doc(tmp_path).replace(b'version="1.14"', b'version="1.5"')
        assert enumerate_event_clips(doc) == ([], 0)

    def test_garbage_refuses(self):
        assert enumerate_event_clips(b"not xml at all") == ([], 0)
        assert enumerate_event_clips(b"<notfcpxml/>") == ([], 0)

    def test_dangling_ref_is_skipped_not_fatal(self, tmp_path):
        doc = _event_doc(tmp_path).replace(b'<mc-clip ref="r10"', b'<mc-clip ref="rNOPE"')
        clips, skipped = enumerate_event_clips(doc)
        assert [c.name for c in clips] == ["Interview B"]
        assert skipped == 2  # dangling mc-clip + the sync-clip


# ---------- wrapper synthesis ------------------------------------------------

class TestSynthesize:
    def test_out_of_range_raises(self, tmp_path):
        with pytest.raises(ParseError, match="out of range"):
            synthesize_wrapper(_event_doc(tmp_path), 5)

    def test_resources_stay_byte_verbatim(self, tmp_path):
        doc = _event_doc(tmp_path)
        wrapper, _ = synthesize_wrapper(doc, 0)
        original_resources = re.search(rb"<resources>.*</resources>", doc, re.S).group(0)
        assert original_resources in wrapper

    def test_absent_start_pins_to_multicam_tcstart(self, tmp_path):
        wrapper, info = synthesize_wrapper(_event_doc(tmp_path), 1)
        root = etree.fromstring(wrapper)
        mc = root.find(".//spine/mc-clip")
        assert mc.get("start") == "3600s"
        assert mc.get("offset") == "0s"
        # And the parse resolves the angle window instead of degrading:
        p = tmp_path / "w.fcpxml"
        p.write_bytes(wrapper)
        parsed = parse_fcpxml(p)
        seg = parsed.spine_segments[0]
        assert len(seg.audio_parts) == 1
        assert seg.audio_parts[0].path.endswith("wav_b.wav")
        assert info["clip_name"] == "Interview B"

    def test_explicit_start_kept_verbatim(self, tmp_path):
        wrapper, _ = synthesize_wrapper(_event_doc(tmp_path), 0)
        root = etree.fromstring(wrapper)
        mc = root.find(".//spine/mc-clip")
        assert mc.get("start") == "0s"
        # mc-source enablement rides through untouched.
        assert mc.find("mc-source").get("angleID") == "A1"

    def test_names_flow_into_wrapper(self, tmp_path):
        wrapper, _ = synthesize_wrapper(_event_doc(tmp_path), 0)
        root = etree.fromstring(wrapper)
        assert root.find(".//project").get("name") == "Interview A"
        assert root.find(".//event").get("name") == "Shoot Day 1"
        seq = root.find(".//sequence")
        assert seq.get("format") == "r1"
        assert seq.get("duration") == "10s"

    @pytest.mark.skipif(not APPLE_DTD.exists(), reason="Final Cut Pro DTD not present")
    def test_wrapper_validates_against_apple_dtd(self, tmp_path):
        wrapper, _ = synthesize_wrapper(_event_doc(tmp_path), 0)
        dtd = etree.DTD(str(APPLE_DTD))
        root = etree.fromstring(wrapper)
        assert dtd.validate(root), dtd.error_log


class TestMultisegParity:
    """The wrapper parse must be numerically identical to the scratch-sequence
    parse of the same multicam — same parts, same fractions, same flags — on
    the field file's shape (4 angles, stop-start takes, 720000 timescale)."""

    def test_part_for_part_parity(self, tmp_path):
        event_doc = _multiseg_event_doc()
        clips, skipped = enumerate_event_clips(event_doc)
        assert len(clips) == 1 and skipped == 0
        assert clips[0].angle_count == 4

        wrapper, _ = synthesize_wrapper(event_doc, 0)
        wpath = tmp_path / "wrapper.fcpxml"
        wpath.write_bytes(wrapper)
        pw = parse_fcpxml(wpath)
        po = parse_fcpxml(MULTISEG_FIXTURE)

        assert len(pw.spine_segments) == 1
        sw, so = pw.spine_segments[0], po.spine_segments[0]
        assert (sw.kind, sw.ref, sw.mc_sources) == (so.kind, so.ref, so.mc_sources)
        assert sw.offset_fraction == so.offset_fraction
        assert sw.start_fraction == so.start_fraction
        assert sw.duration_fraction == so.duration_fraction
        assert pw.is_multi_source and po.is_multi_source
        assert pw.sequence_frame_duration == po.sequence_frame_duration
        assert len(sw.audio_parts) == len(so.audio_parts) == 10
        for a, b in zip(sw.audio_parts, so.audio_parts):
            assert (a.path, a.angle_offset_fraction, a.angle_start_fraction,
                    a.part_duration_fraction) == \
                   (b.path, b.angle_offset_fraction, b.angle_start_fraction,
                    b.part_duration_fraction)


# ---------- Flask: /fcpxml/inspect and /create -------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setitem(app_module.app.config, "PROJECTS_DIR", str(tmp_path / "projects"))
    os.makedirs(app_module.app.config["PROJECTS_DIR"], exist_ok=True)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


def _write_event_file(tmp_path) -> Path:
    p = tmp_path / "Shoot Day 1.fcpxml"
    p.write_bytes(_event_doc(tmp_path))
    return p


class TestInspectRoute:
    def test_lists_event_clips(self, client, tmp_path):
        p = _write_event_file(tmp_path)
        resp = client.post("/fcpxml/inspect", json={"source_path": str(p)})
        assert resp.status_code == 200, resp.data
        data = resp.get_json()
        assert [c["name"] for c in data["clips"]] == ["Interview A", "Interview B"]
        assert data["clips"][1]["index"] == 1
        assert data["skipped_other_clips"] == 1

    def test_sequence_file_returns_empty(self, client):
        resp = client.post("/fcpxml/inspect", json={"source_path": str(MULTISEG_FIXTURE)})
        assert resp.status_code == 200
        assert resp.get_json()["clips"] == []

    def test_missing_file_400(self, client):
        resp = client.post("/fcpxml/inspect", json={"source_path": "/nope/x.fcpxml"})
        assert resp.status_code == 400

    def test_non_fcpxml_400(self, client, tmp_path):
        wav = tmp_path / "a.wav"
        _make_wav(wav)
        resp = client.post("/fcpxml/inspect", json={"source_path": str(wav)})
        assert resp.status_code == 400


class TestCreateFromEvent:
    def _create(self, client, path, **extra):
        return client.post("/create", json={"source_path": str(path), **extra})

    def _meta(self, pid):
        meta_path = Path(app_module.app.config["PROJECTS_DIR"]) / pid / "meta.json"
        return json.loads(meta_path.read_text())

    def test_default_imports_first_clip(self, client, tmp_path):
        resp = self._create(client, _write_event_file(tmp_path))
        assert resp.status_code == 200, resp.data
        meta = self._meta(resp.get_json()["project_id"])
        assert meta["name"] == "Interview A"
        ei = meta["fcpxml_source"]["event_import"]
        assert (ei["clip_index"], ei["total_clips"]) == (0, 2)
        assert meta["fcpxml_source"]["container_type"] == "mc-clip"
        assert meta["source_path"].endswith("wav_a.wav")
        stored = meta["fcpxml_source"]["stored_fcpxml_path"]
        assert stored.endswith("event-import.fcpxml") and os.path.exists(stored)
        # Provenance copy of the raw export sits alongside the wrapper.
        assert (Path(stored).parent / "original-event-export.fcpxml").exists()

    def test_explicit_index_imports_that_clip(self, client, tmp_path):
        resp = self._create(client, _write_event_file(tmp_path), event_clip_index=1)
        assert resp.status_code == 200, resp.data
        meta = self._meta(resp.get_json()["project_id"])
        assert meta["name"] == "Interview B"
        assert meta["source_path"].endswith("wav_b.wav")
        assert meta["fcpxml_source"]["event_import"]["clip_index"] == 1

    def test_out_of_range_index_400_and_no_orphan_dir(self, client, tmp_path):
        resp = self._create(client, _write_event_file(tmp_path), event_clip_index=9)
        assert resp.status_code == 400
        assert "out of range" in resp.get_json()["error"]
        assert os.listdir(app_module.app.config["PROJECTS_DIR"]) == []

    def test_non_integer_index_400(self, client, tmp_path):
        resp = self._create(client, _write_event_file(tmp_path), event_clip_index="two")
        assert resp.status_code == 400

    def test_sequence_file_path_unchanged(self, client, tmp_path):
        # A normal sequence-shaped FCPXML must not grow an event_import block
        # and must keep its exact stored-copy behavior.
        wav = tmp_path / "solo.wav"
        _make_wav(wav)
        doc = textwrap.dedent(f"""\
            <?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE fcpxml>
            <fcpxml version="1.14">
                <resources>
                    <format id="r1" name="FF30" frameDuration="1/30s" width="1920" height="1080"/>
                    <asset id="r2" name="solo" start="0s" duration="100s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                        <media-rep kind="original-media" src="file://{wav}"/>
                    </asset>
                </resources>
                <library><event name="E"><project name="Seq">
                    <sequence format="r1" duration="10s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                        <spine>
                            <asset-clip ref="r2" offset="0s" name="solo" start="0s" duration="10s" audioRole="dialogue"/>
                        </spine>
                    </sequence>
                </project></event></library>
            </fcpxml>
        """)
        p = tmp_path / "seq.fcpxml"
        p.write_text(doc)
        resp = self._create(client, p, event_clip_index=0)
        assert resp.status_code == 200, resp.data
        meta = self._meta(resp.get_json()["project_id"])
        assert "event_import" not in meta["fcpxml_source"]
        assert meta["fcpxml_source"]["stored_fcpxml_path"].endswith("seq.fcpxml")

    def test_unreadable_file_keeps_original_error(self, client, tmp_path):
        p = tmp_path / "broken.fcpxml"
        p.write_text("<fcpxml version='1.14'><resources></fcpxml>")
        resp = self._create(client, p)
        assert resp.status_code == 400
        assert resp.get_json()["error"].startswith("Could not read FCPXML:")


# ---------- Mode A export from an event-imported project ---------------------

class TestModeAExport:
    def test_selects_project_export_from_event_import(self, client, tmp_path, monkeypatch):
        resp = client.post("/create", json={
            "source_path": str(_write_event_file(tmp_path)),
            "event_clip_index": 0,
        })
        assert resp.status_code == 200, resp.data
        pid = resp.get_json()["project_id"]

        meta_path = Path(app_module.app.config["PROJECTS_DIR"]) / pid / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["color_labels"] = {"green": "Best"}
        meta["labeled_sections"] = [
            {"start": 1.0, "end": 3.0, "color": "green", "text": "hook"},
        ]
        meta_path.write_text(json.dumps(meta))

        exports_dir = tmp_path / "exports"
        monkeypatch.setitem(app_module.app.config, "EXPORTS_DIR", str(exports_dir))
        monkeypatch.setattr(app_module, "_reveal_in_finder", lambda p: None)

        resp = client.post(
            f"/project/{pid}/export/fcpxml-multicam",
            json={"mode": "selects_project", "source": "client_selects",
                  "deliver_to": "file"},
        )
        assert resp.status_code == 200, resp.data

        produced = [p for p in exports_dir.rglob("*") if p.name.endswith(".fcpxml")]
        assert produced, "no .fcpxml written to exports dir"
        out_bytes = produced[0].read_bytes()
        root = etree.fromstring(out_bytes)
        mc = root.find(".//spine/mc-clip")
        assert mc is not None and mc.get("ref") == "r10"
        assert mc.find("mc-source").get("angleID") == "A1"

        if APPLE_DTD.exists():
            dtd = etree.DTD(str(APPLE_DTD))
            assert dtd.validate(root), dtd.error_log
