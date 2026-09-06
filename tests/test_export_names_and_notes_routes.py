"""Export naming and verbatim clip notes across every path (1.1).

Raw-media exporter, round-trip writer, Story Builder and the app routes:
timeline names "{Project} – {Kind} {N}", per-project counters that persist,
the Timeline name override, the editor's original event kept on round-trip,
one "Doza Assist" provenance keyword per clip, and a <note> that is the
clip's first child carrying the verbatim transcript (capped, XML-escaped).
Every FCPXML fixture written here is validated against Apple's DTD for the
file's own <fcpxml version> when Final Cut Pro is installed.
"""

import json
import os
import re
import sys
import textwrap
from pathlib import Path

import pytest
from lxml import etree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402
import fcpxml_export  # noqa: E402
from doza_assist.fcpxml import parse_fcpxml, write_selects_as_new_project, write_markers_on_timeline  # noqa: E402
from doza_assist.fcpxml.writer import Select  # noqa: E402

DTD_DIR = "/Applications/Final Cut Pro.app/Contents/Frameworks/Interchange.framework/Versions/A/Resources"


def validate_dtd(xml_bytes: bytes):
    """Validate against the DTD matching the file's <fcpxml version>; skip
    cleanly when Final Cut Pro is not installed."""
    root = etree.fromstring(xml_bytes)
    version = root.get("version") or "1.11"
    dtd_path = os.path.join(DTD_DIR, f"FCPXMLv{version.replace('.', '_')}.dtd")
    if not os.path.exists(dtd_path):
        pytest.skip(f"FCPXML DTD not installed: {dtd_path}")
    with open(dtd_path, "rb") as fh:
        dtd = etree.DTD(fh)
    assert dtd.validate(root), str(dtd.error_log)
    return root


SEGMENTS = [
    {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00", "text": 'We said "yes" & meant it.'},
    {"start": 4.0, "end": 8.0, "speaker": "SPEAKER_01", "text": "Did the fans <love> it?"},
    {"start": 8.0, "end": 12.0, "speaker": "SPEAKER_00", "text": "Every night."},
]
NAMES = {"SPEAKER_00": "Sarah", "SPEAKER_01": "Mike"}


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path):
    app_module.app.config["PROJECTS_DIR"] = str(tmp_path / "projects")
    app_module.app.config["EXPORTS_DIR"] = str(tmp_path / "exports")
    Path(app_module.app.config["PROJECTS_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(app_module.app.config["EXPORTS_DIR"]).mkdir(parents=True, exist_ok=True)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def _quiet_probes(monkeypatch):
    monkeypatch.setattr(app_module, "get_media_duration", lambda p: 60.0)
    monkeypatch.setattr(app_module, "get_video_framerate", lambda p: 25.0)
    monkeypatch.setattr(app_module, "get_video_resolution", lambda p: (1920, 1080))
    monkeypatch.setattr(app_module, "get_video_start_timecode_info", lambda p, fps: (0, "NDF"))
    monkeypatch.setattr(fcpxml_export, "_resolve_audio_decl", lambda p, w=None: ("", "", None, "1"))
    monkeypatch.setattr(fcpxml_export, "_resolve_is_video", lambda p, w=None: True)
    monkeypatch.setattr(app_module, "_reveal_in_finder", lambda p: None)


def _make_project(pid, tmp_path, name="Ella Interview", **extra):
    media = tmp_path / f"{pid}.mov"
    media.write_bytes(b"\0" * 1024)
    pdir = Path(app_module.app.config["PROJECTS_DIR"]) / pid
    pdir.mkdir(parents=True)
    meta = {"id": pid, "name": name, "status": "complete", "source_path": str(media), "filepath": str(media),
            "editing_platform": "fcp",
            "transcript": {"segments": SEGMENTS, "duration": 12.0}, "speaker_names": NAMES,
            "color_labels": {"green": "Best"},
            "labeled_sections": [{"start": 0.0, "end": 7.0, "color": "green", "text": "Opening hook", "title": "Hero Moment"}],
            "analysis": {"story_beats": [{"label": "The Turn", "description": "Where it flips", "start": "00:00:08", "end": "00:00:11"}],
                         "social_clips": [], "strongest_soundbites": []}}
    meta.update(extra)
    (pdir / "meta.json").write_text(json.dumps(meta))
    return pdir


def _meta(pid):
    return json.loads((Path(app_module.app.config["PROJECTS_DIR"]) / pid / "meta.json").read_text())


def _export(client, pid, **body):
    body = {"types": ["labels"], "mode": "cuts", "deliver_to": "file", **body}
    r = client.post(f"/project/{pid}/export/fcpxml", json=body)
    assert r.status_code == 200, r.data
    d = r.get_json()
    return d, Path(d["file"]).read_bytes()


def _project_name_of(xml_bytes):
    return etree.fromstring(xml_bytes).find(".//project").get("name")


def _event_name_of(xml_bytes):
    return etree.fromstring(xml_bytes).find(".//event").get("name")


# ── raw media: names, counter, override, event ──────────────────────────────

def test_raw_selects_name_counter_filename_and_event(client, tmp_path, monkeypatch):
    _quiet_probes(monkeypatch)
    _make_project("p1", tmp_path)
    d1, xml1 = _export(client, "p1")
    assert _project_name_of(xml1) == "Ella Interview – Selects 1"
    assert _event_name_of(xml1) == "Ella Interview"
    assert d1["filename"] == "Ella Interview – Selects 1.fcpxml"
    assert _meta("p1")["export_counts"] == {"selects": 1}
    validate_dtd(xml1)

    d2, xml2 = _export(client, "p1")
    assert _project_name_of(xml2) == "Ella Interview – Selects 2"
    assert d2["filename"] == "Ella Interview – Selects 2.fcpxml"
    assert _meta("p1")["export_counts"]["selects"] == 2
    assert os.path.exists(d1["file"]) and os.path.exists(d2["file"])   # two files, nothing overwritten
    assert "Doza" not in _project_name_of(xml2)


def test_raw_markers_mode_names_and_event_without_suffix(client, tmp_path, monkeypatch):
    _quiet_probes(monkeypatch)
    _make_project("p2", tmp_path)
    d, xml = _export(client, "p2", mode="markers")
    assert _project_name_of(xml) == "Ella Interview – Markers 1"
    assert _event_name_of(xml) == "Ella Interview"                # no " Markers" on the event
    assert d["filename"] == "Ella Interview – Markers 1.fcpxml"
    assert _meta("p2")["export_counts"] == {"markers": 1}
    validate_dtd(xml)


def test_override_wins_and_leaves_the_counter_alone(client, tmp_path, monkeypatch):
    _quiet_probes(monkeypatch)
    _make_project("p3", tmp_path)
    d, xml = _export(client, "p3", timeline_name="  Client   cut  ")
    assert _project_name_of(xml) == "Client cut" and d["filename"] == "Client cut.fcpxml"
    assert "export_counts" not in _meta("p3")
    # slashes in an override are safe in the filename but kept in the timeline
    d, xml = _export(client, "p3", timeline_name="24/7: The Grind")
    assert _project_name_of(xml) == "24/7: The Grind" and d["filename"] == "24-7- The Grind.fcpxml"
    validate_dtd(xml)


def test_missing_project_name_falls_back(client, tmp_path, monkeypatch):
    _quiet_probes(monkeypatch)
    _make_project("p4", tmp_path, name="")
    d, xml = _export(client, "p4")
    assert _project_name_of(xml) == "Interview – Selects 1"
    assert _event_name_of(xml) == "Doza Assist"
    assert "Doza" not in _project_name_of(xml)


def test_timeline_name_endpoint_previews_the_next_name(client, tmp_path, monkeypatch):
    _quiet_probes(monkeypatch)
    _make_project("p5", tmp_path)
    d = client.get("/project/p5/export/timeline-name").get_json()
    assert d == {"timeline_name": "Ella Interview – Selects 1", "kind": "selects", "n": 1,
                 "event_name": "Ella Interview", "filename": "Ella Interview – Selects 1.fcpxml"}
    _export(client, "p5")
    assert client.get("/project/p5/export/timeline-name?kind=selects").get_json()["timeline_name"] == "Ella Interview – Selects 2"
    assert client.get("/project/p5/export/timeline-name?kind=markers").get_json()["timeline_name"] == "Ella Interview – Markers 1"
    story = client.get("/project/p5/export/timeline-name?kind=story&story_title=The%20Grind").get_json()
    assert story["timeline_name"] == "Ella Interview – Story: The Grind" and story["filename"] == "Ella Interview – Story- The Grind.fcpxml"
    assert client.get("/project/nope/export/timeline-name").status_code == 404


# ── raw media: note first child, verbatim, cap, escaping, keyword ───────────

def test_raw_clip_note_is_first_child_with_verbatim_and_provenance_keyword(client, tmp_path, monkeypatch):
    _quiet_probes(monkeypatch)
    _make_project("p6", tmp_path)
    _, xml = _export(client, "p6", types=["labels", "story"])
    root = validate_dtd(xml)
    clips = root.findall(".//spine/asset-clip")
    assert len(clips) == 2
    for clip in clips:
        assert clip[0].tag == "note"                              # first child
        kws = [k for k in clip.findall("keyword") if k.get("value") == "Doza Assist"]
        assert len(kws) == 1                                      # exactly one provenance keyword
        assert kws[0].get("start") == clip.get("start") and kws[0].get("duration") == clip.get("duration")
    hook = next(c for c in clips if c.get("name") == "Best")      # the color label names the clip
    note = hook.find("note").text
    lines = note.split("\n")
    assert lines[0] == "Hero Moment — Sarah"                      # short note + speaker, unchanged
    assert lines[1] == ""                                          # blank line
    assert lines[2] == 'Sarah: We said "yes" & meant it.'          # verbatim, speaker-labelled, unescaped after parse
    assert lines[3] == "Mike: Did the fans <love> it?"
    raw = xml.decode("utf-8")
    assert "&quot;yes&quot; &amp; meant" in raw and "&lt;love&gt;" in raw   # escaped on the wire
    beat = next(c for c in clips if c.get("name") == "The Turn")
    assert beat.find("note").text.startswith("Where it flips — Sarah\n\nSarah: Every night.")


def test_raw_verbatim_is_capped_at_1000_characters(client, tmp_path, monkeypatch):
    _quiet_probes(monkeypatch)
    long_segments = [{"start": 0.0, "end": 12.0, "speaker": "SPEAKER_00", "text": ("Alpha beta gamma. " * 120).strip()}]
    _make_project("p7", tmp_path, transcript={"segments": long_segments, "duration": 12.0})
    _, xml = _export(client, "p7")
    note = validate_dtd(xml).find(".//spine/asset-clip/note").text
    head, _, verbatim = note.partition("\n\n")
    assert head == "Hero Moment — Sarah"
    assert len(verbatim) <= 1000 and verbatim.endswith("gamma.…")


def test_markers_mode_keeps_chapter_marker_notes_unchanged(client, tmp_path, monkeypatch):
    _quiet_probes(monkeypatch)
    _make_project("p8", tmp_path)
    _, xml = _export(client, "p8", mode="markers")
    root = validate_dtd(xml)
    cm = root.find(".//chapter-marker")
    assert cm is not None and cm.get("note") == "Hero Moment [Best] — Sarah"
    assert root.find(".//note") is None


# ── story builder (raw path) ────────────────────────────────────────────────

def test_story_export_names_and_counter(client, tmp_path, monkeypatch):
    _quiet_probes(monkeypatch)
    _make_project("s1", tmp_path)
    clips = [{"title": "Cold open", "start_time": "00:00:00", "end_time": "00:00:04", "editorial_note": "Starts loud", "order": 1},
             {"title": "The turn", "start_time": "00:00:08", "end_time": "00:00:12", "editorial_note": "", "order": 2}]
    body = {"clips": clips, "story_title": "The Grind", "deliver_to": "file"}
    r = client.post("/project/s1/story/export", json=body)
    assert r.status_code == 200, r.data
    d = r.get_json()
    xml = Path(d["file"]).read_bytes()
    root = validate_dtd(xml)
    assert root.find(".//project").get("name") == "Ella Interview – Story: The Grind"
    assert root.find(".//event").get("name") == "Ella Interview"
    assert d["filename"] == "Ella Interview – Story- The Grind.fcpxml"   # ":" is unsafe in a filename
    assert _meta("s1")["export_counts"] == {"story:the grind": 1}
    first = root.findall(".//spine/asset-clip")[0]
    assert first[0].tag == "note" and first.find("note").text.startswith("Starts loud\n\nSarah: We said")
    assert len([k for k in first.findall("keyword") if k.get("value") == "Doza Assist"]) == 1
    # second export of the same story carries the number
    r = client.post("/project/s1/story/export", json=body)
    d2 = r.get_json()
    assert _project_name_of(Path(d2["file"]).read_bytes()) == "Ella Interview – Story: The Grind 2"
    assert d2["filename"] == "Ella Interview – Story- The Grind 2.fcpxml"


# ── round-trip writer ───────────────────────────────────────────────────────

SYNC_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.14">
        <resources>
            <format id="r1" name="FFVideoFormat1080p2398" frameDuration="1001/24000s" width="1920" height="1080"/>
            <asset id="r2" name="dialogue" start="0s" duration="240000/24000s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                <media-rep kind="original-media" src="file:///tmp/audio.wav"/>
            </asset>
            <asset id="r3" name="cam_a" start="0s" duration="240000/24000s" hasVideo="1" videoSources="1">
                <media-rep kind="original-media" src="file:///tmp/cam_a.mov"/>
            </asset>
        </resources>
        <library>
            <event name="Shoot Day 3">
                <project name="Sync Test">
                    <sequence format="r1" duration="240000/24000s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                        <spine>
                            <sync-clip offset="0s" duration="240000/24000s" name="S">
                                <conform-rate scaleEnabled="0"/>
                                <asset-clip ref="r3" offset="0s" duration="240000/24000s"/>
                                <asset-clip ref="r2" offset="0s" duration="240000/24000s" audioRole="dialogue"/>
                                <filter-audio ref="r4" name="Hum"/>
                            </sync-clip>
                        </spine>
                    </sequence>
                </project>
            </event>
        </library>
    </fcpxml>
""")


@pytest.fixture
def parsed(tmp_path):
    p = tmp_path / "sync.fcpxml"
    # the filter-audio needs an effect resource for the DTD
    p.write_text(SYNC_FIXTURE.replace('<asset id="r3"', '<effect id="r4" name="Hum" uid="x.hum"/>\n        <asset id="r3"'))
    return parse_fcpxml(p)


def test_round_trip_keeps_event_names_timeline_and_carries_note_and_keyword(parsed):
    out = write_selects_as_new_project(parsed, [
        Select(start_seconds=2.0, end_seconds=5.0, label="pickup", note="Hero Moment", speaker="Sarah",
               verbatim='Sarah: We said "yes" & meant it.\nMike: Did the fans <love> it?'),
    ], project_name="Ella Interview – Selects 1", event_name="Shoot Day 3")
    root = validate_dtd(out)
    assert root.find(".//event").get("name") == "Shoot Day 3"            # editor's original event, unchanged
    assert root.find(".//project").get("name") == "Ella Interview – Selects 1"
    clip = root.find(".//spine/sync-clip")
    assert clip[0].tag == "note"                                          # note first, before conform-rate
    assert clip[1].tag == "conform-rate"
    assert clip.find("note").text == 'Hero Moment — Sarah\n\nSarah: We said "yes" & meant it.\nMike: Did the fans <love> it?'
    kws = [k for k in clip.findall("keyword") if k.get("value") == "Doza Assist"]
    assert len(kws) == 1
    children = [c.tag for c in clip]
    # keyword after the anchored asset-clips and before filter-audio
    assert children.index("keyword") > max(i for i, t in enumerate(children) if t == "asset-clip")
    assert children.index("keyword") < children.index("filter-audio")
    assert b"&amp; meant" in out and b"&lt;love&gt;" in out


def test_round_trip_defaults_never_say_doza_in_a_timeline_name(parsed):
    out = write_selects_as_new_project(parsed, [Select(start_seconds=2.0, end_seconds=5.0, label="pickup")])
    root = validate_dtd(out)
    assert root.find(".//project").get("name") == "Sync Test - Selects"
    assert root.find(".//event").get("name") == "Shoot Day 3"
    assert "Doza" not in root.find(".//project").get("name")


def test_round_trip_markers_mode_full_name_and_marker_note_unchanged(parsed):
    out = write_markers_on_timeline(parsed, [
        Select(start_seconds=2.0, end_seconds=3.0, label="pickup", note="Hero", speaker="Sarah", verbatim="long text"),
    ], project_name="Ella Interview – Markers 1")
    root = validate_dtd(out)
    assert root.find(".//project").get("name") == "Ella Interview – Markers 1"
    assert root.find(".//event").get("name") == "Shoot Day 3"
    marker = root.find(".//marker")
    assert marker.get("note") == "Hero — Sarah"                          # markers keep the short note only
    assert root.find(".//note") is None


# ── collector: story builder and collections inherit the verbatim ───────────

def test_collector_fills_verbatim_for_story_build_and_labels():
    project = {"name": "P", "transcript": {"segments": SEGMENTS}, "speaker_names": NAMES, "color_labels": {},
               "labeled_sections": [{"start": 0.5, "end": 7.0, "color": "green", "text": "Opening"}]}
    sel = app_module._project_selects_for_fcpxml(project, "client_selects")
    assert sel[0].verbatim.startswith('Sarah: We said "yes" & meant it.\nMike:')
    story = app_module._project_selects_for_fcpxml(
        project, ["story_build"],
        story_build_clips=[{"start_time": "00:00:08", "end_time": "00:00:12", "title": "Turn", "editorial_note": "n", "order": 1}])
    assert story[0].verbatim == "Sarah: Every night."


def test_round_trip_route_names_story_and_bumps_counters(client, tmp_path, monkeypatch):
    _quiet_probes(monkeypatch)
    fx = tmp_path / "stored.fcpxml"
    fx.write_text(SYNC_FIXTURE.replace('<asset id="r3"', '<effect id="r4" name="Hum" uid="x.hum"/>\n        <asset id="r3"'))
    _make_project("rt1", tmp_path, fcpxml_source={"container_type": "sync-clip", "stored_fcpxml_path": str(fx),
                                                   "timeline_audio_rendered": True})
    r = client.post("/project/rt1/export/fcpxml-multicam",
                    json={"mode": "selects_project", "source": "client_selects", "deliver_to": "file"})
    assert r.status_code == 200, r.data
    d = r.get_json()
    assert d["filename"] == "Ella Interview – Selects 1.fcpxml"
    root = validate_dtd(Path(d["file"]).read_bytes())
    assert root.find(".//project").get("name") == "Ella Interview – Selects 1"
    assert root.find(".//event").get("name") == "Shoot Day 3"
    assert _meta("rt1")["export_counts"] == {"selects": 1}
    r = client.post("/project/rt1/export/fcpxml-multicam",
                    json={"mode": "selects_project", "sources": ["story_build"], "preserve_order": True,
                          "story_title": "The Grind", "deliver_to": "file",
                          "story_build_clips": [{"start_time": "00:00:02", "end_time": "00:00:05", "title": "Turn", "order": 1}]})
    d = r.get_json()
    assert d["filename"] == "Ella Interview – Story- The Grind.fcpxml"
    assert _project_name_of(Path(d["file"]).read_bytes()) == "Ella Interview – Story: The Grind"
    r = client.post("/project/rt1/export/fcpxml-multicam",
                    json={"mode": "markers_timeline", "source": "client_selects", "deliver_to": "file",
                          "timeline_name": "Notes for Sam"})
    d = r.get_json()
    assert d["filename"] == "Notes for Sam.fcpxml"
    assert _meta("rt1")["export_counts"] == {"selects": 1, "story:the grind": 1}   # override did not count
