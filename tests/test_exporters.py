"""
Tests for the multi-NLE exporter package.

These tests do not require any source media on disk — they exercise the
exporter logic directly with synthetic marker payloads. Real-NLE import
testing (opening the output in FCP / Premiere / Resolve) is manual and
documented in the plan's verification section.
"""

import os
import sys
import tempfile
import xml.etree.ElementTree as ET

import pytest

# Ensure repo root is importable when running pytest from any cwd.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from exporters import get_exporter, PLATFORMS  # noqa: E402
from exporters.fcpxml import FCPXMLExporter  # noqa: E402
from exporters.premiere_xml import PremiereXMLExporter, _seconds_to_frames  # noqa: E402
from exporters.edl import EDLExporter, _seconds_to_timecode, _sanitize_reel_name  # noqa: E402
import preferences  # noqa: E402


SAMPLE_MARKERS = [
    {"start": 1.5,  "end": 12.3, "text": "Opening hook",   "note": "Strong opener", "color": "green",  "category": "Hook"},
    {"start": 23.4, "end": 38.1, "text": "Personal stakes","note": "Why it matters","color": "purple", "category": "Story Beat"},
    {"start": 60.0, "end": 78.5, "text": "Resolution",     "note": "",              "color": "orange", "category": "Resolution"},
]


# ── Router ────────────────────────────────────────────────────────────

def test_router_returns_correct_exporter():
    assert isinstance(get_exporter("fcp"), FCPXMLExporter)
    assert isinstance(get_exporter("premiere"), PremiereXMLExporter)
    assert isinstance(get_exporter("resolve"), EDLExporter)


def test_router_rejects_unknown_platform():
    with pytest.raises(ValueError):
        get_exporter("avid")


def test_platforms_constant_matches_registry():
    for p in PLATFORMS:
        get_exporter(p)  # must not raise


# ── FCPXML wrapper (regression firewall) ─────────────────────────────

def test_fcpxml_wrapper_byte_identical_to_direct_call():
    """
    The FCPXML wrapper must produce output byte-for-byte identical to a
    direct call to fcpxml_export.generate_fcpxml. Any divergence here is a
    regression for existing FCP users.
    """
    from fcpxml_export import generate_fcpxml

    direct = generate_fcpxml(
        markers=SAMPLE_MARKERS,
        project_name="Test Project",
        framerate=23.976,
        source_path="",
        media_duration=120.0,
        mode="cuts",
        width=1920,
        height=1080,
    )
    with tempfile.TemporaryDirectory() as tmp:
        result = FCPXMLExporter().export_markers(
            SAMPLE_MARKERS,
            project_name="Test Project",
            source_path="",
            media_duration=120.0,
            framerate=23.976,
            width=1920,
            height=1080,
            export_type="all",
            exports_dir=tmp,
            export_mode="cuts",
        )
        with open(result.file_path) as f:
            wrapped = f.read()
    assert direct == wrapped


def test_fcpxml_story_wrapper_byte_identical():
    from fcpxml_export import generate_story_fcpxml

    direct = generate_story_fcpxml(
        markers=SAMPLE_MARKERS,
        project_name="Test Project",
        story_title="My Story",
        framerate=23.976,
        source_path="",
        media_duration=120.0,
        width=1920,
        height=1080,
    )
    with tempfile.TemporaryDirectory() as tmp:
        result = FCPXMLExporter().export_story(
            SAMPLE_MARKERS,
            project_name="Test Project",
            story_title="My Story",
            source_path="",
            media_duration=120.0,
            framerate=23.976,
            width=1920,
            height=1080,
            exports_dir=tmp,
        )
        with open(result.file_path) as f:
            wrapped = f.read()
    assert direct == wrapped


# ── Premiere XML ──────────────────────────────────────────────────────

def _premiere_root(framerate=23.976):
    with tempfile.TemporaryDirectory() as tmp:
        result = PremiereXMLExporter().export_markers(
            SAMPLE_MARKERS,
            project_name="Test Project",
            source_path="/tmp/fake.mov",
            media_duration=120.0,
            framerate=framerate,
            width=1920,
            height=1080,
            export_type="all",
            exports_dir=tmp,
        )
        return ET.parse(result.file_path).getroot()


def test_premiere_xml_root_and_version():
    root = _premiere_root()
    assert root.tag == "xmeml"
    assert root.attrib.get("version") == "5"


def test_premiere_xml_sequence_has_rate_and_format():
    root = _premiere_root(framerate=23.976)
    seq = root.find("sequence")
    assert seq is not None
    rate = seq.find("rate")
    assert rate.find("timebase").text == "24"
    assert rate.find("ntsc").text == "TRUE"
    fmt = seq.find("media/video/format/samplecharacteristics")
    assert fmt.find("width").text == "1920"
    assert fmt.find("height").text == "1080"


def test_premiere_xml_pal_25fps_is_not_ntsc():
    root = _premiere_root(framerate=25.0)
    rate = root.find("sequence/rate")
    assert rate.find("timebase").text == "25"
    assert rate.find("ntsc").text == "FALSE"


def test_premiere_xml_clipitem_count_matches_markers():
    root = _premiere_root()
    video_clipitems = root.findall("sequence/media/video/track/clipitem")
    assert len(video_clipitems) == len(SAMPLE_MARKERS)


def test_premiere_xml_audio_tracks_present():
    root = _premiere_root()
    audio_tracks = root.findall("sequence/media/audio/track")
    assert len(audio_tracks) == 2  # A1 + A2
    assert all(len(t.findall("clipitem")) == len(SAMPLE_MARKERS) for t in audio_tracks)


def test_premiere_xml_first_clipitem_in_out_for_23976():
    """
    A clip starting at 1.5s on a 23.976fps timeline should land at frame 36
    (round(1.5 * 24000/1001) = round(35.964) = 36).
    """
    root = _premiere_root(framerate=23.976)
    first = root.find("sequence/media/video/track/clipitem")
    assert int(first.find("in").text) == 36
    assert int(first.find("out").text) == _seconds_to_frames(12.3, 23.976)


def test_premiere_xml_file_url_uses_localhost():
    root = _premiere_root()
    pathurl = root.find("sequence/media/video/track/clipitem/file/pathurl")
    assert pathurl is not None
    assert pathurl.text.startswith("file://localhost/")


def test_premiere_xml_warnings_present():
    with tempfile.TemporaryDirectory() as tmp:
        result = PremiereXMLExporter().export_markers(
            SAMPLE_MARKERS,
            project_name="Test", source_path="/tmp/x.mov", media_duration=60.0,
            framerate=23.976, width=1920, height=1080,
            export_type="all", exports_dir=tmp,
        )
    assert any("file path" in w.lower() for w in result.warnings)


# ── EDL ───────────────────────────────────────────────────────────────

def _edl_text():
    with tempfile.TemporaryDirectory() as tmp:
        result = EDLExporter().export_markers(
            SAMPLE_MARKERS,
            project_name="Test Project",
            source_path="/tmp/Daybright_Interview_Reel_01.mov",
            media_duration=120.0,
            framerate=23.976,
            width=1920, height=1080,
            export_type="all",
            exports_dir=tmp,
        )
        with open(result.file_path) as f:
            return f.read(), result


def test_edl_header():
    text, _ = _edl_text()
    lines = text.splitlines()
    assert lines[0] == "TITLE: Test Project - Full Export"
    assert lines[1] == "FCM: NON-DROP FRAME"


def test_edl_edit_count_matches_markers():
    text, _ = _edl_text()
    edits = [ln for ln in text.splitlines() if ln[:3].isdigit() and "C        " in ln]
    assert len(edits) == len(SAMPLE_MARKERS)


def test_edl_record_tc_starts_at_one_hour_and_is_monotonic():
    text, _ = _edl_text()
    edits = [ln for ln in text.splitlines() if ln[:3].isdigit() and "C        " in ln]
    # Format: NNN  REEL    AA/V  C        SRC_IN SRC_OUT REC_IN REC_OUT
    rec_ins = [ln.split()[-2] for ln in edits]
    assert rec_ins[0] == "01:00:00:00"
    assert rec_ins == sorted(rec_ins)


def test_edl_timecode_format_strict_8_chars():
    for s, fr, expected in [
        (0.0,    23.976, "00:00:00:00"),
        (1.5,    23.976, "00:00:01:12"),
        (3600.0, 23.976, "01:00:00:00"),
        (10.0,   25.0,   "00:00:10:00"),
        (10.0,   29.97,  "00:00:10:00"),
        (3661.5, 23.976, "01:01:01:12"),
    ]:
        assert _seconds_to_timecode(s, fr) == expected


def test_edl_reel_name_sanitized():
    assert _sanitize_reel_name("/tmp/Daybright Interview Reel 01.mov") == "DAYBRIGHT_INTERVIEW_REEL_01"
    assert _sanitize_reel_name("") == "AX"
    assert len(_sanitize_reel_name("a" * 200)) <= 32


def test_edl_warnings_surface_limitations():
    _, result = _edl_text()
    assert any("multicam" in w.lower() for w in result.warnings)


def test_edl_includes_clip_names_and_comments():
    text, _ = _edl_text()
    assert "* CLIP NAME: Opening hook" in text
    assert "* COMMENT: Strong opener" in text
    assert "* FROM CLIP NAME: Daybright_Interview_Reel_01.mov" in text


# ── Preferences ───────────────────────────────────────────────────────

def test_preferences_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr(preferences, "PREFS_DIR", str(tmp_path))
    monkeypatch.setattr(preferences, "PREFS_PATH", str(tmp_path / "preferences.json"))
    assert preferences.get_default_platform() == "fcp"
    assert preferences.set_default_platform("premiere")
    assert preferences.get_default_platform() == "premiere"
    assert preferences.set_default_platform("resolve")
    assert preferences.get_default_platform() == "resolve"
    assert not preferences.set_default_platform("avid")  # invalid
    assert preferences.get_default_platform() == "resolve"  # unchanged


def test_preferences_default_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(preferences, "PREFS_DIR", str(tmp_path / "doesnt-exist"))
    monkeypatch.setattr(preferences, "PREFS_PATH", str(tmp_path / "doesnt-exist" / "preferences.json"))
    assert preferences.get_default_platform() == "fcp"
