"""
Tests for the multi-NLE exporter package.

Most tests need no source media on disk — they exercise the exporter logic
directly with synthetic marker payloads (probes are monkeypatched). The
final section synthesizes tiny real fixtures with ffmpeg (skipped when a
full ffmpeg/ffprobe isn't available). Real-NLE import testing (opening the
output in FCP / Premiere / Resolve) is manual and documented in the plan's
verification section.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import pytest

# Ensure repo root is importable when running pytest from any cwd.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from exporters import get_exporter, PLATFORMS  # noqa: E402
from exporters.fcpxml import FCPXMLExporter  # noqa: E402
from exporters.premiere_xml import PremiereXMLExporter, _seconds_to_frames  # noqa: E402
from exporters import premiere_xml  # noqa: E402
from exporters import edl  # noqa: E402
from exporters import media_probe  # noqa: E402
from exporters.edl import EDLExporter, _seconds_to_timecode, _sanitize_reel_name, _frames_to_timecode  # noqa: E402
from exporters.resolve_xml import ResolveFCPXMLExporter  # noqa: E402
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
    # "resolve" now defaults to the FCPXML-based Resolve exporter (richer
    # round-trip than EDL). EDL is still available under "resolve-edl".
    assert isinstance(get_exporter("resolve"), ResolveFCPXMLExporter)
    assert isinstance(get_exporter("resolve-edl"), EDLExporter)


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


# ── FCPXML wrapper: export guards, warnings channel, atomic writes ────

_FCP_COMMON = dict(project_name="Test Project", media_duration=120.0,
                   framerate=23.976, width=1920, height=1080)


def _touch_source(tmp_path, name="a.mov"):
    p = tmp_path / name
    p.write_bytes(b"x")  # exporter only checks existence, never decodes
    return str(p)


def test_fcpxml_empty_markers_error(tmp_path):
    # Zero markers used to write (and auto-import) an empty-spine timeline
    # as success — now rejected like the multicam route's empty guard.
    with pytest.raises(ValueError, match="No clips to export"):
        FCPXMLExporter().export_markers(
            [], source_path=_touch_source(tmp_path), export_type="labels",
            exports_dir=str(tmp_path), **_FCP_COMMON)


def test_fcpxml_missing_source_cuts_mode_errors(tmp_path):
    # Unplugged drive / moved file used to silently degrade a pre-cut send
    # into a markers-only gap timeline with a success toast.
    with pytest.raises(ValueError, match="Source media not found") as exc:
        FCPXMLExporter().export_markers(
            SAMPLE_MARKERS, source_path="/Volumes/gone/away.mov",
            export_type="all", exports_dir=str(tmp_path), **_FCP_COMMON)
    assert "/Volumes/gone/away.mov" in str(exc.value)  # names the path


def test_fcpxml_missing_source_markers_mode_still_exports(tmp_path):
    # An EXPLICIT markers-only export never needed the media on disk.
    result = FCPXMLExporter().export_markers(
        SAMPLE_MARKERS, source_path="/Volumes/gone/away.mov",
        export_type="all", exports_dir=str(tmp_path),
        export_mode="markers", **_FCP_COMMON)
    assert os.path.exists(result.file_path)
    assert result.warnings == []


def test_fcpxml_story_missing_source_errors(tmp_path):
    with pytest.raises(ValueError, match="Source media not found"):
        FCPXMLExporter().export_story(
            SAMPLE_MARKERS, story_title="My Story",
            source_path="/Volumes/gone/away.mov",
            exports_dir=str(tmp_path), **_FCP_COMMON)


def test_fcpxml_empty_source_path_degrades_with_warning(tmp_path):
    # source_path='' (project never had media) keeps the legacy markers-only
    # fallback, but the degrade is now SAID, not silent.
    result = FCPXMLExporter().export_markers(
        SAMPLE_MARKERS, source_path="", export_type="all",
        exports_dir=str(tmp_path), export_mode="cuts", **_FCP_COMMON)
    assert any("markers only" in w for w in result.warnings)


def test_fcpxml_dropped_clip_surfaces_warning(tmp_path):
    # A select entirely past the media used to vanish from the timeline
    # with no trace — the export now names it in ExportResult.warnings.
    markers = [
        {"start": 1.0,  "end": 8.0,  "text": "keep me", "category": "x"},
        {"start": 150.0, "end": 160.0, "text": "late clip", "category": "x"},
    ]
    result = FCPXMLExporter().export_markers(
        markers, source_path=_touch_source(tmp_path), export_type="labels",
        exports_dir=str(tmp_path), **_FCP_COMMON)
    assert any("late clip" in w and "past the end" in w
               for w in result.warnings), result.warnings
    with open(result.file_path, encoding="utf-8") as f:
        content = f.read()
    assert "keep me" in content and "late clip" not in content


def test_fcpxml_subframe_clip_surfaces_warning(tmp_path):
    # Both ends snap to the same frame -> dropped, with the reason recorded.
    markers = [
        {"start": 1.0, "end": 8.0, "text": "keep me", "category": "x"},
        {"start": 10.0, "end": 10.004, "text": "blink", "category": "x"},
    ]
    result = FCPXMLExporter().export_markers(
        markers, source_path=_touch_source(tmp_path), export_type="labels",
        exports_dir=str(tmp_path), **_FCP_COMMON)
    assert any("blink" in w and "shorter than one frame" in w
               for w in result.warnings), result.warnings


def test_fcpxml_all_clips_dropped_errors(tmp_path):
    # Every select outside the media: never ship an empty-spine timeline.
    markers = [{"start": 150.0, "end": 160.0, "text": "late", "category": "x"}]
    with pytest.raises(ValueError, match="No exportable clips"):
        FCPXMLExporter().export_markers(
            markers, source_path=_touch_source(tmp_path),
            export_type="labels", exports_dir=str(tmp_path), **_FCP_COMMON)


def test_fcpxml_warnings_are_header_safe_ascii(tmp_path):
    # Warnings ride the ASCII-only X-Export-Warnings header on the legacy
    # attachment path; non-ASCII titles must fold, not crash the response.
    markers = [
        {"start": 1.0, "end": 8.0, "text": "Håkon på Røa", "category": "x"},
        {"start": 150.0, "end": 160.0, "text": "Håkon — sluttscene", "category": "x"},
    ]
    result = FCPXMLExporter().export_markers(
        markers, source_path=_touch_source(tmp_path), export_type="labels",
        exports_dir=str(tmp_path), **_FCP_COMMON)
    assert result.warnings
    for w in result.warnings:
        w.encode("ascii")  # raises if any warning would break the header


def test_fcpxml_atomic_replace_preserves_open_reader(tmp_path):
    # C30: deterministic filenames + threaded Flask meant a second send
    # re-opened the SAME path with a truncating open() while FCP/Resolve
    # could still be parsing the previous file. The write now lands in a
    # temp file and os.replace()s in, so an already-open reader keeps the
    # complete OLD document and the path always shows a complete new one.
    exporter = FCPXMLExporter()
    common = dict(source_path=_touch_source(tmp_path), export_type="all",
                  exports_dir=str(tmp_path), **_FCP_COMMON)
    first = exporter.export_markers(SAMPLE_MARKERS, **common)
    with open(first.file_path, encoding="utf-8") as reader:
        second = exporter.export_markers(SAMPLE_MARKERS[:1], **common)
        assert second.file_path == first.file_path  # deterministic name
        old_doc = reader.read()  # reads the ORIGINAL inode, post-replace
    assert old_doc.rstrip().endswith("</fcpxml>")  # complete, not truncated
    with open(second.file_path, encoding="utf-8") as f:
        assert f.read().rstrip().endswith("</fcpxml>")
    # No temp-file droppings left behind in the exports dir.
    assert not [n for n in os.listdir(tmp_path) if n.startswith(".doza-export-")]


def test_fcpxml_atomic_write_keeps_standard_permissions(tmp_path):
    """R9: mkstemp creates its temp 0600 and os.replace carries that mode
    onto the final export — every file became owner-read-only, and a
    re-export DOWNGRADED an existing 0644 file, so other-UID consumers of
    the exports dir (second macOS account, SMB share, backup daemon) got
    EACCES. _write_atomic must restore what plain open() gave in 1.0.26:
    0666 & ~umask."""
    prior_umask = os.umask(0o022)
    try:
        exporter = FCPXMLExporter()
        common = dict(source_path=_touch_source(tmp_path), export_type="all",
                      exports_dir=str(tmp_path), **_FCP_COMMON)
        first = exporter.export_markers(SAMPLE_MARKERS, **common)
        assert os.stat(first.file_path).st_mode & 0o777 == 0o644

        # Re-export over an existing 0644 file must not downgrade it.
        os.chmod(first.file_path, 0o644)
        second = exporter.export_markers(SAMPLE_MARKERS, **common)
        assert second.file_path == first.file_path
        assert os.stat(second.file_path).st_mode & 0o777 == 0o644

        # The process umask is honored (0666 & ~umask), not a blind 0644.
        os.umask(0o027)
        third = exporter.export_markers(SAMPLE_MARKERS, **common)
        assert os.stat(third.file_path).st_mode & 0o777 == 0o640
    finally:
        os.umask(prior_umask)


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


# ── Premiere XML: source audio routing ────────────────────────────────

def _premiere_export(tmp_path, *, story=False, export_mode="cuts", framerate=23.976):
    exporter = PremiereXMLExporter()
    if story:
        result = exporter.export_story(
            SAMPLE_MARKERS,
            project_name="Test Project", story_title="My Story",
            source_path="/tmp/fake.mov", media_duration=120.0,
            framerate=framerate, width=1920, height=1080,
            exports_dir=str(tmp_path),
        )
    else:
        result = exporter.export_markers(
            SAMPLE_MARKERS,
            project_name="Test Project", source_path="/tmp/fake.mov",
            media_duration=120.0, framerate=framerate, width=1920, height=1080,
            export_type="all", exports_dir=str(tmp_path),
            export_mode=export_mode,
        )
    return ET.parse(result.file_path).getroot()


def _audio_trackindexes(root):
    """Per sequence audio track: the sourcetrack trackindex its clipitems use."""
    indexes = []
    for track in root.findall("sequence/media/audio/track"):
        cis = track.findall("clipitem")
        per_clip = {ci.find("sourcetrack/trackindex").text for ci in cis}
        assert len(per_clip) == 1, "clipitems on one track must share a source track"
        indexes.append(per_clip.pop())
    return indexes


def test_premiere_xml_multimono_routes_dialogue_tracks(monkeypatch, tmp_path):
    """4-mono broadcast MXF, speech detected on source tracks 2+3: one
    sequence track per dialogue channel, trackindex = the REAL source track
    (the old 2-channel first-stream cap never referenced track 2+, so the
    lav mic imported as camera scratch or silence)."""
    monkeypatch.setattr(premiere_xml, "get_audio_layout", lambda p: (4, 4))
    monkeypatch.setattr(premiere_xml, "detect_dialogue_channels", lambda p: [1, 2])
    root = _premiere_export(tmp_path)
    assert _audio_trackindexes(root) == ["2", "3"]
    # File declares the file's REAL total channel count, not the routed subset.
    assert root.find(".//file/media/audio/channelcount").text == "4"


def test_premiere_xml_multimono_unmeasurable_wires_all_tracks(monkeypatch, tmp_path):
    """When loudness can't disambiguate, every source track is wired —
    extra scratch tracks are mutable, a missing dialogue track is not."""
    monkeypatch.setattr(premiere_xml, "get_audio_layout", lambda p: (4, 4))
    monkeypatch.setattr(premiere_xml, "detect_dialogue_channels", lambda p: None)
    root = _premiere_export(tmp_path)
    assert _audio_trackindexes(root) == ["1", "2", "3", "4"]
    assert root.find(".//file/media/audio/channelcount").text == "4"


def test_premiere_xml_mono_gets_single_track(monkeypatch, tmp_path):
    monkeypatch.setattr(premiere_xml, "get_audio_layout", lambda p: (1, 1))
    root = _premiere_export(tmp_path)
    assert _audio_trackindexes(root) == ["1"]
    assert root.find(".//file/media/audio/channelcount").text == "1"


def test_premiere_xml_layout_probe_failure_falls_back_to_stereo(monkeypatch, tmp_path):
    monkeypatch.setattr(premiere_xml, "get_audio_layout", lambda p: None)
    monkeypatch.setattr(premiere_xml, "get_audio_channels", lambda p: None)
    root = _premiere_export(tmp_path)
    assert _audio_trackindexes(root) == ["1", "2"]


def test_premiere_xml_silent_source_wires_no_audio(monkeypatch, tmp_path):
    """get_audio_layout's (0, 0) contract = probed clean, NO audio streams
    (silent B-roll): don't emit audio clipitems against media that can't
    supply them, and drop the file's <audio> declaration to match."""
    monkeypatch.setattr(premiere_xml, "get_audio_layout", lambda p: (0, 0))
    root = _premiere_export(tmp_path)
    assert root.findall("sequence/media/audio/track/clipitem") == []
    assert root.find(".//file/media/audio") is None


def test_premiere_story_probes_audio(monkeypatch, tmp_path):
    """export_story used to skip the audio probe entirely and always built
    A1+A2 — a mono interview WAV got a phantom A2 clipitem referencing a
    channel the file doesn't have."""
    monkeypatch.setattr(premiere_xml, "get_audio_layout", lambda p: (1, 1))
    root = _premiere_export(tmp_path, story=True)
    assert _audio_trackindexes(root) == ["1"]
    assert root.find(".//file/media/audio/channelcount").text == "1"


def test_premiere_story_routes_multimono_dialogue(monkeypatch, tmp_path):
    monkeypatch.setattr(premiere_xml, "get_audio_layout", lambda p: (4, 4))
    monkeypatch.setattr(premiere_xml, "detect_dialogue_channels", lambda p: [1])
    root = _premiere_export(tmp_path, story=True)
    assert _audio_trackindexes(root) == ["2"]


def test_premiere_xml_single_stream_multichannel_keeps_two_track_cap(monkeypatch, tmp_path):
    """R12: a single-STREAM multichannel source (camcorder 5.1 mix = 1
    stream / 6 channels) keeps the historical front-pair wiring. Wiring all
    6 stacked LFE/surround tracks into the stereo master was a regression
    vs the 1.0.26 2-track cap; dialogue-channel detection applies to
    multi-STREAM (multi-mono) sources only."""
    monkeypatch.setattr(premiere_xml, "get_audio_layout", lambda p: (1, 6))
    detect_calls = []
    monkeypatch.setattr(premiere_xml, "detect_dialogue_channels",
                        lambda p: detect_calls.append(p) or [0])
    root = _premiere_export(tmp_path)
    assert _audio_trackindexes(root) == ["1", "2"]
    # <file> still declares the file's REAL total channel count.
    assert root.find(".//file/media/audio/channelcount").text == "6"
    # No per-stream loudness pass (a full-file read) on single-stream files.
    assert detect_calls == []


def test_premiere_story_single_stream_multichannel_keeps_two_track_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(premiere_xml, "get_audio_layout", lambda p: (1, 6))
    root = _premiere_export(tmp_path, story=True)
    assert _audio_trackindexes(root) == ["1", "2"]
    assert root.find(".//file/media/audio/channelcount").text == "6"


# ── Premiere XML: probe-based hasVideo (R10) ──────────────────────────

def test_premiere_xml_audio_only_video_container_omits_video(monkeypatch, tmp_path):
    """R10: hasVideo comes from the stream probe, not the file extension —
    an audio-only .mov/.mp4 (AAC podcast export, field-recorder QuickTime)
    must not emit <video> clipitems against a file with no picture
    (black/offline V1 in Premiere for every select)."""
    monkeypatch.setattr(premiere_xml, "has_video_stream", lambda p: False)
    monkeypatch.setattr(premiere_xml, "get_audio_layout", lambda p: (1, 1))
    root = _premiere_export(tmp_path)  # source is /tmp/fake.mov
    assert root.findall("sequence/media/video/track/clipitem") == []
    assert root.find(".//file/media/video") is None
    assert _audio_trackindexes(root) == ["1"]


def test_premiere_story_audio_only_video_container_omits_video(monkeypatch, tmp_path):
    monkeypatch.setattr(premiere_xml, "has_video_stream", lambda p: False)
    monkeypatch.setattr(premiere_xml, "get_audio_layout", lambda p: (1, 1))
    root = _premiere_export(tmp_path, story=True)
    assert root.findall("sequence/media/video/track/clipitem") == []
    assert root.find(".//file/media/video") is None


def test_premiere_xml_video_probe_none_falls_back_to_extension(monkeypatch, tmp_path):
    """Probe can't answer (missing file / no ffprobe): the extension
    heuristic survives as the fallback — .mov still gets video clipitems."""
    monkeypatch.setattr(premiere_xml, "has_video_stream", lambda p: None)
    monkeypatch.setattr(premiere_xml, "get_audio_layout", lambda p: (1, 2))
    root = _premiere_export(tmp_path)
    assert len(root.findall("sequence/media/video/track/clipitem")) \
        == len(SAMPLE_MARKERS)
    assert root.find(".//file/media/video") is not None


def test_premiere_xml_video_stream_wins_over_audio_extension(monkeypatch, tmp_path):
    """A real video stream in an audio-named container still gets video."""
    monkeypatch.setattr(premiere_xml, "has_video_stream", lambda p: True)
    monkeypatch.setattr(premiere_xml, "get_audio_layout", lambda p: (1, 2))
    exporter = PremiereXMLExporter()
    result = exporter.export_markers(
        SAMPLE_MARKERS,
        project_name="Test Project", source_path="/tmp/fake.wav",
        media_duration=120.0, framerate=23.976, width=1920, height=1080,
        export_type="all", exports_dir=str(tmp_path))
    root = ET.parse(result.file_path).getroot()
    assert len(root.findall("sequence/media/video/track/clipitem")) \
        == len(SAMPLE_MARKERS)


# ── Premiere XML: Cuts + Markers ──────────────────────────────────────

def test_premiere_xml_both_mode_emits_cuts_and_sequence_markers(tmp_path):
    """'Cuts + Markers' used to silently degrade to cuts-only for Premiere —
    'both' now assembles the cut timeline AND a sequence-level marker
    spanning each cut's record range."""
    root = _premiere_export(tmp_path, export_mode="both")
    video_clipitems = root.findall("sequence/media/video/track/clipitem")
    assert len(video_clipitems) == len(SAMPLE_MARKERS)
    seq_markers = root.findall("sequence/marker")
    assert len(seq_markers) == len(SAMPLE_MARKERS)
    # Markers ride the RECORD positions: first cut starts at frame 0 and the
    # second marker begins where the first cut's duration ends.
    first_dur = _seconds_to_frames(12.3, 23.976) - _seconds_to_frames(1.5, 23.976)
    assert seq_markers[0].find("in").text == "0"
    assert seq_markers[0].find("out").text == str(first_dur)
    assert seq_markers[1].find("in").text == str(first_dur)
    assert seq_markers[0].find("name").text == "Opening hook"
    assert seq_markers[0].find("comment").text == "Strong opener"


def test_premiere_xml_cuts_mode_has_no_sequence_markers(tmp_path):
    root = _premiere_export(tmp_path, export_mode="cuts")
    assert root.findall("sequence/marker") == []


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
    # NTSC vectors: the frame index is computed at the ACTUAL rate
    # (24000/1001 for 23.976), then rendered NDF at the integer base — so
    # wall-clock 3600s lands at TC 00:59:56:10, the standard NDF lag.
    # (The old implementation counted frames at the integer rate, which placed
    # every late-interview select ~3.6s/hour too late in Resolve.)
    for s, fr, expected in [
        (0.0,    23.976, "00:00:00:00"),
        (1.5,    23.976, "00:00:01:12"),
        (3600.0, 23.976, "00:59:56:10"),
        (10.0,   25.0,   "00:00:10:00"),
        (10.0,   29.97,  "00:00:10:00"),
        (3661.5, 23.976, "01:00:57:20"),
        (3600.0, 29.97,  "00:59:56:12"),
        (3600.0, 24.0,   "01:00:00:00"),  # integer rates have no lag
        (3600.0, 25.0,   "01:00:00:00"),
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


# ── EDL: export modes (locators) ──────────────────────────────────────

def _edl_export(tmp_path, *, export_mode="cuts", framerate=23.976,
                markers=SAMPLE_MARKERS, tc_format="NDF", start_tc_frames=0,
                source_path="/tmp/interview.mov"):
    result = EDLExporter().export_markers(
        markers,
        project_name="Test Project", source_path=source_path,
        media_duration=120.0, framerate=framerate, width=1920, height=1080,
        export_type="all", exports_dir=str(tmp_path),
        export_mode=export_mode, tc_format=tc_format,
        start_tc_frames=start_tc_frames,
    )
    with open(result.file_path) as f:
        return f.read(), result


def _edl_event_lines(text):
    return [ln for ln in text.splitlines() if ln[:3].isdigit()]


def test_edl_markers_mode_emits_locators(tmp_path):
    """'Markers Only' used to be ignored and emitted a re-cut timeline —
    it now writes ONE full-length event with a * LOC: line per marker
    (Resolve imports locators as timeline markers)."""
    text, _ = _edl_export(tmp_path, export_mode="markers")
    assert len(_edl_event_lines(text)) == 1
    loc_lines = [ln for ln in text.splitlines() if ln.startswith("* LOC: ")]
    assert len(loc_lines) == len(SAMPLE_MARKERS)
    # Marker at 1.5s (frame 36 @ 23.976) rides record TC = source + 1h.
    assert loc_lines[0] == "* LOC: 01:00:01:12 GREEN Opening hook"
    assert loc_lines[1].split()[3] == "PURPLE"
    assert loc_lines[2].split()[3] == "BLUE"  # 'orange' isn't a locator color


def test_edl_both_mode_adds_locators_to_cut_events(tmp_path):
    text, _ = _edl_export(tmp_path, export_mode="both")
    assert len(_edl_event_lines(text)) == len(SAMPLE_MARKERS)
    loc_lines = [ln for ln in text.splitlines() if ln.startswith("* LOC: ")]
    assert len(loc_lines) == len(SAMPLE_MARKERS)
    # Locators sit at each cut's record-in on the sequential record timeline.
    assert loc_lines[0] == "* LOC: 01:00:00:00 GREEN Opening hook"


def test_edl_cuts_mode_has_no_locators(tmp_path):
    text, _ = _edl_export(tmp_path, export_mode="cuts")
    assert "* LOC:" not in text
    assert len(_edl_event_lines(text)) == len(SAMPLE_MARKERS)


# ── EDL: drop-frame timecode ──────────────────────────────────────────

def test_edl_drop_frame_declared_and_rendered(tmp_path):
    """29.97 DF media (embedded TOD TC 10:00:00;00 = 1078920 nominal-grid
    frames): the EDL must declare FCM: DROP FRAME and render ';' labels that
    match the camera TC — the old NDF-only render drifted the labels 108
    frames per TC-hour (source read 09:59:24:00 at hour ten)."""
    text, _ = _edl_export(tmp_path, framerate=29.97, tc_format="DF",
                          start_tc_frames=1078920)
    lines = text.splitlines()
    assert lines[1] == "FCM: DROP FRAME"
    first = _edl_event_lines(text)[0].split()
    # 1.5s @ 29.97 = frame 45; DF label re-inserts the dropped numbers.
    assert first[-4] == "10:00:01;15"   # source in — matches camera TC
    assert first[-2] == "01:00:00;00"   # record still starts at the hour


def test_edl_ndf_and_non_ntsc_rates_stay_ndf(tmp_path):
    text, _ = _edl_export(tmp_path, framerate=29.97, tc_format="NDF")
    assert text.splitlines()[1] == "FCM: NON-DROP FRAME"
    assert ";" not in text
    # A DF tag on a rate with no drop-frame variant is a corrupt probe.
    text, _ = _edl_export(tmp_path, framerate=25.0, tc_format="DF")
    assert text.splitlines()[1] == "FCM: NON-DROP FRAME"
    assert ";" not in text


# ── EDL: CMX 3600 limits ──────────────────────────────────────────────

def test_edl_event_numbers_uniform_width_past_999(tmp_path):
    """>999 events: the number field widens uniformly for the whole file
    (0001…1005, the Premiere zero-pad convention) instead of shifting the
    columns of only the 4-digit tail, and the CMX cap is surfaced as a
    warning rather than a silently malformed file."""
    markers = [{"start": float(i), "end": float(i + 1), "text": f"S{i}"}
               for i in range(1005)]
    text, result = _edl_export(tmp_path, framerate=25.0, markers=markers)
    events = [ln for ln in text.splitlines() if ln[:4].isdigit()]
    assert len(events) == 1005
    assert events[0].startswith("0001  ")
    assert events[999].startswith("1000  ")
    # Uniform columns: the reel field starts at the same offset on every line.
    assert len({ln.index("INTERVIEW") for ln in events}) == 1
    assert any("999" in w for w in result.warnings)


def test_edl_under_1000_events_keeps_three_digit_numbers(tmp_path):
    text, result = _edl_export(tmp_path)
    assert _edl_event_lines(text)[0].startswith("001  ")
    assert not any("999" in w for w in result.warnings)


def test_edl_warnings_are_header_safe_ascii(tmp_path):
    """R8: warnings ride the latin-1-strict X-Export-Warnings header on the
    legacy attachment path — the >999-events warning carried an em-dash,
    which made werkzeug raise mid-header and close the response with no
    body. Every EDL warning must encode to plain ASCII."""
    markers = [{"start": float(i), "end": float(i + 1), "text": f"S{i}"}
               for i in range(1005)]
    _, result = _edl_export(tmp_path, framerate=25.0, markers=markers)
    assert any("CMX 3600" in w for w in result.warnings)  # the >999 warning
    for w in result.warnings:
        w.encode("ascii")  # raises if any warning would break the header


# ── EDL: probe-based AA/V vs AA channel (R11) ─────────────────────────

def test_edl_audio_only_video_container_declares_audio_only(monkeypatch, tmp_path):
    """R11: the 'AA/V ' vs 'AA   ' channel comes from the stream probe, not
    the extension — an audio-only .mov must not claim a video component
    (Resolve shows offline video on every AA/V event whose media has no
    picture)."""
    monkeypatch.setattr(edl, "has_video_stream", lambda p: False)
    text, _ = _edl_export(tmp_path)  # source is /tmp/interview.mov
    events = _edl_event_lines(text)
    assert events and "AA/V" not in text
    assert all(" AA    C " in ln for ln in events)


def test_edl_markers_mode_channel_also_probed(monkeypatch, tmp_path):
    monkeypatch.setattr(edl, "has_video_stream", lambda p: False)
    text, _ = _edl_export(tmp_path, export_mode="markers")
    assert "AA/V" not in text


def test_edl_video_probe_none_falls_back_to_extension(monkeypatch, tmp_path):
    """Probe can't answer (missing file / no ffprobe): keep the extension
    heuristic — .mov claims AA/V, .wav stays audio-only."""
    monkeypatch.setattr(edl, "has_video_stream", lambda p: None)
    text, _ = _edl_export(tmp_path)  # .mov
    assert "AA/V" in text
    text, _ = _edl_export(tmp_path, source_path="/tmp/interview.wav")
    assert "AA/V" not in text


def test_edl_video_stream_wins_over_audio_extension(monkeypatch, tmp_path):
    monkeypatch.setattr(edl, "has_video_stream", lambda p: True)
    text, _ = _edl_export(tmp_path, source_path="/tmp/interview.wav")
    assert "AA/V" in text


def test_edl_record_hours_wrap_modulo_24(tmp_path):
    # TC is a 24h clock: 25h of frames renders 01:00:00:00, not 25:00:00:00.
    assert _frames_to_timecode(25 * 3600 * 24, 24) == "01:00:00:00"
    # End-to-end: 23:30:00:00 TOD start + a select 40 minutes in crosses
    # midnight and must render 00:10:00:00.
    markers = [{"start": 2400.0, "end": 2401.0, "text": "Late select"}]
    text, _ = _edl_export(tmp_path, framerate=25.0, markers=markers,
                          start_tc_frames=int(23.5 * 3600 * 25))
    assert "00:10:00:00" in _edl_event_lines(text)[0]


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


# ── Real-media fixtures: probe-based classification end-to-end ────────
# Everything above monkeypatches the probes; these synthesize tiny REAL
# files and run the shipped ffprobe path, pinning R10/R11/R12 against
# actual stream layouts. Mirrors test_nle_export_routing.py's fixture
# pattern (full ffmpeg for synthesis, any ffprobe for probing).

_FULL_FFMPEG = shutil.which("ffmpeg") or next(
    (p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg")
     if os.path.isfile(p)), None)
_HAVE_FFPROBE = bool(media_probe._find_ffprobe())


def _make_audio_only_mp4(path):
    """A real AAC-in-.mp4 with NO video stream (podcast-export shape)."""
    proc = subprocess.run(
        [_FULL_FFMPEG, "-y", "-v", "error",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=0.8:sample_rate=48000",
         "-c:a", "aac", "-b:a", "96k", str(path)],
        capture_output=True, text=True, timeout=30)
    return proc.returncode == 0 and os.path.exists(path)


def _make_single_stream_5_1_mov(path):
    """ONE 6-channel (5.1) PCM audio stream in a .mov (camcorder-mix shape)."""
    proc = subprocess.run(
        [_FULL_FFMPEG, "-y", "-v", "error",
         "-f", "lavfi", "-i", "anullsrc=channel_layout=5.1:sample_rate=48000",
         "-t", "0.5", "-c:a", "pcm_s16le", str(path)],
        capture_output=True, text=True, timeout=30)
    return proc.returncode == 0 and os.path.exists(path)


@pytest.mark.skipif(_FULL_FFMPEG is None,
                    reason="needs a full ffmpeg to synthesize fixtures")
@pytest.mark.skipif(not _HAVE_FFPROBE, reason="no ffprobe available to probe")
class TestProbeBasedClassificationRealMedia:
    def test_premiere_audio_only_mp4_emits_no_video(self, tmp_path):
        """R10 end-to-end: an audio-only .mp4 (extension says video, probe
        says no picture) exports with no <video> clipitems and no <video>
        block in <file>."""
        src = tmp_path / "podcast.mp4"
        if not _make_audio_only_mp4(src):
            pytest.skip("ffmpeg cannot encode aac/mp4")
        assert media_probe.has_video_stream(str(src)) is False  # fixture sanity
        result = PremiereXMLExporter().export_markers(
            SAMPLE_MARKERS, project_name="Test Project", source_path=str(src),
            media_duration=120.0, framerate=23.976, width=1920, height=1080,
            export_type="all", exports_dir=str(tmp_path))
        root = ET.parse(result.file_path).getroot()
        assert root.findall("sequence/media/video/track/clipitem") == []
        assert root.find(".//file/media/video") is None
        assert root.findall("sequence/media/audio/track/clipitem") != []

    def test_premiere_story_audio_only_mp4_emits_no_video(self, tmp_path):
        src = tmp_path / "podcast.mp4"
        if not _make_audio_only_mp4(src):
            pytest.skip("ffmpeg cannot encode aac/mp4")
        result = PremiereXMLExporter().export_story(
            SAMPLE_MARKERS, project_name="Test Project", story_title="Story",
            source_path=str(src), media_duration=120.0, framerate=23.976,
            width=1920, height=1080, exports_dir=str(tmp_path))
        root = ET.parse(result.file_path).getroot()
        assert root.findall("sequence/media/video/track/clipitem") == []
        assert root.find(".//file/media/video") is None

    def test_edl_audio_only_mp4_declares_audio_only_channel(self, tmp_path):
        """R11 end-to-end: the same audio-only .mp4 gets 'AA' events, never
        'AA/V' (offline video in Resolve)."""
        src = tmp_path / "podcast.mp4"
        if not _make_audio_only_mp4(src):
            pytest.skip("ffmpeg cannot encode aac/mp4")
        text, _ = _edl_export(tmp_path, source_path=str(src))
        events = _edl_event_lines(text)
        assert events and "AA/V" not in text

    def test_premiere_5_1_single_stream_wires_front_pair_only(self, tmp_path):
        """R12 end-to-end: a real single-stream 6-channel (5.1) .mov keeps
        the 2-track cap — front pair wired, file declares all 6 channels."""
        src = tmp_path / "cammix.mov"
        if not _make_single_stream_5_1_mov(src):
            pytest.skip("ffmpeg cannot mux 5.1 pcm in mov")
        assert media_probe.get_audio_layout(str(src)) == (1, 6)  # fixture sanity
        result = PremiereXMLExporter().export_markers(
            SAMPLE_MARKERS, project_name="Test Project", source_path=str(src),
            media_duration=120.0, framerate=23.976, width=1920, height=1080,
            export_type="all", exports_dir=str(tmp_path))
        root = ET.parse(result.file_path).getroot()
        tracks = root.findall("sequence/media/audio/track")
        assert len(tracks) == 2
        indexes = [t.find("clipitem/sourcetrack/trackindex").text for t in tracks]
        assert indexes == ["1", "2"]
        assert root.find(".//file/media/audio/channelcount").text == "6"
