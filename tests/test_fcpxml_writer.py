"""Writer tests: Mode A (selects as new project) and Mode B (markers)."""

import os
import sys
import textwrap
from fractions import Fraction
from pathlib import Path

import pytest
from lxml import etree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from doza_assist.fcpxml import (  # noqa: E402
    Select,
    WriterError,
    parse_fcpxml,
    write_markers_on_timeline,
    write_selects_as_new_project,
)
from doza_assist.fcpxml.parser import strip_file_url  # noqa: E402
from doza_assist.fcpxml.timecode import parse_rational  # noqa: E402
from doza_assist.fcpxml.writer import re_parse  # noqa: E402


ELLA_FCPXML = Path("/Users/dozavisuals/Downloads/Ella Interview.fcpxmld/Info.fcpxml")


SAMPLE_SELECTS = [
    Select(start_seconds=10.0,  end_seconds=25.5,   label="Opening hook",   kind="strong"),
    Select(start_seconds=100.0, end_seconds=130.0,  label="Childhood story", kind="standard"),
    Select(start_seconds=600.0, end_seconds=620.0,  label="Follow-up?",      kind="question", note="ask legal"),
]


# ---------- Mode A ----------------------------------------------------------

@pytest.mark.skipif(not ELLA_FCPXML.exists(), reason="Ella fixture not present")
class TestModeASelectsAsNewProject:
    @pytest.fixture(scope="class")
    def parsed(self):
        return parse_fcpxml(ELLA_FCPXML)

    @pytest.fixture(scope="class")
    def output(self, parsed):
        return write_selects_as_new_project(parsed, SAMPLE_SELECTS)

    def test_output_is_valid_xml(self, output):
        etree.fromstring(output)  # raises on invalid XML

    def test_preserves_fcpxml_version(self, output):
        root = etree.fromstring(output)
        assert root.get("version") == "1.14"

    def test_resources_preserved_byte_for_byte(self, output, parsed):
        # This is the FCP-bookmark-safety invariant: if a single byte of the
        # resources block changes, FCP can reject the import on re-open. We
        # assert the exact verbatim slice appears in the output.
        assert parsed.original_resources_xml in output

    def test_project_name_suffix_applied(self, output):
        root = etree.fromstring(output)
        project = root.find(".//project")
        assert project.get("name") == "Ella Interview - Doza Selects"

    def test_spine_has_one_mc_clip_per_select(self, output):
        root = etree.fromstring(output)
        clips = root.findall(".//spine/mc-clip")
        assert len(clips) == len(SAMPLE_SELECTS)

    def test_mc_clips_reference_original_container(self, output, parsed):
        root = etree.fromstring(output)
        for clip in root.findall(".//spine/mc-clip"):
            assert clip.get("ref") == parsed.container_ref  # "r2"

    def test_mc_clip_source_times_match_selects(self, output, parsed):
        # The first select is at source time 10s; its mc-clip start should
        # decode to 10s (within one frame's worth — we snap to the frame grid).
        root = etree.fromstring(output)
        clip = root.findall(".//spine/mc-clip")[0]
        start = parse_rational(clip.get("start"))
        # Frame-snapped: one frame = 1001/24000 ≈ 0.0417s
        assert abs(float(start) - 10.0) < 0.05

    def test_selects_are_contiguous_on_new_timeline(self, output):
        # Mode A lays selects end-to-end on the new timeline. The nth mc-clip's
        # offset should equal the sum of the previous (n-1) mc-clips' durations.
        root = etree.fromstring(output)
        cursor = Fraction(0)
        for clip in root.findall(".//spine/mc-clip"):
            off = parse_rational(clip.get("offset"))
            dur = parse_rational(clip.get("duration"))
            assert off == cursor
            cursor += dur

    def test_mc_source_enablement_replayed(self, output):
        # Each output mc-clip should carry the same mc-source children as the
        # source timeline had (both video and audio angles enabled).
        root = etree.fromstring(output)
        for clip in root.findall(".//spine/mc-clip"):
            sources = clip.findall("mc-source")
            enables = {s.get("srcEnable") for s in sources}
            assert "audio" in enables
            assert "video" in enables

    def test_note_precedes_mc_source_per_dtd(self, output):
        # FCPXML 1.13/1.14 DTD: <mc-clip> content model is
        # (note?, timing-params, intrinsic-params-audio, mc-source*, ...)
        # so <note> MUST come before <mc-source> children. FCP silently
        # drops the mc-source overrides if the order is wrong, which
        # reproduces as "audio imports but video is missing".
        root = etree.fromstring(output)
        for clip in root.findall(".//spine/mc-clip"):
            children = list(clip)
            note_idx = next((i for i, c in enumerate(children) if c.tag == "note"), None)
            first_mcsource_idx = next(
                (i for i, c in enumerate(children) if c.tag == "mc-source"), None
            )
            if note_idx is not None and first_mcsource_idx is not None:
                assert note_idx < first_mcsource_idx, (
                    "note must precede mc-source per FCPXML DTD"
                )

    def test_round_trips_through_parser(self, output):
        # The proof that the writer emits well-formed, self-consistent FCPXML:
        # feed it back through the parser and confirm the structure.
        re_parsed = re_parse(output)
        assert re_parsed.version == "1.14"
        assert re_parsed.container_type == "mc-clip"
        assert re_parsed.container_ref == "r2"
        assert re_parsed.active_audio_angle_id == "qMugMvsqRpW4mCI2v5CgDA"
        assert len(re_parsed.spine_segments) == len(SAMPLE_SELECTS)
        # The re-parsed audio path still resolves to the same physical file.
        assert re_parsed.audio_file_path == (
            "/Volumes/DOZA EDIT SSD/Trustees/Posey/Studio Visit/"
            "121525_133506/021926_075706_Tr2-esv2-83p-bg-10p.wav"
        )

    def test_empty_selects_rejected(self, parsed):
        with pytest.raises(WriterError, match="no selects"):
            write_selects_as_new_project(parsed, [])

    def test_custom_project_name_used(self, parsed):
        out = write_selects_as_new_project(parsed, SAMPLE_SELECTS, project_name="Custom Cut")
        root = etree.fromstring(out)
        assert root.find(".//project").get("name") == "Custom Cut"


# ---------- Mode B ----------------------------------------------------------

@pytest.mark.skipif(not ELLA_FCPXML.exists(), reason="Ella fixture not present")
class TestModeBMarkersOnTimeline:
    @pytest.fixture(scope="class")
    def parsed(self):
        return parse_fcpxml(ELLA_FCPXML)

    @pytest.fixture(scope="class")
    def output(self, parsed):
        return write_markers_on_timeline(parsed, SAMPLE_SELECTS)

    def test_output_is_valid_xml(self, output):
        etree.fromstring(output)

    def test_resources_preserved(self, output, parsed):
        assert parsed.original_resources_xml in output

    def test_project_renamed_with_notes_suffix(self, output):
        root = etree.fromstring(output)
        assert root.find(".//project").get("name") == "Ella Interview - Doza Notes"

    def test_original_mc_clip_structure_preserved(self, output):
        # Both original mc-clips should still be on the spine, with their
        # original offsets/starts/durations (we only added marker children).
        root = etree.fromstring(output)
        clips = root.findall(".//spine/mc-clip")
        assert len(clips) == 2
        assert parse_rational(clips[0].get("offset")) == 0
        assert parse_rational(clips[1].get("offset")) == Fraction(1017016, 24000)

    def test_markers_injected_in_correct_clips(self, output):
        root = etree.fromstring(output)
        clips = root.findall(".//spine/mc-clip")
        # Select at source 10s → first mc-clip (covers 0–42.4s).
        first_markers = clips[0].findall("marker")
        assert len(first_markers) == 1
        assert first_markers[0].get("value") == "Opening hook"
        # Selects at source 100s and 600s → second mc-clip.
        second_markers = clips[1].findall("marker")
        assert len(second_markers) == 2
        values = {m.get("value") for m in second_markers}
        assert values == {"Childhood story", "Follow-up?"}

    def test_marker_kinds_encoded_as_attributes(self, output):
        # strong → completed="1", standard → no completed attr, question → completed="0"
        root = etree.fromstring(output)
        markers = {m.get("value"): m for m in root.findall(".//marker")}
        assert markers["Opening hook"].get("completed") == "1"
        assert markers["Childhood story"].get("completed") is None
        assert markers["Follow-up?"].get("completed") == "0"

    def test_marker_notes_preserved(self, output):
        root = etree.fromstring(output)
        m = next(m for m in root.findall(".//marker") if m.get("value") == "Follow-up?")
        assert m.get("note") == "ask legal"

    def test_marker_times_match_source_selects(self, output):
        root = etree.fromstring(output)
        markers = {m.get("value"): m for m in root.findall(".//marker")}
        # Source time 10s → container time 10s (angle offsets are 0 in Ella).
        assert abs(float(parse_rational(markers["Opening hook"].get("start"))) - 10.0) < 0.05
        assert abs(float(parse_rational(markers["Childhood story"].get("start"))) - 100.0) < 0.05
        assert abs(float(parse_rational(markers["Follow-up?"].get("start"))) - 600.0) < 0.05

    def test_round_trips_through_parser(self, output):
        re_parsed = re_parse(output)
        assert re_parsed.container_type == "mc-clip"
        assert len(re_parsed.spine_segments) == 2  # original structure intact
        assert re_parsed.audio_asset_id == "r4"

    def test_selects_in_gaps_are_dropped(self, parsed):
        # Construct a select that falls past the end of the timeline — should
        # be silently dropped (no marker emitted anywhere).
        out = write_markers_on_timeline(
            parsed,
            [Select(start_seconds=99999.0, end_seconds=100000.0, label="unused")],
        )
        root = etree.fromstring(out)
        assert root.findall(".//marker") == []


# ---------- sync-clip writer (synthetic fixture) ----------------------------

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
            <event name="E">
                <project name="Sync Test">
                    <sequence format="r1" duration="240000/24000s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                        <spine>
                            <sync-clip offset="0s" duration="240000/24000s" name="S">
                                <asset-clip ref="r3" offset="0s" duration="240000/24000s"/>
                                <asset-clip ref="r2" offset="0s" duration="240000/24000s" audioRole="dialogue"/>
                            </sync-clip>
                        </spine>
                    </sequence>
                </project>
            </event>
        </library>
    </fcpxml>
""")


class TestSyncClipWriter:
    @pytest.fixture
    def parsed(self, tmp_path):
        p = tmp_path / "sync.fcpxml"
        p.write_text(SYNC_FIXTURE)
        return parse_fcpxml(p)

    def test_mode_a_emits_asset_clips(self, parsed):
        # Sync-clip Mode A emits asset-clips pointing at the dialogue audio
        # (see writer.py docstring for the rationale / tradeoff).
        out = write_selects_as_new_project(parsed, [
            Select(start_seconds=2.0, end_seconds=5.0, label="pickup"),
        ])
        root = etree.fromstring(out)
        clips = root.findall(".//spine/asset-clip")
        assert len(clips) == 1
        assert clips[0].get("ref") == "r2"
        assert clips[0].get("audioRole") == "dialogue"

    def test_mode_b_injects_marker_in_sync_clip(self, parsed):
        out = write_markers_on_timeline(parsed, [
            Select(start_seconds=2.0, end_seconds=3.0, label="pickup", kind="standard"),
        ])
        root = etree.fromstring(out)
        sync_clips = root.findall(".//spine/sync-clip")
        assert len(sync_clips) == 1
        markers = sync_clips[0].findall("marker")
        assert len(markers) == 1
        assert markers[0].get("value") == "pickup"
