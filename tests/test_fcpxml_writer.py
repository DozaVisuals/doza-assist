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

    def test_mode_a_emits_full_sync_clip_with_video_and_audio(self, parsed):
        # Sync-clip Mode A emits the whole <sync-clip> structure (inner
        # asset-clips preserved) so video + audio lanes reattach when FCP
        # re-imports. Emitting audio-only asset-clips would silently drop
        # the video component on import.
        out = write_selects_as_new_project(parsed, [
            Select(start_seconds=2.0, end_seconds=5.0, label="pickup"),
        ])
        root = etree.fromstring(out)
        # A single <sync-clip> on the new spine, carrying the original's
        # video (r3) + audio (r2) asset-clips underneath.
        sync_clips = root.findall(".//spine/sync-clip")
        assert len(sync_clips) == 1
        inner_refs = {ac.get("ref") for ac in sync_clips[0].findall("asset-clip")}
        assert "r2" in inner_refs  # dialogue audio
        assert "r3" in inner_refs  # video
        # Top-level attributes retargeted to the select range.
        assert sync_clips[0].get("name") == "pickup"
        from doza_assist.fcpxml.timecode import parse_rational
        assert abs(float(parse_rational(sync_clips[0].get("start"))) - 2.0) < 0.05
        # No stray bare asset-clips at the top of the spine — that's the old
        # audio-only shape.
        assert root.findall(".//spine/asset-clip") == []

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


# ---------- mixed-container spines (multi-source) ---------------------------

# Mixed spine: mc-clip on [0,100), sync-clip on [100,150) both with resolvable
# dialogue audio. Multi-source projects use timeline-relative select times.
MIXED_SPINE_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.14">
        <resources>
            <format id="r1" name="FF" frameDuration="1001/24000s" width="1920" height="1080"/>
            <asset id="rEllaA" name="ella_audio" start="0s" duration="240000/24000s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                <media-rep kind="original-media" src="file:///tmp/ella.wav"/>
            </asset>
            <asset id="rEllaV" name="ella_video" start="0s" duration="240000/24000s" hasVideo="1" videoSources="1">
                <media-rep kind="original-media" src="file:///tmp/ella_video.mov"/>
            </asset>
            <asset id="rDlg" name="dialogue" start="0s" duration="240000/24000s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                <media-rep kind="original-media" src="file:///tmp/dialogue.wav"/>
            </asset>
            <media id="mcE" name="Ella">
                <multicam>
                    <mc-angle name="V" angleID="vE"><asset-clip ref="rEllaV" offset="0s" duration="240000/24000s"/></mc-angle>
                    <mc-angle name="A" angleID="aE"><asset-clip ref="rEllaA" offset="0s" duration="240000/24000s" audioRole="dialogue"/></mc-angle>
                </multicam>
            </media>
        </resources>
        <library>
            <event name="E">
                <project name="Mixed">
                    <sequence format="r1" duration="150s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                        <spine>
                            <mc-clip ref="mcE" offset="0s" name="Ella" duration="100s">
                                <mc-source angleID="vE" srcEnable="video"/>
                                <mc-source angleID="aE" srcEnable="audio"/>
                            </mc-clip>
                            <sync-clip offset="100s" name="Pickup" duration="50s">
                                <spine>
                                    <asset-clip ref="rDlg" offset="0s" duration="50s" audioRole="dialogue"/>
                                </spine>
                            </sync-clip>
                        </spine>
                    </sequence>
                </project>
            </event>
        </library>
    </fcpxml>
""")


class TestMixedSpineWriter:
    @pytest.fixture
    def parsed(self, tmp_path):
        p = tmp_path / "mixed.fcpxml"
        p.write_text(MIXED_SPINE_FIXTURE)
        return parse_fcpxml(p)

    def test_parsed_is_multi_source(self, parsed):
        assert parsed.is_multi_source is True

    # --- Mode A --------------------------------------------------------------

    def test_mode_a_mixed_output_emits_correct_clip_types(self, parsed):
        # Two selects: one on the mc-clip (timeline 10–20) and one on the
        # sync-clip (timeline 110–120). Expect one mc-clip + one full sync-clip
        # (the sync-clip path now preserves the original inline structure so
        # video + audio reattach on FCP re-import).
        selects = [
            Select(start_seconds=10.0, end_seconds=20.0, label="mc pick", kind="standard"),
            Select(start_seconds=110.0, end_seconds=120.0, label="sync pick", kind="strong"),
        ]
        out = write_selects_as_new_project(parsed, selects)
        root = etree.fromstring(out)
        # Count direct children of the top-level (project) spine only — the
        # sync-clip's own inner spine legitimately contains asset-clips.
        top_spine = root.find(".//sequence/spine")
        top_tags = [c.tag for c in top_spine]
        assert top_tags.count("mc-clip") == 1
        assert top_tags.count("sync-clip") == 1
        assert "asset-clip" not in top_tags

    def test_mode_a_mc_clip_references_segment_ref(self, parsed):
        out = write_selects_as_new_project(parsed, [
            Select(start_seconds=10.0, end_seconds=20.0, label="pick"),
        ])
        root = etree.fromstring(out)
        mc = root.find(".//spine/mc-clip")
        assert mc.get("ref") == "mcE"
        # start should decode to 10s (container time = timeline time for first segment)
        assert abs(float(parse_rational(mc.get("start"))) - 10.0) < 0.05

    def test_mode_a_sync_clip_preserves_inner_structure(self, parsed):
        out = write_selects_as_new_project(parsed, [
            Select(start_seconds=110.0, end_seconds=120.0, label="pick"),
        ])
        root = etree.fromstring(out)
        sc = root.find(".//spine/sync-clip")
        # Top-level sync-clip retargeted to the select's source range.
        # Container time at timeline 110s = segment.start (0) + (110 - 100) = 10s,
        # which for sync-clips equals source time (angle offsets zero).
        assert abs(float(parse_rational(sc.get("start"))) - 10.0) < 0.05
        assert abs(float(parse_rational(sc.get("duration"))) - 10.0) < 0.05
        # Inner structure preserved from the original (deep-copy): the
        # dialogue asset-clip is still nested inside the sync-clip's <spine>,
        # so FCP reattaches video + audio on re-import.
        inner_asset_clips = sc.findall(".//asset-clip")
        assert len(inner_asset_clips) >= 1
        refs = {ac.get("ref") for ac in inner_asset_clips}
        assert "rDlg" in refs

    def test_mode_a_mc_source_enablement_comes_from_owning_segment(self, parsed):
        out = write_selects_as_new_project(parsed, [
            Select(start_seconds=10.0, end_seconds=20.0, label="pick"),
        ])
        root = etree.fromstring(out)
        mc = root.find(".//spine/mc-clip")
        srcs = mc.findall("mc-source")
        enables = {s.get("srcEnable"): s.get("angleID") for s in srcs}
        assert enables.get("audio") == "aE"
        assert enables.get("video") == "vE"

    def test_mode_a_cross_boundary_heterogeneous_raises(self, parsed):
        # Select 95-110 crosses mc-clip (mcE) → sync-clip (rDlg). Different
        # sources → still rejected; the editor needs to trim or split.
        with pytest.raises(WriterError, match="different sources"):
            write_selects_as_new_project(parsed, [
                Select(start_seconds=95.0, end_seconds=110.0, label="bad"),
            ])

    def test_mode_a_select_in_gap_raises(self, parsed):
        # Past the end of the sequence (> 150s) — no covering segment.
        with pytest.raises(WriterError, match="outside"):
            write_selects_as_new_project(parsed, [
                Select(start_seconds=200.0, end_seconds=210.0, label="bad"),
            ])

    def test_mode_a_selects_contiguous_on_new_timeline(self, parsed):
        # Each select's offset on the new timeline = sum of previous durations.
        selects = [
            Select(start_seconds=10.0, end_seconds=20.0, label="first"),
            Select(start_seconds=110.0, end_seconds=120.0, label="second"),
        ]
        out = write_selects_as_new_project(parsed, selects)
        root = etree.fromstring(out)
        children = [c for c in root.find(".//spine") if c.tag in ("mc-clip", "sync-clip", "asset-clip")]
        cursor = Fraction(0)
        for c in children:
            off = parse_rational(c.get("offset"))
            dur = parse_rational(c.get("duration"))
            assert off == cursor
            cursor += dur

    # --- Mode B --------------------------------------------------------------

    def test_mode_b_markers_route_to_correct_segment(self, parsed):
        selects = [
            Select(start_seconds=15.0, end_seconds=16.0, label="mc marker", kind="standard"),
            Select(start_seconds=115.0, end_seconds=116.0, label="sync marker", kind="strong"),
        ]
        out = write_markers_on_timeline(parsed, selects)
        root = etree.fromstring(out)
        mc_markers = root.findall(".//spine/mc-clip/marker")
        sync_markers = root.findall(".//spine/sync-clip/marker")
        assert len(mc_markers) == 1
        assert mc_markers[0].get("value") == "mc marker"
        assert len(sync_markers) == 1
        assert sync_markers[0].get("value") == "sync marker"

    def test_mode_b_marker_container_time_is_relative_to_segment(self, parsed):
        out = write_markers_on_timeline(parsed, [
            Select(start_seconds=115.0, end_seconds=116.0, label="sync marker"),
        ])
        root = etree.fromstring(out)
        m = root.find(".//sync-clip/marker")
        # timeline 115 → segment 2 (offset 100, start 0) → container time 15.
        assert abs(float(parse_rational(m.get("start"))) - 15.0) < 0.05


# ---------- same-source boundary crossing (story beats spanning FCP splits) ----
#
# Real-world pattern (Trustees project): FCP often splits one continuous
# recording into several adjacent sync-clips on the spine. An AI-generated
# story beat can naturally span that split — it's still one semantic take.
# The writer should collapse same-source spans into one clip on the new
# project instead of rejecting them.

SAME_SOURCE_SYNC_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.14">
        <resources>
            <format id="r1" name="FF" frameDuration="1001/24000s" width="1920" height="1080"/>
            <asset id="rCam" name="cam" start="0s" duration="1000s" hasVideo="1" hasAudio="1" videoSources="1" audioSources="1" audioChannels="2" audioRate="48000">
                <media-rep kind="original-media" src="file:///tmp/cam.mov"/>
            </asset>
            <asset id="rExt" name="ext" start="0s" duration="1000s" hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000">
                <media-rep kind="original-media" src="file:///tmp/external.wav"/>
            </asset>
            <asset id="rOther" name="otherAudio" start="0s" duration="1000s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                <media-rep kind="original-media" src="file:///tmp/other.wav"/>
            </asset>
        </resources>
        <library>
            <event name="E">
                <project name="Split Take">
                    <sequence format="r1" duration="500s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                        <spine>
                            <!-- Two adjacent sync-clips from one continuous
                                 external recording (rExt) split by the editor. -->
                            <sync-clip offset="0s" name="take A" start="100s" duration="200s">
                                <spine>
                                    <asset-clip ref="rExt" lane="-1" offset="0s" duration="1000s" audioRole="dialogue"/>
                                </spine>
                                <sync-source sourceID="storyline">
                                    <audio-role-source role="dialogue.dialogue-1" active="1"/>
                                </sync-source>
                            </sync-clip>
                            <sync-clip offset="200s" name="take B" start="300s" duration="200s">
                                <spine>
                                    <asset-clip ref="rExt" lane="-1" offset="0s" duration="1000s" audioRole="dialogue"/>
                                </spine>
                                <sync-source sourceID="storyline">
                                    <audio-role-source role="dialogue.dialogue-1" active="1"/>
                                </sync-source>
                            </sync-clip>
                            <!-- Third segment with a different source — its sole
                                 purpose is to force is_multi_source=True so the
                                 timeline-coordinate select path is exercised. -->
                            <sync-clip offset="400s" name="other" start="0s" duration="100s">
                                <spine>
                                    <asset-clip ref="rOther" offset="0s" duration="1000s" audioRole="dialogue"/>
                                </spine>
                            </sync-clip>
                        </spine>
                    </sequence>
                </project>
            </event>
        </library>
    </fcpxml>
""")


class TestSameSourceBoundaryCrossing:
    @pytest.fixture
    def parsed(self, tmp_path):
        p = tmp_path / "same_src.fcpxml"
        p.write_text(SAME_SOURCE_SYNC_FIXTURE)
        return parse_fcpxml(p)

    def test_parsed_is_multi_source(self, parsed):
        # Third segment forces is_multi_source=True so select times are
        # interpreted in timeline coordinates.
        assert parsed.is_multi_source is True

    def test_parsed_first_two_sync_clips_share_source(self, parsed):
        s1, s2, _ = parsed.spine_segments
        assert s1.audio_source.asset_id == s2.audio_source.asset_id == "rExt"

    def test_cross_boundary_select_collapses_to_one_sync_clip(self, parsed):
        # Select at timeline 150–250 spans segments 1 and 2 (boundary at 200).
        # Both reference rExt → collapse into ONE sync-clip on the new spine.
        selects = [Select(start_seconds=150.0, end_seconds=250.0, label="story beat")]
        out = write_selects_as_new_project(parsed, selects)
        root = etree.fromstring(out)
        top_spine = root.find(".//sequence/spine")
        top_children = [c for c in top_spine if c.tag in ("mc-clip", "sync-clip")]
        assert len(top_children) == 1
        assert top_children[0].tag == "sync-clip"
        sc = top_children[0]
        # Duration matches the full select span (100s), not the intersection
        # with one segment.
        assert abs(float(parse_rational(sc.get("duration"))) - 100.0) < 0.1
        # start is the source time at the select's in-point: segment 1 has
        # offset=0 and start=100, so timeline 150 → container time 250.
        assert abs(float(parse_rational(sc.get("start"))) - 250.0) < 0.1


MULTI_MC_SAME_REF_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.14">
        <resources>
            <format id="r1" name="FF" frameDuration="1001/24000s" width="1920" height="1080"/>
            <asset id="rA" name="audio" start="0s" duration="1000s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                <media-rep kind="original-media" src="file:///tmp/a.wav"/>
            </asset>
            <asset id="rV" name="video" start="0s" duration="1000s" hasVideo="1" videoSources="1">
                <media-rep kind="original-media" src="file:///tmp/v.mov"/>
            </asset>
            <media id="mcX" name="Interview MC">
                <multicam>
                    <mc-angle name="V" angleID="vX"><asset-clip ref="rV" offset="0s" duration="1000s"/></mc-angle>
                    <mc-angle name="A" angleID="aX"><asset-clip ref="rA" offset="0s" duration="1000s" audioRole="dialogue"/></mc-angle>
                </multicam>
            </media>
        </resources>
        <library>
            <event name="E">
                <project name="Same MC Twice">
                    <sequence format="r1" duration="200s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                        <spine>
                            <!-- Two adjacent mc-clips referencing the same multicam. -->
                            <mc-clip ref="mcX" offset="0s" name="first" duration="100s">
                                <mc-source angleID="vX" srcEnable="video"/>
                                <mc-source angleID="aX" srcEnable="audio"/>
                            </mc-clip>
                            <mc-clip ref="mcX" offset="100s" name="second" start="100s" duration="100s">
                                <mc-source angleID="vX" srcEnable="video"/>
                                <mc-source angleID="aX" srcEnable="audio"/>
                            </mc-clip>
                        </spine>
                    </sequence>
                </project>
            </event>
        </library>
    </fcpxml>
""")


class TestSameMulticamBoundaryCrossing:
    @pytest.fixture
    def parsed(self, tmp_path):
        p = tmp_path / "same_mc.fcpxml"
        p.write_text(MULTI_MC_SAME_REF_FIXTURE)
        return parse_fcpxml(p)

    def test_cross_boundary_mc_select_collapses_to_one_mc_clip(self, parsed):
        # Select 80–120 crosses the boundary at 100 but both segments reference
        # the same multicam (mcX). Collapse into one mc-clip.
        selects = [Select(start_seconds=80.0, end_seconds=120.0, label="beat")]
        out = write_selects_as_new_project(parsed, selects)
        root = etree.fromstring(out)
        mc_clips = root.findall(".//spine/mc-clip")
        assert len(mc_clips) == 1
        assert mc_clips[0].get("ref") == "mcX"
        assert abs(float(parse_rational(mc_clips[0].get("duration"))) - 40.0) < 0.1
