"""Tests for the doza_assist.fcpxml parser and timecode modules.

The canonical multicam fixture is the Ella Interview sample at
``/Users/dozavisuals/Downloads/Ella Interview.fcpxmld/Info.fcpxml``. Timecode
expectations in this file mirror the requirements in the pass-A brief: audio
file time 0 should map to timeline time 0, audio file time 100s should map to
timeline time 100s for the contiguous multicam spine.

Sync-clip coverage uses a synthetic FCPXML generated in-test so the suite stays
self-contained (no on-disk sync-clip fixture required).
"""

import os
import sys
import textwrap
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from doza_assist.fcpxml import (  # noqa: E402
    ParseError,
    audio_source_to_timeline,
    parse_fcpxml,
    parse_rational,
    rational_to_seconds,
    seconds_to_rational,
    timeline_to_segment,
)
from doza_assist.fcpxml.parser import SpineSegment, strip_file_url  # noqa: E402


ELLA_FCPXML = Path("/Users/dozavisuals/Downloads/Ella Interview.fcpxmld/Info.fcpxml")


# ---------- timecode ---------------------------------------------------------

class TestRationalParsing:
    def test_fractional_value(self):
        assert parse_rational("1017016/24000s") == Fraction(1017016, 24000)

    def test_integer_seconds(self):
        assert parse_rational("28626s") == Fraction(28626)

    def test_zero(self):
        assert parse_rational("0s") == 0
        assert parse_rational(None) == 0
        assert parse_rational("") == 0

    def test_bare_number(self):
        # Some producers omit the trailing 's'.
        assert parse_rational("1017016/24000") == Fraction(1017016, 24000)

    def test_to_seconds(self):
        assert rational_to_seconds("1001/24000s") == pytest.approx(1001 / 24000)


class TestSecondsToRational:
    FD_2398 = Fraction(1001, 24000)

    def test_zero(self):
        assert seconds_to_rational(0, self.FD_2398) == "0s"

    def test_round_trips_known_value(self):
        out = seconds_to_rational(Fraction(1017016, 24000), self.FD_2398)
        # 1017016 / 24000 s = 1016 frames * (1001/24000). 1016 * 1001 = 1017016.
        assert out == "1017016/24000s"
        assert parse_rational(out) == Fraction(1017016, 24000)

    def test_snaps_to_frame_grid(self):
        # A non-frame-aligned seconds value should snap to the nearest frame.
        out = seconds_to_rational(1.0, self.FD_2398)
        num, den = out.rstrip("s").split("/")
        num_i = int(num)
        assert num_i % 1001 == 0  # always a multiple of frame_duration numerator

    def test_rejects_nonpositive_frame_duration(self):
        with pytest.raises(ValueError):
            seconds_to_rational(1.0, Fraction(0))


# ---------- audio_source_to_timeline ----------------------------------------

def _seg(offset, start, duration):
    return SpineSegment(
        kind="mc-clip", ref="r2", name="",
        offset_fraction=Fraction(offset),
        start_fraction=Fraction(start),
        duration_fraction=Fraction(duration),
    )


class TestAudioSourceToTimeline:
    """Mirrors the Ella Interview timeline: two contiguous mc-clips."""

    def setup_method(self):
        self.first = Fraction(1017016, 24000)
        self.total = Fraction(52079027, 24000)
        self.segments = [
            _seg(0, 0, self.first),
            _seg(self.first, self.first, self.total - self.first),
        ]

    def test_source_zero_maps_to_timeline_zero(self):
        assert audio_source_to_timeline(0, self.segments) == 0

    def test_source_100s_maps_to_timeline_100s(self):
        assert audio_source_to_timeline(100, self.segments) == 100

    def test_within_first_segment(self):
        assert audio_source_to_timeline(20, self.segments) == 20

    def test_past_end_returns_none(self):
        assert audio_source_to_timeline(10_000, self.segments) is None

    def test_gap_between_segments_returns_none(self):
        # Construct a gapped spine: [0,10) and [20,30) on the timeline, same
        # container-internal ranges. Source time 15 falls in the gap.
        gapped = [_seg(0, 0, 10), _seg(20, 20, 10)]
        assert audio_source_to_timeline(15, gapped) is None

    def test_container_start_offset(self):
        # A single mc-clip that starts 5s into its container.
        segs = [_seg(offset=0, start=5, duration=10)]
        # Source time 7 is at container time 7, which is at offset 0 + (7-5) = 2.
        assert audio_source_to_timeline(7, segs) == 2
        # Source time 4 is before the segment starts → None.
        assert audio_source_to_timeline(4, segs) is None

    def test_audio_angle_offset_shifts_mapping(self):
        # Audio angle asset-clip offset inside the container = 3s. So source
        # time T corresponds to container time T + 3.
        segs = [_seg(offset=0, start=0, duration=10)]
        assert audio_source_to_timeline(2, segs, audio_angle_offset=Fraction(3)) == 5

    def test_accepts_dict_segments(self):
        # Segments serialized to project meta.json round-trip through dicts.
        dicts = [s.to_dict() for s in self.segments]
        assert audio_source_to_timeline(100, dicts) == 100


# ---------- file URL decoding -----------------------------------------------

class TestStripFileUrl:
    def test_strips_and_decodes(self):
        src = "file:///Volumes/DOZA%20EDIT%20SSD/Trustees/Ella/021926_075706/021926_075706_Tr2.WAV"
        assert strip_file_url(src) == "/Volumes/DOZA EDIT SSD/Trustees/Ella/021926_075706/021926_075706_Tr2.WAV"

    def test_passthrough_when_no_scheme(self):
        assert strip_file_url("/tmp/foo.wav") == "/tmp/foo.wav"


# ---------- multicam parsing: Ella Interview --------------------------------

@pytest.mark.skipif(not ELLA_FCPXML.exists(), reason="Ella fixture not present")
class TestEllaMulticam:
    @pytest.fixture(scope="class")
    def parsed(self):
        return parse_fcpxml(ELLA_FCPXML)

    def test_version_supported(self, parsed):
        assert parsed.version == "1.14"

    def test_container_is_multicam(self, parsed):
        assert parsed.container_type == "mc-clip"
        assert parsed.container_ref == "r2"

    def test_active_audio_angle(self, parsed):
        # In the Ella timeline the audio-enabled mc-source is the
        # esv2-83p-bg-10p angle (qMugMvsqRpW4mCI2v5CgDA → asset r4).
        assert parsed.active_audio_angle_id == "qMugMvsqRpW4mCI2v5CgDA"
        assert parsed.audio_asset_id == "r4"

    def test_audio_path_is_decoded_and_absolute(self, parsed):
        assert parsed.audio_file_path == (
            "/Volumes/DOZA EDIT SSD/Trustees/Posey/Studio Visit/"
            "121525_133506/021926_075706_Tr2-esv2-83p-bg-10p.wav"
        )
        assert not parsed.audio_file_path.startswith("file://")
        assert "%20" not in parsed.audio_file_path

    def test_sequence_framerate(self, parsed):
        # 24000/1001 ≈ 23.976
        assert parsed.sequence_frame_duration == Fraction(1001, 24000)

    def test_spine_segments_contiguous(self, parsed):
        assert len(parsed.spine_segments) == 2
        first, second = parsed.spine_segments
        assert first.offset_fraction == 0
        assert first.start_fraction == 0
        assert second.offset_fraction == Fraction(1017016, 24000)
        assert second.start_fraction == Fraction(1017016, 24000)
        # Contiguous: second.offset == first.offset + first.duration
        assert second.offset_fraction == first.offset_fraction + first.duration_fraction

    def test_source_time_equals_timeline_time_on_contiguous_spine(self, parsed):
        # Brief's canonical assertion: 0 → 0 and 100 → 100.
        assert audio_source_to_timeline(
            0, parsed.spine_segments,
            audio_angle_offset=parsed.audio_angle_offset_fraction,
            audio_angle_start=parsed.audio_angle_start_fraction,
        ) == 0
        assert audio_source_to_timeline(
            100, parsed.spine_segments,
            audio_angle_offset=parsed.audio_angle_offset_fraction,
            audio_angle_start=parsed.audio_angle_start_fraction,
        ) == 100

    def test_source_time_600s_matches_timeline_600s(self, parsed):
        # 10-minute mark: audio 00:10:00 → timeline 00:10:00 for contiguous multicam.
        assert audio_source_to_timeline(
            600, parsed.spine_segments,
            audio_angle_offset=parsed.audio_angle_offset_fraction,
            audio_angle_start=parsed.audio_angle_start_fraction,
        ) == 600

    def test_resources_preserved_verbatim(self, parsed):
        # The verbatim slice must appear exactly in the original bytes.
        assert parsed.original_resources_xml in parsed.original_fcpxml_bytes
        assert parsed.original_resources_xml.startswith(b"<resources")
        assert parsed.original_resources_xml.endswith(b"</resources>")
        # And it must contain the bookmark base64 blobs verbatim — that's what
        # FCP uses to locate media on re-import; any rewrite breaks the import.
        assert b"<bookmark>" in parsed.original_resources_xml

    def test_metadata_dict_is_json_serializable(self, parsed):
        import json
        data = parsed.to_metadata_dict()
        round_tripped = json.loads(json.dumps(data))
        assert round_tripped["container_type"] == "mc-clip"
        assert round_tripped["active_audio_angle_id"] == "qMugMvsqRpW4mCI2v5CgDA"


# ---------- sync-clip: synthetic FCPXML -------------------------------------

SYNC_CLIP_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.13">
        <resources>
            <format id="r1" name="FFVideoFormat1080p2398" frameDuration="1001/24000s" width="1920" height="1080"/>
            <asset id="r2" name="interview_audio" start="0s" duration="240000/24000s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                <media-rep kind="original-media" src="file:///tmp/A%20Folder/interview%20audio.wav"/>
            </asset>
            <asset id="r3" name="cam_a" start="0s" duration="240000/24000s" hasVideo="1" videoSources="1">
                <media-rep kind="original-media" src="file:///tmp/cam_a.mov"/>
            </asset>
        </resources>
        <library>
            <event name="Test Event">
                <project name="Sync Test">
                    <sequence format="r1" duration="240000/24000s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                        <spine>
                            <sync-clip offset="0s" name="Sync Clip 1" duration="240000/24000s">
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


class TestSyncClip:
    @pytest.fixture
    def fixture_path(self, tmp_path):
        p = tmp_path / "sync.fcpxml"
        p.write_text(SYNC_CLIP_FIXTURE, encoding="utf-8")
        return p

    def test_parses_sync_clip(self, fixture_path):
        parsed = parse_fcpxml(fixture_path)
        assert parsed.container_type == "sync-clip"
        assert parsed.version == "1.13"

    def test_resolves_dialogue_audio(self, fixture_path):
        parsed = parse_fcpxml(fixture_path)
        assert parsed.audio_asset_id == "r2"
        assert parsed.audio_file_path == "/tmp/A Folder/interview audio.wav"

    def test_has_no_audio_angle(self, fixture_path):
        parsed = parse_fcpxml(fixture_path)
        assert parsed.active_audio_angle_id is None

    def test_sync_clip_has_one_spine_segment(self, fixture_path):
        parsed = parse_fcpxml(fixture_path)
        assert len(parsed.spine_segments) == 1
        seg = parsed.spine_segments[0]
        assert seg.kind == "sync-clip"

    def test_contiguous_source_to_timeline(self, fixture_path):
        parsed = parse_fcpxml(fixture_path)
        assert audio_source_to_timeline(
            5, parsed.spine_segments,
            audio_angle_offset=parsed.audio_angle_offset_fraction,
            audio_angle_start=parsed.audio_angle_start_fraction,
        ) == 5


# ---------- error handling --------------------------------------------------

class TestParseErrors:
    def test_rejects_unsupported_version(self, tmp_path):
        p = tmp_path / "old.fcpxml"
        p.write_text('<?xml version="1.0"?><fcpxml version="1.9"><resources/></fcpxml>')
        with pytest.raises(ParseError, match="unsupported FCPXML version"):
            parse_fcpxml(p)

    def test_rejects_missing_spine(self, tmp_path):
        p = tmp_path / "empty.fcpxml"
        p.write_text(textwrap.dedent("""\
            <?xml version="1.0"?>
            <fcpxml version="1.14">
                <resources>
                    <format id="r1" name="X" frameDuration="1001/24000s" width="1920" height="1080"/>
                </resources>
                <library>
                    <event><project>
                        <sequence format="r1" duration="0s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k"/>
                    </project></event>
                </library>
            </fcpxml>
        """))
        with pytest.raises(ParseError, match="no <spine>"):
            parse_fcpxml(p)

    def test_rejects_invalid_xml(self, tmp_path):
        p = tmp_path / "broken.fcpxml"
        p.write_text("<fcpxml version=\"1.14\"><resources></fcpxml>")
        with pytest.raises(ParseError, match="invalid FCPXML"):
            parse_fcpxml(p)


# ---------- timeline_to_segment ---------------------------------------------

class TestTimelineToSegment:
    """Inverse of audio_source_to_timeline — find the owning segment for a
    timeline time, returning container-internal time within it."""

    def setup_method(self):
        # Two contiguous segments: [0,10) and [10,30) on the timeline, matching
        # the same ranges in container time.
        self.segments = [_seg(0, 0, 10), _seg(10, 10, 20)]

    def test_maps_into_first_segment(self):
        seg, ct = timeline_to_segment(self.segments, 5)
        assert seg is self.segments[0]
        assert ct == 5

    def test_maps_into_second_segment(self):
        seg, ct = timeline_to_segment(self.segments, 20)
        assert seg is self.segments[1]
        assert ct == 20

    def test_exact_start_of_segment_inclusive(self):
        seg, ct = timeline_to_segment(self.segments, 10)
        assert seg is self.segments[1]
        assert ct == 10

    def test_gap_returns_none(self):
        gapped = [_seg(0, 0, 5), _seg(10, 0, 5)]
        seg, ct = timeline_to_segment(gapped, 7)
        assert seg is None
        assert ct is None

    def test_past_end_returns_none(self):
        seg, ct = timeline_to_segment(self.segments, 100)
        assert seg is None

    def test_segment_with_container_start_offset(self):
        # Segment that starts 5s into its container and plays 10s of it.
        segs = [_seg(offset=0, start=5, duration=10)]
        seg, ct = timeline_to_segment(segs, 3)
        # container time = segment.start + (timeline - segment.offset) = 5 + 3 = 8
        assert seg is segs[0]
        assert ct == 8


# ---------- mixed-spine parsing ---------------------------------------------

MIXED_SPINE_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.14">
        <resources>
            <format id="r1" name="FF" frameDuration="1001/24000s" width="1920" height="1080"/>
            <asset id="r2" name="ella_audio" start="0s" duration="240000/24000s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                <media-rep kind="original-media" src="file:///tmp/ella.wav"/>
            </asset>
            <asset id="r3" name="cam_a" start="0s" duration="240000/24000s" hasVideo="1" videoSources="1">
                <media-rep kind="original-media" src="file:///tmp/cam_a.mov"/>
            </asset>
            <asset id="r4" name="ella_video" start="0s" duration="240000/24000s" hasVideo="1" videoSources="1">
                <media-rep kind="original-media" src="file:///tmp/ella_video.mov"/>
            </asset>
            <asset id="r5" name="posey_audio" start="0s" duration="240000/24000s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                <media-rep kind="original-media" src="file:///tmp/posey.wav"/>
            </asset>
            <asset id="r6" name="dialogue_asset" start="0s" duration="240000/24000s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                <media-rep kind="original-media" src="file:///tmp/dialogue.wav"/>
            </asset>
            <media id="mcA" name="Ella Interview">
                <multicam>
                    <mc-angle name="Video" angleID="vidA">
                        <asset-clip ref="r4" offset="0s" duration="240000/24000s"/>
                    </mc-angle>
                    <mc-angle name="Audio" angleID="audA">
                        <asset-clip ref="r2" offset="0s" duration="240000/24000s" audioRole="dialogue"/>
                    </mc-angle>
                </multicam>
            </media>
        </resources>
        <library>
            <event name="E">
                <project name="Mixed Interview">
                    <sequence format="r1" duration="600s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                        <spine>
                            <mc-clip ref="mcA" offset="0s" name="Ella" duration="100s">
                                <mc-source angleID="vidA" srcEnable="video"/>
                                <mc-source angleID="audA" srcEnable="audio"/>
                            </mc-clip>
                            <sync-clip offset="100s" name="Pickup" duration="50s">
                                <spine>
                                    <asset-clip ref="r6" offset="0s" duration="50s" audioRole="dialogue"/>
                                </spine>
                                <sync-source sourceID="storyline">
                                    <audio-role-source role="dialogue.dialogue-1" active="1"/>
                                </sync-source>
                            </sync-clip>
                            <sync-clip offset="150s" name="Muted pickup" duration="25s">
                                <spine>
                                    <asset-clip ref="r6" offset="0s" duration="25s" audioRole="dialogue"/>
                                </spine>
                                <sync-source sourceID="storyline">
                                    <audio-role-source role="dialogue.dialogue-1" active="0"/>
                                </sync-source>
                            </sync-clip>
                        </spine>
                    </sequence>
                </project>
            </event>
        </library>
    </fcpxml>
""")


class TestMixedSpine:
    @pytest.fixture
    def parsed(self, tmp_path):
        p = tmp_path / "mixed.fcpxml"
        p.write_text(MIXED_SPINE_FIXTURE)
        return parse_fcpxml(p)

    def test_three_segments_preserved(self, parsed):
        assert len(parsed.spine_segments) == 3
        kinds = [s.kind for s in parsed.spine_segments]
        assert kinds == ["mc-clip", "sync-clip", "sync-clip"]

    def test_is_multi_source_true(self, parsed):
        assert parsed.is_multi_source is True

    def test_per_segment_audio_resolved(self, parsed):
        ella, pickup, muted = parsed.spine_segments
        assert ella.audio_source.path == "/tmp/ella.wav"
        assert ella.audio_source.asset_id == "r2"
        assert ella.audio_source.active_audio_angle_id == "audA"
        assert pickup.audio_source.path == "/tmp/dialogue.wav"
        assert pickup.audio_source.asset_id == "r6"
        assert muted.audio_source.path == "/tmp/dialogue.wav"

    def test_sync_clip_nested_spine_is_resolved(self, parsed):
        # The sync-clips wrap their asset-clip inside <spine>; resolver must
        # find it in either shape.
        pickup = parsed.spine_segments[1]
        assert pickup.audio_source.asset_id == "r6"

    def test_muted_sync_clip_detected(self, parsed):
        ella, pickup, muted = parsed.spine_segments
        assert ella.audio_source.is_muted is False
        assert pickup.audio_source.is_muted is False
        assert muted.audio_source.is_muted is True

    def test_representative_audio_is_first_non_muted(self, parsed):
        assert parsed.audio_file_path == "/tmp/ella.wav"
        assert parsed.container_type == "mc-clip"
        assert parsed.container_ref == "mcA"

    def test_unique_audio_sources(self, parsed):
        sources = parsed.unique_audio_sources()
        paths = {s.path for s in sources}
        assert paths == {"/tmp/ella.wav", "/tmp/dialogue.wav"}

    def test_metadata_dict_includes_per_segment_audio(self, parsed):
        data = parsed.to_metadata_dict()
        assert data["is_multi_source"] is True
        assert data["spine_segments"][0]["audio_source"]["asset_id"] == "r2"
        assert data["spine_segments"][2]["audio_source"]["is_muted"] is True


# ---------- mc-clips referencing different containers -----------------------

MULTI_MC_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.14">
        <resources>
            <format id="r1" name="FF" frameDuration="1001/24000s" width="1920" height="1080"/>
            <asset id="rA" name="ella_audio" start="0s" duration="240000/24000s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                <media-rep kind="original-media" src="file:///tmp/ella.wav"/>
            </asset>
            <asset id="rB" name="posey_audio" start="0s" duration="240000/24000s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                <media-rep kind="original-media" src="file:///tmp/posey.wav"/>
            </asset>
            <asset id="rAV" name="ella_v" start="0s" duration="240000/24000s" hasVideo="1" videoSources="1">
                <media-rep kind="original-media" src="file:///tmp/ella_v.mov"/>
            </asset>
            <asset id="rBV" name="posey_v" start="0s" duration="240000/24000s" hasVideo="1" videoSources="1">
                <media-rep kind="original-media" src="file:///tmp/posey_v.mov"/>
            </asset>
            <media id="mcE" name="Ella MC">
                <multicam>
                    <mc-angle name="V" angleID="vE"><asset-clip ref="rAV" offset="0s" duration="240000/24000s"/></mc-angle>
                    <mc-angle name="A" angleID="aE"><asset-clip ref="rA" offset="0s" duration="240000/24000s" audioRole="dialogue"/></mc-angle>
                </multicam>
            </media>
            <media id="mcP" name="Posey MC">
                <multicam>
                    <mc-angle name="V" angleID="vP"><asset-clip ref="rBV" offset="0s" duration="240000/24000s"/></mc-angle>
                    <mc-angle name="A" angleID="aP"><asset-clip ref="rB" offset="0s" duration="240000/24000s" audioRole="dialogue"/></mc-angle>
                </multicam>
            </media>
        </resources>
        <library>
            <event name="E">
                <project name="Two Multicams">
                    <sequence format="r1" duration="200s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                        <spine>
                            <mc-clip ref="mcE" offset="0s" name="Ella" duration="100s">
                                <mc-source angleID="vE" srcEnable="video"/>
                                <mc-source angleID="aE" srcEnable="audio"/>
                            </mc-clip>
                            <mc-clip ref="mcP" offset="100s" name="Posey" duration="100s">
                                <mc-source angleID="vP" srcEnable="video"/>
                                <mc-source angleID="aP" srcEnable="audio"/>
                            </mc-clip>
                        </spine>
                    </sequence>
                </project>
            </event>
        </library>
    </fcpxml>
""")


class TestMultipleMulticamsOnOneSpine:
    @pytest.fixture
    def parsed(self, tmp_path):
        p = tmp_path / "two_mc.fcpxml"
        p.write_text(MULTI_MC_FIXTURE)
        return parse_fcpxml(p)

    def test_two_segments_different_refs(self, parsed):
        assert len(parsed.spine_segments) == 2
        assert parsed.spine_segments[0].ref == "mcE"
        assert parsed.spine_segments[1].ref == "mcP"

    def test_is_multi_source(self, parsed):
        assert parsed.is_multi_source is True

    def test_each_segment_has_own_angle(self, parsed):
        ella, posey = parsed.spine_segments
        assert ella.audio_source.active_audio_angle_id == "aE"
        assert posey.audio_source.active_audio_angle_id == "aP"
        assert ella.audio_source.path == "/tmp/ella.wav"
        assert posey.audio_source.path == "/tmp/posey.wav"


# ---------- sync-clip with muted camera + lane-attached external audio ------
#
# Real-world pattern (Trustees project): a sync-clip pairs a camera clip
# (low-quality scratch mic) with an external recorder on a connected lane.
# FCP mutes the camera mic via <sync-source>/<audio-role-source active="0">,
# and plays the external WAV. Doza Assist must route transcription to the
# external audio, not silence.

LANE_AUDIO_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.14">
        <resources>
            <format id="r1" name="FF" frameDuration="1001/24000s" width="1920" height="1080"/>
            <format id="r8" name="FFVideoFormatRateUndefined"/>
            <asset id="rCam" name="C1234" start="0s" duration="240000/24000s" hasVideo="1" hasAudio="1" videoSources="1" audioSources="1" audioChannels="2" audioRate="48000" format="r8">
                <media-rep kind="original-media" src="file:///tmp/C1234.MP4"/>
            </asset>
            <asset id="rExt" name="external_wav" start="0s" duration="600s" hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000">
                <media-rep kind="original-media" src="file:///tmp/external.wav"/>
            </asset>
        </resources>
        <library>
            <event name="E">
                <project name="Lane Rescue">
                    <sequence format="r1" duration="200s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                        <spine>
                            <sync-clip offset="0s" name="C1234 - Synchronized" start="100s" duration="50s" format="r8" tcFormat="NDF">
                                <spine>
                                    <gap name="Gap" offset="0s" start="3600s" duration="23s">
                                        <asset-clip ref="rExt" lane="-1" offset="3600s" name="external_wav" duration="600s" audioRole="dialogue"/>
                                    </gap>
                                    <asset-clip ref="rCam" offset="23s" name="C1234" duration="200s" audioRole="dialogue"/>
                                </spine>
                                <sync-source sourceID="storyline">
                                    <audio-role-source role="dialogue.dialogue-1" active="0"/>
                                </sync-source>
                            </sync-clip>
                        </spine>
                    </sequence>
                </project>
            </event>
        </library>
    </fcpxml>
""")


class TestSyncClipLaneAudioRescue:
    """When the camera mic is muted and a connected external recorder is
    attached on a lane, we should route audio to the external clip and the
    segment should NOT be marked muted."""

    @pytest.fixture
    def parsed(self, tmp_path):
        p = tmp_path / "lane.fcpxml"
        p.write_text(LANE_AUDIO_FIXTURE)
        return parse_fcpxml(p)

    def test_picks_lane_attached_external_audio(self, parsed):
        seg = parsed.spine_segments[0]
        assert seg.audio_source.asset_id == "rExt"
        assert seg.audio_source.path == "/tmp/external.wav"

    def test_segment_not_muted_when_lane_rescues(self, parsed):
        # FCP is actively playing the external recorder, so ingest must not
        # treat this as silent.
        seg = parsed.spine_segments[0]
        assert seg.audio_source.is_muted is False

    def test_source_time_is_sync_clip_start(self, parsed):
        # For sync-clips the `start` attribute is already source time into the
        # chosen audio asset. angle_offset/angle_start collapse to zero so the
        # renderer's math reduces to source_time = segment.start.
        seg = parsed.spine_segments[0]
        assert seg.audio_source.angle_offset_fraction == 0
        assert seg.audio_source.angle_start_fraction == 0
        assert seg.start_fraction == 100  # seconds


class TestSyncClipFullyMuted:
    """When the camera mic is muted and no lane replacement is available,
    the segment IS silent in FCP and should be marked muted."""

    FIXTURE = textwrap.dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE fcpxml>
        <fcpxml version="1.14">
            <resources>
                <format id="r1" name="FF" frameDuration="1001/24000s" width="1920" height="1080"/>
                <asset id="rCam" name="cam" start="0s" duration="240000/24000s" hasVideo="1" hasAudio="1" videoSources="1" audioSources="1" audioChannels="2" audioRate="48000">
                    <media-rep kind="original-media" src="file:///tmp/cam.mov"/>
                </asset>
            </resources>
            <library>
                <event name="E">
                    <project name="Muted">
                        <sequence format="r1" duration="50s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                            <spine>
                                <sync-clip offset="0s" name="Muted only" duration="50s">
                                    <spine>
                                        <asset-clip ref="rCam" offset="0s" duration="50s" audioRole="dialogue"/>
                                    </spine>
                                    <sync-source sourceID="storyline">
                                        <audio-role-source role="dialogue.dialogue-1" active="0"/>
                                    </sync-source>
                                </sync-clip>
                            </spine>
                        </sequence>
                    </project>
                </event>
            </library>
        </fcpxml>
    """)

    def test_marks_muted_when_no_lane_rescue(self, tmp_path):
        p = tmp_path / "muted.fcpxml"
        p.write_text(self.FIXTURE)
        parsed = parse_fcpxml(p)
        seg = parsed.spine_segments[0]
        assert seg.audio_source.is_muted is True
