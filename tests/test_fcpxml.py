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
