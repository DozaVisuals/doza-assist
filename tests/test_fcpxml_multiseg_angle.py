"""Segmented multicam angles: every camera file of the angle, not just one.

Field bug (2026-08-21): multicam angles built from many short stop-start
camera files (the camera recording in takes — 43 files across 4 angles of a
2h44 interview) resolved to a SINGLE file per mc-clip, so Doza transcribed
~10 minutes of the interview and the player showed a 9:38 timeline.

The parser now emits one ``SegmentAudioSource`` part per angle file
overlapping the mc-clip's window (``SpineSegment.audio_parts``), such a
project flips ``is_multi_source`` so ingest composes the timeline WAV, and
``plan_render`` emits one windowed entry per part. Parts hang UNDER the one
segment — the parser/writer 1:1 index lockstep is untouched, so the writer
needs no change: multi-source mc-clip selects are emitted in container time
and FCP resolves whichever internal file plays there.

The fixture is an ANONYMIZED copy of the reporting user's XML (names, paths,
angle IDs, and content-signature hashes replaced; every timing attribute
preserved verbatim): 4 angles (11+11+11 camera files, 10 recorder files),
the recorder angle active via mc-source, a leading 4.28s gap, non-zero
in-points, 720000-timescale audio assets. The expected numbers below were
cross-checked against the verbatim user XML — the fixture's render plan is
numerically identical to it.
"""

import os
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from lxml import etree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from doza_assist.fcpxml import (  # noqa: E402
    Select,
    parse_fcpxml,
    parse_rational,
    write_selects_as_new_project,
)
from doza_assist.fcpxml.timeline_audio import (  # noqa: E402
    build_ffmpeg_command,
    plan_render,
)


FIXTURE = Path(__file__).parent / "fixtures" / "multiseg_angle.fcpxml"

# The recorder angle's first two files, straight from the (anonymized) XML:
#   part 1: offset 3081600/720000s (4.28s), start 251831711/720000s, len 577.23s
#   part 2: offset 495417600/720000s (688.08s), start 57181283/720000s
PART1_SRC_START = Fraction(251831711, 720000)      # ~349.766s into file 1
PART1_DELAY_MS = 4280                              # the leading angle gap
PART2_SRC_START = Fraction(57181283, 720000)       # ~79.418s into file 2
PART2_DELAY_MS = 688080


@pytest.fixture(scope="module")
def parsed():
    return parse_fcpxml(FIXTURE)


class TestSegmentedAngleParsing:
    def test_one_segment_ten_parts(self, parsed):
        assert len(parsed.spine_segments) == 1
        seg = parsed.spine_segments[0]
        assert seg.kind == "mc-clip"
        assert len(seg.audio_parts) == 10
        # Ten asset-clip occurrences over SEVEN distinct recorder files — the
        # editor windows the same long recorder file more than once (duplicate
        # inputs are explicitly supported by build_ffmpeg_command).
        assert len({p.path for p in seg.audio_parts}) == 7
        offsets = [p.angle_offset_fraction for p in seg.audio_parts]
        assert offsets == sorted(offsets)

    def test_flips_multi_source(self, parsed):
        # Transcription must run on the composed timeline WAV (and selects in
        # timeline coordinates) — one file of ten is the field bug.
        assert parsed.is_multi_source

    def test_representative_is_first_part(self, parsed):
        # The window starts inside the leading gap (no part covers t=0), so
        # the representative falls back to the first overlapping part — the
        # same file the legacy single-pick resolved.
        seg = parsed.spine_segments[0]
        assert seg.audio_source is seg.audio_parts[0]
        assert parsed.audio_file_path.endswith("SRC_034.wav")


class TestSegmentedAnglePlan:
    def test_plan_emits_one_entry_per_part(self, parsed):
        plan = plan_render(parsed)
        assert len(plan) == 10
        assert all(e["segment_index"] == 0 for e in plan)

    def test_exact_part_windows(self, parsed):
        plan = plan_render(parsed)
        # Part 1: placed at its true timeline position AFTER the leading gap,
        # seeking to its recorder in-point. (The legacy math bled the gap:
        # src 345.49s at delay 0 — everything 4.28s early.)
        assert plan[0]["source_start_fraction"] == PART1_SRC_START
        assert plan[0]["timeline_offset_ms"] == PART1_DELAY_MS
        assert plan[0]["source_start_seconds"] == pytest.approx(349.766, abs=1e-3)
        assert plan[0]["duration_seconds"] == pytest.approx(577.234, abs=1e-3)
        # Part 2.
        assert plan[1]["source_start_fraction"] == PART2_SRC_START
        assert plan[1]["timeline_offset_ms"] == PART2_DELAY_MS
        assert plan[1]["source_start_seconds"] == pytest.approx(79.418, abs=1e-3)

    def test_last_part_clipped_to_segment_window(self, parsed):
        # The final recorder file runs a hair past the mc-clip's duration —
        # its render window must clip to the segment, not the file.
        plan = plan_render(parsed)
        assert plan[-1]["timeline_offset_ms"] == 9556640
        assert plan[-1]["duration_seconds"] == pytest.approx(293.08, abs=1e-3)
        end = plan[-1]["timeline_offset_fraction"] + plan[-1]["duration_fraction"]
        assert end == parsed.spine_segments[0].duration_fraction

    def test_ffmpeg_command_builds_with_all_parts(self, parsed):
        argv = build_ffmpeg_command(parsed, "/dev/null")
        assert argv.count("-i") == 11  # anullsrc base + 10 part inputs


class TestRoundTripSelect:
    def test_select_lands_mid_second_file(self, parsed):
        # Timeline 700s falls inside recorder file 2's window (688.08–961.4s).
        # Multi-source mc-clip selects are emitted in CONTAINER time — FCP
        # resolves which internal angle file plays — so the writer needs no
        # per-file logic.
        out = write_selects_as_new_project(
            parsed, [Select(start_seconds=700.0, end_seconds=710.0, label="Beat")],
        )
        root = etree.fromstring(out)
        clips = root.findall(".//project/sequence/spine/mc-clip")
        assert len(clips) == 1
        assert parse_rational(clips[0].get("start")) == 700
        assert parse_rational(clips[0].get("duration")) == 10
        # The source segment's angle enablement is replayed (recorder angle).
        ms = clips[0].find("mc-source")
        assert ms is not None
        assert ms.get("angleID") == "ANGLE_4"


SINGLE_FILE_ANGLE_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.13">
    <resources>
        <format id="r1" name="FFVideoFormat1080p25" frameDuration="100/2500s" width="1920" height="1080"/>
        <asset id="r2" name="cam" start="0s" duration="600s" hasVideo="1" hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000">
            <media-rep kind="original-media" src="file:///Volumes/EditDrive/Single/cam.mov"/>
        </asset>
        <media id="r3" name="MC">
            <multicam format="r1" tcStart="0s">
                <mc-angle name="A" angleID="A1">
                    <asset-clip name="cam" ref="r2" offset="0s" start="0s" duration="600s"/>
                </mc-angle>
            </multicam>
        </media>
    </resources>
    <library location="file:///Users/t/Movies/Single.fcpbundle/">
        <event name="Single">
            <project name="Single MC">
                <sequence format="r1" duration="300s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                    <spine>
                        <mc-clip ref="r3" offset="0s" name="MC" start="0s" duration="300s">
                            <mc-source angleID="A1" srcEnable="all"/>
                        </mc-clip>
                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>
"""


class TestSingleFileAngleRegression:
    """Single-file angles must behave byte-for-byte as before the part model:
    one part, no multi-source flip, the legacy plan shape — previously-ingested
    projects keep their stored-select coordinate convention."""

    @pytest.fixture
    def single(self, tmp_path):
        p = tmp_path / "single.fcpxml"
        p.write_text(SINGLE_FILE_ANGLE_FIXTURE)
        return parse_fcpxml(p)

    def test_one_part_identity(self, single):
        seg = single.spine_segments[0]
        assert len(seg.audio_parts) == 1
        assert seg.audio_parts[0] is seg.audio_source

    def test_multi_source_unchanged(self, single):
        assert not single.is_multi_source

    def test_legacy_plan_shape(self, single):
        plan = plan_render(single)
        assert len(plan) == 1
        assert plan[0]["timeline_offset_ms"] == 0
        assert plan[0]["source_start_fraction"] == 0
        assert plan[0]["duration_fraction"] == Fraction(300)
