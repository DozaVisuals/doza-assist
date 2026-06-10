"""Lane-1 B-roll-gap regression: connected clips inside primary-storyline
gaps are real timeline content. The parser used to skip <gap> children
entirely, so selects landing in a gap were silently dropped from Mode A
exports, never marked in Mode B, and the gap rendered as silence in the
ingest WAV (dialogue never transcribed)."""

import os
import sys
import textwrap

import pytest
from lxml import etree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from doza_assist.fcpxml.parser import parse_fcpxml  # noqa: E402
from doza_assist.fcpxml.writer import (  # noqa: E402
    Select, write_selects_as_new_project, write_markers_on_timeline,
)
from doza_assist.fcpxml.timeline_audio import plan_render  # noqa: E402


def _fixture(tmp_path, broll_ref="r3"):
    """A[0-10s) / gap[10-20s) carrying lane-1 B-roll at 12-20s / C[20-30s)."""
    fcpxml = tmp_path / "lane_gap.fcpxml"
    fcpxml.write_text(textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE fcpxml>
        <fcpxml version="1.11">
            <resources>
                <format id="r1" frameDuration="1001/24000s" width="1920" height="1080"/>
                <asset id="r2" name="camA" start="0s" duration="600s" hasVideo="1" hasAudio="1">
                    <media-rep kind="original-media" src="file:///tmp/camA.mov"/>
                </asset>
                <asset id="r3" name="broll" start="0s" duration="600s" hasVideo="1" hasAudio="1">
                    <media-rep kind="original-media" src="file:///tmp/broll.mov"/>
                </asset>
                <asset id="r4" name="camC" start="0s" duration="600s" hasVideo="1" hasAudio="1">
                    <media-rep kind="original-media" src="file:///tmp/camC.mov"/>
                </asset>
            </resources>
            <library>
                <event name="Lane Gap">
                    <project name="Lane Gap">
                        <sequence format="r1" duration="720720/24000s" tcStart="0s" tcFormat="NDF">
                            <spine>
                                <asset-clip name="A" ref="r2" offset="0s" start="0s" duration="240240/24000s" format="r1"/>
                                <gap name="Gap" offset="240240/24000s" start="3600s" duration="240240/24000s">
                                    <asset-clip name="Broll" ref="{broll_ref}" lane="1" offset="3602s" start="120120/24000s" duration="192192/24000s" format="r1"/>
                                </gap>
                                <asset-clip name="C" ref="r4" offset="480480/24000s" start="0s" duration="240240/24000s" format="r1"/>
                            </spine>
                        </sequence>
                    </project>
                </event>
            </library>
        </fcpxml>
    """))
    return str(fcpxml)


class TestLaneGapParsing:
    def test_three_segments_with_lane_metadata(self, tmp_path):
        parsed = parse_fcpxml(_fixture(tmp_path))
        assert len(parsed.spine_segments) == 3
        a, broll, c = parsed.spine_segments
        assert (a.name, a.lane) == ("A", "")
        assert (broll.name, broll.lane) == ("Broll", "1")
        assert (c.name, c.lane) == ("C", "")
        # gap-local anchor: gap@10.01s (240 NTSC frames), clip offset
        # 3602s, gap start 3600s -> 10.01 + 2 = 12.01s
        assert float(broll.offset_fraction) == pytest.approx(12.01)
        # three distinct primary sources? A + C only decide multi-source
        assert parsed.is_multi_source  # A vs C differ -> timeline coords

    def test_lane_segment_excluded_from_multi_source_basis(self, tmp_path):
        # One primary + lane B-roll from another source must stay
        # single-source: re-parses of old ingests must not flip select
        # coordinate semantics (migration guard).
        fcpxml = tmp_path / "single.fcpxml"
        fcpxml.write_text(textwrap.dedent("""\
            <?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE fcpxml>
            <fcpxml version="1.11">
                <resources>
                    <format id="r1" frameDuration="1001/24000s" width="1920" height="1080"/>
                    <asset id="r2" name="camA" start="0s" duration="600s" hasAudio="1">
                        <media-rep kind="original-media" src="file:///tmp/camA.mov"/>
                    </asset>
                    <asset id="r3" name="broll" start="0s" duration="600s" hasAudio="1">
                        <media-rep kind="original-media" src="file:///tmp/broll.mov"/>
                    </asset>
                </resources>
                <library><event name="E"><project name="P">
                    <sequence format="r1" duration="480480/24000s" tcStart="0s" tcFormat="NDF">
                        <spine>
                            <asset-clip name="A" ref="r2" offset="0s" start="0s" duration="240240/24000s"/>
                            <gap name="Gap" offset="240240/24000s" start="3600s" duration="240240/24000s">
                                <asset-clip name="Broll" ref="r3" lane="1" offset="3600s" start="0s" duration="240240/24000s"/>
                            </gap>
                        </spine>
                    </sequence>
                </project></event></library>
            </fcpxml>
        """))
        parsed = parse_fcpxml(str(fcpxml))
        assert len(parsed.spine_segments) == 2
        assert not parsed.is_multi_source

    def test_plan_render_covers_the_gap(self, tmp_path):
        parsed = parse_fcpxml(_fixture(tmp_path))
        plan = plan_render(parsed)
        offsets = sorted(round(p["timeline_offset_ms"]) for p in plan)
        assert offsets == [0, 12010, 20020]  # gap B-roll audio is rendered


class TestLaneGapModeA:
    def test_gap_select_is_emitted_lane_stripped(self, tmp_path):
        parsed = parse_fcpxml(_fixture(tmp_path))
        skipped = []
        out = write_selects_as_new_project(
            parsed,
            [
                Select(start_seconds=1.0, end_seconds=2.0, label="In A"),
                Select(start_seconds=12.5, end_seconds=15.0, label="Gap Broll"),
            ],
            skipped_out=skipped,
        )
        assert skipped == []  # the gap select used to be silently dropped
        root = etree.fromstring(out)
        spine = root.find(".//project/sequence/spine")
        clips = list(spine)
        assert len(clips) == 2
        broll_clip = next(c for c in clips if c.get("ref") == "r3")
        assert broll_clip.get("lane") is None  # stripped on copy
        # in-point: clip start 5.005s + (12.5s - 12.01s anchor) = 5.495s
        num, den = broll_clip.get("start").rstrip("s").split("/")
        assert int(num) / int(den) == pytest.approx(5.495, abs=0.05)


class TestLaneGapModeB:
    def test_marker_lands_inside_the_lane_clip(self, tmp_path):
        parsed = parse_fcpxml(_fixture(tmp_path))
        skipped = []
        out = write_markers_on_timeline(
            parsed,
            [Select(start_seconds=13.0, end_seconds=14.0, label="Gap note")],
            skipped_out=skipped,
        )
        assert skipped == []
        root = etree.fromstring(out)
        gap = root.find(".//project/sequence/spine/gap")
        lane_clip = gap.find("asset-clip")
        markers = lane_clip.findall("marker")
        assert len(markers) == 1
        assert markers[0].get("value") == "Gap note"
