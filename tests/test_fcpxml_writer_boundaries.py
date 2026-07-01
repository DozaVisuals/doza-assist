"""Boundary-split / coordinate-space regressions for the round-trip writer.

Covers the 1.0.27 export-hardening cluster:

- cross-source boundary selects split into per-segment pieces instead of
  aborting the whole export (the live 1.0.26 'purple' failure);
- same-source collapse only when the span is container-CONTIGUOUS on
  multi-source timelines (a jump cut must not export the removed footage);
- single-source select matching only against segments sharing the
  representative coordinate space (foreign-asset lane B-roll must not
  swallow interview selects);
- snapped out-points clamped to the segment/media end (sample-aligned
  boundaries must not overshoot into FCP's "Invalid edit" rejection);
- gap-start selects keep their covered tail;
- zero-length selects surfaced through skipped_out.
"""

import os
import sys
import textwrap
from fractions import Fraction

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
from doza_assist.fcpxml.timecode import parse_rational  # noqa: E402


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return str(p)


def _spine_clips(output_bytes):
    root = etree.fromstring(output_bytes)
    spine = root.find(".//sequence/spine")
    return [c for c in spine if c.tag in ("mc-clip", "sync-clip", "asset-clip")]


# ---------- multi-source jump cut: collapse vs split ------------------------

# camA is jump-cut: source 60-120s removed at the clip1|clip2 join. camB makes
# the project multi-source (select times = timeline seconds).
JUMPCUT_FIXTURE = """\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.11">
        <resources>
            <format id="r1" frameDuration="1/30s" width="1920" height="1080"/>
            <asset id="r2" name="camA" start="0s" duration="600s" hasVideo="1" hasAudio="1">
                <media-rep kind="original-media" src="file:///tmp/camA.mov"/>
            </asset>
            <asset id="r3" name="camB" start="0s" duration="600s" hasVideo="1" hasAudio="1">
                <media-rep kind="original-media" src="file:///tmp/camB.mov"/>
            </asset>
        </resources>
        <library><event name="E"><project name="Jump">
            <sequence format="r1" duration="180s" tcStart="0s" tcFormat="NDF">
                <spine>
                    <asset-clip name="A1" ref="r2" offset="0s" start="0s" duration="60s"/>
                    <asset-clip name="A2" ref="r2" offset="60s" start="120s" duration="60s"/>
                    <asset-clip name="B" ref="r3" offset="120s" start="0s" duration="60s"/>
                </spine>
            </sequence>
        </project></event></library>
    </fcpxml>
"""

# Same shape but A2 is container-contiguous with A1 (no material removed).
CONTIGUOUS_FIXTURE = JUMPCUT_FIXTURE.replace(
    '<asset-clip name="A2" ref="r2" offset="60s" start="120s" duration="60s"/>',
    '<asset-clip name="A2" ref="r2" offset="60s" start="60s" duration="60s"/>',
)


class TestMultiSourceJumpCut:
    def test_jump_cut_select_splits_at_the_join(self, tmp_path):
        parsed = parse_fcpxml(_write(tmp_path, "jump.fcpxml", JUMPCUT_FIXTURE))
        assert parsed.is_multi_source
        out = write_selects_as_new_project(parsed, [
            Select(start_seconds=55.0, end_seconds=65.0, label="beat"),
        ])
        clips = _spine_clips(out)
        # Two pieces of camA: source [55,60) then source [120,125) — NOT one
        # collapsed clip playing the removed source [60,65).
        assert len(clips) == 2
        assert [c.get("ref") for c in clips] == ["r2", "r2"]
        s0 = float(parse_rational(clips[0].get("start")))
        s1 = float(parse_rational(clips[1].get("start")))
        assert s0 == pytest.approx(55.0, abs=0.05)
        assert s1 == pytest.approx(120.0, abs=0.05)
        assert float(parse_rational(clips[0].get("duration"))) == pytest.approx(5.0, abs=0.05)
        assert float(parse_rational(clips[1].get("duration"))) == pytest.approx(5.0, abs=0.05)

    def test_contiguous_same_source_still_collapses(self, tmp_path):
        parsed = parse_fcpxml(_write(tmp_path, "contig.fcpxml", CONTIGUOUS_FIXTURE))
        assert parsed.is_multi_source
        out = write_selects_as_new_project(parsed, [
            Select(start_seconds=55.0, end_seconds=65.0, label="beat"),
        ])
        clips = _spine_clips(out)
        assert len(clips) == 1
        assert float(parse_rational(clips[0].get("start"))) == pytest.approx(55.0, abs=0.05)
        assert float(parse_rational(clips[0].get("duration"))) == pytest.approx(10.0, abs=0.05)

    def test_heterogeneous_boundary_still_splits(self, tmp_path):
        parsed = parse_fcpxml(_write(tmp_path, "jump2.fcpxml", JUMPCUT_FIXTURE))
        out = write_selects_as_new_project(parsed, [
            Select(start_seconds=115.0, end_seconds=125.0, label="cross"),
        ])
        clips = _spine_clips(out)
        assert [c.get("ref") for c in clips] == ["r2", "r3"]
        # camA piece plays source [175,180); camB piece plays source [0,5).
        assert float(parse_rational(clips[0].get("start"))) == pytest.approx(175.0, abs=0.05)
        assert float(parse_rational(clips[1].get("start"))) == pytest.approx(0.0, abs=0.05)

    def test_pieces_contiguous_on_new_timeline(self, tmp_path):
        parsed = parse_fcpxml(_write(tmp_path, "jump3.fcpxml", JUMPCUT_FIXTURE))
        out = write_selects_as_new_project(parsed, [
            Select(start_seconds=10.0, end_seconds=20.0, label="plain"),
            Select(start_seconds=55.0, end_seconds=65.0, label="split"),
        ])
        cursor = Fraction(0)
        for c in _spine_clips(out):
            assert parse_rational(c.get("offset")) == cursor
            cursor += parse_rational(c.get("duration"))


# ---------- single-source: foreign-asset lane B-roll must not match ---------

# Primary camA is TRIMMED (only source [50,60) is on the timeline). The lane-1
# B-roll (different asset) sits in the gap with its own source range [0,10) —
# numerically overlapping camA's untimelined head. Single-source per the
# migration guard, so select times are camA-file seconds.
TRIMMED_LANE_FIXTURE = """\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.11">
        <resources>
            <format id="r1" frameDuration="1/25s" width="1920" height="1080"/>
            <asset id="r2" name="camA" start="0s" duration="600s" hasAudio="1">
                <media-rep kind="original-media" src="file:///tmp/camA.mov"/>
            </asset>
            <asset id="r3" name="broll" start="0s" duration="600s" hasVideo="1" hasAudio="1">
                <media-rep kind="original-media" src="file:///tmp/broll.mov"/>
            </asset>
        </resources>
        <library><event name="E"><project name="Trimmed">
            <sequence format="r1" duration="20s" tcStart="0s" tcFormat="NDF">
                <spine>
                    <asset-clip name="A" ref="r2" offset="0s" start="50s" duration="10s"/>
                    <gap name="Gap" offset="10s" start="3600s" duration="10s">
                        <asset-clip name="Broll" ref="r3" lane="1" offset="3600s" start="0s" duration="10s"/>
                    </gap>
                </spine>
            </sequence>
        </project></event></library>
    </fcpxml>
"""


class TestSingleSourceLaneCoordinates:
    @pytest.fixture
    def parsed(self, tmp_path):
        return parse_fcpxml(_write(tmp_path, "trimmed.fcpxml", TRIMMED_LANE_FIXTURE))

    def test_fixture_is_single_source(self, parsed):
        assert not parsed.is_multi_source

    def test_untimelined_head_select_skipped_not_broll(self, parsed):
        # camA source 5s is NOT on the timeline. The B-roll's own-asset range
        # [0,10) numerically covers 5 but is a different file — matching it
        # exported B-roll footage in place of the interview.
        skipped = []
        out = write_selects_as_new_project(parsed, [
            Select(start_seconds=5.0, end_seconds=8.0, label="head"),
            Select(start_seconds=52.0, end_seconds=58.0, label="body"),
        ], skipped_out=skipped)
        assert [s.label for s in skipped] == ["head"]
        clips = _spine_clips(out)
        assert [c.get("ref") for c in clips] == ["r2"]
        assert float(parse_rational(clips[0].get("start"))) == pytest.approx(52.0, abs=0.05)

    def test_mode_b_marker_never_lands_on_foreign_lane_clip(self, parsed):
        skipped = []
        out = write_markers_on_timeline(parsed, [
            Select(start_seconds=5.0, end_seconds=6.0, label="head"),
            Select(start_seconds=52.0, end_seconds=53.0, label="body"),
        ], skipped_out=skipped)
        assert [s.label for s in skipped] == ["head"]
        root = etree.fromstring(out)
        broll_clips = [c for c in root.iter("asset-clip") if c.get("ref") == "r3"]
        assert all(c.find("marker") is None for c in broll_clips)
        cam_markers = [m for c in root.iter("asset-clip") if c.get("ref") == "r2"
                       for m in c.findall("marker")]
        assert len(cam_markers) == 1


# ---------- Mode B: marker DTD position relative to <sync-source> -----------

SYNC_SOURCE_FIXTURE = """\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.11">
        <resources>
            <format id="r1" frameDuration="1001/24000s" width="1920" height="1080"/>
            <asset id="r2" name="cam" start="0s" duration="600s" hasVideo="1" hasAudio="1">
                <media-rep kind="original-media" src="file:///tmp/cam.mov"/>
            </asset>
            <asset id="r3" name="ext" start="0s" duration="600s" hasAudio="1">
                <media-rep kind="original-media" src="file:///tmp/ext.wav"/>
            </asset>
        </resources>
        <library><event name="E"><project name="Synced">
            <sequence format="r1" duration="60s" tcStart="0s" tcFormat="NDF">
                <spine>
                    <sync-clip name="Interview" offset="0s" duration="60s">
                        <spine>
                            <asset-clip name="cam" ref="r2" offset="0s" start="0s" duration="60s">
                                <asset-clip name="ext" ref="r3" lane="-1" offset="0s" start="0s" duration="60s" audioRole="dialogue"/>
                            </asset-clip>
                        </spine>
                        <sync-source sourceID="connected">
                            <audio-role-source role="dialogue"/>
                        </sync-source>
                    </sync-clip>
                </spine>
            </sequence>
        </project></event></library>
    </fcpxml>
"""


class TestModeBMarkerOrder:
    def test_marker_inserted_before_sync_source(self, tmp_path):
        parsed = parse_fcpxml(_write(tmp_path, "sync.fcpxml", SYNC_SOURCE_FIXTURE))
        out = write_markers_on_timeline(parsed, [
            Select(start_seconds=5.0, end_seconds=6.0, label="note here"),
        ])
        root = etree.fromstring(out)
        sc = root.find(".//sequence/spine/sync-clip")
        tags = [c.tag for c in sc]
        assert "marker" in tags and "sync-source" in tags
        # DTD content model: (marker | ...)* comes BEFORE sync-source*.
        assert tags.index("marker") < tags.index("sync-source")

    def test_marker_label_with_control_char_is_scrubbed_not_fatal(self, tmp_path):
        parsed = parse_fcpxml(_write(tmp_path, "sync2.fcpxml", SYNC_SOURCE_FIXTURE))
        out = write_markers_on_timeline(parsed, [
            Select(start_seconds=5.0, end_seconds=6.0, label="bad\x0cchar"),
        ])
        root = etree.fromstring(out)
        m = root.find(".//sync-clip/marker")
        assert m is not None
        assert "\x0c" not in (m.get("value") or "")


# ---------- snapped out-point clamped to sample-aligned media ends ----------

SAMPLE_ALIGNED_FIXTURE = """\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.11">
        <resources>
            <format id="r1" frameDuration="1001/24000s" width="1920" height="1080"/>
            <asset id="r2" name="tail" start="0s" duration="1441/600s" hasAudio="1">
                <media-rep kind="original-media" src="file:///tmp/tail.wav"/>
            </asset>
        </resources>
        <library><event name="E"><project name="Tail">
            <sequence format="r1" duration="1441/600s" tcStart="0s" tcFormat="NDF">
                <spine>
                    <asset-clip name="tail" ref="r2" offset="0s" start="0s" duration="1441/600s"/>
                </spine>
            </sequence>
        </project></event></library>
    </fcpxml>
"""


class TestSnapClampAtMediaTail:
    def test_select_to_media_tail_never_overshoots(self, tmp_path):
        parsed = parse_fcpxml(_write(tmp_path, "tail.fcpxml", SAMPLE_ALIGNED_FIXTURE))
        # Whisper's last segment ends exactly at the audio duration
        # (2.401666...s — sample-aligned, not frame-aligned at 23.976).
        out = write_selects_as_new_project(parsed, [
            Select(start_seconds=0.0, end_seconds=float(Fraction(1441, 600)), label="all"),
        ])
        clips = _spine_clips(out)
        assert len(clips) == 1
        start = parse_rational(clips[0].get("start"))
        dur = parse_rational(clips[0].get("duration"))
        # Out point must sit at or before the media end — nearest-frame
        # rounding used to overshoot by +0.017s and FCP rejected the edit.
        assert start + dur <= Fraction(1441, 600)
        assert dur >= parsed.sequence_frame_duration


# ---------- gap-start selects keep their covered tail ------------------------

LEADING_GAP_FIXTURE = """\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.11">
        <resources>
            <format id="r1" frameDuration="1/25s" width="1920" height="1080"/>
            <asset id="r2" name="camA" start="0s" duration="600s" hasAudio="1">
                <media-rep kind="original-media" src="file:///tmp/camA.mov"/>
            </asset>
            <asset id="r3" name="camB" start="0s" duration="600s" hasAudio="1">
                <media-rep kind="original-media" src="file:///tmp/camB.mov"/>
            </asset>
        </resources>
        <library><event name="E"><project name="Lead">
            <sequence format="r1" duration="130s" tcStart="0s" tcFormat="NDF">
                <spine>
                    <gap name="Gap" offset="0s" start="3600s" duration="10s"/>
                    <asset-clip name="A" ref="r2" offset="10s" start="0s" duration="60s"/>
                    <asset-clip name="B" ref="r3" offset="70s" start="0s" duration="60s"/>
                </spine>
            </sequence>
        </project></event></library>
    </fcpxml>
"""


class TestGapStartSelect:
    def test_select_starting_in_gap_keeps_covered_tail(self, tmp_path):
        parsed = parse_fcpxml(_write(tmp_path, "lead.fcpxml", LEADING_GAP_FIXTURE))
        assert parsed.is_multi_source
        skipped = []
        out = write_selects_as_new_project(parsed, [
            # Whisper word-start drift: starts 0.4s before the first cut.
            Select(start_seconds=9.6, end_seconds=25.0, label="hook"),
        ], skipped_out=skipped)
        assert skipped == []
        clips = _spine_clips(out)
        assert len(clips) == 1
        assert clips[0].get("ref") == "r2"
        # Covered tail = timeline [10, 25) -> camA source [0, 15).
        assert float(parse_rational(clips[0].get("start"))) == pytest.approx(0.0, abs=0.05)
        assert float(parse_rational(clips[0].get("duration"))) == pytest.approx(15.0, abs=0.05)

    def test_fully_uncovered_select_still_skipped(self, tmp_path):
        parsed = parse_fcpxml(_write(tmp_path, "lead2.fcpxml", LEADING_GAP_FIXTURE))
        skipped = []
        out = write_selects_as_new_project(parsed, [
            Select(start_seconds=2.0, end_seconds=8.0, label="in gap"),
            Select(start_seconds=20.0, end_seconds=30.0, label="fine"),
        ], skipped_out=skipped)
        assert [s.label for s in skipped] == ["in gap"]
        assert len(_spine_clips(out)) == 1


# ---------- zero-length selects surfaced -------------------------------------

class TestZeroLengthSelects:
    def test_zero_length_select_lands_in_skipped_out(self, tmp_path):
        parsed = parse_fcpxml(_write(tmp_path, "z.fcpxml", JUMPCUT_FIXTURE))
        skipped = []
        out = write_selects_as_new_project(parsed, [
            Select(start_seconds=10.0, end_seconds=10.0, label="stray click"),
            Select(start_seconds=20.0, end_seconds=30.0, label="fine"),
        ], skipped_out=skipped)
        assert [s.label for s in skipped] == ["stray click"]
        assert len(_spine_clips(out)) == 1

    def test_all_zero_length_gets_specific_error(self, tmp_path):
        parsed = parse_fcpxml(_write(tmp_path, "z2.fcpxml", JUMPCUT_FIXTURE))
        with pytest.raises(WriterError, match="zero length"):
            write_selects_as_new_project(parsed, [
                Select(start_seconds=10.0, end_seconds=10.0, label="a"),
            ])

    def test_mode_b_zero_length_also_surfaced(self, tmp_path):
        parsed = parse_fcpxml(_write(tmp_path, "z3.fcpxml", JUMPCUT_FIXTURE))
        skipped = []
        write_markers_on_timeline(parsed, [
            Select(start_seconds=10.0, end_seconds=10.0, label="stray"),
            Select(start_seconds=20.0, end_seconds=21.0, label="fine"),
        ], skipped_out=skipped)
        assert [s.label for s in skipped] == ["stray"]


# ---------- review round 2: collapse clamp, dup-multicam, snap bounds --------

# One continuous recording FCP split into two asset-clips whose summed length
# IS the asset's full (sample-aligned) duration — a select to the media tail
# used to overshoot after the same-source collapse skipped the clamp.
SPLIT_TAIL_FIXTURE = """\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.11">
        <resources>
            <format id="r1" frameDuration="1001/24000s" width="1920" height="1080"/>
            <asset id="r2" name="rec" start="0s" duration="117700/24000s" hasAudio="1">
                <media-rep kind="original-media" src="file:///tmp/rec.wav"/>
            </asset>
        </resources>
        <library><event name="E"><project name="SplitTail">
            <sequence format="r1" duration="117700/24000s" tcStart="0s" tcFormat="NDF">
                <spine>
                    <asset-clip name="A1" ref="r2" offset="0s" start="0s" duration="60060/24000s"/>
                    <asset-clip name="A2" ref="r2" offset="60060/24000s" start="60060/24000s" duration="57640/24000s"/>
                </spine>
            </sequence>
        </project></event></library>
    </fcpxml>
"""


class TestCollapseClampAtMediaTail:
    def test_collapsed_span_to_media_tail_never_overshoots(self, tmp_path):
        parsed = parse_fcpxml(_write(tmp_path, "split_tail.fcpxml", SPLIT_TAIL_FIXTURE))
        assert not parsed.is_multi_source
        media_end = Fraction(117700, 24000)
        out = write_selects_as_new_project(parsed, [
            Select(start_seconds=1.0, end_seconds=float(media_end), label="all"),
        ])
        clips = _spine_clips(out)
        assert len(clips) == 1  # same-source collapse still applies
        start = parse_rational(clips[0].get("start"))
        dur = parse_rational(clips[0].get("duration"))
        assert start + dur <= media_end


# FCP "Duplicate" of a multicam: a second <media> id over the SAME angle
# assets. Selects on the copy's clips must still resolve (same audio asset =>
# same container coordinates), not silently skip.
DUP_MULTICAM_FIXTURE = """\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.11">
        <resources>
            <format id="r1" frameDuration="1/30s" width="1920" height="1080"/>
            <asset id="rA" name="audio" start="0s" duration="600s" hasAudio="1">
                <media-rep kind="original-media" src="file:///tmp/audio.wav"/>
            </asset>
            <asset id="rV" name="video" start="0s" duration="600s" hasVideo="1">
                <media-rep kind="original-media" src="file:///tmp/video.mov"/>
            </asset>
            <media id="mc1" name="MC">
                <multicam>
                    <mc-angle name="V" angleID="v1"><asset-clip ref="rV" offset="0s" duration="600s"/></mc-angle>
                    <mc-angle name="A" angleID="a1"><asset-clip ref="rA" offset="0s" duration="600s" audioRole="dialogue"/></mc-angle>
                </multicam>
            </media>
            <media id="mc2" name="MC copy">
                <multicam>
                    <mc-angle name="V" angleID="v1"><asset-clip ref="rV" offset="0s" duration="600s"/></mc-angle>
                    <mc-angle name="A" angleID="a1"><asset-clip ref="rA" offset="0s" duration="600s" audioRole="dialogue"/></mc-angle>
                </multicam>
            </media>
        </resources>
        <library><event name="E"><project name="Dup">
            <sequence format="r1" duration="120s" tcStart="0s" tcFormat="NDF">
                <spine>
                    <mc-clip ref="mc1" offset="0s" name="M1" duration="60s">
                        <mc-source angleID="v1" srcEnable="video"/>
                        <mc-source angleID="a1" srcEnable="audio"/>
                    </mc-clip>
                    <mc-clip ref="mc2" offset="60s" name="M2" start="60s" duration="60s">
                        <mc-source angleID="v1" srcEnable="video"/>
                        <mc-source angleID="a1" srcEnable="audio"/>
                    </mc-clip>
                </spine>
            </sequence>
        </project></event></library>
    </fcpxml>
"""


class TestDuplicatedMulticam:
    def test_select_on_duplicate_multicam_still_exports(self, tmp_path):
        parsed = parse_fcpxml(_write(tmp_path, "dup.fcpxml", DUP_MULTICAM_FIXTURE))
        assert not parsed.is_multi_source
        skipped = []
        out = write_selects_as_new_project(parsed, [
            Select(start_seconds=70.0, end_seconds=75.0, label="on M2"),
        ], skipped_out=skipped)
        assert skipped == []
        clips = _spine_clips(out)
        assert len(clips) == 1
        assert clips[0].get("ref") == "mc2"
        assert abs(float(parse_rational(clips[0].get("start"))) - 70.0) < 0.05


class TestSnapClipTimesBounds:
    """Unit coverage of the snap clamp — the media bounds passed per piece."""

    def _snap(self, *a, **k):
        from doza_assist.fcpxml.writer import _snap_clip_times
        return _snap_clip_times(*a, **k)

    def test_start_below_min_start_bumps_up_to_grid(self):
        fd = Fraction(1001, 24000)
        min_start = Fraction(239, 100)  # 2.39s — not frame-aligned
        # 2.39 rounds DOWN to 57 frames (2.377) — before the media in-point.
        start_str, dur_str, _ = self._snap(
            Fraction(239, 100), Fraction(339, 100), fd, None, min_start)
        from doza_assist.fcpxml.timecode import parse_rational as pr
        assert pr(start_str) >= min_start

    def test_one_frame_minimum_never_dips_below_min_start(self):
        fd = Fraction(1001, 24000)
        min_start = Fraction(10)
        max_end = Fraction(10) + fd / 2  # half a frame of media
        start_str, dur_str, _ = self._snap(
            Fraction(10), Fraction(10) + fd / 2, fd, max_end, min_start)
        from doza_assist.fcpxml.timecode import parse_rational as pr
        assert pr(start_str) >= min_start
        assert pr(dur_str) >= fd  # still a legal 1-frame edit
