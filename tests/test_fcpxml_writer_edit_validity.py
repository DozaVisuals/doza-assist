"""Regression tests for FCPXML *writer* edit validity (multicam/sync-clip round-trip).

The selects round-trip writer (``doza_assist.fcpxml.writer``) shares the same
edit-overshoot class fixed in the direct exporter: ``start`` and ``duration``
were snapped to the frame grid independently, so ``start + duration`` could land
one frame past the snapped out point — referencing media the source asset does
not have, which FCP rejects on import as "Invalid edit with no respective media".

``_snap_clip_times`` now derives duration from the snapped endpoints
(``duration = snapped_end - snapped_start``) so the out point is exactly the
snapped end and never overshoots; the timeline cursor advances by that same
snapped duration so the new spine stays frame-contiguous.
"""

import os
import sys
import textwrap
from fractions import Fraction

import pytest
from lxml import etree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from doza_assist.fcpxml import Select, parse_fcpxml, write_selects_as_new_project  # noqa: E402
from doza_assist.fcpxml.timecode import parse_rational  # noqa: E402
from doza_assist.fcpxml.writer import _snap_clip_times  # noqa: E402

FD = Fraction(1001, 24000)  # 23.976 fps frame duration


# ── unit: _snap_clip_times invariant ─────────────────────────────────

def test_duration_derived_from_snapped_endpoints_never_overshoots():
    # container_start = 10.6 frames, container_end = 20.4 frames.
    # Independent rounding (the OLD bug) would give:
    #   start = round(10.6) = 11, duration = round(20.4 - 10.6) = round(9.8) = 10
    #   => out = 11 + 10 = 21 frames, ONE FRAME PAST the snapped end (20).
    # The fix snaps both ends and takes duration = end - start:
    #   start = 11, end = round(20.4) = 20  => duration = 9, out = 20. No overshoot.
    start_str, dur_str, dur_frac = _snap_clip_times(
        Fraction(106, 10) * FD, Fraction(204, 10) * FD, FD
    )
    start = parse_rational(start_str)
    dur = parse_rational(dur_str)
    assert start == 11 * FD
    assert dur == 9 * FD            # NOT 10 frames (the independent-rounding result)
    assert start + dur == 20 * FD   # out point is exactly the snapped end
    assert dur_frac == 9 * FD


def test_snapped_times_are_frame_aligned():
    start_str, dur_str, _ = _snap_clip_times(Fraction(7, 3), Fraction(11, 3), FD)
    assert (parse_rational(start_str) / FD).denominator == 1
    assert (parse_rational(dur_str) / FD).denominator == 1


def test_sub_frame_select_gets_at_least_one_frame():
    # A zero-length (or sub-frame) select must still emit a >= 1-frame edit,
    # never a zero/negative duration that FCP would reject.
    _, dur_str, dur_frac = _snap_clip_times(Fraction(5) * FD, Fraction(5) * FD, FD)
    assert parse_rational(dur_str) == FD
    assert dur_frac == FD


# ── integration: the fix is actually wired through the writer ────────

# Single sync-clip, single-source (zero angle offsets) so container time equals
# the select's source time — lets us drive an exact frame-divergence case end to
# end. Media files need not exist; the parser only reads the XML.
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


@pytest.fixture
def parsed(tmp_path):
    p = tmp_path / "sync.fcpxml"
    p.write_text(SYNC_FIXTURE)
    return parse_fcpxml(p)


def test_writer_out_point_is_snapped_end_not_a_frame_past(parsed):
    # Drive the exact 10.6 -> 20.4 frame divergence through the full writer.
    start_s = float(Fraction(106, 10) * FD)   # 10.6 frames
    end_s = float(Fraction(204, 10) * FD)     # 20.4 frames
    out = write_selects_as_new_project(parsed, [
        Select(start_seconds=start_s, end_seconds=end_s, label="edge"),
    ])
    sc = etree.fromstring(out).find(".//spine/sync-clip")
    start = parse_rational(sc.get("start"))
    dur = parse_rational(sc.get("duration"))
    assert start == 11 * FD
    assert dur == 9 * FD                 # the fix: 9 frames, not the buggy 10
    assert start + dur == 20 * FD        # out point lands on the snapped end, never 21
    assert (start / FD).denominator == 1
    assert (dur / FD).denominator == 1


def test_writer_spine_is_frame_contiguous(parsed):
    selects = [
        Select(start_seconds=0.5, end_seconds=1.3,  label="a"),
        Select(start_seconds=2.0, end_seconds=3.7,  label="b"),
        Select(start_seconds=5.1, end_seconds=6.05, label="c"),
    ]
    out = write_selects_as_new_project(parsed, selects)
    clips = etree.fromstring(out).findall(".//spine/sync-clip")
    assert len(clips) == 3
    cursor = Fraction(0)
    for clip in clips:
        off = parse_rational(clip.get("offset"))
        dur = parse_rational(clip.get("duration"))
        assert off == cursor, "spine not contiguous (sub-frame gap/overlap)"
        cursor += dur


def test_writer_in_range_edits_stay_within_source(parsed):
    # Asset declares 240000/24000s of media; in-range edits must not reference
    # past it (out point = start + duration).
    asset_dur = Fraction(240000, 24000)
    selects = [
        Select(start_seconds=1.0, end_seconds=4.4,  label="a"),
        Select(start_seconds=6.7, end_seconds=9.95, label="b"),
    ]
    out = write_selects_as_new_project(parsed, selects)
    for clip in etree.fromstring(out).findall(".//spine/sync-clip"):
        start = parse_rational(clip.get("start"))
        dur = parse_rational(clip.get("duration"))
        assert start + dur <= asset_dur, "edit out point exceeds source media"
