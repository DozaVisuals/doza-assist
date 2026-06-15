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
from doza_assist.fcpxml.writer import _set_clip_note, _snap_clip_times  # noqa: E402

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


# ── DTD child-order validity: <note> must lead, before <conform-rate> ────────
#
# FCPXML's content model for every clip kind we copy here leads with
#   (note?, (conform-rate?, timeMap?), …)
# so the select note must be inserted BEFORE a leading <conform-rate>/<timeMap>.
# Rate-conformed sources (e.g. 25fps footage dropped into a 23.976 timeline)
# carry a <conform-rate> child; the old writer inserted the note AFTER it,
# producing <conform-rate/><note/>, which FCP rejects on import with
# "Element asset-clip content does not follow the DTD, expecting
#  (note?, (conform-rate?, timeMap?), …)".

def test_set_clip_note_precedes_leading_conform_rate():
    clip = etree.fromstring(
        '<asset-clip ref="r2" start="0s" duration="100s">'
        '<conform-rate srcFrameRate="25"/>'
        '<keyword start="0s" duration="100s" value="k"/>'
        '</asset-clip>'
    )
    _set_clip_note(clip, "Opening Hook — Jack")
    tags = [c.tag for c in clip]
    assert tags[0] == "note", f"<note> must be first child, got order {tags}"
    assert tags.index("note") < tags.index("conform-rate"), (
        "<note> must precede <conform-rate> per the FCPXML DTD"
    )


def test_set_clip_note_replaces_inherited_note_at_front():
    # An inherited note is dropped (0-or-1 in the DTD) and ours lands first,
    # still before conform-rate.
    clip = etree.fromstring(
        '<asset-clip ref="r2" start="0s" duration="100s">'
        '<note>old</note>'
        '<conform-rate srcFrameRate="25"/>'
        '</asset-clip>'
    )
    _set_clip_note(clip, "new")
    notes = clip.findall("note")
    assert len(notes) == 1 and notes[0].text == "new"
    assert list(clip)[0].tag == "note"


# Single-source plain <asset-clip> spine whose clip is rate-conformed (25fps
# source in a 23.976 timeline) — exactly the shape that broke a real NEC export.
CONFORM_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.14">
        <resources>
            <format id="r1" name="FFVideoFormat1080p2398" frameDuration="1001/24000s" width="1920" height="1080"/>
            <format id="r3" name="FFVideoFormat1080p25" frameDuration="1/25s" width="1920" height="1080"/>
            <asset id="r2" name="rebecca" start="0s" duration="240000/24000s" hasVideo="1" hasAudio="1" videoSources="1" audioSources="1" audioChannels="2" audioRate="48000" format="r3">
                <media-rep kind="original-media" src="file:///tmp/rebecca.mov"/>
            </asset>
        </resources>
        <library>
            <event name="E">
                <project name="Conform Test">
                    <sequence format="r1" duration="240000/24000s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                        <spine>
                            <asset-clip ref="r2" offset="0s" name="Rebecca" start="0s" duration="240000/24000s" format="r3" tcFormat="NDF" audioRole="dialogue">
                                <conform-rate srcFrameRate="25"/>
                            </asset-clip>
                        </spine>
                    </sequence>
                </project>
            </event>
        </library>
    </fcpxml>
""")


@pytest.fixture
def conform_parsed(tmp_path):
    p = tmp_path / "conform.fcpxml"
    p.write_text(CONFORM_FIXTURE)
    return parse_fcpxml(p)


def test_writer_note_before_conform_rate_through_full_writer(conform_parsed):
    out = write_selects_as_new_project(conform_parsed, [
        Select(start_seconds=1.0, end_seconds=3.0, label="green",
               note="for me, NEC Prep was kind of where I found"),
    ])
    clip = etree.fromstring(out).find(".//spine/asset-clip")
    assert clip is not None
    tags = [c.tag for c in clip]
    assert tags[0] == "note", f"<note> must lead the asset-clip, got {tags}"
    assert tags.index("note") < tags.index("conform-rate"), (
        "DTD-invalid: <conform-rate> emitted before <note>"
    )
