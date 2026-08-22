"""Spine-level ``<clip>`` import support.

Field bug (2026-08-19, trial user): a timeline whose spine holds ``<clip>``
wrappers — Resolve's connected-clip form, and the exact shape our OWN flat
exporter emits when it routes a detected dialogue channel
(``fcpxml_export._spine_clip``) — was rejected with "spine has no clips Doza
Assist can read". ``<clip>`` is now a first-class spine segment tag, so the
shared parser/writer walker also lifts lane-connected ``<clip>`` B-roll out
of primary-storyline gaps for free.

Coordinate mapping under test (the container convention the renderer/writer
already use): ``seek = seg.start - angle_offset + (angle_start - asset_start)``
with the clip's media child supplying angle_offset (clip-local position) and
angle_start (asset-local in-point).
"""

import os
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from lxml import etree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from doza_assist.fcpxml import (  # noqa: E402
    ParseError,
    Select,
    parse_fcpxml,
    write_selects_as_new_project,
)
from doza_assist.fcpxml.timeline_audio import _segment_source_window  # noqa: E402
from doza_assist.fcpxml.writer import re_parse  # noqa: E402

import fcpxml_export  # noqa: E402


# The basic connected-clip shape: a spine <clip> windowing a <video> over the
# asset's full span, dialogue on a nested lane <audio>.
BASIC_CLIP_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.13">
    <resources>
        <format id="r1" name="FFVideoFormat1080p30" frameDuration="1/30s" width="1920" height="1080"/>
        <asset id="r2" name="drone_001" start="0s" duration="100s" hasVideo="1" hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000">
            <media-rep kind="original-media" src="file:///Volumes/MyDrive/Drone/drone_001.mp4"/>
        </asset>
    </resources>
    <library location="file:///Users/t/Movies/Drone.fcpbundle/">
        <event name="Drone">
            <project name="Clip Spine">
                <sequence format="r1" duration="40s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                    <spine>
                        <clip name="drone_001" offset="0s" start="10s" duration="40s" format="r1" tcFormat="NDF" enabled="1">
                            <video ref="r2" offset="0s" start="0s" duration="100s">
                                <audio lane="-1" ref="r2" offset="0s" start="0s" duration="100s" role="dialogue"/>
                            </video>
                        </clip>
                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>
"""

# Offset-mapping variant: the media child sits 5s into the clip's local
# timeline with an asset-local in-point on TC-carrying media (asset start
# 3600s). seek = 10 - 5 + (3605 - 3600) = 10s.
OFFSET_CLIP_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.13">
    <resources>
        <format id="r1" name="FFVideoFormat1080p30" frameDuration="1/30s" width="1920" height="1080"/>
        <asset id="r2" name="cam_a" start="3600s" duration="100s" hasVideo="1" hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000">
            <media-rep kind="original-media" src="file:///Volumes/MyDrive/Cam/cam_a.mov"/>
        </asset>
    </resources>
    <library location="file:///Users/t/Movies/Cam.fcpbundle/">
        <event name="Cam">
            <project name="Offset Clip">
                <sequence format="r1" duration="40s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                    <spine>
                        <clip name="cam_a" offset="0s" start="10s" duration="40s" format="r1" tcFormat="NDF" enabled="1">
                            <video ref="r2" offset="5s" start="3605s" duration="95s"/>
                        </clip>
                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>
"""

# A primary asset-clip plus a lane-1 <clip> B-roll bridging a gap in the
# primary storyline — the shape the shared walker must lift (video-only
# asset -> muted, matching what FCP plays there).
LANE_GAP_CLIP_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.13">
    <resources>
        <format id="r1" name="FFVideoFormat1080p30" frameDuration="1/30s" width="1920" height="1080"/>
        <asset id="r2" name="interview" start="0s" duration="60s" hasVideo="1" hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000">
            <media-rep kind="original-media" src="file:///Volumes/MyDrive/Int/interview.mov"/>
        </asset>
        <asset id="r3" name="broll" start="0s" duration="50s" hasVideo="1">
            <media-rep kind="original-media" src="file:///Volumes/MyDrive/Int/broll.mov"/>
        </asset>
    </resources>
    <library location="file:///Users/t/Movies/Int.fcpbundle/">
        <event name="Int">
            <project name="Lane Gap Clip">
                <sequence format="r1" duration="30s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                    <spine>
                        <asset-clip name="interview" ref="r2" offset="0s" start="0s" duration="20s" format="r1"/>
                        <gap name="Gap" offset="20s" start="3600s" duration="10s">
                            <clip lane="1" name="broll" offset="3602s" start="0s" duration="5s" format="r1">
                                <video ref="r3" offset="0s" start="0s" duration="50s"/>
                            </clip>
                        </gap>
                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>
"""


def _parse_text(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return parse_fcpxml(p)


class TestBasicClipSpine:
    @pytest.fixture
    def parsed(self, tmp_path):
        return _parse_text(tmp_path, "clip.fcpxml", BASIC_CLIP_FIXTURE)

    def test_segment_shape(self, parsed):
        assert len(parsed.spine_segments) == 1
        seg = parsed.spine_segments[0]
        assert seg.kind == "clip"
        assert seg.name == "drone_001"
        assert seg.offset_fraction == 0
        assert seg.start_fraction == 10
        assert seg.duration_fraction == 40
        assert not parsed.is_multi_source
        assert parsed.container_type == "clip"

    def test_audio_resolution(self, parsed):
        src = parsed.spine_segments[0].audio_source
        assert src.path == "/Volumes/MyDrive/Drone/drone_001.mp4"
        assert src.asset_id == "r2"
        assert not src.is_muted

    def test_source_window_starts_at_clip_start(self, parsed):
        start, dur = _segment_source_window(parsed.spine_segments[0])
        assert start == Fraction(10)
        assert dur == Fraction(40)

    def test_mode_a_select_round_trips_as_clip(self, parsed):
        out = write_selects_as_new_project(
            parsed, [Select(start_seconds=12.0, end_seconds=15.0, label="Pick")],
        )
        # The deep copy keeps the media children (the srcCh/role routing).
        root = etree.fromstring(out)
        clips = root.findall(".//project/sequence/spine/clip")
        assert len(clips) == 1
        assert clips[0].get("name") == "Pick"
        assert clips[0].find("video") is not None
        assert clips[0].find("video/audio") is not None
        # And re-parses with the select's exact window (12s + 3s @ 30fps).
        reparsed = re_parse(out)
        seg = reparsed.spine_segments[0]
        assert seg.kind == "clip"
        assert seg.start_fraction == 12
        assert seg.duration_fraction == 3
        assert seg.audio_source.path == "/Volumes/MyDrive/Drone/drone_001.mp4"


class TestClipOffsetMapping:
    def test_media_child_offsets_map_to_source_window(self, tmp_path):
        parsed = _parse_text(tmp_path, "offset.fcpxml", OFFSET_CLIP_FIXTURE)
        seg = parsed.spine_segments[0]
        src = seg.audio_source
        assert src.angle_offset_fraction == 5
        assert src.angle_start_fraction == 3605
        assert src.asset_start_fraction == 3600
        start, dur = _segment_source_window(seg)
        assert start == Fraction(10)   # 10 - 5 + (3605 - 3600)
        assert dur == Fraction(40)


class TestFlatExportSelfReimport:
    def test_channel_picked_flat_export_reimports(self, tmp_path, monkeypatch):
        # Force the dialogue-channel <clip><video><audio srcCh> form without
        # probing real media — the exact spine our flat export ships and the
        # trial user's rejected re-import.
        monkeypatch.setattr(
            fcpxml_export, "_resolve_audio_decl",
            lambda source_path, warnings_out=None: (
                ' audioSources="1" audioChannels="4" audioRate="48000"',
                ' audioRole="dialogue"', [2], 1),
        )
        monkeypatch.setattr(
            fcpxml_export, "_resolve_is_video",
            lambda source_path, warnings_out=None: True,
        )
        # generate_fcpxml degrades to markers-only when the source is not on
        # disk — the probes are patched, so a stand-in file is enough.
        source = tmp_path / "interview.mxf"
        source.write_bytes(b"\x00" * 1024)
        markers = [
            {"start": 2.0, "end": 5.0, "text": "One"},
            {"start": 10.0, "end": 12.0, "text": "Two"},
            {"start": 20.0, "end": 30.0, "text": "Three"},
        ]
        xml = fcpxml_export.generate_fcpxml(
            markers, project_name="Reimport", framerate=25,
            source_path=str(source),
            media_duration=120.0, mode="cuts",
        )
        assert "<clip " in xml  # the channel-routed form actually engaged
        p = tmp_path / "flat.fcpxml"
        p.write_text(xml)
        parsed = parse_fcpxml(p)
        assert [s.kind for s in parsed.spine_segments] == ["clip"] * 3
        assert {s.audio_source.path for s in parsed.spine_segments} \
            == {str(source)}
        windows = [_segment_source_window(s) for s in parsed.spine_segments]
        assert windows == [
            (Fraction(2), Fraction(3)),
            (Fraction(10), Fraction(2)),
            (Fraction(20), Fraction(10)),
        ]


class TestLaneGapClip:
    def test_lane_clip_is_lifted_without_warning(self, tmp_path):
        parsed = _parse_text(tmp_path, "lanegap.fcpxml", LANE_GAP_CLIP_FIXTURE)
        assert len(parsed.spine_segments) == 2
        lane_seg = parsed.spine_segments[1]
        assert lane_seg.kind == "clip"
        assert lane_seg.lane == "1"
        # Anchored in the gap's LOCAL timeline: 20 + (3602 - 3600) = 22s.
        assert lane_seg.offset_fraction == 22
        # Video-only asset — plays as silence on the timeline, so muted.
        assert lane_seg.audio_source.is_muted
        assert not any("<clip>" in w for w in parsed.parse_warnings)
