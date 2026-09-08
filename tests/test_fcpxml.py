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
from lxml import etree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from doza_assist.fcpxml import (  # noqa: E402
    ParseError,
    Select,
    audio_source_to_timeline,
    parse_fcpxml,
    parse_rational,
    rational_to_seconds,
    seconds_to_rational,
    timeline_to_segment,
    write_markers_on_timeline,
    write_selects_as_new_project,
)
from doza_assist.fcpxml.parser import SpineSegment, _resolve_asset_path, strip_file_url  # noqa: E402
from doza_assist.fcpxml.timeline_audio import plan_render  # noqa: E402
from doza_assist.fcpxml.writer import re_parse  # noqa: E402


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

    def test_fcp_triple_slash_form(self):
        assert strip_file_url("file:///tmp/foo.wav") == "/tmp/foo.wav"

    def test_strips_localhost_authority(self):
        # Legal RFC 8089 authority form written by some older exporters /
        # translators. Keeping 'localhost' produced a relative garbage path
        # that silently failed the on-disk original-media preference.
        assert (strip_file_url("file://localhost/Volumes/X/a.wav")
                == "/Volumes/X/a.wav")

    def test_strips_localhost_authority_and_decodes(self):
        assert (strip_file_url("file://localhost/Volumes/EDIT%20SSD/a.wav")
                == "/Volumes/EDIT SSD/a.wav")

    def test_windows_drive_letter_authority_kept(self):
        # FCPXML written on a Windows Resolve/translator box, two-slash
        # drive form. 'C:' must NOT be swallowed as a URL authority: the
        # mangled '/Users/ed/a.mov' can silently COLLIDE with a path that
        # exists on this Mac, while 'C:/Users/...' can never exist here and
        # errors honestly with the path the XML actually contained.
        assert (strip_file_url("file://C:/Users/ed/a.mov")
                == "C:/Users/ed/a.mov")

    def test_windows_standard_triple_slash_drive_passthrough(self):
        # The standard Windows file URL has an EMPTY authority; the path
        # (leading slash included) passes through untouched.
        assert (strip_file_url("file:///C:/Users/ed/a.mov")
                == "/C:/Users/ed/a.mov")

    def test_slashless_volume_authority_is_first_path_component(self):
        # Missing-third-slash form: 'Volumes' is the first path component,
        # not a hostname — re-prepend it instead of dropping it.
        assert (strip_file_url("file://Volumes/X/a.mov")
                == "/Volumes/X/a.mov")

    def test_unc_ish_host_reprepended_not_dropped(self):
        # Any other authority is treated as the first path component too, so
        # the missing-media error names something traceable to the XML.
        assert (strip_file_url("file://Server/share/x.mov")
                == "/Server/share/x.mov")

    def test_bare_scheme_stays_empty(self):
        assert strip_file_url("file://") == ""


# ---------- original-media (audio source) preference -----------------------

class TestResolveAssetPath:
    """``_resolve_asset_path`` resolves to ``original-media`` so the audio
    source is the camera/recorder master. Camera proxies (``proxy-media``)
    routinely carry silent, reference-only audio, so transcribing the proxy
    yields an empty WAV. Audio is extracted with video disabled, so the
    proxy's faster decode is irrelevant to the audio path. Falls back to a
    proxy only when the original is offline but the proxy is on disk."""

    def _asset(self, *reps):
        rep_xml = "\n".join(f'    <media-rep kind="{kind}" src="{src}"/>' for kind, src in reps)
        xml = f'<asset id="r1">\n{rep_xml}\n</asset>'
        return etree.fromstring(xml)

    def test_picks_first_when_only_one_rep(self):
        asset = self._asset(("original-media", "file:///tmp/clip.mov"))
        assert _resolve_asset_path(asset) == "/tmp/clip.mov"

    def test_prefers_original_when_both_exist(self, tmp_path):
        # The bug fix: both reps on disk -> the ORIGINAL (real mic) wins, not
        # the silent Sony proxy.
        original = tmp_path / "LCO.MP4"
        original.write_bytes(b"")
        proxy = tmp_path / "LCO.proxy.mov"
        proxy.write_bytes(b"")
        asset = self._asset(
            ("original-media", f"file://{original}"),
            ("proxy-media", f"file://{proxy}"),
        )
        assert _resolve_asset_path(asset) == str(original)

    def test_prefers_original_even_when_proxy_listed_first(self, tmp_path):
        # Order-independent: a proxy declared first must not win.
        original = tmp_path / "LCO.MP4"
        original.write_bytes(b"")
        proxy = tmp_path / "LCO.proxy.mov"
        proxy.write_bytes(b"")
        asset = self._asset(
            ("proxy-media", f"file://{proxy}"),
            ("original-media", f"file://{original}"),
        )
        assert _resolve_asset_path(asset) == str(original)

    def test_falls_back_to_proxy_when_original_offline(self, tmp_path):
        # Graceful fallback: original not mounted, proxy on disk -> use proxy
        # (at least the import succeeds; the silent guard catches it if silent).
        proxy = tmp_path / "clip.proxy.mov"
        proxy.write_bytes(b"")
        asset = self._asset(
            ("original-media", "file:///tmp/missing-original.mov"),
            ("proxy-media", f"file://{proxy}"),
        )
        assert _resolve_asset_path(asset) == str(proxy)

    def test_falls_back_to_original_when_proxy_missing(self):
        asset = self._asset(
            ("original-media", "file:///tmp/clip.mov"),
            ("proxy-media", "file:///tmp/does-not-exist.proxy.mov"),
        )
        assert _resolve_asset_path(asset) == "/tmp/clip.mov"

    def test_returns_original_path_when_nothing_on_disk(self):
        # Neither on disk -> never raises; returns the declared original so the
        # app-side existence check produces the "drive not mounted" message.
        asset = self._asset(
            ("original-media", "file:///tmp/missing.MP4"),
            ("proxy-media", "file:///tmp/missing.proxy.mov"),
        )
        assert _resolve_asset_path(asset) == "/tmp/missing.MP4"

    def test_localhost_authority_original_still_beats_proxy(self, tmp_path):
        # file://localhost/... used to mangle to 'localhost/...' — the on-disk
        # check failed and the (possibly silent) proxy won despite the
        # original being mounted.
        original = tmp_path / "LCO.MP4"
        original.write_bytes(b"")
        proxy = tmp_path / "LCO.proxy.mov"
        proxy.write_bytes(b"")
        asset = self._asset(
            ("original-media", f"file://localhost{original}"),
            ("proxy-media", f"file://{proxy}"),
        )
        assert _resolve_asset_path(asset) == str(original)


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
        # 1.8–1.14 are supported now (FCP + Resolve span). Use a version
        # below that floor so the rejection path is actually exercised.
        p = tmp_path / "old.fcpxml"
        p.write_text('<?xml version="1.0"?><fcpxml version="1.5"><resources/></fcpxml>')
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


# ---------- sync-clip with external audio nested INSIDE the primary clip ----
#
# Real-world pattern (Meeting and Interview project): FCP writes a sync-clip
# with NO inner <spine>; the external recorder is nested as a lane=-1
# asset-clip INSIDE the primary camera asset-clip rather than as a sibling.
# Before the fix, the parser scanned only direct children of <sync-clip>
# and children of <sync-clip>/<spine>, so the lane-attached external audio
# was never collected. With the camera mic muted, the segment fell to
# is_muted=True and got dropped from the timeline-audio plan — Parakeet
# transcribed silence for that half of the interview.

NESTED_LANE_AUDIO_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.14">
        <resources>
            <format id="r1" name="FF" frameDuration="1001/24000s" width="1920" height="1080"/>
            <format id="r4" name="audioOnly"/>
            <asset id="rCam" name="C9892" start="0s" duration="35820000/24000s" hasVideo="1" hasAudio="1" videoSources="1" audioSources="1" audioChannels="2" audioRate="48000">
                <media-rep kind="original-media" src="file:///tmp/C9892.MP4"/>
            </asset>
            <asset id="rExt" name="external_wav" start="0s" duration="1476s" hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000" format="r4">
                <media-rep kind="original-media" src="file:///tmp/external.wav"/>
            </asset>
        </resources>
        <library>
            <event name="E">
                <project name="Nested Lane">
                    <sequence format="r1" duration="200s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                        <spine>
                            <sync-clip offset="0s" name="C9892 - Synchronized Clip" start="92100/2400s" duration="426500/2400s">
                                <asset-clip ref="rCam" offset="0s" name="C9892" duration="35820000/24000s" audioRole="dialogue">
                                    <asset-clip ref="rExt" lane="-1" offset="156893/8000s" name="external_wav" duration="1476s" format="r4" audioRole="dialogue"/>
                                </asset-clip>
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


class TestSyncClipNestedLaneAudio:
    """Sync-clip with no inner <spine>, external audio nested as lane=-1
    asset-clip INSIDE the primary asset-clip, camera mic muted via
    audio-role-source active='0'. The parser must still surface the
    external recorder so the segment is transcribed instead of dropping
    to silence."""

    @pytest.fixture
    def parsed(self, tmp_path):
        p = tmp_path / "nested_lane.fcpxml"
        p.write_text(NESTED_LANE_AUDIO_FIXTURE)
        return parse_fcpxml(p)

    def test_picks_nested_external_audio(self, parsed):
        seg = parsed.spine_segments[0]
        assert seg.audio_source.asset_id == "rExt", (
            "Expected the external recorder nested inside the primary asset-clip "
            f"to be chosen; got {seg.audio_source.asset_id!r} instead"
        )
        assert seg.audio_source.path == "/tmp/external.wav"

    def test_segment_not_muted_when_nested_lane_rescues(self, parsed):
        seg = parsed.spine_segments[0]
        assert seg.audio_source.is_muted is False, (
            "Nested lane=-1 external audio must rescue the muted-camera segment "
            "from is_muted=True (which would drop it from the timeline-audio plan)"
        )


# ---------- asset-clip: plain single-camera spine (the reported bug) --------
#
# Single-cam footage (Meta Glasses, a mirrorless body, a screen capture, a
# drone) imports into FCP as plain <asset-clip> elements on the spine — no
# multicam, no sync-clip. A "rush" timeline laying many such clips end-to-end is
# exactly what the tester (Larry) dragged in; before the fix the parser skipped
# every spine child and raised "spine contains no <mc-clip> or <sync-clip>
# elements". 30 fps so whole-second times are frame-aligned (Meta Glasses rate).

META_RUSH_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.13">
        <resources>
            <format id="r1" name="FFVideoFormat1080p30" frameDuration="1/30s" width="1920" height="1080"/>
            <asset id="r2" name="meta_001" start="0s" duration="100s" hasVideo="1" hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000">
                <media-rep kind="original-media" src="file:///Volumes/MyDrive/Meta/meta_001.mp4"/>
            </asset>
            <asset id="r3" name="meta_002" start="0s" duration="100s" hasVideo="1" hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000">
                <media-rep kind="original-media" src="file:///Volumes/MyDrive/Meta/meta_002.mp4"/>
            </asset>
            <asset id="r4" name="meta_003" start="0s" duration="100s" hasVideo="1" hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000">
                <media-rep kind="original-media" src="file:///Volumes/MyDrive/Meta/meta_003.mp4"/>
            </asset>
        </resources>
        <library location="file:///Users/larry/Movies/Meta.fcpbundle/">
            <event name="Meta Glasses">
                <project name="Rush Full Unedited">
                    <sequence format="r1" duration="300s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                        <spine>
                            <asset-clip ref="r2" offset="0s" name="meta_001" start="0s" duration="100s" tcFormat="NDF" audioRole="dialogue"/>
                            <asset-clip ref="r3" offset="100s" name="meta_002" start="0s" duration="100s" tcFormat="NDF" audioRole="dialogue"/>
                            <asset-clip ref="r4" offset="200s" name="meta_003" start="0s" duration="100s" tcFormat="NDF" audioRole="dialogue"/>
                        </spine>
                    </sequence>
                </project>
            </event>
        </library>
    </fcpxml>
""")


class TestAssetClipSpine:
    """A spine of plain single-cam <asset-clip>s (Meta Glasses rush). Before the
    fix this raised ParseError; now each clip becomes a segment with its own
    resolved audio source."""

    @pytest.fixture
    def parsed(self, tmp_path):
        p = tmp_path / "rush.fcpxml"
        p.write_text(META_RUSH_FIXTURE)
        return parse_fcpxml(p)

    def test_all_three_clips_become_segments(self, parsed):
        assert len(parsed.spine_segments) == 3
        assert [s.kind for s in parsed.spine_segments] == ["asset-clip"] * 3

    def test_each_segment_resolves_its_own_asset(self, parsed):
        s0, s1, s2 = parsed.spine_segments
        assert (s0.audio_source.asset_id, s0.audio_source.path) == (
            "r2", "/Volumes/MyDrive/Meta/meta_001.mp4")
        assert (s1.audio_source.asset_id, s1.audio_source.path) == (
            "r3", "/Volumes/MyDrive/Meta/meta_002.mp4")
        assert (s2.audio_source.asset_id, s2.audio_source.path) == (
            "r4", "/Volumes/MyDrive/Meta/meta_003.mp4")

    def test_distinct_assets_are_multi_source(self, parsed):
        # Each clip is a different file → ingest renders a composed timeline WAV.
        assert parsed.is_multi_source is True

    def test_container_type_is_asset_clip(self, parsed):
        assert parsed.container_type == "asset-clip"
        assert parsed.container_ref == "r2"

    def test_representative_audio_is_first_clip(self, parsed):
        assert parsed.audio_file_path == "/Volumes/MyDrive/Meta/meta_001.mp4"
        assert parsed.audio_asset_id == "r2"
        assert parsed.active_audio_angle_id is None

    def test_project_and_event_names(self, parsed):
        assert parsed.project_name == "Rush Full Unedited"
        assert parsed.event_name == "Meta Glasses"

    def test_metadata_dict_is_json_serializable(self, parsed):
        import json
        data = parsed.to_metadata_dict()
        json.dumps(data)  # must not raise
        assert data["container_type"] == "asset-clip"
        assert len(data["spine_segments"]) == 3

    def test_timeline_audio_plan_one_entry_per_clip(self, parsed):
        plan = plan_render(parsed)
        assert len(plan) == 3
        # Untrimmed (start=0, asset.start=0) → seek 0 into each file, placed at
        # 0/100/200s on the timeline.
        assert [round(p["source_start_seconds"], 3) for p in plan] == [0.0, 0.0, 0.0]
        assert [p["timeline_offset_ms"] for p in plan] == [0, 100000, 200000]
        assert [p["input_path"] for p in plan] == [
            "/Volumes/MyDrive/Meta/meta_001.mp4",
            "/Volumes/MyDrive/Meta/meta_002.mp4",
            "/Volumes/MyDrive/Meta/meta_003.mp4",
        ]


SINGLE_ASSET_CLIP_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.13">
        <resources>
            <format id="r1" name="FF30" frameDuration="1/30s" width="1920" height="1080"/>
            <asset id="r2" name="solo" start="0s" duration="60s" hasVideo="1" hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000">
                <media-rep kind="original-media" src="file:///tmp/solo.mp4"/>
            </asset>
        </resources>
        <library location="file:///Users/x/Movies/X.fcpbundle/">
            <event name="E"><project name="Solo">
                <sequence format="r1" duration="60s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                    <spine>
                        <asset-clip ref="r2" offset="0s" name="solo" start="0s" duration="60s" audioRole="dialogue"/>
                    </spine>
                </sequence>
            </project></event>
        </library>
    </fcpxml>
""")


class TestSingleAssetClip:
    """One plain asset-clip → single-source: transcription runs against the file
    directly (no composed timeline WAV)."""

    @pytest.fixture
    def parsed(self, tmp_path):
        p = tmp_path / "solo.fcpxml"
        p.write_text(SINGLE_ASSET_CLIP_FIXTURE)
        return parse_fcpxml(p)

    def test_single_source_uses_the_file_directly(self, parsed):
        assert parsed.is_multi_source is False
        assert parsed.audio_file_path == "/tmp/solo.mp4"
        assert len(parsed.spine_segments) == 1
        assert parsed.spine_segments[0].kind == "asset-clip"

    def test_single_source_select_round_trips(self, parsed):
        # Single-source → select time is audio-source (file) seconds. A 10–30s
        # select rebuilds as an asset-clip starting 10s into r2, length 20s.
        out = write_selects_as_new_project(parsed, [Select(10, 30, label="Bit")])
        root = etree.fromstring(out)
        clip = root.find(".//sequence/spine/asset-clip")
        assert clip is not None and clip.get("ref") == "r2"
        assert parse_rational(clip.get("start")) == 10
        assert parse_rational(clip.get("duration")) == 20


ASSET_CLIP_TOD_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.13">
        <resources>
            <format id="r1" name="FF30" frameDuration="1/30s" width="1920" height="1080"/>
            <asset id="r2" name="tod_cam" start="3600s" duration="100s" hasVideo="1" hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000">
                <media-rep kind="original-media" src="file:///tmp/tod.mov"/>
            </asset>
        </resources>
        <library location="file:///Users/x/Movies/X.fcpbundle/">
            <event name="E"><project name="TOD">
                <sequence format="r1" duration="20s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                    <spine>
                        <asset-clip ref="r2" offset="0s" name="tod_cam" start="3610s" duration="20s" audioRole="dialogue"/>
                    </spine>
                </sequence>
            </project></event>
        </library>
    </fcpxml>
""")


class TestAssetClipTimecodeOffset:
    """An asset whose media carries embedded start timecode (asset@start != 0).
    The renderer must seek to ``asset-clip.start - asset.start`` so the audio
    pulled for the segment is the right slice of the file."""

    @pytest.fixture
    def parsed(self, tmp_path):
        p = tmp_path / "tod.fcpxml"
        p.write_text(ASSET_CLIP_TOD_FIXTURE)
        return parse_fcpxml(p)

    def test_asset_start_carried_through(self, parsed):
        src = parsed.spine_segments[0].audio_source
        assert src.asset_start_fraction == Fraction(3600)
        assert src.angle_offset_fraction == 0
        assert src.angle_start_fraction == 0

    def test_source_seek_subtracts_asset_start(self, parsed):
        plan = plan_render(parsed)
        assert len(plan) == 1
        # start 3610s into an asset whose media TC starts at 3600s → 10s in.
        assert round(plan[0]["source_start_seconds"], 3) == 10.0

    def test_representative_carries_asset_start(self, parsed):
        # The single-source locate inverts the seek using these.
        assert parsed.audio_asset_start_fraction == Fraction(3600)
        assert parsed.audio_container_tc_start_fraction == 0

    def test_single_source_tod_select_round_trips(self, parsed):
        # The clip (asset@start=3600, clip start=3610) occupies file/player time
        # [10s, 30s). A select at file time 15–25s must locate (container time
        # 15 + 3600 = 3615s, inside the clip's [3610s, 3630s) range) and rebuild
        # as an asset-clip starting at 3615s. Without adding asset_start in the
        # single-source locate this select falls outside every segment and the
        # export aborts with "all selects fall outside" — so this guards the fix.
        out = write_selects_as_new_project(parsed, [Select(15, 25, label="TOD bit")])
        root = etree.fromstring(out)
        clip = root.find(".//sequence/spine/asset-clip")
        assert clip is not None and clip.get("ref") == "r2"
        assert parse_rational(clip.get("start")) == 3615
        assert parse_rational(clip.get("duration")) == 10


VIDEO_ONLY_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.13">
        <resources>
            <format id="r1" name="FF30" frameDuration="1/30s" width="1920" height="1080"/>
            <asset id="r2" name="talker" start="0s" duration="100s" hasVideo="1" hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000">
                <media-rep kind="original-media" src="file:///tmp/has_audio.mp4"/>
            </asset>
            <asset id="r3" name="broll" start="0s" duration="100s" hasVideo="1" hasAudio="0" videoSources="1">
                <media-rep kind="original-media" src="file:///tmp/silent_broll.mp4"/>
            </asset>
        </resources>
        <library location="file:///Users/x/Movies/X.fcpbundle/">
            <event name="E"><project name="With B-roll">
                <sequence format="r1" duration="200s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                    <spine>
                        <asset-clip ref="r2" offset="0s" name="talker" start="0s" duration="100s" audioRole="dialogue"/>
                        <asset-clip ref="r3" offset="100s" name="broll" start="0s" duration="100s"/>
                    </spine>
                </sequence>
            </project></event>
        </library>
    </fcpxml>
""")


class TestAssetClipVideoOnly:
    """A silent b-roll asset-clip (hasAudio='0') occupies timeline space but
    contributes silence — it must be marked muted and dropped from the render
    plan, never sent to ffmpeg as a missing audio stream."""

    @pytest.fixture
    def parsed(self, tmp_path):
        p = tmp_path / "broll.fcpxml"
        p.write_text(VIDEO_ONLY_FIXTURE)
        return parse_fcpxml(p)

    def test_video_only_clip_is_muted(self, parsed):
        s0, s1 = parsed.spine_segments
        assert s0.audio_source.is_muted is False
        assert s1.audio_source.is_muted is True

    def test_muted_clip_dropped_from_plan(self, parsed):
        plan = plan_render(parsed)
        assert len(plan) == 1
        assert plan[0]["input_path"] == "/tmp/has_audio.mp4"


GAP_ONLY_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.13">
        <resources>
            <format id="r1" name="FF30" frameDuration="1/30s" width="1920" height="1080"/>
        </resources>
        <library>
            <event name="E"><project name="Empty">
                <sequence format="r1" duration="300s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                    <spine>
                        <gap offset="0s" name="Gap" duration="300s"/>
                    </spine>
                </sequence>
            </project></event>
        </library>
    </fcpxml>
""")


class TestUnsupportedSpineError:
    """A spine with no readable clips must fail with a message that names what
    it actually found — so the next unsupported shape (e.g. compound clips)
    points us straight at what to support next, instead of a dead end."""

    def test_error_is_diagnostic(self, tmp_path):
        p = tmp_path / "gap.fcpxml"
        p.write_text(GAP_ONLY_FIXTURE)
        with pytest.raises(ParseError, match="no clips Doza Assist can read"):
            parse_fcpxml(p)

    def test_error_names_present_tags(self, tmp_path):
        p = tmp_path / "gap.fcpxml"
        p.write_text(GAP_ONLY_FIXTURE)
        with pytest.raises(ParseError) as ei:
            parse_fcpxml(p)
        assert "gap" in str(ei.value)


class TestAssetClipRoundTrip:
    """Selects on a single-cam rush round-trip back to FCP-importable
    <asset-clip>s (Mode A) and as markers on the original spine (Mode B).
    META_RUSH is multi-source, so Select times are TIMELINE seconds."""

    @pytest.fixture
    def parsed(self, tmp_path):
        p = tmp_path / "rush.fcpxml"
        p.write_text(META_RUSH_FIXTURE)
        return parse_fcpxml(p)

    def test_selects_export_as_asset_clips(self, parsed):
        selects = [
            Select(start_seconds=10, end_seconds=30, label="Intro"),     # clip 0 (r2)
            Select(start_seconds=120, end_seconds=160, label="Middle"),  # clip 1 (r3)
        ]
        out = write_selects_as_new_project(parsed, selects)
        reparsed = re_parse(out)
        assert [s.kind for s in reparsed.spine_segments] == ["asset-clip", "asset-clip"]
        # Sorted chronologically by source in-point; each keeps its own asset.
        assert reparsed.spine_segments[0].audio_source.asset_id == "r2"
        assert reparsed.spine_segments[1].audio_source.asset_id == "r3"

    def test_select_source_in_point_and_placement(self, parsed):
        # 120–160s on the timeline lands 20s into clip 1 (offset 100s) → start
        # 20s into asset r3, length 40s, placed first on the new timeline.
        selects = [Select(start_seconds=120, end_seconds=160, label="Middle")]
        out = write_selects_as_new_project(parsed, selects)
        root = etree.fromstring(out)
        clips = root.findall(".//sequence/spine/asset-clip")
        assert len(clips) == 1
        assert clips[0].get("ref") == "r3"
        assert parse_rational(clips[0].get("start")) == 20
        assert parse_rational(clips[0].get("duration")) == 40
        assert parse_rational(clips[0].get("offset")) == 0
        # audioRole survives the deep copy; the source's name is overwritten.
        assert clips[0].get("audioRole") == "dialogue"
        assert clips[0].get("name") == "Middle"

    def test_label_and_speaker_become_name_and_note(self, parsed):
        selects = [Select(start_seconds=10, end_seconds=30, label="Intro", speaker="Larry")]
        out = write_selects_as_new_project(parsed, selects)
        root = etree.fromstring(out)
        clip = root.find(".//sequence/spine/asset-clip")
        assert clip.get("name") == "Intro"
        note = clip.find("note")
        assert note is not None and "Larry" in note.text

    def test_original_resources_preserved_verbatim(self, parsed):
        selects = [Select(start_seconds=10, end_seconds=30, label="Intro")]
        out = write_selects_as_new_project(parsed, selects)
        assert b'src="file:///Volumes/MyDrive/Meta/meta_001.mp4"' in out

    def test_markers_mode_attaches_to_the_right_clip(self, parsed):
        selects = [Select(start_seconds=150, end_seconds=151, label="Beat", kind="strong")]
        out = write_markers_on_timeline(parsed, selects)
        root = etree.fromstring(out)
        spine_clips = root.findall(".//sequence/spine/asset-clip")
        assert len(spine_clips) == 3
        # timeline 150s → clip 1 (100–200s) at container time 50s.
        markers = [c.findall("marker") for c in spine_clips]
        assert [len(m) for m in markers] == [0, 1, 0]
        assert parse_rational(markers[1][0].get("start")) == 50


# A single-cam asset-clip carrying inherited annotations: a marker before the
# select, one inside, one after, and a keyword range that straddles the select.
ANNOTATED_ASSET_CLIP_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.13">
        <resources>
            <format id="r1" name="FF30" frameDuration="1/30s" width="1920" height="1080"/>
            <asset id="r2" name="cam" start="0s" duration="100s" hasVideo="1" hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000">
                <media-rep kind="original-media" src="file:///tmp/cam.mp4"/>
            </asset>
        </resources>
        <library location="file:///Users/x/Movies/X.fcpbundle/">
            <event name="E"><project name="Annotated">
                <sequence format="r1" duration="100s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                    <spine>
                        <asset-clip ref="r2" offset="0s" name="cam" start="0s" duration="100s" audioRole="dialogue">
                            <marker start="5s" duration="1/30s" value="before"/>
                            <marker start="25s" duration="1/30s" value="inside"/>
                            <marker start="60s" duration="1/30s" value="after"/>
                            <keyword start="10s" duration="40s" value="spanning"/>
                        </asset-clip>
                    </spine>
                </sequence>
            </project></event>
        </library>
    </fcpxml>
""")


class TestSelectTrimsStaleAnnotations:
    """A select keeps only a sub-range of its source clip. Inherited markers /
    keyword ranges that fall outside that sub-range must be dropped (and
    straddling ranges clamped) — otherwise the round-tripped select carries
    metadata pointing at footage it no longer contains."""

    @pytest.fixture
    def parsed(self, tmp_path):
        p = tmp_path / "annotated.fcpxml"
        p.write_text(ANNOTATED_ASSET_CLIP_FIXTURE)
        return parse_fcpxml(p)

    def test_out_of_range_markers_dropped_and_keyword_clamped(self, parsed):
        # Single-source → select time is file seconds. Keep 20–40s.
        out = write_selects_as_new_project(parsed, [Select(20, 40, label="Mid")])
        clip = etree.fromstring(out).find(".//sequence/spine/asset-clip")
        # Only the marker inside [20s, 40s) survives.
        markers = clip.findall("marker")
        assert [m.get("value") for m in markers] == ["inside"]
        assert parse_rational(markers[0].get("start")) == 25
        # The keyword [10s, 50s) is clamped to the kept range [20s, 40s).
        # (The writer's own "Doza Assist" provenance keyword spans the whole
        # kept range and is not the subject here.)
        kws = [k for k in clip.findall("keyword") if k.get("value") != "Doza Assist"]
        assert len(kws) == 1
        assert parse_rational(kws[0].get("start")) == 20
        assert parse_rational(kws[0].get("duration")) == 20

    def test_no_annotation_escapes_the_clip_bounds(self, parsed):
        out = write_selects_as_new_project(parsed, [Select(20, 40, label="Mid")])
        clip = etree.fromstring(out).find(".//sequence/spine/asset-clip")
        clip_start = parse_rational(clip.get("start"))
        clip_end = clip_start + parse_rational(clip.get("duration"))
        for child in clip:
            if child.tag in ("marker", "keyword"):
                s = parse_rational(child.get("start"))
                e = s + parse_rational(child.get("duration") or "0s")
                assert clip_start <= s < clip_end and e <= clip_end


# A captioned single-cam clip: a wide caption spanning the whole clip (so two
# selects both keep it) plus a tail caption neither select reaches. The shape a
# Pro tester hit — a 186-caption interview round-tripped through Story Builder
# emitted ts1…tsN once per select, and FCP rejected the import with
# "DTD validation failed. ID ts1 already defined".
CAPTIONED_CLIP_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.13">
        <resources>
            <format id="r1" name="FF30" frameDuration="1/30s" width="1920" height="1080"/>
            <asset id="r2" name="cam" start="0s" duration="60s" hasVideo="1" hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000">
                <media-rep kind="original-media" src="file:///tmp/cam.mp4"/>
            </asset>
        </resources>
        <library location="file:///Users/x/Movies/X.fcpbundle/">
            <event name="E"><project name="Captioned">
                <sequence format="r1" duration="60s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                    <spine>
                        <asset-clip ref="r2" offset="0s" name="cam" start="0s" duration="60s" audioRole="dialogue">
                            <caption lane="1" offset="0s" name="wide" start="3600s" duration="60s" role="iTT?captionFormat=ITT.en-US">
                                <text><text-style ref="ts1">Whole clip</text-style></text>
                                <text-style-def id="ts1"><text-style font="Helvetica" fontSize="63"/></text-style-def>
                            </caption>
                            <caption lane="1" offset="50s" name="tail" start="3600s" duration="5s" role="iTT?captionFormat=ITT.en-US">
                                <text><text-style ref="ts2">Tail only</text-style></text>
                                <text-style-def id="ts2"><text-style font="Helvetica" fontSize="63"/></text-style-def>
                            </caption>
                        </asset-clip>
                    </spine>
                </sequence>
            </project></event>
        </library>
    </fcpxml>
""")


class TestCaptionedClipRoundTrip:
    """Two selects cut from the same captioned clip must not both re-define the
    same <text-style-def> ids, and each must carry only its overlapping
    captions."""

    @pytest.fixture
    def parsed(self, tmp_path):
        p = tmp_path / "captioned.fcpxml"
        p.write_text(CAPTIONED_CLIP_FIXTURE)
        return parse_fcpxml(p)

    def test_text_style_def_ids_unique_across_copies(self, parsed):
        # Two selects both overlap the wide caption (ts1); neither reaches the
        # tail caption (ts2, offset 50s).
        out = write_selects_as_new_project(
            parsed, [Select(0, 10, label="A"), Select(20, 30, label="B")])
        root = etree.fromstring(out)
        ids = [el.get("id") for el in root.iter() if el.get("id")]
        # No id appears twice — exactly what FCP's DTD enforces.
        assert len(ids) == len(set(ids)), f"duplicate ids: {ids}"
        # The shared caption's id was suffixed per copy, not left as bare ts1.
        tsd_ids = [t.get("id") for t in root.iter("text-style-def")]
        assert tsd_ids == ["ts1_s0", "ts1_s1"]
        # Every <text-style ref> still resolves.
        defined = set(ids)
        for ts in root.iter("text-style"):
            if ts.get("ref"):
                assert ts.get("ref") in defined

    def test_non_overlapping_captions_pruned(self, parsed):
        # The tail caption (offset 50s) overlaps neither select → dropped from
        # both copies; only the wide caption survives.
        out = write_selects_as_new_project(
            parsed, [Select(0, 10, label="A"), Select(20, 30, label="B")])
        root = etree.fromstring(out)
        for clip in root.findall(".//sequence/spine/asset-clip"):
            caps = clip.findall("caption")
            assert len(caps) == 1
            assert caps[0].get("name") == "wide"


# A sync-clip pairing a camera with a TC-carrying field-recorder WAV: the
# asset declares start="3600s" (01:00:00:00 jam-sync TC) and the dialogue
# asset-clip's start attr is TC-based (3610s = 10s into the file).
SYNC_CLIP_TOD_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.13">
        <resources>
            <format id="r1" name="FF30" frameDuration="1/30s" width="1920" height="1080"/>
            <asset id="r2" name="recorder" start="3600s" duration="120s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                <media-rep kind="original-media" src="file:///tmp/recorder_tod.wav"/>
            </asset>
        </resources>
        <library location="file:///Users/x/Movies/X.fcpbundle/">
            <event name="E"><project name="TOD Sync">
                <sequence format="r1" duration="20s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                    <spine>
                        <sync-clip offset="0s" name="S" duration="20s">
                            <asset-clip ref="r2" offset="0s" start="3610s" duration="20s" audioRole="dialogue"/>
                        </sync-clip>
                    </spine>
                </sequence>
            </project></event>
        </library>
    </fcpxml>
""")


class TestSyncClipRecorderTC:
    """Sync-clip dialogue resolution used to hardcode asset_start=0 while
    returning the clip's TC-based start attr as angle_start — the renderer
    then over-seeked by exactly asset@start (an hour for 01:00:00:00 jam-sync
    TC), silencing that interview in the timeline WAV with no error."""

    @pytest.fixture
    def parsed(self, tmp_path):
        p = tmp_path / "tod_sync.fcpxml"
        p.write_text(SYNC_CLIP_TOD_FIXTURE)
        return parse_fcpxml(p)

    def test_asset_start_resolved_from_asset(self, parsed):
        src = parsed.spine_segments[0].audio_source
        assert src.angle_start_fraction == Fraction(3610)
        assert src.asset_start_fraction == Fraction(3600)

    def test_source_seek_is_zero_based(self, parsed):
        plan = plan_render(parsed)
        assert len(plan) == 1
        # start 3610s TC into an asset whose TC origin is 3600s → 10s into
        # the actual file (was 3610s — hours past EOF of a 120s WAV).
        assert round(plan[0]["source_start_seconds"], 3) == 10.0

    def test_no_clip_start_attr_keeps_source_time_convention(self, tmp_path):
        # Without an explicit clip start, sync-clip times already live in
        # source time; asset@start must NOT be subtracted then.
        p = tmp_path / "plain_sync.fcpxml"
        p.write_text(SYNC_CLIP_TOD_FIXTURE.replace(' start="3610s"', ''))
        parsed = parse_fcpxml(p)
        src = parsed.spine_segments[0].audio_source
        assert src.asset_start_fraction == Fraction(0)
        assert plan_render(parsed)[0]["source_start_seconds"] == 0.0

    def test_single_source_tod_select_locates(self, parsed):
        # The clip occupies file time [10s, 30s). A select at 12–18s inverts to
        # container time 12 − 3610 + 3600 = 2s, inside the sync-clip's [0, 20s)
        # range. With asset_start hardcoded to 0 this came out at −3598s —
        # outside every segment — and the select was skipped.
        skipped = []
        out = write_selects_as_new_project(
            parsed, [Select(12, 18, label="TOD bit")], skipped_out=skipped)
        assert skipped == []
        root = etree.fromstring(out)
        assert root.find(".//sequence/spine/sync-clip") is not None


# Two readable asset-clips with an unsupported <ref-clip> (compound clip) and
# a spine <title> between them.
MIXED_UNSUPPORTED_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.13">
        <resources>
            <format id="r1" name="FF30" frameDuration="1/30s" width="1920" height="1080"/>
            <asset id="rA" name="camA" start="0s" duration="100s" hasVideo="1" hasAudio="1" audioSources="1">
                <media-rep kind="original-media" src="file:///tmp/camA.mov"/>
            </asset>
            <asset id="rB" name="camB" start="0s" duration="100s" hasVideo="1" hasAudio="1" audioSources="1">
                <media-rep kind="original-media" src="file:///tmp/camB.mov"/>
            </asset>
        </resources>
        <library location="file:///Users/x/Movies/X.fcpbundle/">
            <event name="E"><project name="Mixed">
                <sequence format="r1" duration="60s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                    <spine>
                        <asset-clip ref="rA" offset="0s" name="A" start="0s" duration="10s" audioRole="dialogue"/>
                        <ref-clip ref="rComp" offset="10s" name="Montage" duration="20s"/>
                        <title ref="rT" offset="30s" name="Lower Third" duration="5s"/>
                        <asset-clip ref="rB" offset="35s" name="B" start="0s" duration="25s" audioRole="dialogue"/>
                    </spine>
                </sequence>
            </project></event>
        </library>
    </fcpxml>
""")


class TestUnsupportedSpineEntryWarning:
    """A mixed spine (readable clips + a compound <ref-clip>) parses fine, but
    the dropped clip leaves silence in the timeline WAV and a hole in the
    transcript — the drop must surface as a parse warning, not vanish."""

    @pytest.fixture
    def parsed(self, tmp_path):
        p = tmp_path / "mixed_unsupported.fcpxml"
        p.write_text(MIXED_UNSUPPORTED_FIXTURE)
        return parse_fcpxml(p)

    def test_supported_clips_still_parse(self, parsed):
        assert [s.name for s in parsed.spine_segments] == ["A", "B"]

    def test_ref_clip_drop_is_warned_with_hint(self, parsed):
        ref_warnings = [w for w in parsed.parse_warnings if "ref-clip" in w]
        assert len(ref_warnings) == 1
        assert "Montage" in ref_warnings[0]
        assert "Break Apart" in ref_warnings[0]

    def test_titles_do_not_warn(self, parsed):
        # <title> carries no dialogue; warning on every lower third would
        # bury the real signal.
        assert not any("title" in w for w in parsed.parse_warnings)

    def test_warnings_reach_project_metadata(self, parsed):
        meta = parsed.to_metadata_dict()
        assert meta["parse_warnings"] == parsed.parse_warnings
        assert len(meta["parse_warnings"]) == 1

    def test_clean_timeline_has_no_warnings(self, tmp_path):
        p = tmp_path / "clean.fcpxml"
        p.write_text(VIDEO_ONLY_FIXTURE)
        assert parse_fcpxml(p).parse_warnings == []


# One readable asset-clip plus a <gap> carrying a connected compound clip —
# the lane-1 B-roll-gap timeline class. The segment walk descends into gaps
# only for supported tags, so the ref-clip is dropped; the drop must warn
# exactly like a top-level one (R7).
GAP_NESTED_UNSUPPORTED_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.13">
        <resources>
            <format id="r1" name="FF30" frameDuration="1/30s" width="1920" height="1080"/>
            <asset id="rA" name="camA" start="0s" duration="100s" hasVideo="1" hasAudio="1" audioSources="1">
                <media-rep kind="original-media" src="file:///tmp/camA.mov"/>
            </asset>
        </resources>
        <library location="file:///Users/x/Movies/X.fcpbundle/">
            <event name="E"><project name="GapNested">
                <sequence format="r1" duration="60s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                    <spine>
                        <asset-clip ref="rA" offset="0s" name="A" start="0s" duration="10s" audioRole="dialogue"/>
                        <gap name="Gap" offset="10s" duration="20s" start="0s">
                            <ref-clip ref="rComp" lane="1" offset="0s" name="InterviewCompound" duration="20s"/>
                            <title ref="rT" lane="2" offset="0s" name="Lower Third" duration="5s"/>
                        </gap>
                    </spine>
                </sequence>
            </project></event>
        </library>
    </fcpxml>
""")


class TestGapNestedUnsupportedClipWarning:
    """An unsupported dialogue-capable clip connected INSIDE a <gap> is
    dropped just as silently as a top-level one — the unsupported-spine scan
    must descend into gaps the same way iter_spine_clip_elements does."""

    @pytest.fixture
    def parsed(self, tmp_path):
        p = tmp_path / "gap_nested_unsupported.fcpxml"
        p.write_text(GAP_NESTED_UNSUPPORTED_FIXTURE)
        return parse_fcpxml(p)

    def test_parse_succeeds_without_the_nested_clip(self, parsed):
        assert [s.name for s in parsed.spine_segments] == ["A"]

    def test_gap_nested_ref_clip_drop_is_warned(self, parsed):
        ref_warnings = [w for w in parsed.parse_warnings if "ref-clip" in w]
        assert len(ref_warnings) == 1
        assert "InterviewCompound" in ref_warnings[0]
        assert "Break Apart" in ref_warnings[0]

    def test_gap_nested_titles_still_do_not_warn(self, parsed):
        assert not any("Lower Third" in w for w in parsed.parse_warnings)


class TestTimeMapWarning:
    """<timeMap> retimes are not applied by the renderer/locator (1:1 math),
    so a speed-changed clip must at least be flagged at parse time."""

    def _fixture(self, tmp_path, clip_xml):
        p = tmp_path / "retimed.fcpxml"
        # Dedent BEFORE substituting: a multi-line clip_xml would defeat the
        # common-prefix dedent and leave the XML declaration indented.
        template = textwrap.dedent("""\
            <?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE fcpxml>
            <fcpxml version="1.13">
                <resources>
                    <format id="r1" name="FF30" frameDuration="1/30s" width="1920" height="1080"/>
                    <asset id="r2" name="cam" start="0s" duration="100s" hasVideo="1" hasAudio="1" audioSources="1">
                        <media-rep kind="original-media" src="file:///tmp/cam.mov"/>
                    </asset>
                </resources>
                <library location="file:///Users/x/Movies/X.fcpbundle/">
                    <event name="E"><project name="Retimed">
                        <sequence format="r1" duration="60s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                            <spine>
                            {clip_xml}
                            </spine>
                        </sequence>
                    </project></event>
                </library>
            </fcpxml>
        """)
        p.write_text(template.format(clip_xml=clip_xml))
        return parse_fcpxml(p)

    def test_retimed_asset_clip_warns_by_name(self, tmp_path):
        parsed = self._fixture(tmp_path, textwrap.dedent("""\
            <asset-clip ref="r2" offset="0s" name="Speed Ramp" start="0s" duration="30s" audioRole="dialogue">
                <timeMap>
                    <timept time="0s" value="0s" interp="smooth2"/>
                    <timept time="30s" value="60s" interp="smooth2"/>
                </timeMap>
            </asset-clip>
        """))
        warnings = [w for w in parsed.parse_warnings if "retimed" in w]
        assert len(warnings) == 1
        assert "Speed Ramp" in warnings[0]
        assert "misaligned" in warnings[0]

    def test_timemap_on_inner_sync_clip_child_detected(self, tmp_path):
        # FCP can put the timeMap on the clip nested inside the sync-clip.
        parsed = self._fixture(tmp_path, textwrap.dedent("""\
            <sync-clip offset="0s" name="Synced Ramp" duration="30s">
                <asset-clip ref="r2" offset="0s" start="0s" duration="30s" audioRole="dialogue">
                    <timeMap>
                        <timept time="0s" value="0s" interp="smooth2"/>
                        <timept time="30s" value="60s" interp="smooth2"/>
                    </timeMap>
                </asset-clip>
            </sync-clip>
        """))
        assert any("Synced Ramp" in w and "retimed" in w
                   for w in parsed.parse_warnings)

    def test_normal_speed_clip_does_not_warn(self, tmp_path):
        parsed = self._fixture(
            tmp_path,
            '<asset-clip ref="r2" offset="0s" name="Normal" start="0s" '
            'duration="30s" audioRole="dialogue"/>',
        )
        assert parsed.parse_warnings == []


class TestConformRateWarning:
    """<conform-rate> with scaleEnabled defaulting to "1" means FCP plays the
    media frame-for-frame at the sequence rate (25p in 23.976 runs ~4.3%
    slow); our literal-seconds math drifts progressively inside such clips, so
    a genuine close-rate conform must be flagged."""

    def _fixture(self, tmp_path, conform_xml):
        p = tmp_path / "conformed.fcpxml"
        p.write_text(textwrap.dedent(f"""\
            <?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE fcpxml>
            <fcpxml version="1.13">
                <resources>
                    <format id="r1" name="FF2398" frameDuration="1001/24000s" width="1920" height="1080"/>
                    <asset id="r2" name="cam25p" start="0s" duration="1800s" hasVideo="1" hasAudio="1" audioSources="1">
                        <media-rep kind="original-media" src="file:///tmp/cam25p.mov"/>
                    </asset>
                </resources>
                <library location="file:///Users/x/Movies/X.fcpbundle/">
                    <event name="E"><project name="Conformed">
                        <sequence format="r1" duration="60s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                            <spine>
                                <asset-clip ref="r2" offset="0s" name="PAL Clip" start="0s" duration="30s" audioRole="dialogue">
                                    {conform_xml}
                                </asset-clip>
                            </spine>
                        </sequence>
                    </project></event>
                </library>
            </fcpxml>
        """))
        return parse_fcpxml(p)

    def test_default_scale_enabled_mismatch_warns(self, tmp_path):
        # 25p media in a 23.976 sequence, scaleEnabled absent (DTD default 1).
        parsed = self._fixture(tmp_path, '<conform-rate srcFrameRate="25"/>')
        warnings = [w for w in parsed.parse_warnings if "rate-conformed" in w]
        assert len(warnings) == 1
        assert "PAL Clip" in warnings[0]
        assert "4.3%" in warnings[0]  # 25/23.976 − 1 ≈ 4.27% drift

    def test_scale_disabled_does_not_warn(self, tmp_path):
        parsed = self._fixture(
            tmp_path, '<conform-rate scaleEnabled="0" srcFrameRate="25"/>')
        assert parsed.parse_warnings == []

    def test_matching_rate_token_does_not_warn(self, tmp_path):
        # FCP writes rounded rate tokens; "23.98" in a 23.976 sequence is not
        # a conform.
        parsed = self._fixture(tmp_path, '<conform-rate srcFrameRate="23.98"/>')
        assert parsed.parse_warnings == []


class TestMalformedRationalIsParseError:
    """Malformed time attributes raised bare ValueError/ZeroDivisionError from
    Fraction, bypassing the routes' `except ParseError` handlers (HTTP 500
    with a raw traceback + a leaked half-created project dir)."""

    def _parse(self, tmp_path, offset='0s', duration='10s'):
        p = tmp_path / "malformed.fcpxml"
        p.write_text(textwrap.dedent(f"""\
            <?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE fcpxml>
            <fcpxml version="1.13">
                <resources>
                    <format id="r1" name="FF30" frameDuration="1/30s" width="1920" height="1080"/>
                    <asset id="r2" name="cam" start="0s" duration="100s" hasVideo="1" hasAudio="1" audioSources="1">
                        <media-rep kind="original-media" src="file:///tmp/cam.mov"/>
                    </asset>
                </resources>
                <library location="file:///Users/x/Movies/X.fcpbundle/">
                    <event name="E"><project name="Bad">
                        <sequence format="r1" duration="60s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                            <spine>
                                <asset-clip ref="r2" offset="{offset}" name="C" start="0s" duration="{duration}" audioRole="dialogue"/>
                            </spine>
                        </sequence>
                    </project></event>
                </library>
            </fcpxml>
        """))
        return parse_fcpxml(p)

    def test_zero_denominator_duration(self, tmp_path):
        # Fraction(240240, 0) raises ZeroDivisionError — NOT a ValueError
        # subclass, so it used to escape every handler.
        with pytest.raises(ParseError, match="duration"):
            self._parse(tmp_path, duration="240240/0s")

    def test_decimal_offset(self, tmp_path):
        with pytest.raises(ParseError, match="offset.*1.5"):
            self._parse(tmp_path, offset="1.5s")

    def test_error_names_offending_value(self, tmp_path):
        with pytest.raises(ParseError, match="240240/0s"):
            self._parse(tmp_path, duration="240240/0s")


# Interview with audio + B-roll whose asset omits hasAudio entirely — FCP's
# encoding of a video-only file (it never writes hasAudio="0" itself).
ABSENT_HASAUDIO_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.13">
        <resources>
            <format id="r1" name="FF30" frameDuration="1/30s" width="1920" height="1080"/>
            <asset id="r2" name="talker" start="0s" duration="100s" hasVideo="1" hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000">
                <media-rep kind="original-media" src="file:///tmp/has_audio.mp4"/>
            </asset>
            <asset id="r3" name="broll" start="0s" duration="100s" hasVideo="1" videoSources="1">
                <media-rep kind="original-media" src="file:///tmp/video_only_broll.mov"/>
            </asset>
        </resources>
        <library location="file:///Users/x/Movies/X.fcpbundle/">
            <event name="E"><project name="With Silent B-roll">
                <sequence format="r1" duration="200s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                    <spine>
                        <asset-clip ref="r2" offset="0s" name="talker" start="0s" duration="100s" audioRole="dialogue"/>
                        <asset-clip ref="r3" offset="100s" name="broll" start="0s" duration="100s"/>
                    </spine>
                </sequence>
            </project></event>
        </library>
    </fcpxml>
""")


class TestAssetClipAbsentHasAudio:
    """FCP omits hasAudio for video-only assets. Treating absence as
    audio-bearing fed an audio-less input to the ffprobe gate and the
    timeline render — one silent graphics/drone clip killed the whole
    import."""

    @pytest.fixture
    def parsed(self, tmp_path):
        p = tmp_path / "absent_hasaudio.fcpxml"
        p.write_text(ABSENT_HASAUDIO_FIXTURE)
        return parse_fcpxml(p)

    def test_absent_hasaudio_reads_as_muted(self, parsed):
        s0, s1 = parsed.spine_segments
        assert s0.audio_source.is_muted is False
        assert s1.audio_source.is_muted is True

    def test_muted_broll_dropped_from_plan(self, parsed):
        plan = plan_render(parsed)
        assert [p["input_path"] for p in plan] == ["/tmp/has_audio.mp4"]

    def test_audio_sources_attr_counts_as_audio_evidence(self, tmp_path):
        # Producers that declare the layout without hasAudio still transcribe.
        p = tmp_path / "layout_only.fcpxml"
        p.write_text(ABSENT_HASAUDIO_FIXTURE.replace(
            'name="broll" start="0s" duration="100s" hasVideo="1" videoSources="1"',
            'name="broll" start="0s" duration="100s" hasVideo="1" videoSources="1" audioSources="1"',
        ))
        parsed = parse_fcpxml(p)
        assert parsed.spine_segments[1].audio_source.is_muted is False


DISABLED_CLIP_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.13">
        <resources>
            <format id="r1" name="FF30" frameDuration="1/30s" width="1920" height="1080"/>
            <asset id="rA" name="camA" start="0s" duration="600s" hasVideo="1" hasAudio="1" audioSources="1">
                <media-rep kind="original-media" src="file:///tmp/camA.mov"/>
            </asset>
            <asset id="rB" name="camB" start="0s" duration="600s" hasVideo="1" hasAudio="1" audioSources="1">
                <media-rep kind="original-media" src="file:///tmp/camB.mov"/>
            </asset>
            <asset id="rC" name="camC" start="0s" duration="600s" hasVideo="1" hasAudio="1" audioSources="1">
                <media-rep kind="original-media" src="file:///tmp/camC.mov"/>
            </asset>
        </resources>
        <library location="file:///Users/x/Movies/X.fcpbundle/">
            <event name="E"><project name="Disabled Middle">
                <sequence format="r1" duration="90s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                    <spine>
                        <asset-clip ref="rA" offset="0s" name="A" start="0s" duration="30s" audioRole="dialogue"/>
                        <asset-clip ref="rB" offset="30s" name="B" start="0s" duration="30s" enabled="0" audioRole="dialogue"/>
                        <asset-clip ref="rC" offset="60s" name="C" start="0s" duration="30s" audioRole="dialogue"/>
                    </spine>
                </sequence>
            </project></event>
        </library>
    </fcpxml>
""")


class TestDisabledSpineClips:
    """Clips disabled with V in FCP (enabled="0") play as black/silence —
    they must be neither transcribed nor selectable. They STAY in the
    segment list (dropping them would shift indices and flip
    is_multi_source for projects ingested before this behavior existed —
    stored FCPXML is re-parsed at export time), marked disabled with muted
    audio, and the writer's select locator refuses to match them."""

    @pytest.fixture
    def parsed(self, tmp_path):
        p = tmp_path / "disabled.fcpxml"
        p.write_text(DISABLED_CLIP_FIXTURE)
        return parse_fcpxml(p)

    def test_disabled_clip_stays_a_segment_but_disabled(self, parsed):
        assert [s.name for s in parsed.spine_segments] == ["A", "B", "C"]
        a, b, c = parsed.spine_segments
        assert a.enabled and c.enabled and not b.enabled
        assert b.audio_source is not None and b.audio_source.is_muted

    def test_disabled_clip_does_not_flip_multi_source_on_reparse(self, parsed):
        # The 1.0.26 parser counted the disabled clip in the multi-source
        # basis; re-parses must keep the same coordinate convention or
        # stored selects export the wrong footage.
        assert parsed.is_multi_source

    def test_disabled_clip_is_not_rendered(self, parsed):
        # Not in the plan → its dialogue never reaches transcription.
        plan = plan_render(parsed)
        assert "/tmp/camB.mov" not in [p["input_path"] for p in plan]
        assert len(plan) == 2

    def test_writer_spine_index_stays_lockstep(self, parsed):
        from doza_assist.fcpxml.writer import _index_original_spine
        elements = _index_original_spine(parsed)
        assert [el.get("name") for el in elements] == ["A", "B", "C"]
        assert len(elements) == len(parsed.spine_segments)

    def test_select_on_enabled_clip_exports_right_footage(self, parsed):
        # Multi-source: timeline 65s is 5s into C. With the disabled B
        # filtered on BOTH sides, the select must still deep-copy rC (an
        # index skew would grab the wrong original element).
        out = write_selects_as_new_project(
            parsed, [Select(start_seconds=65, end_seconds=70, label="In C")])
        clips = etree.fromstring(out).findall(".//sequence/spine/asset-clip")
        assert [c.get("ref") for c in clips] == ["rC"]

    def test_select_on_disabled_clip_is_skipped(self, parsed):
        # Timeline 35s falls where B sits — B is not timeline content, so the
        # select is skipped and surfaced, not exported as disabled footage.
        skipped = []
        out = write_selects_as_new_project(
            parsed,
            [Select(start_seconds=5, end_seconds=8, label="In A"),
             Select(start_seconds=35, end_seconds=40, label="On B")],
            skipped_out=skipped,
        )
        assert [s.label for s in skipped] == ["On B"]
        clips = etree.fromstring(out).findall(".//sequence/spine/asset-clip")
        assert [c.get("ref") for c in clips] == ["rA"]

    def test_disabled_lane_clip_in_gap_is_skipped(self, tmp_path):
        p = tmp_path / "disabled_lane.fcpxml"
        p.write_text(DISABLED_CLIP_FIXTURE.replace(
            '<asset-clip ref="rB" offset="30s" name="B" start="0s" duration="30s" enabled="0" audioRole="dialogue"/>',
            '<gap name="Gap" offset="30s" start="3600s" duration="30s">'
            '<asset-clip ref="rB" lane="1" offset="3600s" name="B" start="0s" duration="30s" enabled="0"/>'
            '</gap>',
        ))
        parsed = parse_fcpxml(p)
        assert [s.name for s in parsed.spine_segments] == ["A", "B", "C"]
        b = parsed.spine_segments[1]
        assert not b.enabled
        assert b.audio_source is None or b.audio_source.is_muted

    def test_all_disabled_spine_parses_fully_muted(self, tmp_path):
        # Parse succeeds (segments keep their indices); nothing is renderable,
        # so the ingest-level no-audio gate rejects the project with its own
        # friendlier message.
        p = tmp_path / "all_disabled.fcpxml"
        p.write_text(DISABLED_CLIP_FIXTURE
                     .replace('name="A" start="0s" duration="30s"',
                              'name="A" start="0s" duration="30s" enabled="0"')
                     .replace('name="C" start="0s" duration="30s"',
                              'name="C" start="0s" duration="30s" enabled="0"'))
        parsed = parse_fcpxml(p)
        assert all(not s.enabled for s in parsed.spine_segments)
        assert all(s.audio_source is None or s.audio_source.is_muted
                   for s in parsed.spine_segments)
        assert plan_render(parsed) == []
