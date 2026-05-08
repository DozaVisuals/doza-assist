"""Tests for the timeline audio renderer.

The render_timeline_audio function shells out to ffmpeg — these tests cover
the planning and argv construction to avoid requiring ffmpeg + real WAVs in
CI. A companion integration assertion (`test_renders_real_wav`) is marked as
requiring ffmpeg on the PATH and is skipped when unavailable.
"""

import os
import shutil
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from doza_assist.fcpxml import parse_fcpxml  # noqa: E402
from doza_assist.fcpxml.timeline_audio import (  # noqa: E402
    TimelineAudioError,
    build_ffmpeg_command,
    plan_render,
    render_timeline_audio,
)


MIXED_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.14">
        <resources>
            <format id="r1" name="FF" frameDuration="1001/24000s" width="1920" height="1080"/>
            <asset id="rA" name="a1" start="0s" duration="240000/24000s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                <media-rep kind="original-media" src="file://{audio_a}"/>
            </asset>
            <asset id="rV" name="v" start="0s" duration="240000/24000s" hasVideo="1" videoSources="1">
                <media-rep kind="original-media" src="file:///tmp/v.mov"/>
            </asset>
            <asset id="rB" name="a2" start="0s" duration="240000/24000s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                <media-rep kind="original-media" src="file://{audio_b}"/>
            </asset>
            <media id="mcA" name="MC">
                <multicam>
                    <mc-angle name="V" angleID="v1"><asset-clip ref="rV" offset="0s" duration="240000/24000s"/></mc-angle>
                    <mc-angle name="A" angleID="a1"><asset-clip ref="rA" offset="0s" duration="240000/24000s" audioRole="dialogue"/></mc-angle>
                </multicam>
            </media>
        </resources>
        <library>
            <event name="E">
                <project name="Mix">
                    <sequence format="r1" duration="200s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                        <spine>
                            <mc-clip ref="mcA" offset="0s" name="mc" duration="50s">
                                <mc-source angleID="v1" srcEnable="video"/>
                                <mc-source angleID="a1" srcEnable="audio"/>
                            </mc-clip>
                            <sync-clip offset="50s" name="active" duration="30s">
                                <spine>
                                    <asset-clip ref="rB" offset="0s" duration="30s" audioRole="dialogue"/>
                                </spine>
                                <sync-source sourceID="storyline">
                                    <audio-role-source role="dialogue.dialogue-1" active="1"/>
                                </sync-source>
                            </sync-clip>
                            <sync-clip offset="100s" name="muted" duration="40s">
                                <spine>
                                    <asset-clip ref="rB" offset="0s" duration="40s" audioRole="dialogue"/>
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


@pytest.fixture
def parsed_mixed(tmp_path):
    audio_a = tmp_path / "a.wav"
    audio_b = tmp_path / "b.wav"
    # Empty files are enough for plan/argv tests; render test writes real WAVs.
    audio_a.write_bytes(b"")
    audio_b.write_bytes(b"")
    fcpxml = tmp_path / "mixed.fcpxml"
    fcpxml.write_text(MIXED_FIXTURE.format(audio_a=str(audio_a), audio_b=str(audio_b)))
    return parse_fcpxml(fcpxml), audio_a, audio_b


class TestPlanRender:
    def test_muted_segments_are_skipped(self, parsed_mixed):
        parsed, _, _ = parsed_mixed
        plan = plan_render(parsed)
        # 3 segments, 1 muted → 2 in the plan.
        assert len(plan) == 2

    def test_plan_has_timeline_offsets_in_ms(self, parsed_mixed):
        parsed, _, _ = parsed_mixed
        plan = plan_render(parsed)
        offsets = [p["timeline_offset_ms"] for p in plan]
        # Segments at offsets 0s and 50s.
        assert offsets == [0, 50000]

    def test_plan_preserves_source_paths(self, parsed_mixed):
        parsed, audio_a, audio_b = parsed_mixed
        plan = plan_render(parsed)
        paths = [p["input_path"] for p in plan]
        assert paths == [str(audio_a), str(audio_b)]


class TestBuildFfmpegCommand:
    def test_argv_has_anullsrc_base(self, parsed_mixed):
        parsed, _, _ = parsed_mixed
        argv = build_ffmpeg_command(parsed, "/tmp/out.wav", ffmpeg_bin="ffmpeg")
        # First -i must be anullsrc with sequence duration and target sample rate.
        assert argv[:5] == ["ffmpeg", "-y", "-nostdin", "-f", "lavfi"]
        anullsrc_idx = argv.index("-i") + 1
        assert argv[anullsrc_idx].startswith("anullsrc=")
        assert "d=200.000000" in argv[anullsrc_idx]
        assert "r=16000" in argv[anullsrc_idx]

    def test_argv_includes_one_input_per_unmuted_segment(self, parsed_mixed):
        parsed, audio_a, audio_b = parsed_mixed
        argv = build_ffmpeg_command(parsed, "/tmp/out.wav", ffmpeg_bin="ffmpeg")
        inputs = [argv[i + 1] for i, v in enumerate(argv) if v == "-i"]
        # anullsrc + 2 unmuted segments (third is muted and skipped).
        assert len(inputs) == 3
        assert str(audio_a) in inputs
        assert str(audio_b) in inputs

    def test_filter_complex_has_one_delay_per_segment(self, parsed_mixed):
        parsed, _, _ = parsed_mixed
        argv = build_ffmpeg_command(parsed, "/tmp/out.wav", ffmpeg_bin="ffmpeg")
        fc_idx = argv.index("-filter_complex") + 1
        fc = argv[fc_idx]
        # One atrim + one adelay per unmuted segment, plus one amix terminus.
        assert fc.count("atrim=") == 2
        assert fc.count("adelay=") == 2
        assert "amix=inputs=3" in fc  # anullsrc base + 2 segments

    def test_output_is_mono_16k_wav(self, parsed_mixed):
        parsed, _, _ = parsed_mixed
        argv = build_ffmpeg_command(parsed, "/tmp/out.wav", ffmpeg_bin="ffmpeg")
        assert "-ac" in argv and argv[argv.index("-ac") + 1] == "1"
        assert "-ar" in argv and argv[argv.index("-ar") + 1] == "16000"
        assert argv[-1] == "/tmp/out.wav"


# Multicam with non-zero tcStart: jam-synced / time-of-day timecode.
# The multicam tcStart and the asset's own start are both 186612/25s
# (= 7464.48s, mirroring the original Panasonic / RED / ARRI time-of-day TC
# from the bug report). Four mc-clips on the spine each pick a different
# one-quarter slice of the multicam by advancing `start` along the
# multicam's tc-space. The first two slices land in asset rA1; the last
# two in asset rA2.
TC_START_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.14">
        <resources>
            <format id="r1" name="FF" frameDuration="1001/24000s" width="1920" height="1080"/>
            <asset id="rA1" name="cam1" start="186612/25s" duration="100s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                <media-rep kind="original-media" src="file://{audio_1}"/>
            </asset>
            <asset id="rA2" name="cam2" start="186612/25s" duration="100s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                <media-rep kind="original-media" src="file://{audio_2}"/>
            </asset>
            <asset id="rV" name="v" start="0s" duration="200s" hasVideo="1" videoSources="1">
                <media-rep kind="original-media" src="file:///tmp/v.mov"/>
            </asset>
            <media id="mcTOD" name="TOD MC">
                <multicam tcStart="186612/25s">
                    <mc-angle name="V" angleID="v1">
                        <asset-clip ref="rV" offset="0s" duration="200s"/>
                    </mc-angle>
                    <mc-angle name="A" angleID="a1">
                        <asset-clip ref="rA1" offset="0s" start="186612/25s" duration="50s" audioRole="dialogue"/>
                        <asset-clip ref="rA2" offset="50s" start="186612/25s" duration="50s" audioRole="dialogue"/>
                    </mc-angle>
                </multicam>
            </media>
        </resources>
        <library>
            <event name="E">
                <project name="TOD">
                    <sequence format="r1" duration="100s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                        <spine>
                            <mc-clip ref="mcTOD" offset="0s" start="186612/25s" name="seg1" duration="25s">
                                <mc-source angleID="v1" srcEnable="video"/>
                                <mc-source angleID="a1" srcEnable="audio"/>
                            </mc-clip>
                            <mc-clip ref="mcTOD" offset="25s" start="187237/25s" name="seg2" duration="25s">
                                <mc-source angleID="v1" srcEnable="video"/>
                                <mc-source angleID="a1" srcEnable="audio"/>
                            </mc-clip>
                            <mc-clip ref="mcTOD" offset="50s" start="187862/25s" name="seg3" duration="25s">
                                <mc-source angleID="v1" srcEnable="video"/>
                                <mc-source angleID="a1" srcEnable="audio"/>
                            </mc-clip>
                            <mc-clip ref="mcTOD" offset="75s" start="188487/25s" name="seg4" duration="25s">
                                <mc-source angleID="v1" srcEnable="video"/>
                                <mc-source angleID="a1" srcEnable="audio"/>
                            </mc-clip>
                        </spine>
                    </sequence>
                </project>
            </event>
        </library>
    </fcpxml>
""")


class TestMulticamNonZeroTcStart:
    """Bug fix: multicam jam-synced / time-of-day timecode (tcStart != 0).

    Pre-fix: the asset-clip selector compared multicam-tc-space starts against
    zero-based asset-clip offsets, so every spine segment fell through to
    ``all_clips[0]`` (only the first .MOV got transcribed). Seek positions
    were also wrong because ``asset.start`` was added without subtracting
    ``asset.start``. This suite locks in the corrected behavior.
    """

    @pytest.fixture
    def parsed(self, tmp_path):
        audio_1 = tmp_path / "cam1.wav"
        audio_2 = tmp_path / "cam2.wav"
        audio_1.write_bytes(b"")
        audio_2.write_bytes(b"")
        fcpxml = tmp_path / "tod.fcpxml"
        fcpxml.write_text(TC_START_FIXTURE.format(audio_1=str(audio_1), audio_2=str(audio_2)))
        return parse_fcpxml(fcpxml), audio_1, audio_2

    def test_each_segment_resolves_to_correct_asset_clip(self, parsed):
        # First two spine segments fall in the first multicam asset-clip
        # (zero-based [0,50)); last two fall in the second ([50,100)).
        parsed_obj, audio_1, audio_2 = parsed
        paths = [s.audio_source.path for s in parsed_obj.spine_segments]
        assert paths == [str(audio_1), str(audio_1), str(audio_2), str(audio_2)]

    def test_all_segments_carry_container_tc_start(self, parsed):
        from fractions import Fraction
        parsed_obj, _, _ = parsed
        for seg in parsed_obj.spine_segments:
            # 7464.48 = 186612/25
            assert seg.audio_source.container_tc_start_fraction == Fraction(186612, 25)
            assert seg.audio_source.asset_start_fraction == Fraction(186612, 25)

    def test_seek_positions_are_zero_based_into_source(self, parsed):
        parsed_obj, _, _ = parsed
        plan = plan_render(parsed_obj)
        # 4 spine clips slicing the multicam at 0/25/50/75s. After tcStart
        # subtraction the first asset-clip provides [0,25) and [25,50);
        # the second asset-clip provides [0,25) and [25,50) again.
        starts = [round(p["source_start_seconds"], 4) for p in plan]
        assert starts == [0.0, 25.0, 0.0, 25.0]

    def test_is_multi_source_true_with_distinct_files(self, parsed):
        parsed_obj, _, _ = parsed
        assert parsed_obj.is_multi_source is True

    def test_metadata_dict_round_trips_tc_fields(self, parsed):
        import json
        parsed_obj, _, _ = parsed
        data = parsed_obj.to_metadata_dict()
        round_tripped = json.loads(json.dumps(data))
        first = round_tripped["spine_segments"][0]["audio_source"]
        assert first["container_tc_start_fraction"] == "186612/25"
        assert first["asset_start_fraction"] == "186612/25"


class TestMulticamZeroTcStartUnchanged:
    """Regression guard: tcStart=0 path must produce identical seek math to
    the pre-fix formula (``seg.start - angle_offset + angle_start``)."""

    def test_existing_mixed_fixture_seek_unchanged(self, parsed_mixed):
        parsed, _, _ = parsed_mixed
        plan = plan_render(parsed)
        # mc-clip starts at 0s, sync-clip at 50s — same as before the fix.
        starts = [round(p["source_start_seconds"], 6) for p in plan]
        assert starts == [0.0, 0.0]


class TestRenderErrors:
    def test_raises_when_source_missing(self, tmp_path):
        # Point the fixture at a path that doesn't exist; parse succeeds
        # because the parser only resolves refs, not disk. The renderer checks.
        audio_a = tmp_path / "missing_a.wav"
        audio_b = tmp_path / "missing_b.wav"
        fcpxml = tmp_path / "x.fcpxml"
        fcpxml.write_text(MIXED_FIXTURE.format(audio_a=str(audio_a), audio_b=str(audio_b)))
        parsed = parse_fcpxml(fcpxml)

        with pytest.raises(TimelineAudioError, match="missing audio source"):
            render_timeline_audio(parsed, str(tmp_path / "out.wav"))


# ---------- integration: real ffmpeg -----------------------------------------

HAS_FFMPEG = shutil.which("ffmpeg") is not None


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
class TestRealRender:
    """End-to-end: build actual synthetic WAVs, render, verify output exists
    and has roughly the right duration. Skips without ffmpeg on PATH."""

    def _make_silent_wav(self, path: Path, duration_seconds: int = 60):
        import subprocess
        subprocess.run([
            "ffmpeg", "-y", "-nostdin",
            "-f", "lavfi", "-i", f"anullsrc=r=16000:cl=mono:d={duration_seconds}",
            "-acodec", "pcm_s16le", str(path),
        ], capture_output=True, check=True)

    def test_renders_wav_of_expected_duration(self, tmp_path):
        audio_a = tmp_path / "a.wav"
        audio_b = tmp_path / "b.wav"
        self._make_silent_wav(audio_a, 60)
        self._make_silent_wav(audio_b, 60)

        fcpxml = tmp_path / "mixed.fcpxml"
        fcpxml.write_text(MIXED_FIXTURE.format(audio_a=str(audio_a), audio_b=str(audio_b)))
        parsed = parse_fcpxml(fcpxml)

        out = tmp_path / "timeline.wav"
        render_timeline_audio(parsed, str(out))

        assert out.exists()
        # Probe duration with ffprobe (skip cleanly if unavailable).
        if shutil.which("ffprobe"):
            import subprocess
            result = subprocess.run([
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(out),
            ], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                dur = float(result.stdout.strip())
                # Sequence is 200s; allow a small margin.
                assert 199.0 <= dur <= 201.0
