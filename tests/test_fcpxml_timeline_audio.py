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
