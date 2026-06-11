"""
Source media probing helpers (extracted from app.py).

Both export routes need the source video's resolution and frame rate to
generate timeline metadata. Previously this logic was duplicated inline in
two places; now it lives here and both routes call into it.
"""

import json
import os
import re
import shutil
import subprocess

STANDARD_FRAMERATES = [
    23.976, 24.0, 25.0, 29.97, 30.0,
    48.0, 50.0, 59.94, 60.0,
    100.0, 119.88, 120.0,
]


def snap_framerate(fps: float) -> float:
    """Snap a raw probed fps to the nearest supported standard rate.

    Exact-rate footage (24/25/30/48/50/60/100/120 and their NTSC pulldowns)
    snaps to itself. Without 48/50/100/120 in the table a 50fps clip used to
    snap to 59.94, which exported on the wrong frame grid."""
    return min(STANDARD_FRAMERATES, key=lambda s: abs(s - fps))


def _find_ffprobe() -> str | None:
    # Bundled binary first: packaged installs may have NO ffprobe on PATH,
    # and every probe falling back silently (default fps, 1920x1080, no
    # start TC -> FCP "invalid edit" rejections) is the worst failure mode.
    # Same resolution order as app.py's duration route.
    bundled_dir = os.environ.get("DOZA_FFMPEG_DIR")
    if bundled_dir:
        candidate = os.path.join(bundled_dir, "ffprobe")
        if os.path.isfile(candidate):
            return candidate
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        return ffprobe
    for candidate in ("/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe"):
        if os.path.isfile(candidate):
            return candidate
    return None


def get_video_resolution(path: str) -> tuple[int, int]:
    """Detect (width, height) using ffprobe. Falls back to 1920x1080."""
    if not path or not os.path.exists(path):
        return 1920, 1080
    ffprobe = _find_ffprobe()
    if not ffprobe:
        return 1920, 1080
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "quiet",
                # Capital V excludes attached pictures — album art in
                # mp3/m4a/flac otherwise probes as a "video" stream and
                # poisons the timeline with cover-art dimensions.
                "-select_streams", "V:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            if len(parts) >= 2:
                width, height = int(parts[0]), int(parts[1])
                if width > 0 and height > 0:
                    return width, height
    except Exception:
        pass
    return 1920, 1080


def get_video_framerate(path: str) -> float | None:
    """Detect frame rate using ffprobe, snapped to nearest standard rate."""
    if not path or not os.path.exists(path):
        return None
    ffprobe = _find_ffprobe()
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "quiet",
                # Capital V: skip attached_pic streams (see resolution probe).
                "-select_streams", "V:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            num, den = result.stdout.strip().split("/")
            fps = float(num) / float(den)
            # Implausible rates (cover-art 90000/1, broken streams) must not
            # snap to a "standard" rate — treat as no-video-stream instead.
            if fps < 1.0 or fps > 240.0:
                return None
            return snap_framerate(fps)
    except Exception:
        pass
    return None


def get_audio_channels(path: str) -> int | None:
    """Channel count of the first audio stream (None if no audio/probe fail).

    The Premiere exporter previously hardcoded stereo: mono interview WAVs
    got an A2 clipitem referencing a channel the file doesn't have.
    """
    if not path or not os.path.exists(path):
        return None
    ffprobe = _find_ffprobe()
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "quiet",
                "-select_streams", "a:0",
                "-show_entries", "stream=channels",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            channels = int(result.stdout.strip().split(",")[0])
            if channels > 0:
                return channels
    except Exception:
        pass
    return None


def get_media_duration(path: str) -> float | None:
    """Detect the container duration in seconds using ffprobe.

    Exports used to fall back to ``transcript['duration']`` — the end of the
    last *spoken word* — as the media length. Any selected clip extending past
    the final sentence (trailing B-roll, music, room tone) was clamped to that
    shorter grid, and a clip living entirely in the tail was silently dropped.
    The real container duration is the correct clamp bound.
    """
    if not path or not os.path.exists(path):
        return None
    ffprobe = _find_ffprobe()
    if not ffprobe:
        return None
    try:
        # Prefer the VIDEO stream's own duration: the container duration is
        # >= the stream's, and rounding a longer container value up declares
        # a final frame the video doesn't have (import-time edit validation
        # checks the declared range). Audio-only files fall through to the
        # container value.
        result = subprocess.run(
            [
                ffprobe, "-v", "quiet",
                "-select_streams", "V:0",
                "-show_entries", "stream=duration",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            try:
                duration = float(result.stdout.strip().split(",")[0])
                if duration > 0:
                    return duration
            except ValueError:
                pass  # e.g. "N/A" — fall through to container duration
        result = subprocess.run(
            [
                ffprobe, "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            duration = float(result.stdout.strip())
            if duration > 0:
                return duration
    except Exception:
        pass
    return None


# Separator before FF: ':' = non-drop; ';' (and the rarer '.'/',') = drop.
_TIMECODE_RE = re.compile(r"^(\d+):(\d+):(\d+)([:;.,])(\d+)$")


def timecode_to_frames(tc: str, framerate: float) -> int | None:
    """Convert an SMPTE timecode string ("HH:MM:SS:FF", or ";"/"." for
    drop-frame) to a whole-frame count on the framerate's grid. Returns None if
    the string isn't a timecode.

    Frames are counted at the nominal integer rate (29.97 -> 30, 23.976 -> 24).
    Drop-frame (`;`) drops 2 frames per minute except every tenth minute (4 at
    59.94). This frame index is exactly what FCP stores as an asset's source
    timecode: asset.start = frames * frameDuration."""
    m = _TIMECODE_RE.match((tc or "").strip())
    if not m:
        return None
    hh, mm, ss, sep, ff = (int(m.group(1)), int(m.group(2)), int(m.group(3)),
                           m.group(4), int(m.group(5)))
    nominal = int(round(framerate))
    if nominal <= 0:
        return None
    # Reject malformed fields instead of converting them to garbage frame
    # counts (a corrupt tag is worse silent than absent).
    if mm > 59 or ss > 59 or ff >= max(nominal, 1):
        return None
    total = ((hh * 3600 + mm * 60 + ss) * nominal) + ff
    if sep in (";", ".", ",") and nominal % 30 == 0:
        drop_per_min = 2 * (nominal // 30)
        minutes = hh * 60 + mm
        total -= drop_per_min * (minutes - minutes // 10)
    return total


def get_video_start_timecode_frames(path: str, framerate: float) -> int:
    """Whole-frame start timecode (0 if none) — see get_video_start_timecode_info."""
    return get_video_start_timecode_info(path, framerate)[0]


def get_video_start_timecode_info(path: str, framerate: float) -> tuple[int, str]:
    """Return (start timecode in whole frames, "DF"|"NDF") for the media.

    DJI, Sony, and many cameras stamp time-of-day timecode. Final Cut keys an
    asset's source timecode off the media's real timecode TRACK (``tmcd``), so
    an FCPXML that exports 0-based edits against such media is rejected with
    "Invalid edit with no respective media" — the edits fall outside the
    media's real timecode range.

    The inverse bit us on Sony XAVC-S: MP4 containers cannot carry a ``tmcd``
    track, but Sony stamps a ``timecode`` metadata TAG (alongside an ``rtmd``
    data track). FCP ignores the tag and treats such files as starting at 0 —
    exporting tag-derived TC offsets put every clip outside the media and
    produced the same rejection in the other direction. So: honor the embedded
    timecode ONLY when the container carries a real ``tmcd`` stream, matching
    what FCP itself keys off.
    """
    if not path or not os.path.exists(path):
        return 0, "NDF"
    ffprobe = _find_ffprobe()
    if not ffprobe:
        return 0, "NDF"
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "quiet",
                "-print_format", "json",
                "-show_entries",
                "stream=codec_type,codec_tag_string,sample_rate:stream_tags=timecode:format_tags=timecode,time_reference",
                path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return 0, "NDF"
        data = json.loads(result.stdout or "{}")
        streams = data.get("streams") or []
        fmt_tags = (data.get("format") or {}).get("tags") or {}
        tmcd_streams = [
            s for s in streams
            if (s.get("codec_tag_string") or "").lower() == "tmcd"
        ]
        if tmcd_streams:
            # Prefer the tmcd stream's own tag, then any other stream tag,
            # then the container-level tag (muxers vary in where they stamp
            # it). The separator before FF carries drop-frame-ness.
            candidates = []
            for s in tmcd_streams + streams:
                tc = ((s.get("tags") or {}).get("timecode") or "").strip()
                if tc:
                    candidates.append(tc)
            fmt_tc = (fmt_tags.get("timecode") or "").strip()
            if fmt_tc:
                candidates.append(fmt_tc)
            for tc in candidates:
                frames = timecode_to_frames(tc, framerate)
                if frames:
                    m = _TIMECODE_RE.match(tc.strip())
                    is_drop = bool(m) and m.group(4) in (";", ".", ",")
                    return frames, ("DF" if is_drop else "NDF")
            return 0, "NDF"
        # No tmcd track. BWF field-recorder WAV/AIFF stamp time-of-day TC as
        # bext time_reference (samples since midnight) — FCP anchors such
        # assets there, so 0-based exports hit the same "invalid edit"
        # rejection the tmcd path fixes for camera files.
        time_ref = (fmt_tags.get("time_reference") or "").strip()
        if time_ref and time_ref.isdigit() and int(time_ref) > 0:
            sample_rate = 0
            for s in streams:
                if s.get("codec_type") == "audio":
                    try:
                        sample_rate = int(s.get("sample_rate") or 0)
                    except (TypeError, ValueError):
                        sample_rate = 0
                    break
            if sample_rate > 0:
                seconds = int(time_ref) / sample_rate
                return int(round(seconds * framerate)), "NDF"
    except Exception:
        pass
    return 0, "NDF"
