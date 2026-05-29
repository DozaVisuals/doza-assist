"""
Source media probing helpers (extracted from app.py).

Both export routes need the source video's resolution and frame rate to
generate timeline metadata. Previously this logic was duplicated inline in
two places; now it lives here and both routes call into it.
"""

import os
import re
import shutil
import subprocess

STANDARD_FRAMERATES = [
    23.976, 24.0, 25.0, 29.97, 30.0,
    48.0, 50.0, 59.94, 60.0,
    100.0, 120.0,
]


def snap_framerate(fps: float) -> float:
    """Snap a raw probed fps to the nearest supported standard rate.

    Exact-rate footage (24/25/30/48/50/60/100/120 and their NTSC pulldowns)
    snaps to itself. Without 48/50/100/120 in the table a 50fps clip used to
    snap to 59.94, which exported on the wrong frame grid."""
    return min(STANDARD_FRAMERATES, key=lambda s: abs(s - fps))


def _find_ffprobe() -> str | None:
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
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
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
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            num, den = result.stdout.strip().split("/")
            fps = float(num) / float(den)
            return snap_framerate(fps)
    except Exception:
        pass
    return None


_TIMECODE_RE = re.compile(r"^(\d+):(\d+):(\d+)([:;])(\d+)$")


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
    total = ((hh * 3600 + mm * 60 + ss) * nominal) + ff
    if sep == ";" and nominal % 30 == 0:
        drop_per_min = 2 * (nominal // 30)
        minutes = hh * 60 + mm
        total -= drop_per_min * (minutes - minutes // 10)
    return total


def get_video_start_timecode_frames(path: str, framerate: float) -> int:
    """Return the media's embedded start timecode in whole frames (0 if none).

    DJI, Sony, and many cameras stamp time-of-day timecode. Final Cut keys an
    asset's source timecode off it, so an FCPXML that exports 0-based edits
    against such media is rejected with "Invalid edit with no respective media"
    — the edits fall outside the media's real timecode range. Reads the
    `timecode` tag from the format or any stream (e.g. a `tmcd` track)."""
    if not path or not os.path.exists(path):
        return 0
    ffprobe = _find_ffprobe()
    if not ffprobe:
        return 0
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "quiet",
                "-show_entries", "format_tags=timecode:stream_tags=timecode",
                "-of", "default=nw=1:nk=1",
                path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                frames = timecode_to_frames(line.strip(), framerate)
                if frames:
                    return frames
    except Exception:
        pass
    return 0
