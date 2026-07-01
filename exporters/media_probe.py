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


def _find_ffmpeg() -> str | None:
    """Resolve ffmpeg (bundled first), mirroring _find_ffprobe. Needed for the
    per-stream loudness probe (volumedetect) that finds the dialogue channel."""
    bundled_dir = os.environ.get("DOZA_FFMPEG_DIR")
    if bundled_dir:
        candidate = os.path.join(bundled_dir, "ffmpeg")
        if os.path.isfile(candidate):
            return candidate
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    for candidate in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if os.path.isfile(candidate):
            return candidate
    return None


def _first_csv_row(stdout: str) -> str:
    """First non-empty row of ffprobe csv output.

    ffprobe prints each stream once per enclosing section: containers with
    programs (MPEG-TS) list streams under the program AND at top level, so
    even a ``V:0``-selected probe yields its row twice. Parsing the joined
    blob raised ValueError and silently dropped every probe below to its
    fallback for TS sources."""
    for line in stdout.splitlines():
        if line.strip():
            return line.strip()
    return ""


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
            parts = _first_csv_row(result.stdout).split(",")
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
            num, den = _first_csv_row(result.stdout).split("/")
            fps = float(num) / float(den)
            # Implausible rates (cover-art 90000/1, broken streams) must not
            # snap to a "standard" rate — treat as no-video-stream instead.
            if fps < 1.0 or fps > 240.0:
                return None
            return snap_framerate(fps)
    except Exception:
        pass
    return None


def has_video_stream(path: str) -> bool | None:
    """Whether the file carries a REAL video stream (attached-picture cover
    art excluded), or None when the question can't be answered (missing
    file / no ffprobe / probe failure).

    The FCPXML exporters used to decide ``hasVideo`` from the file
    EXTENSION, so an audio-only .mp4/.mov (AAC podcast export, multi-mono
    field-recorder QuickTime) declared a video component the media doesn't
    have — FCP/Resolve import such an asset offline/invalid instead of as a
    clean audio clip. Callers keep the extension heuristic only for the
    None case.
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
                # Capital V excludes attached pictures (see resolution probe).
                "-select_streams", "V",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
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
            channels = int(_first_csv_row(result.stdout).split(",")[0])
            if channels > 0:
                return channels
    except Exception:
        pass
    return None


def get_audio_sample_rate(path: str) -> int | None:
    """Sample rate (Hz) of the first audio stream, or None if no audio/probe fail.

    Used to declare an FCPXML asset's ``audioRate`` so Resolve routes the
    clip audio (Resolve maps FCPXML audio from the declared rate/channels,
    not the file's track table). Pro media is overwhelmingly 48000.
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
                "-show_entries", "stream=sample_rate",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            rate = int(_first_csv_row(result.stdout).split(",")[0])
            if rate > 0:
                return rate
    except Exception:
        pass
    return None


def get_audio_layout(path: str):
    """``(num_audio_streams, total_channels)`` across ALL audio streams,
    ``(0, 0)`` when the file probed clean but has NO audio streams (silent
    B-roll / FX plate), or None when the probe itself failed (missing file,
    no ffprobe, timeout) — callers use the distinction to declare
    ``hasAudio="0"`` for genuinely silent media without failing open on a
    probe error.

    FCP declares a clip's asset as ``audioSources=<#streams> audioChannels=<total>``:
    a stereo camera file is ``(1, 2)``, a mono file ``(1, 1)``, and a 4-mono-track
    MXF ``(4, 4)``. Resolve needs the TOTAL channel count to bring every track's
    audio onto the timeline — declaring only the FIRST stream's channels
    (``get_audio_channels`` → 1 for multi-mono MXF) imports just track 1, which is
    silent when the dialogue is on another mono track or the editor transcribed
    "all tracks (mixed)" and the lav lives elsewhere.
    """
    if not path or not os.path.exists(path):
        return None
    ffprobe = _find_ffprobe()
    if not ffprobe:
        return None
    try:
        # Query the stream INDEX alongside channels so we can dedup: MPEG-TS
        # lists each stream twice (top-level + program section), which would
        # otherwise double the source/channel count for the .ts (and
        # .ts-content-named-.mp4) media this app ingests — the same double-
        # listing _first_csv_row guards against for the single-row probes.
        result = subprocess.run(
            [
                ffprobe, "-v", "quiet",
                "-select_streams", "a",
                "-show_entries", "stream=index,channels",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        by_index = {}  # stream index -> channel count (dedups TS double-listing)
        rows_seen = False
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            rows_seen = True
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                idx, ch = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if ch > 0:
                by_index[idx] = ch
        if not by_index:
            # No rows at all = the probe ANSWERED "no audio streams" -> (0, 0).
            # Rows that all failed to parse (N/A channels) = audio exists but
            # we can't describe it -> None, same as a probe failure.
            return (0, 0) if not rows_seen else None
        return (len(by_index), sum(by_index.values()))
    except Exception:
        return None


# Per-(path,size,mtime) memo so re-exporting a source doesn't re-probe loudness.
_dialogue_channel_cache: dict = {}


def _mean_volume_db(ffmpeg: str, path: str, stream_idx: int,
                    start: float, seg: float):
    """Mean volume (dB) of ONE audio stream over a sample window, via ffmpeg
    `volumedetect`. None on failure. Digital silence reads ≈ -91 dB / -inf.

    Input-seek (`-ss` before `-i`) so a multi-GB master isn't decoded from the
    top — sample-window accuracy is irrelevant for a loudness average."""
    try:
        cmd = [ffmpeg, "-nostdin", "-hide_banner", "-nostats"]
        if start and start > 0:
            cmd += ["-ss", f"{start:.3f}"]
        cmd += ["-t", f"{max(1.0, seg):.3f}", "-i", path,
                "-map", f"0:a:{stream_idx}", "-af", "volumedetect",
                "-f", "null", "-"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        m = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", result.stderr or "")
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


def detect_dialogue_channels(path: str, sample_seconds: float = 90.0,
                             floor_db: float = -60.0, rel_db: float = 25.0):
    """0-based indices of the audio streams carrying usable audio (the lav /
    dialogue track[s]), loudest first — or None.

    For a broadcast multi-mono MXF (e.g. 4 discrete mono tracks, lav on one,
    the rest silent scratch/room), this returns the speech-bearing track(s) so
    the FCPXML export can route ONLY them and the editor gets clean dialogue
    instead of an empty channel. Returns None for single-stream sources (no
    disambiguation needed) and when nothing can be measured.

    Method: ffmpeg `volumedetect` on a ~90s window per stream (export-time
    cost, not a full decode). "Active" = the loudest stream plus any within
    `rel_db` of it and above `floor_db`; empty PCM tracks read ≈ -91 dB and
    drop out. Loudness can't separate speech from music — but it cleanly
    separates SILENT tracks, which is the multi-mono case here; the editor can
    always override via the audio selector. Result is memoized per file.
    """
    if not path or not os.path.exists(path):
        return None
    layout = get_audio_layout(path)
    if not layout or layout[0] <= 1:
        return None  # single audio stream — nothing to disambiguate
    n_streams, n_channels = layout
    # Only the all-mono case maps stream index -> source channel cleanly
    # (stream i == srcCh i+1). Mixed/multi-channel streams: leave alone.
    if n_streams != n_channels:
        return None
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return None
    try:
        st = os.stat(path)
        key = (path, st.st_size, int(st.st_mtime))
    except OSError:
        key = None
    if key is not None and key in _dialogue_channel_cache:
        return _dialogue_channel_cache[key]

    dur = get_media_duration(path) or 0.0
    start = max(0.0, dur * 0.15) if dur else 0.0
    seg = min(sample_seconds, max(10.0, dur - start)) if dur else sample_seconds

    measured = []  # (stream_idx, mean_db)
    for i in range(n_streams):
        db = _mean_volume_db(ffmpeg, path, i, start, seg)
        if db is not None:
            measured.append((i, db))

    result = None
    if measured:
        loudest = max(db for _, db in measured)
        active = [i for (i, db) in measured
                  if db >= loudest - rel_db and db > floor_db]
        active.sort(key=lambda i: dict(measured)[i], reverse=True)
        result = active or None
    if key is not None:
        _dialogue_channel_cache[key] = result
    return result


def get_media_container_format(path: str) -> str | None:
    """Container format name(s) via ffprobe, or None on any probe trouble.

    Returns the raw first-row string, e.g. ``'mpegts'``, ``'wav'``, or the
    comma-separated demuxer family ``'mov,mp4,m4a,3gp,3g2,mj2'`` (csv=p=0
    quotes comma-containing values — callers should substring-match, not
    compare equality). Used to warn editors when an export references
    MPEG-TS broadcast media, which FCP/Premiere/Resolve cannot decode.
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
                "-show_entries", "format=format_name",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            row = _first_csv_row(result.stdout)
            if row:
                return row
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
                duration = float(_first_csv_row(result.stdout).split(",")[0])
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
    """Whole-frame start timecode (0 if none) — see get_video_start_timecode."""
    return get_video_start_timecode_info(path, framerate)[0]


def get_video_start_timecode_info(path: str, framerate: float) -> tuple[int, str]:
    """Return (start timecode in whole frames, "DF"|"NDF") — thin export
    wrapper over get_video_start_timecode, kept BYTE-COMPATIBLE with the
    pre-refactor behavior so the FCPXML export call sites and their v3.5.7
    invariant are untouched: zero frames always reports "NDF" (the old
    code returned the (0, "NDF") fallthrough for zero tags regardless of
    the tag's separator)."""
    tc = get_video_start_timecode(path, framerate)
    if not tc or tc["frames"] == 0:
        return 0, "NDF"
    return tc["frames"], ("DF" if tc["drop"] else "NDF")


def get_video_start_timecode(path: str, framerate: float) -> dict | None:
    """Embedded start timecode, or None when absent.

    Shape: ``{'frames': int, 'drop': bool, 'raw': str, 'source': str}`` —
    frames on the nominal grid, drop-frame flag, the raw tag/derivation,
    and which mechanism carried it ('tmcd' or 'bwf'). A legitimate
    ``00:00:00:00`` tmcd tag returns ``{'frames': 0, ...}`` — callers must
    distinguish "zero TC" (display layers still know TC exists) from
    "no TC" (None). Dual-tag media prefers the first NONZERO candidate.

    DJI, Sony, and many cameras stamp time-of-day timecode. Final Cut keys
    an asset's source timecode off the media's real timecode TRACK
    (``tmcd``), so an FCPXML that exports 0-based edits against such media
    is rejected with "Invalid edit with no respective media".

    The inverse bit us on Sony XAVC-S: MP4 containers cannot carry a
    ``tmcd`` track, but Sony stamps a ``timecode`` metadata TAG (alongside
    an ``rtmd`` data track). FCP ignores the tag and treats such files as
    starting at 0 — so for QuickTime-family containers we honor embedded
    timecode ONLY when a real ``tmcd`` stream exists, matching what FCP
    keys off. The same gate governs the DISPLAY layer so on-screen TC
    always matches what an export (and FCP) will say.

    MXF is the exception: it carries SMPTE-12M timecode in its structural
    metadata, which ffprobe surfaces as a format-level (or data-stream)
    ``timecode`` tag rather than a ``tmcd`` track. FCP and Resolve both
    anchor an MXF asset to that embedded TC, so we honor the tag for MXF
    even without a tmcd stream — broadcast/camera MXF routinely starts at a
    non-zero record TC (e.g. 00:54:44:12), and exporting 0-based edits
    against it makes every clip land outside the asset's range.

    BWF field-recorder WAV/AIFF stamp time-of-day TC as bext
    time_reference (samples since midnight) — FCP anchors such assets
    there, so those are honored too.
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
                "-print_format", "json",
                "-show_entries",
                "stream=codec_type,codec_tag_string,sample_rate:format=format_name:stream_tags=timecode:format_tags=timecode,time_reference",
                path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout or "{}")
        streams = data.get("streams") or []
        fmt = data.get("format") or {}
        fmt_tags = fmt.get("tags") or {}
        tmcd_streams = [
            s for s in streams
            if (s.get("codec_tag_string") or "").lower() == "tmcd"
        ]
        # MXF anchors to its embedded SMPTE-12M timecode (surfaced as a
        # format-/stream-level `timecode` tag, no tmcd track), so honor the
        # tag for MXF even without tmcd. The tmcd gate stays for QuickTime
        # containers to keep rejecting the Sony XAVC-S MP4 tag FCP ignores.
        container = (fmt.get("format_name") or "").lower()
        is_mxf = ("mxf" in container
                  or os.path.splitext(path)[1].lower() == ".mxf")
        if tmcd_streams or is_mxf:
            # Prefer a tmcd stream's own tag, then any other stream tag, then
            # the container-level tag (muxers — MXF especially — vary in where
            # they stamp it). The separator before FF carries drop-frame-ness.
            candidates = []
            for s in tmcd_streams + streams:
                tc = ((s.get("tags") or {}).get("timecode") or "").strip()
                if tc:
                    candidates.append(tc)
            fmt_tc = (fmt_tags.get("timecode") or "").strip()
            if fmt_tc:
                candidates.append(fmt_tc)
            source = "tmcd" if tmcd_streams else "mxf"
            zero_tc = None
            for tc in candidates:
                frames = timecode_to_frames(tc, framerate)
                if frames is None:
                    continue
                m = _TIMECODE_RE.match(tc.strip())
                is_drop = bool(m) and m.group(4) in (";", ".", ",")
                if frames > 0:
                    # First NONZERO wins: dual-tag media (a zero container
                    # tag plus the real camera TC on the tmcd stream, or
                    # vice versa) must keep resolving to the real one.
                    return {"frames": frames, "drop": is_drop,
                            "raw": tc.strip(), "source": source}
                if zero_tc is None:
                    zero_tc = {"frames": 0, "drop": is_drop,
                               "raw": tc.strip(), "source": source}
            # Only zero tags found: the media genuinely starts at zero TC.
            return zero_tc
        # No tmcd track — BWF bext time_reference (samples since midnight).
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
                frames = int(round(seconds * framerate))
                return {"frames": frames, "drop": False,
                        "raw": f"bext:{time_ref}", "source": "bwf"}
    except Exception:
        pass
    return None


def frames_to_timecode_label(total_frames: int, framerate: float, drop: bool = False) -> str:
    """Inverse of timecode_to_frames: render a nominal-grid frame count as
    an SMPTE label, re-inserting drop-frame skips when ``drop`` is True."""
    nominal = int(round(framerate))
    if nominal <= 0:
        return "00:00:00:00"
    frames = max(0, int(total_frames))
    if drop and nominal % 30 == 0:
        # Re-insert the dropped frame numbers: 2 per minute (×nominal/30),
        # except every tenth minute.
        drop_per_min = 2 * (nominal // 30)
        frames_per_min = nominal * 60 - drop_per_min
        frames_per_10min = nominal * 600 - drop_per_min * 9
        d10 = frames // frames_per_10min
        rem = frames % frames_per_10min
        if rem < nominal * 60:
            m_extra = 0
        else:
            m_extra = 1 + (rem - nominal * 60) // frames_per_min
        frames += drop_per_min * (d10 * 9 + m_extra)
        sep = ';'
    else:
        sep = ';' if drop else ':'
    ff = frames % nominal
    total_seconds = frames // nominal
    ss = total_seconds % 60
    mm = (total_seconds // 60) % 60
    hh = total_seconds // 3600
    return f"{hh:02d}:{mm:02d}:{ss:02d}{sep}{ff:02d}"
