"""FCPXML ingestion for My Style.

Wraps the existing ``doza_assist.fcpxml`` parser/renderer (the same one the
project ingest path uses) so that an editor can drop in a finished FCPXML
export and we extract the spoken-word audio that survived the cut. That audio
goes through the normal Parakeet/Whisper transcription pipeline and the
resulting transcript represents the "story DNA" of the finished piece.

Two responsibilities live here that the project ingest path doesn't need:

  1. We always render a single timeline-ordered WAV (even for single-source
     spines), because ``transcribe_file`` needs one input and we want
     timeline-relative timestamps regardless of how the original timeline was
     structured. The project path can short-circuit to the source asset for
     single-source cases; My Style cannot.

  2. Missing-media reporting is non-blocking. The PRD says: if a clip
     references a file that's not on disk, skip the clip, log it, and continue
     with what we can resolve. We surface the missing files in the returned
     metadata so the UI can display a non-blocking notice.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Add the parent of editorial_dna/ so we can import the sibling fcpxml module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from doza_assist.fcpxml import (  # noqa: E402
    parse_fcpxml,
    ParseError,
)
from doza_assist.fcpxml.timeline_audio import (  # noqa: E402
    render_timeline_audio,
    TimelineAudioError,
)


SUPPORTED_FCPXML_EXTENSIONS = {'.fcpxml', '.fcpxmld'}


@dataclass
class IngestResult:
    """Result of staging an FCPXML for transcription.

    audio_path        : the WAV that should be passed to transcribe_file()
    stored_fcpxml_path: a copy of the FCPXML inside the staging directory
    fcpxml_metadata   : parsed metadata dict suitable for source_files.json
    duration_seconds  : timeline duration of the rendered audio
    missing_media     : list of audio paths the FCPXML referenced that weren't
                        on disk. Empty when nothing is missing.
    """

    audio_path: str
    stored_fcpxml_path: str
    fcpxml_metadata: dict
    duration_seconds: float
    missing_media: List[str]


def is_fcpxml_filename(name: str) -> bool:
    if not name:
        return False
    lower = name.lower().rstrip('/')
    return any(lower.endswith(ext) for ext in SUPPORTED_FCPXML_EXTENSIONS)


def resolve_fcpxml_inner_path(source_path: str) -> str:
    """``.fcpxmld`` is a bundle directory; return the path to ``Info.fcpxml``
    inside. ``.fcpxml`` files pass through unchanged.
    """
    p = source_path.rstrip('/')
    if os.path.isdir(p) and p.lower().endswith('.fcpxmld'):
        inner = os.path.join(p, 'Info.fcpxml')
        if not os.path.isfile(inner):
            raise ValueError(f'FCPXML bundle missing Info.fcpxml: {p}')
        return inner
    return p


def stage_fcpxml(fcpxml_path: str, staging_dir: str) -> IngestResult:
    """Parse the FCPXML, render its dialogue timeline to a WAV, and stage both
    inside ``staging_dir`` so they outlive any temp file cleanup.

    Raises:
        ValueError: when the FCPXML can't be parsed at all, or when EVERY
            referenced audio file is missing from disk (nothing to transcribe).
        TimelineAudioError: when ffmpeg fails on an otherwise-valid timeline.

    Partial-missing-media is non-blocking — those segments get reported via
    ``IngestResult.missing_media`` and the rendered audio just omits them.
    """
    inner_path = resolve_fcpxml_inner_path(fcpxml_path)

    try:
        parsed = parse_fcpxml(inner_path)
    except ParseError as e:
        raise ValueError(f'Could not read FCPXML: {e}') from e

    os.makedirs(staging_dir, exist_ok=True)

    # Stash a copy of the FCPXML so the source folder is self-contained even
    # after the original drop is gone.
    fcpxml_copy_name = os.path.basename(inner_path) or 'source.fcpxml'
    stored_fcpxml_path = os.path.join(staging_dir, fcpxml_copy_name)
    if os.path.abspath(inner_path) != os.path.abspath(stored_fcpxml_path):
        shutil.copy2(inner_path, stored_fcpxml_path)

    # Walk every referenced audio source and split into present/missing.
    referenced_paths: list[str] = []
    seen: set[str] = set()
    for seg in parsed.spine_segments:
        if seg.audio_source is None:
            continue
        p = seg.audio_source.path
        if p in seen:
            continue
        seen.add(p)
        referenced_paths.append(p)

    missing_media = [p for p in referenced_paths if not os.path.exists(p)]
    present_count = len(referenced_paths) - len(missing_media)

    if referenced_paths and present_count == 0:
        raise ValueError(
            'FCPXML parsed OK, but none of the audio files it references are '
            'on disk right now. Re-mount the edit drive(s) and try again.'
        )

    timeline_wav = os.path.join(staging_dir, 'timeline_audio.wav')
    try:
        render_timeline_audio(parsed, timeline_wav)
    except TimelineAudioError as e:
        raise ValueError(f'Could not render timeline audio: {e}') from e

    duration = _wav_duration_seconds(timeline_wav) or float(
        parsed.timeline_duration_seconds or 0.0
    )

    metadata = parsed.to_metadata_dict()
    metadata['stored_fcpxml_path'] = stored_fcpxml_path
    metadata['original_fcpxml_path'] = inner_path
    metadata['missing_media'] = missing_media
    metadata['referenced_media_count'] = len(referenced_paths)
    metadata['present_media_count'] = present_count

    return IngestResult(
        audio_path=timeline_wav,
        stored_fcpxml_path=stored_fcpxml_path,
        fcpxml_metadata=metadata,
        duration_seconds=duration,
        missing_media=missing_media,
    )


def _wav_duration_seconds(wav_path: str) -> Optional[float]:
    """Read a WAV header and return its duration in seconds, or None on error."""
    try:
        import wave
        with wave.open(wav_path, 'rb') as w:
            frames = w.getnframes()
            rate = w.getframerate() or 1
            return frames / float(rate)
    except Exception:
        return None
