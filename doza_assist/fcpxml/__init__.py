"""FCPXML parsing, timecode translation, and writing.

Layer one of FCP integration: ingest FCPXML containing multicam or sync-clip
containers, extract the active audio source, and round-trip selects back out.
"""

from .parser import (
    ParsedFCPXML,
    SpineSegment,
    SegmentAudioSource,
    parse_fcpxml,
    ParseError,
)
from .timecode import (
    parse_rational,
    rational_to_seconds,
    seconds_to_rational,
    audio_source_to_timeline,
    timeline_to_segment,
)
from .writer import (
    Select,
    WriterError,
    write_selects_as_new_project,
    write_markers_on_timeline,
)

__all__ = [
    "ParsedFCPXML",
    "SpineSegment",
    "SegmentAudioSource",
    "parse_fcpxml",
    "ParseError",
    "parse_rational",
    "rational_to_seconds",
    "seconds_to_rational",
    "audio_source_to_timeline",
    "timeline_to_segment",
    "Select",
    "WriterError",
    "write_selects_as_new_project",
    "write_markers_on_timeline",
]
