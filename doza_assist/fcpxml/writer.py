"""FCPXML writer — round-trips Doza Assist selects back into FCP-importable XML.

Two output modes:

- **Mode A (``write_selects_as_new_project``)**: emits a fresh project whose
  spine contains one ``<mc-clip>`` (or ``<asset-clip>`` for sync-clip sources)
  per select. The original ``<resources>`` block is spliced in byte-for-byte
  from the source FCPXML so that asset IDs and bookmark base64 blobs survive
  untouched — FCP is strict about bookmark mismatch on import.

- **Mode B (``write_markers_on_timeline``)**: emits the original project
  structure with ``<marker>`` elements injected into the existing spine clips
  at each select's in-point. Marker style (standard / completion / to-do)
  encodes select type per the pass-B brief.

Both modes take a :class:`~doza_assist.fcpxml.parser.ParsedFCPXML` (produced
during ingest) and a list of :class:`Select` objects whose timecodes are
relative to the audio source file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, List, Optional

from lxml import etree

from .parser import ParsedFCPXML, SpineSegment
from .timecode import parse_rational, seconds_to_rational


# ---------- public data types -----------------------------------------------

@dataclass(frozen=True)
class Select:
    """One editor-facing selection.

    Times are in seconds, relative to the audio source file (the same
    coordinate system used by the transcription pipeline).
    """

    start_seconds: float
    end_seconds: float
    label: str = "Select"
    note: str = ""
    kind: str = "standard"        # 'strong' | 'standard' | 'question'

    @property
    def duration_seconds(self) -> float:
        return max(0.0, float(self.end_seconds) - float(self.start_seconds))


# Map Select.kind → marker attribute style.
# FCPXML expresses marker color/type through the ``completed`` attribute on
# ``<marker>`` (and a separate ``<chapter-marker>`` element for chapters —
# not used here).
_MARKER_KIND_ATTRS = {
    "strong":   {"completed": "1"},     # green / completion
    "standard": {},                     # blue / standard
    "question": {"completed": "0"},     # red / to-do
}


_XML_PROLOGUE = b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n\n'


class WriterError(ValueError):
    """Raised when selects cannot be written (e.g. unsupported source container)."""


# ---------- helpers ---------------------------------------------------------

def _source_to_container_fraction(
    source_seconds: float, parsed: ParsedFCPXML
) -> Fraction:
    """Translate an audio-source timecode to container-internal time.

    For the Ella sample both ``audio_angle_offset`` and ``audio_angle_start``
    are zero, so source time equals container time. Non-zero values happen
    when the audio angle asset-clip is positioned inside the multicam with a
    leading gap or a non-zero source start.
    """
    return (
        Fraction(source_seconds).limit_denominator(10_000_000)
        + parsed.audio_angle_offset_fraction
        - parsed.audio_angle_start_fraction
    )


def _iter_selects(selects: Iterable[Select]) -> List[Select]:
    out = [s for s in selects if s.duration_seconds > 0]
    # Sort by source in-point so the new timeline reads chronologically.
    out.sort(key=lambda s: s.start_seconds)
    return out


def _format_suffix(original: Optional[str], suffix: str) -> str:
    base = (original or "Doza Project").strip()
    return f"{base} - {suffix}"


# ---------- Mode A: selects as a new project --------------------------------

def _build_select_mc_clip(
    parsed: ParsedFCPXML,
    select: Select,
    timeline_offset: Fraction,
) -> etree._Element:
    fd = parsed.sequence_frame_duration
    container_in = _source_to_container_fraction(select.start_seconds, parsed)
    duration = Fraction(select.duration_seconds).limit_denominator(10_000_000)

    mc = etree.Element("mc-clip")
    mc.set("ref", parsed.container_ref)
    mc.set("offset", seconds_to_rational(timeline_offset, fd))
    mc.set("name", select.label or "Select")
    mc.set("start", seconds_to_rational(container_in, fd))
    mc.set("duration", seconds_to_rational(duration, fd))

    # Replay the source timeline's mc-source enablement so both the video and
    # audio angles match what the editor had on the original timeline.
    source_sample = parsed.spine_segments[0] if parsed.spine_segments else None
    if source_sample and source_sample.mc_sources:
        for ms in source_sample.mc_sources:
            sub = etree.SubElement(mc, "mc-source")
            sub.set("angleID", ms.get("angleID", ""))
            sub.set("srcEnable", ms.get("srcEnable", ""))
    elif parsed.active_audio_angle_id:
        # Fallback: at least enable audio if we somehow lost the full enablement.
        sub = etree.SubElement(mc, "mc-source")
        sub.set("angleID", parsed.active_audio_angle_id)
        sub.set("srcEnable", "audio")

    if select.note:
        note = etree.SubElement(mc, "note")
        note.text = select.note

    return mc


def _build_select_sync_asset_clip(
    parsed: ParsedFCPXML,
    select: Select,
    timeline_offset: Fraction,
) -> etree._Element:
    """For sync-clip sources, emit an ``<asset-clip>`` cut of the dialogue audio.

    Rebuilding the full inline sync-clip (video + audio) per select would
    require carrying the original inline children through to the writer, which
    is deferred. The emitted asset-clip lands on an audio track in FCP; the
    editor can sync it against source video manually if needed. Multicam
    exports (the demo path) keep video and audio sync intact via Mode A's
    mc-clip output above.
    """
    fd = parsed.sequence_frame_duration
    container_in = _source_to_container_fraction(select.start_seconds, parsed)
    duration = Fraction(select.duration_seconds).limit_denominator(10_000_000)

    ac = etree.Element("asset-clip")
    ac.set("ref", parsed.audio_asset_id)
    ac.set("offset", seconds_to_rational(timeline_offset, fd))
    ac.set("name", select.label or "Select")
    ac.set("start", seconds_to_rational(container_in, fd))
    ac.set("duration", seconds_to_rational(duration, fd))
    ac.set("audioRole", "dialogue")
    if select.note:
        note = etree.SubElement(ac, "note")
        note.text = select.note
    return ac


def _build_selects_spine(parsed: ParsedFCPXML, selects: List[Select]) -> etree._Element:
    spine = etree.Element("spine")
    timeline_cursor = Fraction(0)
    for s in selects:
        if parsed.container_type == "mc-clip":
            node = _build_select_mc_clip(parsed, s, timeline_cursor)
        else:
            node = _build_select_sync_asset_clip(parsed, s, timeline_cursor)
        spine.append(node)
        timeline_cursor += Fraction(s.duration_seconds).limit_denominator(10_000_000)
    return spine


def write_selects_as_new_project(
    parsed: ParsedFCPXML,
    selects: Iterable[Select],
    *,
    project_name: Optional[str] = None,
    event_name: Optional[str] = None,
) -> bytes:
    """Mode A — emit an FCPXML where the selects are a new project's spine.

    The original ``<resources>`` block is preserved byte-for-byte; only the
    ``<library>`` is rebuilt. Returns UTF-8 encoded FCPXML bytes.
    """
    selects = _iter_selects(selects)
    if not selects:
        raise WriterError("no selects provided")

    project_title = project_name or _format_suffix(parsed.project_name, "Doza Selects")
    event_title = event_name or (parsed.event_name or "Doza Selects")

    total_duration = Fraction(0)
    for s in selects:
        total_duration += Fraction(s.duration_seconds).limit_denominator(10_000_000)
    fd = parsed.sequence_frame_duration

    sequence = etree.Element("sequence")
    sequence.set("format", parsed.sequence_format_id)
    sequence.set("duration", seconds_to_rational(total_duration, fd))
    sequence.set("tcStart", "0s")
    sequence.set("tcFormat", "NDF")
    sequence.set("audioLayout", "stereo")
    sequence.set("audioRate", "48k")
    sequence.append(_build_selects_spine(parsed, selects))

    project = etree.Element("project")
    project.set("name", project_title)
    project.append(sequence)

    event = etree.Element("event")
    event.set("name", event_title)
    event.append(project)

    library = etree.Element("library")
    if parsed.library_location:
        library.set("location", parsed.library_location)
    library.append(event)

    library_bytes = etree.tostring(library, pretty_print=True, encoding="utf-8")

    # Assemble the final document: prologue + <fcpxml> + verbatim <resources>
    # + freshly serialized <library> + closing tag. Preserving the original
    # resources byte-for-byte is what keeps FCP's bookmark validation happy.
    buf = bytearray()
    buf += _XML_PROLOGUE
    buf += f'<fcpxml version="{parsed.version}">\n    '.encode("utf-8")
    buf += parsed.original_resources_xml
    buf += b"\n    "
    buf += library_bytes
    buf += b"</fcpxml>\n"
    return bytes(buf)


# ---------- Mode B: markers on the existing timeline ------------------------

def _find_segment_for_source_time(
    parsed: ParsedFCPXML, source_seconds: float
) -> Optional[SpineSegment]:
    container_time = _source_to_container_fraction(source_seconds, parsed)
    for seg in parsed.spine_segments:
        if seg.start_fraction <= container_time < seg.start_fraction + seg.duration_fraction:
            return seg
    return None


def _marker_element(
    parsed: ParsedFCPXML,
    select: Select,
    container_time: Fraction,
) -> etree._Element:
    fd = parsed.sequence_frame_duration
    # A 1-frame marker duration keeps FCP from stretching the marker across time.
    marker = etree.Element("marker")
    marker.set("start", seconds_to_rational(container_time, fd))
    marker.set("duration", seconds_to_rational(fd, fd))
    marker.set("value", select.label or "Marker")
    for k, v in _MARKER_KIND_ATTRS.get(select.kind, {}).items():
        marker.set(k, v)
    if select.note:
        marker.set("note", select.note)
    return marker


def write_markers_on_timeline(
    parsed: ParsedFCPXML,
    selects: Iterable[Select],
    *,
    project_name_suffix: str = "Doza Notes",
) -> bytes:
    """Mode B — copy the original structure and inject markers at each select.

    The original resources block is preserved byte-for-byte; only the
    ``<library>`` structure is re-serialized with marker children added to the
    appropriate spine clips. Selects that fall outside any spine segment are
    silently dropped (their source audio is unused on the timeline).
    """
    selects = _iter_selects(selects)
    if not selects:
        raise WriterError("no selects provided")

    root = etree.fromstring(parsed.original_fcpxml_bytes)
    library = root.find("library")
    if library is None:
        raise WriterError("source FCPXML has no <library> to annotate")

    # Rename the project so the marker-annotated copy is obviously distinct
    # from the original when both appear in the FCP event browser.
    project_el = library.find(".//project")
    if project_el is not None and project_name_suffix:
        original = project_el.get("name") or "Doza Project"
        project_el.set("name", _format_suffix(original, project_name_suffix))

    spine = library.find(".//sequence/spine")
    if spine is None:
        raise WriterError("source FCPXML sequence has no <spine>")

    # Build an index from segment identity → the actual lxml element on the spine.
    # The parser's SpineSegment list is in document order, so we can zip them up
    # with spine children of the right tags.
    spine_clip_elements: List[etree._Element] = [
        c for c in spine if c.tag in ("mc-clip", "sync-clip")
    ]
    if len(spine_clip_elements) != len(parsed.spine_segments):
        raise WriterError(
            "spine structure changed since parse; cannot attach markers safely"
        )
    element_for_segment = {
        id(seg): el for seg, el in zip(parsed.spine_segments, spine_clip_elements)
    }

    for s in selects:
        seg = _find_segment_for_source_time(parsed, s.start_seconds)
        if seg is None:
            # Select falls in a gap — source audio is never on the timeline here.
            continue
        container_time = _source_to_container_fraction(s.start_seconds, parsed)
        marker = _marker_element(parsed, s, container_time)
        element_for_segment[id(seg)].append(marker)

    library_bytes = etree.tostring(library, pretty_print=True, encoding="utf-8")

    # Splice the original <resources> in verbatim (lxml would otherwise re-serialize
    # the bookmark blobs; in practice text content round-trips, but byte-splicing
    # costs nothing and removes any doubt from an FCP import reviewer's mind).
    buf = bytearray()
    buf += _XML_PROLOGUE
    buf += f'<fcpxml version="{parsed.version}">\n    '.encode("utf-8")
    buf += parsed.original_resources_xml
    buf += b"\n    "
    buf += library_bytes
    buf += b"</fcpxml>\n"
    return bytes(buf)


# ---------- round-trip helper used by tests ---------------------------------

def re_parse(output_bytes: bytes):
    """Re-parse writer output through :func:`parse_fcpxml` to validate round-trip.

    Writes to a temp file since :func:`parse_fcpxml` takes a path (so it can
    also preserve the source file's raw bytes). Used by the round-trip tests.
    """
    import tempfile
    from pathlib import Path
    from .parser import parse_fcpxml

    with tempfile.NamedTemporaryFile(
        "wb", suffix=".fcpxml", delete=False
    ) as fh:
        fh.write(output_bytes)
        path = fh.name
    try:
        return parse_fcpxml(path)
    finally:
        Path(path).unlink(missing_ok=True)
