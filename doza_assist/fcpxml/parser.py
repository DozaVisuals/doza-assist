"""FCPXML parser for multicam and sync-clip containers.

Walks an FCPXML document to:

- Identify the spine's container elements (``<mc-clip>`` or ``<sync-clip>``).
- For multicam: resolve the container's ``ref`` to a ``<media>/<multicam>``,
  find the angle enabled for audio via ``<mc-source srcEnable="audio">``, and
  resolve that angle's asset to a filesystem path.
- For sync-clip: find the inline ``<asset-clip audioRole="dialogue">`` and
  resolve it to a filesystem path.
- Preserve the verbatim ``<resources>`` block and full source bytes so the
  writer module (pass B) can round-trip Mode A output without regenerating
  asset IDs or bookmark data.

Supports FCPXML 1.13 and 1.14.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import List, Optional
from urllib.parse import unquote

from lxml import etree

from .timecode import parse_rational


SUPPORTED_VERSIONS = {"1.13", "1.14"}


class ParseError(ValueError):
    """Raised when an FCPXML document cannot be interpreted."""


@dataclass
class SpineSegment:
    """One ``<mc-clip>`` or ``<sync-clip>`` entry in the sequence spine.

    ``mc_sources`` captures the full ``<mc-source>`` enablement on this spine
    mc-clip (typically one audio + one video angle). The writer replays these
    verbatim on each emitted select so the new project shows the same angle
    mix as the source timeline.
    """

    kind: str                     # 'mc-clip' | 'sync-clip'
    ref: str                      # reference into <resources> (mc-clip) or "" (inline sync-clip)
    name: str
    offset_fraction: Fraction
    start_fraction: Fraction
    duration_fraction: Fraction
    mc_sources: List[dict] = field(default_factory=list)

    @property
    def offset_seconds(self) -> float:
        return float(self.offset_fraction)

    @property
    def start_seconds(self) -> float:
        return float(self.start_fraction)

    @property
    def duration_seconds(self) -> float:
        return float(self.duration_fraction)

    def to_dict(self) -> dict:
        def _frac(f: Fraction) -> str:
            return f"{f.numerator}/{f.denominator}"
        return {
            "kind": self.kind,
            "ref": self.ref,
            "name": self.name,
            "offset_fraction": _frac(self.offset_fraction),
            "start_fraction": _frac(self.start_fraction),
            "duration_fraction": _frac(self.duration_fraction),
            "offset_seconds": self.offset_seconds,
            "start_seconds": self.start_seconds,
            "duration_seconds": self.duration_seconds,
            "mc_sources": list(self.mc_sources),
        }


@dataclass
class ParsedFCPXML:
    """Everything the transcription and writer pipelines need from an FCPXML."""

    version: str
    source_path: str

    container_type: str                           # 'mc-clip' | 'sync-clip'
    container_ref: str                            # resource id (mc-clip) or '' (inline sync-clip)

    audio_file_path: str                          # decoded absolute filesystem path
    audio_asset_id: str                           # resource id of the audio asset
    active_audio_angle_id: Optional[str]          # only set for mc-clip containers

    # Parameters for source-time → timeline-time translation.
    audio_angle_offset_fraction: Fraction         # offset of audio asset-clip inside its container
    audio_angle_start_fraction: Fraction          # start attribute of audio asset-clip (usually 0)

    sequence_format_id: str
    sequence_frame_duration: Fraction             # e.g. 1001/24000
    timeline_duration_fraction: Fraction

    project_name: Optional[str]
    event_name: Optional[str]
    library_location: Optional[str]

    spine_segments: List[SpineSegment]

    original_resources_xml: bytes                 # verbatim byte-slice from the source
    original_fcpxml_bytes: bytes

    @property
    def timeline_duration_seconds(self) -> float:
        return float(self.timeline_duration_fraction)

    @property
    def sequence_framerate(self) -> float:
        fd = self.sequence_frame_duration
        if fd == 0:
            return 0.0
        return float(Fraction(fd.denominator, fd.numerator))

    def to_metadata_dict(self) -> dict:
        """A JSON-serializable snapshot for project meta.json. Excludes raw bytes."""
        def _frac(f: Fraction) -> str:
            return f"{f.numerator}/{f.denominator}"
        return {
            "version": self.version,
            "source_path": self.source_path,
            "container_type": self.container_type,
            "container_ref": self.container_ref,
            "audio_file_path": self.audio_file_path,
            "audio_asset_id": self.audio_asset_id,
            "active_audio_angle_id": self.active_audio_angle_id,
            "audio_angle_offset_fraction": _frac(self.audio_angle_offset_fraction),
            "audio_angle_start_fraction": _frac(self.audio_angle_start_fraction),
            "sequence_format_id": self.sequence_format_id,
            "sequence_frame_duration": _frac(self.sequence_frame_duration),
            "sequence_framerate": self.sequence_framerate,
            "timeline_duration_fraction": _frac(self.timeline_duration_fraction),
            "timeline_duration_seconds": self.timeline_duration_seconds,
            "project_name": self.project_name,
            "event_name": self.event_name,
            "library_location": self.library_location,
            "spine_segments": [s.to_dict() for s in self.spine_segments],
        }


def strip_file_url(src: str) -> str:
    """Convert a ``media-rep`` ``src`` attribute to an absolute filesystem path."""
    if src.startswith("file://"):
        src = src[len("file://"):]
    return unquote(src)


def _extract_resources_bytes(fcpxml_bytes: bytes) -> bytes:
    """Slice out ``<resources>...</resources>`` from the raw source bytes.

    Byte-exact, so bookmark base64 blobs, asset UIDs, whitespace, and attribute
    ordering are all preserved — FCP is strict about mismatched bookmarks.
    """
    open_match = re.search(rb"<resources(\s[^>]*)?>", fcpxml_bytes)
    if not open_match:
        raise ParseError("no <resources> block found in FCPXML")
    close_tag = b"</resources>"
    end_idx = fcpxml_bytes.find(close_tag, open_match.end())
    if end_idx == -1:
        raise ParseError("<resources> block is not closed")
    return fcpxml_bytes[open_match.start(): end_idx + len(close_tag)]


def _resolve_asset_path(asset_el) -> str:
    """Extract and decode the filesystem path from an ``<asset>`` element."""
    media_rep = asset_el.find("media-rep")
    if media_rep is None:
        raise ParseError(f"asset {asset_el.get('id')!r} has no <media-rep>")
    src = media_rep.get("src")
    if not src:
        raise ParseError(f"asset {asset_el.get('id')!r} has no media-rep/@src")
    return strip_file_url(src)


def _resolve_multicam_audio(
    resource_by_id: dict,
    container_ref: str,
    angle_id: Optional[str],
) -> dict:
    """Resolve the active audio angle within a ``<media>/<multicam>`` → asset path."""
    media_el = resource_by_id.get(container_ref)
    if media_el is None:
        raise ParseError(f"mc-clip ref {container_ref!r} not found in <resources>")
    multicam = media_el.find("multicam")
    if multicam is None:
        raise ParseError(f"resource {container_ref!r} is not a <multicam> media")

    angles = multicam.findall("mc-angle")
    chosen = None
    if angle_id:
        for a in angles:
            if a.get("angleID") == angle_id:
                chosen = a
                break
        if chosen is None:
            raise ParseError(
                f"mc-source references angleID {angle_id!r}, "
                f"no matching <mc-angle> in multicam {container_ref!r}"
            )
    else:
        # No explicit audio mc-source: fall back to the first angle with an audio asset-clip.
        for a in angles:
            if a.find("asset-clip[@audioRole]") is not None or a.find("asset-clip") is not None:
                chosen = a
                break
        if chosen is None:
            raise ParseError(
                f"multicam {container_ref!r} has no angles with asset-clips; "
                "cannot resolve audio source"
            )

    asset_clip = chosen.find("asset-clip")
    if asset_clip is None:
        raise ParseError(
            f"mc-angle {chosen.get('name')!r} has no <asset-clip>; "
            "audio-only angle formats are not supported"
        )

    asset_ref = asset_clip.get("ref")
    asset_el = resource_by_id.get(asset_ref)
    if asset_el is None or asset_el.tag != "asset":
        raise ParseError(f"asset-clip ref {asset_ref!r} does not resolve to an <asset>")

    return {
        "path": _resolve_asset_path(asset_el),
        "asset_id": asset_ref,
        "angle_offset": parse_rational(asset_clip.get("offset")),
        "angle_start": parse_rational(asset_clip.get("start")),
    }


def _resolve_sync_clip_audio(sync_clip_el, resource_by_id: dict) -> dict:
    """Resolve the dialogue asset-clip inside an inline ``<sync-clip>``."""
    # Prefer audioRole="dialogue"; fall back to the first asset-clip with audio enabled.
    candidates = sync_clip_el.findall("asset-clip")
    if not candidates:
        raise ParseError("sync-clip contains no <asset-clip> children")

    dialogue = None
    for ac in candidates:
        if ac.get("audioRole") == "dialogue":
            dialogue = ac
            break
    if dialogue is None:
        # Fall back to the first asset-clip. A stricter check (hasAudio on asset) is
        # possible but editors sometimes omit audioRole on mono interview sources.
        dialogue = candidates[0]

    asset_ref = dialogue.get("ref")
    asset_el = resource_by_id.get(asset_ref)
    if asset_el is None or asset_el.tag != "asset":
        raise ParseError(f"sync-clip asset-clip ref {asset_ref!r} does not resolve to an <asset>")

    return {
        "path": _resolve_asset_path(asset_el),
        "asset_id": asset_ref,
        "angle_offset": parse_rational(dialogue.get("offset")),
        "angle_start": parse_rational(dialogue.get("start")),
    }


def _find_first_spine_sync_clip(spine) -> Optional[object]:
    for c in spine:
        if c.tag == "sync-clip":
            return c
    return None


def parse_fcpxml(path) -> ParsedFCPXML:
    """Parse an FCPXML file and return a :class:`ParsedFCPXML` snapshot.

    Raises :class:`ParseError` for unsupported versions, missing structures,
    or unresolved references.
    """
    path_str = str(path)
    raw_bytes = Path(path_str).read_bytes()

    try:
        root = etree.fromstring(raw_bytes)
    except etree.XMLSyntaxError as e:
        raise ParseError(f"invalid FCPXML: {e}") from e

    if root.tag != "fcpxml":
        raise ParseError(f"root element is <{root.tag}>, expected <fcpxml>")

    version = root.get("version")
    if version not in SUPPORTED_VERSIONS:
        raise ParseError(
            f"unsupported FCPXML version {version!r}; supported: {sorted(SUPPORTED_VERSIONS)}"
        )

    resources = root.find("resources")
    if resources is None:
        raise ParseError("no <resources> block")
    resource_by_id = {el.get("id"): el for el in resources if el.get("id")}

    sequence = root.find(".//project/sequence")
    if sequence is None:
        raise ParseError("no <sequence> inside <library>/<event>/<project>")

    sequence_format_id = sequence.get("format") or ""
    sequence_duration = parse_rational(sequence.get("duration"))
    fmt_el = resource_by_id.get(sequence_format_id)
    if fmt_el is None:
        raise ParseError(f"sequence references missing format id {sequence_format_id!r}")
    frame_duration = parse_rational(fmt_el.get("frameDuration"))

    spine = sequence.find("spine")
    if spine is None:
        raise ParseError("sequence has no <spine>")

    segments: List[SpineSegment] = []
    container_type: Optional[str] = None
    container_ref: Optional[str] = None
    active_audio_angle_id: Optional[str] = None

    for child in spine:
        if child.tag not in ("mc-clip", "sync-clip"):
            continue

        mc_sources: List[dict] = []
        if child.tag == "mc-clip":
            for ms in child.findall("mc-source"):
                mc_sources.append({
                    "angleID": ms.get("angleID") or "",
                    "srcEnable": ms.get("srcEnable") or "",
                })

        seg = SpineSegment(
            kind=child.tag,
            ref=child.get("ref") or "",
            name=child.get("name") or "",
            offset_fraction=parse_rational(child.get("offset")),
            start_fraction=parse_rational(child.get("start")),
            duration_fraction=parse_rational(child.get("duration")),
            mc_sources=mc_sources,
        )
        segments.append(seg)

        if container_type is None:
            container_type = child.tag
            container_ref = seg.ref
            if child.tag == "mc-clip":
                for ms in mc_sources:
                    enable = (ms.get("srcEnable") or "").lower()
                    # srcEnable can be "video", "audio", "all", or a mix like "audio video"
                    if "audio" in enable or enable == "all":
                        active_audio_angle_id = ms.get("angleID")
                        break
        else:
            # Pass-A simplification: all spine containers must point at the same source.
            if child.tag != container_type:
                raise ParseError(
                    "spine mixes mc-clip and sync-clip containers; not supported in pass A"
                )
            if child.tag == "mc-clip" and seg.ref != container_ref:
                raise ParseError(
                    f"spine mc-clips reference multiple containers ({container_ref!r} and {seg.ref!r}); "
                    "not supported in pass A"
                )

    if not segments:
        raise ParseError("spine contains no <mc-clip> or <sync-clip> elements")

    if container_type == "mc-clip":
        audio_info = _resolve_multicam_audio(resource_by_id, container_ref, active_audio_angle_id)
    else:
        first_sync = _find_first_spine_sync_clip(spine)
        # _find_first returns the same element that was walked above; always present here.
        audio_info = _resolve_sync_clip_audio(first_sync, resource_by_id)

    library_el = root.find("library")
    library_location = library_el.get("location") if library_el is not None else None
    event_el = root.find(".//event")
    event_name = event_el.get("name") if event_el is not None else None
    project_el = root.find(".//project")
    project_name = project_el.get("name") if project_el is not None else None

    original_resources_xml = _extract_resources_bytes(raw_bytes)

    return ParsedFCPXML(
        version=version,
        source_path=path_str,
        container_type=container_type,
        container_ref=container_ref or "",
        audio_file_path=audio_info["path"],
        audio_asset_id=audio_info["asset_id"],
        active_audio_angle_id=active_audio_angle_id if container_type == "mc-clip" else None,
        audio_angle_offset_fraction=audio_info["angle_offset"],
        audio_angle_start_fraction=audio_info["angle_start"],
        sequence_format_id=sequence_format_id,
        sequence_frame_duration=frame_duration,
        timeline_duration_fraction=sequence_duration,
        project_name=project_name,
        event_name=event_name,
        library_location=library_location,
        spine_segments=segments,
        original_resources_xml=original_resources_xml,
        original_fcpxml_bytes=raw_bytes,
    )
