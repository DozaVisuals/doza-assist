"""FCPXML parser for multicam and sync-clip containers.

Walks an FCPXML document to:

- Identify every ``<mc-clip>`` or ``<sync-clip>`` element on the main spine.
- For each segment, resolve the audio source:
  - mc-clip → find the angle enabled by ``<mc-source srcEnable="audio">`` on
    that segment, locate it in the referenced ``<media>/<multicam>``, and
    resolve the angle's ``<asset-clip>`` to a filesystem path.
  - sync-clip → find the inline ``<asset-clip audioRole="dialogue">`` (either a
    direct child of ``<sync-clip>`` or nested in ``<sync-clip>/<spine>``) and
    resolve it to a filesystem path. Also read
    ``<sync-source>/<audio-role-source@active>`` to detect FCP-muted segments.
- Preserve the verbatim ``<resources>`` block and full source bytes so the
  writer module can round-trip output without regenerating asset IDs or
  bookmark data.

Mixed-container spines (e.g. two interview multicams plus pick-up sync-clips
on the same storyline) are supported: each segment carries its own resolved
:class:`SegmentAudioSource`. Callers that only need a single representative
audio path still find it at :attr:`ParsedFCPXML.audio_file_path` (the first
non-muted segment).

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
class SegmentAudioSource:
    """Audio resolved for a single spine segment.

    ``is_muted`` reflects FCP's own play state for the segment — currently
    derived from ``<audio-role-source@active>`` on sync-clips (default active).
    Muted segments still carry a path so callers can inspect the source, but
    the timeline-audio renderer skips them.
    """

    path: str
    asset_id: str
    angle_offset_fraction: Fraction
    angle_start_fraction: Fraction
    active_audio_angle_id: Optional[str] = None
    is_muted: bool = False

    def to_dict(self) -> dict:
        def _frac(f: Fraction) -> str:
            return f"{f.numerator}/{f.denominator}"
        return {
            "path": self.path,
            "asset_id": self.asset_id,
            "angle_offset_fraction": _frac(self.angle_offset_fraction),
            "angle_start_fraction": _frac(self.angle_start_fraction),
            "active_audio_angle_id": self.active_audio_angle_id,
            "is_muted": self.is_muted,
        }


@dataclass
class SpineSegment:
    """One ``<mc-clip>`` or ``<sync-clip>`` entry in the sequence spine.

    ``mc_sources`` captures the full ``<mc-source>`` enablement on this spine
    mc-clip (typically one audio + one video angle). The writer replays these
    verbatim on each emitted select so the new project shows the same angle
    mix as the source timeline.

    ``audio_source`` is the resolved audio for this segment specifically.
    """

    kind: str                     # 'mc-clip' | 'sync-clip'
    ref: str                      # reference into <resources> (mc-clip) or "" (inline sync-clip)
    name: str
    offset_fraction: Fraction
    start_fraction: Fraction
    duration_fraction: Fraction
    mc_sources: List[dict] = field(default_factory=list)
    audio_source: Optional[SegmentAudioSource] = None

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
        d = {
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
        if self.audio_source is not None:
            d["audio_source"] = self.audio_source.to_dict()
        return d


@dataclass
class ParsedFCPXML:
    """Everything the transcription and writer pipelines need from an FCPXML."""

    version: str
    source_path: str

    container_type: str                           # 'mc-clip' | 'sync-clip' (first segment's kind)
    container_ref: str                            # first segment's ref ("" for inline sync-clip)

    audio_file_path: str                          # first non-muted segment's audio path
    audio_asset_id: str                           # corresponding asset id
    active_audio_angle_id: Optional[str]          # first mc-clip segment's active angle, if any

    # Parameters for source-time → timeline-time translation on the
    # representative (first non-muted) segment. For multi-source spines, use
    # each segment's own ``audio_source`` instead.
    audio_angle_offset_fraction: Fraction
    audio_angle_start_fraction: Fraction

    sequence_format_id: str
    sequence_frame_duration: Fraction             # e.g. 1001/24000
    timeline_duration_fraction: Fraction

    project_name: Optional[str]
    event_name: Optional[str]
    library_location: Optional[str]

    spine_segments: List[SpineSegment]
    is_multi_source: bool                         # True if segments reference >1 distinct audio asset

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

    def unique_audio_sources(self) -> List[SegmentAudioSource]:
        """Distinct audio sources across all segments, keyed by (path, asset_id)."""
        seen = set()
        out: List[SegmentAudioSource] = []
        for seg in self.spine_segments:
            src = seg.audio_source
            if src is None:
                continue
            key = (src.path, src.asset_id)
            if key in seen:
                continue
            seen.add(key)
            out.append(src)
        return out

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
            "is_multi_source": self.is_multi_source,
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


def _sync_source_dialogue_muted(sync_clip_el) -> bool:
    """True when ``<sync-source>/<audio-role-source>`` mutes the dialogue role.

    FCP uses this on sync-clips that pair a camera clip (with its built-in
    scratch mic) with a higher-quality external recorder on a connected lane.
    The camera mic's dialogue role is marked ``active="0"`` so only the
    external audio plays.
    """
    sync_source = sync_clip_el.find("sync-source")
    if sync_source is None:
        return False
    for ars in sync_source.findall("audio-role-source"):
        role = ars.get("role", "")
        if role.startswith("dialogue"):
            return ars.get("active", "1") == "0"
    return False


def _pick_dialogue_asset_clip(clips, resource_by_id):
    """From a list of asset-clip elements, return the one most likely to be
    dialogue audio. Prefers ``audioRole="dialogue"``; falls back to the first
    whose asset declares ``hasAudio="1"``; finally falls back to the first.
    Returns None on an empty list.
    """
    if not clips:
        return None
    for ac in clips:
        if ac.get("audioRole") == "dialogue":
            return ac
    for ac in clips:
        asset = resource_by_id.get(ac.get("ref"))
        if asset is not None and asset.tag == "asset" and asset.get("hasAudio") == "1":
            return ac
    return clips[0]


def _resolve_sync_clip_audio(sync_clip_el, resource_by_id: dict) -> dict:
    """Resolve the dialogue audio FCP actually plays for a ``<sync-clip>``.

    FCP produces two common shapes:

    1. Asset-clips as direct children of ``<sync-clip>`` (older/synthetic form).
    2. A nested ``<sync-clip>/<spine>`` with the main camera asset-clip, plus
       an externally-recorded audio asset-clip attached on a connected lane
       (``lane != 0``) nested inside a ``<gap>``.

    When ``<sync-source>`` marks the primary dialogue role as ``active="0"``,
    the camera mic is muted and the external recorder is what FCP plays. We
    pick that lane-attached clip in that case, so the timeline-audio render
    includes the real dialogue instead of silence.

    Sync-clip's own ``start`` attribute already measures the source time into
    the chosen audio asset, so the returned ``angle_offset`` / ``angle_start``
    are zero — the segment's ``start`` is used directly as the source time.

    Returns ``{path, asset_id, angle_offset, angle_start, is_muted}``. Sets
    ``is_muted`` only when the primary is muted and no lane replacement is
    available — i.e. when FCP itself plays silence there.
    """
    primary_candidates = list(sync_clip_el.findall("asset-clip"))
    lane_candidates: List = []
    inner_spine = sync_clip_el.find("spine")
    if inner_spine is not None:
        primary_candidates.extend(inner_spine.findall("asset-clip"))
        # Connected clips: asset-clips with a non-zero lane attribute, usually
        # nested inside a <gap> (the way FCP records "attached external audio").
        for gap in inner_spine.findall("gap"):
            for ac in gap.findall("asset-clip"):
                lane = ac.get("lane", "")
                if lane and lane != "0":
                    lane_candidates.append(ac)
        # Some FCP exports attach the lane clip directly under the inner spine.
        for ac in inner_spine.findall("asset-clip"):
            lane = ac.get("lane", "")
            if lane and lane != "0":
                lane_candidates.append(ac)

    primary = _pick_dialogue_asset_clip(primary_candidates, resource_by_id)
    lane = _pick_dialogue_asset_clip(lane_candidates, resource_by_id)

    if primary is None and lane is None:
        raise ParseError("sync-clip contains no <asset-clip> children")

    primary_muted = _sync_source_dialogue_muted(sync_clip_el)

    if primary_muted and lane is not None:
        # Camera mic muted, external recorder takes over — what FCP plays.
        dialogue = lane
        is_muted = False
    elif primary is not None:
        dialogue = primary
        is_muted = primary_muted
    else:
        dialogue = lane
        is_muted = False

    asset_ref = dialogue.get("ref")
    asset_el = resource_by_id.get(asset_ref)
    if asset_el is None or asset_el.tag != "asset":
        raise ParseError(f"sync-clip asset-clip ref {asset_ref!r} does not resolve to an <asset>")

    return {
        "path": _resolve_asset_path(asset_el),
        "asset_id": asset_ref,
        # Sync-clip@start maps directly to source time; collapse the angle
        # offsets so downstream math reduces to source_time = segment.start.
        "angle_offset": Fraction(0),
        "angle_start": Fraction(0),
        "is_muted": is_muted,
    }


def _resolve_segment_audio(
    child, resource_by_id: dict, mc_sources: List[dict]
) -> SegmentAudioSource:
    """Resolve the audio source for a single spine segment."""
    if child.tag == "mc-clip":
        angle_id = None
        for ms in mc_sources:
            enable = (ms.get("srcEnable") or "").lower()
            # srcEnable can be "video", "audio", "all", or a mix like "audio video"
            if "audio" in enable or enable == "all":
                angle_id = ms.get("angleID")
                break
        info = _resolve_multicam_audio(resource_by_id, child.get("ref") or "", angle_id)
        return SegmentAudioSource(
            path=info["path"],
            asset_id=info["asset_id"],
            angle_offset_fraction=info["angle_offset"],
            angle_start_fraction=info["angle_start"],
            active_audio_angle_id=angle_id,
            is_muted=False,
        )
    else:  # sync-clip
        info = _resolve_sync_clip_audio(child, resource_by_id)
        return SegmentAudioSource(
            path=info["path"],
            asset_id=info["asset_id"],
            angle_offset_fraction=info["angle_offset"],
            angle_start_fraction=info["angle_start"],
            active_audio_angle_id=None,
            is_muted=info["is_muted"],
        )


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

        audio_source = _resolve_segment_audio(child, resource_by_id, mc_sources)

        seg = SpineSegment(
            kind=child.tag,
            ref=child.get("ref") or "",
            name=child.get("name") or "",
            offset_fraction=parse_rational(child.get("offset")),
            start_fraction=parse_rational(child.get("start")),
            duration_fraction=parse_rational(child.get("duration")),
            mc_sources=mc_sources,
            audio_source=audio_source,
        )
        segments.append(seg)

    if not segments:
        raise ParseError("spine contains no <mc-clip> or <sync-clip> elements")

    # Representative source: first non-muted segment, else first segment.
    representative = next(
        (s for s in segments if s.audio_source and not s.audio_source.is_muted),
        segments[0],
    )
    rep_audio = representative.audio_source
    assert rep_audio is not None  # every segment resolves audio above

    # Multi-source when segments span more than one distinct audio asset, or
    # mix container kinds (so ingest renders a composed timeline WAV).
    distinct_sources = {(s.audio_source.path, s.audio_source.asset_id)
                        for s in segments if s.audio_source is not None}
    distinct_kinds = {s.kind for s in segments}
    is_multi_source = len(distinct_sources) > 1 or len(distinct_kinds) > 1

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
        container_type=segments[0].kind,
        container_ref=segments[0].ref,
        audio_file_path=rep_audio.path,
        audio_asset_id=rep_audio.asset_id,
        active_audio_angle_id=rep_audio.active_audio_angle_id,
        audio_angle_offset_fraction=rep_audio.angle_offset_fraction,
        audio_angle_start_fraction=rep_audio.angle_start_fraction,
        sequence_format_id=sequence_format_id,
        sequence_frame_duration=frame_duration,
        timeline_duration_fraction=sequence_duration,
        project_name=project_name,
        event_name=event_name,
        library_location=library_location,
        spine_segments=segments,
        is_multi_source=is_multi_source,
        original_resources_xml=original_resources_xml,
        original_fcpxml_bytes=raw_bytes,
    )
