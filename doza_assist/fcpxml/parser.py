"""FCPXML parser for multicam, sync-clip, and plain asset-clip spines.

Walks an FCPXML document to:

- Identify every ``<mc-clip>``, ``<sync-clip>``, or ``<asset-clip>`` element on
  the main spine.
- For each segment, resolve the audio source:
  - mc-clip → find the angle enabled by ``<mc-source srcEnable="audio">`` on
    that segment, locate it in the referenced ``<media>/<multicam>``, and
    resolve the angle's ``<asset-clip>`` to a filesystem path.
  - sync-clip → find the inline ``<asset-clip audioRole="dialogue">`` (either a
    direct child of ``<sync-clip>`` or nested in ``<sync-clip>/<spine>``) and
    resolve it to a filesystem path. Also read
    ``<sync-source>/<audio-role-source@active>`` to detect FCP-muted segments.
  - asset-clip → a plain single-camera clip on the spine (Meta Glasses, a
    mirrorless body, a screen capture, a drone — anything that isn't a multicam
    or a synced pair). Resolve its ``ref`` straight to the ``<asset>`` and use
    that file as both video and audio. A spine of these laid end-to-end (a
    "rush" timeline) is the common single-cam case.
- Preserve the verbatim ``<resources>`` block and full source bytes so the
  writer module can round-trip output without regenerating asset IDs or
  bookmark data.

Mixed-container spines (e.g. two interview multicams plus pick-up sync-clips
on the same storyline) are supported: each segment carries its own resolved
:class:`SegmentAudioSource`. Callers that only need a single representative
audio path still find it at :attr:`ParsedFCPXML.audio_file_path` (the first
non-muted segment).

Supports FCPXML 1.8–1.14 (DaVinci Resolve exports 1.8–1.11; Final Cut Pro
exports 1.13–1.14).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import List, Optional
from urllib.parse import unquote

from lxml import etree

from .timecode import parse_rational


SUPPORTED_VERSIONS = {
    "1.8", "1.9", "1.10", "1.11", "1.12",
    "1.13", "1.14",
}

NLE_FCP = "fcp"
NLE_RESOLVE = "resolve"
NLE_UNKNOWN = "unknown"

# The spine child tags this parser turns into :class:`SpineSegment` entries, in
# the order it visits them (document order). The writer's original-spine walker
# MUST filter on this exact tuple so its element index lines up one-for-one with
# ``ParsedFCPXML.spine_segments`` — keep them sourced from here, never inline.
SPINE_SEGMENT_TAGS = ("mc-clip", "sync-clip", "asset-clip")


def iter_spine_clip_elements(spine):
    """Yield ``(clip_element, parent_gap_or_None)`` for every spine entry the
    pipeline treats as a segment, in document order.

    Direct children whose tag is in :data:`SPINE_SEGMENT_TAGS` are primary-
    storyline segments. ``<gap>`` children are transparent containers:
    connected clips (``lane != 0``) anchored inside a gap — B-roll bridging a
    hole in the primary storyline — are real timeline content whose dialogue
    is selectable; skipping them silently dropped any select landing in the
    gap (the long-standing "lane-1 B-roll-gap" bug). The writer's
    ``_index_original_spine`` and Mode B's element indexing MUST iterate via
    this same function — index N corresponds to ``spine_segments[N]``.
    """
    for child in spine:
        if child.tag in SPINE_SEGMENT_TAGS:
            yield child, None
        elif child.tag == "gap":
            for sub in child:
                if (sub.tag in SPINE_SEGMENT_TAGS
                        and (sub.get("lane") or "0") != "0"):
                    yield sub, child

_log = logging.getLogger(__name__)


def _safe_parse_rational(value: Optional[str], *, what: str) -> Fraction:
    """Parse an FCPXML rational, defaulting to 0 with a warning on failure.

    Used for ``tcStart`` and asset ``start`` attributes where a malformed value
    should not abort ingest — falling back to 0 yields the original (tcStart=0)
    code path.
    """
    if value is None:
        return Fraction(0)
    try:
        return parse_rational(value)
    except (ValueError, ZeroDivisionError) as e:
        _log.warning("could not parse %s=%r (%s); defaulting to 0", what, value, e)
        return Fraction(0)


def _detect_nle(root) -> str:
    """Guess which NLE produced this FCPXML.

    FCP always wraps content in ``<library location="...">``.  Resolve omits
    ``<library>`` entirely (pre-19) or includes it without a ``location``.
    The version number is also a strong signal: 1.13+ is FCP territory,
    1.8–1.11 is Resolve territory.
    """
    version = root.get("version") or ""
    library = root.find("library")
    if library is not None and library.get("location"):
        return NLE_FCP
    if version in {"1.13", "1.14"}:
        return NLE_FCP
    if version in {"1.8", "1.9", "1.10", "1.11"}:
        return NLE_RESOLVE
    if library is None:
        return NLE_RESOLVE
    return NLE_UNKNOWN


class ParseError(ValueError):
    """Raised when an FCPXML document cannot be interpreted."""


@dataclass
class SegmentAudioSource:
    """Audio resolved for a single spine segment.

    ``is_muted`` reflects FCP's own play state for the segment — currently
    derived from ``<audio-role-source@active>`` on sync-clips (default active).
    Muted segments still carry a path so callers can inspect the source, but
    the timeline-audio renderer skips them.

    ``container_tc_start_fraction`` and ``asset_start_fraction`` capture the
    timecode origins of, respectively, the multicam container and the
    underlying asset. Time-of-day timecode (jam-synced cameras, sync boxes,
    Panasonic / RED / ARRI / Sony FX recorders) makes both of these non-zero;
    seek math must subtract them to get a zero-based offset into the actual
    media file. Sync-clips leave both at zero — their existing seek formula
    (``angle_offset = angle_start = 0``) already operates in source-time.
    """

    path: str
    asset_id: str
    angle_offset_fraction: Fraction
    angle_start_fraction: Fraction
    active_audio_angle_id: Optional[str] = None
    is_muted: bool = False
    container_tc_start_fraction: Fraction = Fraction(0)
    asset_start_fraction: Fraction = Fraction(0)

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
            "container_tc_start_fraction": _frac(self.container_tc_start_fraction),
            "asset_start_fraction": _frac(self.asset_start_fraction),
        }


@dataclass
class SpineSegment:
    """One ``<mc-clip>``, ``<sync-clip>``, or ``<asset-clip>`` entry in the spine.

    ``mc_sources`` captures the full ``<mc-source>`` enablement on this spine
    mc-clip (typically one audio + one video angle). The writer replays these
    verbatim on each emitted select so the new project shows the same angle
    mix as the source timeline. Empty for sync-clip and asset-clip segments.

    ``audio_source`` is the resolved audio for this segment specifically.
    """

    kind: str                     # 'mc-clip' | 'sync-clip' | 'asset-clip'
    ref: str                      # <resources> id (mc-clip/asset-clip) or "" (inline sync-clip)
    name: str
    offset_fraction: Fraction
    start_fraction: Fraction
    duration_fraction: Fraction
    mc_sources: List[dict] = field(default_factory=list)
    audio_source: Optional[SegmentAudioSource] = None
    # Non-empty for connected clips lifted out of a primary-storyline <gap>
    # (e.g. "1" for lane-1 B-roll). Empty for primary spine segments.
    lane: str = ""

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
            "lane": self.lane,
        }
        if self.audio_source is not None:
            d["audio_source"] = self.audio_source.to_dict()
        return d


@dataclass
class ParsedFCPXML:
    """Everything the transcription and writer pipelines need from an FCPXML."""

    version: str
    source_path: str

    container_type: str                           # first segment's kind: mc-clip | sync-clip | asset-clip
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
    nle_source: str = NLE_UNKNOWN                 # 'fcp' | 'resolve' | 'unknown'

    original_resources_xml: bytes = b""           # verbatim byte-slice from the source
    original_fcpxml_bytes: bytes = b""

    # Representative segment's container / asset timecode origins. Needed to
    # invert the source→container mapping when locating selects on a SINGLE-
    # source spine (multi-source spines locate via timeline offsets instead).
    # Zero for sync-clips and tcStart-zero media; non-zero only for embedded /
    # jam-synced timecode (pro cameras, time-of-day TC). Mirror audio_angle_*.
    audio_container_tc_start_fraction: Fraction = Fraction(0)
    audio_asset_start_fraction: Fraction = Fraction(0)

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
            "nle_source": self.nle_source,
            "container_type": self.container_type,
            "container_ref": self.container_ref,
            "audio_file_path": self.audio_file_path,
            "audio_asset_id": self.audio_asset_id,
            "active_audio_angle_id": self.active_audio_angle_id,
            "audio_angle_offset_fraction": _frac(self.audio_angle_offset_fraction),
            "audio_angle_start_fraction": _frac(self.audio_angle_start_fraction),
            "audio_container_tc_start_fraction": _frac(self.audio_container_tc_start_fraction),
            "audio_asset_start_fraction": _frac(self.audio_asset_start_fraction),
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
    """Convert a ``media-rep`` ``src`` attribute to an absolute filesystem path.

    Handles FCP's ``file:///Volume/...`` URLs, Resolve's occasional bare paths,
    and percent-encoded characters in either form.
    """
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
    """Extract and decode the filesystem path from an ``<asset>`` element.

    Resolves to the ORIGINAL media, never the proxy. This path feeds audio
    extraction (``ffmpeg`` run with video disabled) and the preview-proxy
    build. The audio source MUST be the camera/recorder master: Sony/Lumix
    camera proxies (``kind="proxy-media"``) routinely carry SILENT,
    reference-only audio, so transcribing the proxy yields an empty WAV even
    though the clip plainly has speech in an NLE (which auditions the linked
    original). Because audio is extracted with video disabled, the proxy's
    faster Apple-Silicon decode gives the audio path zero benefit; for the
    rare master the browser cannot decode, the app builds its own playable
    preview proxy from the original (see ``_build_preview_proxy``).

    [Was proxy-preferred — that fed Whisper the silent Sony FX proxy and, post
    the 1.0.25 silent-audio guard, hard-errored "audio track appears silent"
    on footage that clearly has a mic.]

    Preference order: ``kind="original-media"`` → first non-proxy rep → any
    proxy that exists on disk → first declared rep. The ``os.path.exists``
    gate preserves the graceful fallback: when the original is offline but a
    proxy is on disk, the proxy is still used rather than failing the import.
    """
    media_reps = asset_el.findall("media-rep")
    if not media_reps:
        raise ParseError(f"asset {asset_el.get('id')!r} has no <media-rep>")

    def _on_disk(mr) -> bool:
        src = mr.get("src")
        return bool(src) and os.path.exists(strip_file_url(src))

    originals = [mr for mr in media_reps if mr.get("kind") == "original-media"]
    non_proxy = [mr for mr in media_reps if mr.get("kind") != "proxy-media"]
    proxies = [mr for mr in media_reps if mr.get("kind") == "proxy-media"]

    # NOTE: do not use ``a or b`` to pick between media-rep elements — an
    # Element with no children is FALSY, so a found-but-childless <media-rep>
    # would be skipped. Compare against None explicitly.
    def _first_on_disk(reps):
        for mr in reps:
            if _on_disk(mr):
                return mr
        return None

    chosen = _first_on_disk(originals)
    if chosen is None:
        chosen = _first_on_disk(non_proxy)
    if chosen is None:
        chosen = _first_on_disk(proxies)
    if chosen is None:
        chosen = (originals or non_proxy or media_reps)[0]

    src = chosen.get("src")
    if not src:
        raise ParseError(f"asset {asset_el.get('id')!r} has no media-rep/@src")
    return strip_file_url(src)


def _resolve_multicam_audio(
    resource_by_id: dict,
    container_ref: str,
    angle_id: Optional[str],
    segment_start: Fraction = Fraction(0),
) -> dict:
    """Resolve the active audio angle within a ``<media>/<multicam>`` → asset path.

    When a multicam angle contains multiple asset-clips (e.g. several camera
    files stitched into one angle), ``segment_start`` — the mc-clip's ``start``
    attribute — is used to pick the asset-clip whose time range covers that
    position within the multicam container.

    ``segment_start`` is in the multicam's timecode space, which is offset by
    the multicam's ``tcStart`` (jam-synced cameras and time-of-day TC make this
    non-zero — e.g. 7464.48s for an interview that started at 02:04:24:11). The
    asset-clips inside the multicam are positioned in zero-based container
    time, so the comparison must subtract ``tcStart`` first; otherwise every
    spine clip lands past every asset-clip range and the loop falls through to
    ``all_clips[0]``, sending every segment to the first .MOV's audio.
    """
    media_el = resource_by_id.get(container_ref)
    if media_el is None:
        raise ParseError(f"mc-clip ref {container_ref!r} not found in <resources>")
    multicam = media_el.find("multicam")
    if multicam is None:
        raise ParseError(f"resource {container_ref!r} is not a <multicam> media")

    mcam_tc_start = _safe_parse_rational(
        multicam.get("tcStart"),
        what=f"multicam {container_ref!r} tcStart",
    )

    # FCP uses <mc-angle angleID="...">, Resolve may also use <mc-angle> but
    # older versions (1.8–1.9) sometimes use <angle> instead.
    angles = multicam.findall("mc-angle")
    if not angles:
        angles = multicam.findall("angle")
        if angles:
            _log.debug("multicam %r uses <angle> elements (Resolve-style)", container_ref)

    chosen = None
    if angle_id:
        for a in angles:
            if a.get("angleID") == angle_id:
                chosen = a
                break
        # Resolve may match on name instead of angleID.
        if chosen is None:
            for a in angles:
                if a.get("name") == angle_id:
                    _log.debug(
                        "matched angle by name=%r instead of angleID in multicam %r",
                        angle_id, container_ref,
                    )
                    chosen = a
                    break
        if chosen is None:
            raise ParseError(
                f"mc-source references angleID {angle_id!r}, "
                f"no matching angle in multicam {container_ref!r}"
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

    # Resolve may nest asset-clips inside a child <clip> rather than directly
    # under the angle — unwrap one level if needed.
    all_clips = chosen.findall("asset-clip")
    if not all_clips:
        for clip_wrapper in chosen.findall("clip"):
            all_clips.extend(clip_wrapper.findall("asset-clip"))
        if all_clips:
            _log.debug("found asset-clips inside <clip> wrapper in angle %r", chosen.get("name"))
    if not all_clips:
        raise ParseError(
            f"angle {chosen.get('name')!r} has no <asset-clip>; "
            "audio-only angle formats are not supported"
        )

    # When an angle has multiple asset-clips, pick the one whose range covers
    # the segment's start position within the multicam container. Compare in
    # zero-based container time, not multicam-tc space.
    zero_based_start = segment_start - mcam_tc_start
    asset_clip = all_clips[0]
    if len(all_clips) > 1:
        for ac in all_clips:
            ac_offset = _safe_parse_rational(ac.get("offset"), what="asset-clip offset")
            ac_duration = _safe_parse_rational(ac.get("duration"), what="asset-clip duration")
            if ac_duration <= 0:
                continue
            if ac_offset <= zero_based_start < ac_offset + ac_duration:
                asset_clip = ac
                break

    asset_ref = asset_clip.get("ref")
    asset_el = resource_by_id.get(asset_ref)
    if asset_el is None or asset_el.tag != "asset":
        raise ParseError(f"asset-clip ref {asset_ref!r} does not resolve to an <asset>")

    return {
        "path": _resolve_asset_path(asset_el),
        "asset_id": asset_ref,
        "angle_offset": parse_rational(asset_clip.get("offset")),
        "angle_start": parse_rational(asset_clip.get("start")),
        "container_tc_start": mcam_tc_start,
        "asset_start": _safe_parse_rational(
            asset_el.get("start"), what=f"asset {asset_ref!r} start",
        ),
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


def _offset_within_sync_clip(clip_el, sync_clip_el) -> Fraction:
    """Absolute offset of ``clip_el`` within ``sync_clip_el``'s internal
    timeline — the coordinate the sync-clip's ``start`` attribute is measured
    in. A select's source time is shifted by this so the emitted clip lands on
    the right footage.

    Walks up the parent chain. A clip directly on the sync-clip's inner
    ``<spine>`` (or a direct child of the sync-clip) contributes its own
    ``offset``. A connected/nested clip (``lane != 0`` — external audio nested
    inside the camera asset-clip, or attached in a ``<gap>``) is anchored in
    its parent's local timeline, so its position is the parent's position plus
    ``(offset - parent.start)``. Covers every FCP sync shape: camera-on-spine
    (Meeting), external-recorder-nested-in-camera (Interview), lane-in-gap.
    """
    if clip_el is None or clip_el is sync_clip_el:
        return Fraction(0)
    parent = clip_el.getparent()
    off = parse_rational(clip_el.get("offset"))
    if parent is None or parent is sync_clip_el or parent.tag == "spine":
        return off
    parent_start = parse_rational(parent.get("start"))
    return _offset_within_sync_clip(parent, sync_clip_el) + (off - parent_start)


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
    # Third shape: FCP also writes sync-clips with NO inner <spine>, where
    # the external lane-attached audio sits nested INSIDE the primary
    # asset-clip itself (lane=-1 child of the camera/video asset-clip).
    # Without scanning here, sync-clips of this shape whose camera mic was
    # muted via <sync-source>/<audio-role-source active="0"> resolve to
    # is_muted=True and drop out of the timeline-audio plan entirely.
    for primary_clip in list(sync_clip_el.findall("asset-clip")):
        for ac in primary_clip.findall("asset-clip"):
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

    # Where the chosen dialogue clip sits inside the sync-clip's internal
    # timeline (the coordinate sync-clip@start is measured in). Selects are in
    # the transcribed source's time, so shifting by this offset is what makes
    # the emitted clip land on the right footage. Works for every sync shape:
    #   - camera/dialogue clip on the sync-clip's main spine: its own offset
    #     (0 for "video at the top"; non-zero when FCP places it below a
    #     leading gap — the Meeting case);
    #   - external recorder nested inside the camera clip on a lane (the
    #     Interview case) or attached in a <gap> — composed via the parent
    #     chain by _offset_within_sync_clip.
    # tcStart/asset-start stay zero: the inner spine is zero-based and the
    # chosen clip's own start already expresses its media in-point.
    angle_offset = _offset_within_sync_clip(dialogue, sync_clip_el)
    angle_start = parse_rational(dialogue.get("start"))
    return {
        "path": _resolve_asset_path(asset_el),
        "asset_id": asset_ref,
        "angle_offset": angle_offset,
        "angle_start": angle_start,
        "container_tc_start": Fraction(0),
        "asset_start": Fraction(0),
        "is_muted": is_muted,
    }


def _resolve_asset_clip_audio(asset_clip_el, resource_by_id: dict) -> dict:
    """Resolve the audio source for a plain spine ``<asset-clip>``.

    A single-camera clip on the main spine references one ``<asset>`` directly;
    that asset's media file carries both the video and the audio. So the
    "audio source" is simply that asset's media-rep path — no multicam angle or
    sync pairing to disambiguate.

    The asset-clip's ``start`` (carried on the :class:`SpineSegment`) is measured
    in the asset's local timeline, whose origin is the asset's own ``start``.
    That is non-zero only for media with embedded start timecode (rare on
    single-cam consumer footage like Meta Glasses, but real for pro cameras), so
    we surface it as ``asset_start`` and let the timeline-audio renderer subtract
    it to seek to the right place in the file. There is no container indirection,
    so ``angle_offset`` / ``angle_start`` / ``container_tc_start`` are all zero —
    the same convention sync-clips use.

    ``is_muted`` is True when the asset declares ``hasAudio="0"`` (a video-only
    clip — silent b-roll): it still occupies timeline space but contributes
    silence, which is exactly what FCP plays there.
    """
    asset_ref = asset_clip_el.get("ref")
    asset_el = resource_by_id.get(asset_ref)
    if asset_el is None or asset_el.tag != "asset":
        raise ParseError(
            f"asset-clip ref {asset_ref!r} does not resolve to an <asset> "
            "(compound clips / <ref-clip> are not supported)"
        )
    asset_start = _safe_parse_rational(
        asset_el.get("start"), what=f"asset {asset_ref!r} start",
    )
    return {
        "path": _resolve_asset_path(asset_el),
        "asset_id": asset_ref,
        "angle_offset": Fraction(0),
        "angle_start": Fraction(0),
        "container_tc_start": Fraction(0),
        "asset_start": asset_start,
        "is_muted": asset_el.get("hasAudio") == "0",
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
        segment_start = parse_rational(child.get("start"))
        info = _resolve_multicam_audio(
            resource_by_id, child.get("ref") or "", angle_id,
            segment_start=segment_start,
        )
        return SegmentAudioSource(
            path=info["path"],
            asset_id=info["asset_id"],
            angle_offset_fraction=info["angle_offset"],
            angle_start_fraction=info["angle_start"],
            active_audio_angle_id=angle_id,
            is_muted=False,
            container_tc_start_fraction=info["container_tc_start"],
            asset_start_fraction=info["asset_start"],
        )
    elif child.tag == "asset-clip":
        info = _resolve_asset_clip_audio(child, resource_by_id)
        return SegmentAudioSource(
            path=info["path"],
            asset_id=info["asset_id"],
            angle_offset_fraction=info["angle_offset"],
            angle_start_fraction=info["angle_start"],
            active_audio_angle_id=None,
            is_muted=info["is_muted"],
            container_tc_start_fraction=info["container_tc_start"],
            asset_start_fraction=info["asset_start"],
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
            container_tc_start_fraction=info["container_tc_start"],
            asset_start_fraction=info["asset_start"],
        )


def parse_fcpxml(path) -> ParsedFCPXML:
    """Parse an FCPXML file and return a :class:`ParsedFCPXML` snapshot.

    Raises :class:`ParseError` for unsupported versions, missing structures,
    or unresolved references.
    """
    path_str = str(path)
    raw_bytes = Path(path_str).read_bytes()

    try:
        # Harden against entity-expansion DoS and external resource fetches.
        # FCPXML never relies on DTDs, internal/external entities, or remote
        # content, so disabling these defaults is safe for valid input.
        parser = etree.XMLParser(resolve_entities=False, huge_tree=False, no_network=True)
        root = etree.fromstring(raw_bytes, parser=parser)
    except etree.XMLSyntaxError as e:
        raise ParseError(f"invalid FCPXML: {e}") from e

    if root.tag != "fcpxml":
        raise ParseError(f"root element is <{root.tag}>, expected <fcpxml>")

    version = root.get("version")
    if version not in SUPPORTED_VERSIONS:
        raise ParseError(
            f"unsupported FCPXML version {version!r}; supported: {sorted(SUPPORTED_VERSIONS)}"
        )

    nle = _detect_nle(root)
    _log.debug("detected NLE source: %s (version %s)", nle, version)

    resources = root.find("resources")
    if resources is None:
        raise ParseError("no <resources> block")
    resource_by_id = {el.get("id"): el for el in resources if el.get("id")}

    # FCP always nests <sequence> inside <library>/<event>/<project>.
    # Resolve may omit <library> or <event>, putting <project>/<sequence>
    # directly under <fcpxml> or under a bare <event>.
    sequence = root.find(".//project/sequence")
    if sequence is None:
        raise ParseError("no <sequence> inside a <project> element")

    sequence_format_id = sequence.get("format") or ""
    sequence_duration = parse_rational(sequence.get("duration"))
    fmt_el = resource_by_id.get(sequence_format_id)
    if fmt_el is None:
        if nle == NLE_RESOLVE:
            _log.warning(
                "sequence format %r not in <resources>; "
                "Resolve export may use inline format — defaulting frame duration to 1001/24000",
                sequence_format_id,
            )
            frame_duration = Fraction(1001, 24000)
        else:
            raise ParseError(f"sequence references missing format id {sequence_format_id!r}")
    else:
        frame_duration = parse_rational(fmt_el.get("frameDuration"))

    spine = sequence.find("spine")
    if spine is None:
        raise ParseError("sequence has no <spine>")

    segments: List[SpineSegment] = []

    for child, parent_gap in iter_spine_clip_elements(spine):
        mc_sources: List[dict] = []
        if child.tag == "mc-clip":
            for ms in child.findall("mc-source"):
                mc_sources.append({
                    "angleID": ms.get("angleID") or "",
                    "srcEnable": ms.get("srcEnable") or "",
                })
            if not mc_sources:
                _log.debug(
                    "mc-clip %r has no <mc-source> children — will fall back to first audio angle",
                    child.get("name"),
                )

        audio_source = _resolve_segment_audio(child, resource_by_id, mc_sources)

        offset_fraction = parse_rational(child.get("offset"))
        lane = ""
        if parent_gap is not None:
            # Connected clips are anchored in the gap's LOCAL timeline, whose
            # origin is the gap's own start — the same composition rule
            # _offset_within_sync_clip uses for nested sync-clip audio.
            gap_offset = parse_rational(parent_gap.get("offset"))
            gap_start = parse_rational(parent_gap.get("start"))
            offset_fraction = gap_offset + (offset_fraction - gap_start)
            lane = child.get("lane") or ""
            if child.get("enabled") == "0" and audio_source is not None:
                audio_source.is_muted = True

        seg = SpineSegment(
            kind=child.tag,
            ref=child.get("ref") or "",
            name=child.get("name") or "",
            offset_fraction=offset_fraction,
            start_fraction=parse_rational(child.get("start")),
            duration_fraction=parse_rational(child.get("duration")),
            mc_sources=mc_sources,
            audio_source=audio_source,
            lane=lane,
        )
        segments.append(seg)

    if not segments:
        # Report what the spine *did* contain so the message is actionable —
        # e.g. a compound-clip ("<ref-clip>") or titles-only timeline tells us
        # exactly which shape to support next, instead of a dead-end error.
        present = sorted({
            c.tag for c in spine if isinstance(c.tag, str)
        })
        found = f" (spine contains: {', '.join(present)})" if present else " (spine is empty)"
        hint = ""
        if "ref-clip" in present:
            # Compound clips wrap their real clips a level down; flattening on
            # export turns them back into the asset-/mc-/sync-clips we read.
            hint = (" — this looks like a compound-clip timeline; in Final Cut "
                    "choose Clip ▸ Break Apart Clip Items (or flatten compound "
                    "clips) before exporting XML")
        raise ParseError(
            "spine has no clips Doza Assist can read — expected one or more "
            "<asset-clip>, <mc-clip>, or <sync-clip> elements" + found + hint
        )

    # Representative source: first non-muted PRIMARY segment (connected
    # lane segments never represent the timeline), else first segment.
    representative = next(
        (s for s in segments
         if not s.lane and s.audio_source and not s.audio_source.is_muted),
        next((s for s in segments
              if s.audio_source and not s.audio_source.is_muted),
             segments[0]),
    )
    rep_audio = representative.audio_source
    assert rep_audio is not None  # every segment resolves audio above

    # Multi-source when segments span more than one distinct audio asset, or
    # mix container kinds (so ingest renders a composed timeline WAV).
    # Connected (lane) segments are EXCLUDED here: they only matter on
    # timelines that are already multi-source, and including them would flip
    # is_multi_source for previously-ingested single-source projects whose
    # stored selects are in source-time coordinates (exports re-parse the
    # stored FCPXML at export time).
    primary_segments = [s for s in segments if not s.lane] or segments
    distinct_sources = {(s.audio_source.path, s.audio_source.asset_id)
                        for s in primary_segments if s.audio_source is not None}
    distinct_kinds = {s.kind for s in primary_segments}
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
        nle_source=nle,
        container_type=primary_segments[0].kind,
        container_ref=primary_segments[0].ref,
        audio_file_path=rep_audio.path,
        audio_asset_id=rep_audio.asset_id,
        active_audio_angle_id=rep_audio.active_audio_angle_id,
        audio_angle_offset_fraction=rep_audio.angle_offset_fraction,
        audio_angle_start_fraction=rep_audio.angle_start_fraction,
        audio_container_tc_start_fraction=rep_audio.container_tc_start_fraction,
        audio_asset_start_fraction=rep_audio.asset_start_fraction,
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
