"""FCPXML writer — round-trips Doza Assist selects back into FCP-importable XML.

Two output modes:

- **Mode A (``write_selects_as_new_project``)**: emits a fresh project whose
  spine contains one clip per select, routed by the owning spine segment's
  kind — a fresh ``<mc-clip>`` for mc-clip segments (preserving the multicam
  angle enablement), or a deep copy of the original ``<sync-clip>`` /
  ``<asset-clip>`` for sync-clip / plain single-cam segments (so the source's
  format, conform-rate, audio role, and any reattached audio survive verbatim;
  only the in/out/position attributes are rewritten). The original
  ``<resources>`` block is spliced in byte-for-byte from the source FCPXML so
  that asset IDs and bookmark base64 blobs survive untouched — FCP is strict
  about bookmark mismatch on import.

- **Mode B (``write_markers_on_timeline``)**: emits the original project
  structure with ``<marker>`` elements injected into the existing spine clips
  at each select's in-point. Marker style (standard / completion / to-do)
  encodes select type.

Both modes take a :class:`~doza_assist.fcpxml.parser.ParsedFCPXML` (produced
during ingest) and a list of :class:`Select` objects.

Select time semantics depend on the parsed project:

- If :attr:`~ParsedFCPXML.is_multi_source` is False (one audio source drives
  the whole spine), :attr:`Select.start_seconds` / :attr:`Select.end_seconds`
  are in audio-source seconds — the coordinate system of the original source
  file that the transcription ran against.
- If ``is_multi_source`` is True, select times are in timeline seconds (from
  the sequence start). Ingest renders a composed timeline WAV in this case,
  so the transcript the editor clicks on is already timeline-aligned.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, List, Optional, Tuple

from lxml import etree

from exporters.xml_text import scrub_xml_text
from .parser import ParsedFCPXML, SpineSegment, SPINE_SEGMENT_TAGS
from .timecode import parse_rational, seconds_to_rational, timeline_to_segment


# ---------- public data types -----------------------------------------------

@dataclass(frozen=True)
class Select:
    """One editor-facing selection.

    Times are in seconds. See the module docstring for how they are
    interpreted relative to single-source vs. multi-source projects.

    ``speaker`` is the resolved display name of whoever is talking at the
    select's start (looked up from the transcript + ``speaker_names``
    rename map before the select is built). Empty when the project carries
    no speaker info; when set, round-trip writers append the name to the
    clip/marker note ("note — Speaker"). (Direct-media exports additionally
    emit a "Speaker: …" keyword; the round-trip writers do not.)
    """

    start_seconds: float
    end_seconds: float
    label: str = "Select"
    note: str = ""
    kind: str = "standard"        # 'strong' | 'standard' | 'question'
    speaker: str = ""

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
    """Raised when selects cannot be written (e.g. cross-boundary select)."""


# ---------- select → (segment, container time) locator ---------------------

def _locate_select_start(
    parsed: ParsedFCPXML, select_seconds: float
) -> Tuple[Optional[SpineSegment], Optional[Fraction]]:
    """Find the segment and container-internal time for a select start.

    For multi-source projects, ``select_seconds`` is a timeline time; find the
    segment whose timeline range covers it and translate to container time.
    For single-source projects, ``select_seconds`` is audio-source time (0-based
    into the file the transcription ran against); convert to container time by
    inverting the renderer's source-seek formula, then find the covering
    segment. Adding the representative's container-tc / asset start is what keeps
    selects landing correctly when the media carries embedded timecode (pro
    cameras, time-of-day TC) — both terms are zero for sync-clips and ordinary
    tcStart-zero footage, so this is a no-op there.

    Returns ``(None, None)`` if the time falls outside any spine segment.
    """
    if parsed.is_multi_source:
        seg, container_time = timeline_to_segment(parsed.spine_segments, select_seconds)
        return seg, container_time

    t = Fraction(select_seconds).limit_denominator(10_000_000)
    container_time = (
        t
        + parsed.audio_angle_offset_fraction
        - parsed.audio_angle_start_fraction
        + parsed.audio_container_tc_start_fraction
        + parsed.audio_asset_start_fraction
    )
    for seg in parsed.spine_segments:
        if seg.start_fraction <= container_time < seg.start_fraction + seg.duration_fraction:
            return seg, container_time
    return None, None


def _all_share_source(segments: List[SpineSegment]) -> bool:
    """True when every segment references the same underlying media source.

    Same source means: same kind, same container ref for mc-clips, same
    resolved audio asset for sync-clips. Adjacent sync-clips that FCP split
    from one continuous external recording satisfy this — they're effectively
    one take and a cross-boundary select over them is meaningful.
    """
    if len(segments) <= 1:
        return True
    first = segments[0]
    first_asset = first.audio_source.asset_id if first.audio_source else None
    for s in segments[1:]:
        if s.kind != first.kind:
            return False
        if first.kind == "mc-clip" and s.ref != first.ref:
            return False
        s_asset = s.audio_source.asset_id if s.audio_source else None
        if s_asset != first_asset:
            return False
    return True


def _locate_select_range(
    parsed: ParsedFCPXML, select: Select
) -> Tuple[SpineSegment, Fraction, Fraction]:
    """Resolve a select to (segment, container_start, container_end).

    Returns container-time fractions on the **starting** segment suitable for
    building a new clip. The container_end may extend past the starting
    segment's duration when the select spans multiple segments that all share
    the same underlying source (common when FCP split one continuous recording
    into several sync-clips; a story beat naturally spans those splits).

    Raises :class:`WriterError` when the select falls outside any segment or
    crosses a boundary between heterogeneous sources (e.g. mc-clip → sync-clip,
    or two mc-clips from different multicams) — those need to be split into
    separate selects.
    """
    seg, container_start = _locate_select_start(parsed, select.start_seconds)
    if seg is None or container_start is None:
        raise WriterError(
            f"select {select.label!r} at {select.start_seconds}s falls outside "
            "any spine segment"
        )
    duration = Fraction(select.duration_seconds).limit_denominator(10_000_000)
    container_end = container_start + duration
    if container_end <= seg.start_fraction + seg.duration_fraction:
        return seg, container_start, container_end

    # Multi-segment span. Collapse into one clip when all spanned segments
    # share the same source; otherwise reject with a trim-or-split hint.
    start_idx = parsed.spine_segments.index(seg)
    end_time = Fraction(select.end_seconds).limit_denominator(10_000_000)
    end_idx = start_idx
    for i in range(start_idx, len(parsed.spine_segments)):
        s = parsed.spine_segments[i]
        if s.offset_fraction < end_time:
            end_idx = i
        else:
            break
    spanned = parsed.spine_segments[start_idx:end_idx + 1]

    if _all_share_source(spanned):
        # Container-internal time is continuous across same-source segments
        # (mc-clips of the same multicam share a container timeline; sync-clip
        # math zeroes the angle offsets so container_time maps directly to
        # source time on the shared asset). Returning the starting segment +
        # the full container range lets the builder emit one clip that plays
        # the whole select duration from a single source.
        return seg, container_start, container_end

    raise WriterError(
        f"select {select.label!r} ({select.start_seconds}–{select.end_seconds}s) "
        "crosses a segment boundary between different sources; "
        "trim it to land within one source clip"
    )


def _iter_selects(selects: Iterable[Select], *, preserve_order: bool = False) -> List[Select]:
    out = [s for s in selects if s.duration_seconds > 0]
    if not preserve_order:
        # Default: sort by in-point so the new timeline reads chronologically.
        # Story Builder exports pass preserve_order=True so the build's
        # narrative ordering survives the round-trip.
        out.sort(key=lambda s: s.start_seconds)
    return out


def _format_suffix(original: Optional[str], suffix: str) -> str:
    base = (original or "Doza Project").strip()
    return f"{base} - {suffix}"


def _snap_clip_times(
    container_start: Fraction, container_end: Fraction, fd: Fraction
) -> Tuple[str, str, Fraction]:
    """Snap a select's in/out to the frame grid and derive duration from the
    *snapped* endpoints.

    ``start`` and ``duration`` are separate FCPXML attributes. Rounding each to
    the nearest frame independently lets ``start + duration`` land one frame past
    the snapped out point — and therefore past the source media — which FCP
    rejects on import as "Invalid edit with no respective media". Taking
    ``duration = snapped_end - snapped_start`` keeps the edit's out point exactly
    on the snapped end, so it can never overshoot.

    Returns ``(start_str, duration_str, duration_fraction)``. The fraction is an
    exact multiple of ``fd``, so the caller advances the timeline cursor with it
    and the new spine stays frame-contiguous (no sub-frame gaps/overlaps).
    """
    start_frac = parse_rational(seconds_to_rational(container_start, fd))
    end_frac = parse_rational(seconds_to_rational(container_end, fd))
    if end_frac - start_frac < fd:
        end_frac = start_frac + fd        # never emit a zero / sub-frame edit
    dur_frac = end_frac - start_frac
    return (
        seconds_to_rational(start_frac, fd),
        seconds_to_rational(dur_frac, fd),
        dur_frac,
    )


# ---------- Mode A: selects as a new project --------------------------------

def _build_mc_clip_node(
    parsed: ParsedFCPXML,
    select: Select,
    segment: SpineSegment,
    start_str: str,
    duration_str: str,
    offset_str: str,
) -> etree._Element:
    mc = etree.Element("mc-clip")
    mc.set("ref", segment.ref)
    mc.set("offset", offset_str)
    mc.set("name", scrub_xml_text(select.label) or "Select")
    mc.set("start", start_str)
    mc.set("duration", duration_str)

    # FCPXML 1.13/1.14 DTD requires children in order:
    #   (note?, timing-params, intrinsic-params-audio, mc-source*, anchor_items*, ...)
    # <note> must come first; <mc-source> must come before anchor-items.
    # Violating this order makes FCP silently drop the mc-source overrides
    # and fall back to the multicam's default angle — which manifests as
    # "audio but no video" on import.
    note_text = scrub_xml_text(select.note)
    speaker = scrub_xml_text(select.speaker)
    if speaker:
        note_text = f"{note_text} — {speaker}" if note_text else speaker
    if note_text:
        note = etree.SubElement(mc, "note")
        note.text = note_text

    # Replay the source segment's mc-source enablement so both video and audio
    # angles match what the editor had on the original timeline.
    if segment.mc_sources:
        for ms in segment.mc_sources:
            sub = etree.SubElement(mc, "mc-source")
            sub.set("angleID", ms.get("angleID", ""))
            sub.set("srcEnable", ms.get("srcEnable", ""))
    elif segment.audio_source and segment.audio_source.active_audio_angle_id:
        sub = etree.SubElement(mc, "mc-source")
        sub.set("angleID", segment.audio_source.active_audio_angle_id)
        sub.set("srcEnable", "audio")

    return mc


def _set_clip_note(clip_el: etree._Element, note_text: str) -> None:
    """Put our select note on a deep-copied clip at a DTD-valid position.

    ``<note>`` is a 0-or-1 child in the FCPXML DTD, so any note inherited from
    the source clip is dropped first (two notes would fail import). It then
    belongs at the front for ``<sync-clip>`` / ``<mc-clip>``, but after the
    required-leading ``<conform-rate>`` / ``<timeMap>`` for ``<asset-clip>`` —
    skipping those lands the note in the first slot the DTD allows for every
    clip kind.
    """
    for existing in clip_el.findall("note"):
        clip_el.remove(existing)
    if not note_text:
        return
    idx = 0
    for child in clip_el:
        if child.tag in ("conform-rate", "timeMap"):
            idx += 1
        else:
            break
    note = etree.Element("note")
    note.text = note_text
    clip_el.insert(idx, note)


# Direct-child elements that carry their own position inside the clip's (asset)
# timeline. When a select trims the source clip to a sub-range, these can fall
# outside the kept range; FCP still imports the clip but silently clips
# out-of-range markers and ignores overshooting durations, leaving stale,
# misleading metadata on the select. Structure-critical children (conform-rate,
# format, audio-channel-source config, the sync-clip inner <spine>, and filters
# with no explicit range — which apply to the whole clip) are NOT in this set
# and are preserved verbatim.
_TRIMMABLE_TIMED_TAGS = (
    "marker", "chapter-marker", "keyword", "rating", "analysis-marker",
    "filter-video", "filter-audio",
)


def _trim_nested_timing(
    clip_el: etree._Element,
    new_start: Fraction,
    new_end: Fraction,
    frame_duration: Fraction,
) -> None:
    """Drop / clamp inherited timed children that fall outside a trimmed select.

    Only DIRECT children are touched — a sync-clip's inner ``<spine>`` and the
    camera/audio clips nested within it are part of the source structure and are
    preserved verbatim. A child's ``start`` / ``duration`` is in the clip's own
    (asset) timeline, the same coordinate as the clip's rewritten ``start``, so
    the range comparison is direct. Children fully outside ``[new_start,
    new_end)`` are removed; those that partially overlap are clamped to it and
    re-snapped to the frame grid.
    """
    for child in list(clip_el):
        if child.tag not in _TRIMMABLE_TIMED_TAGS:
            continue
        start_attr = child.get("start")
        if start_attr is None:
            continue  # no position of its own (e.g. a whole-clip filter) — keep
        a_start = parse_rational(start_attr)
        dur_attr = child.get("duration")
        if dur_attr is not None:
            a_end = a_start + parse_rational(dur_attr)
            outside = a_end <= new_start or a_start >= new_end
        else:
            a_end = a_start
            outside = a_start < new_start or a_start >= new_end
        if outside:
            clip_el.remove(child)
            continue
        clamped_start = max(a_start, new_start)
        if clamped_start != a_start:
            child.set("start", seconds_to_rational(clamped_start, frame_duration))
        if dur_attr is not None:
            clamped_end = min(a_end, new_end)
            child.set("duration", seconds_to_rational(clamped_end - clamped_start, frame_duration))


def _uniquify_text_style_defs(clip_el: etree._Element, suffix: str) -> None:
    """Make every inline ``<text-style-def id>`` in a copied clip unique.

    Captions / titles define their styling inline as
    ``<text-style-def id="ts1">`` and reference it with
    ``<text-style ref="ts1">``. Those ids are document-unique only ONCE — but a
    Story Builder export emits several selects from the SAME captioned source
    clip, so each deep copy re-defines ts1…tsN and FCP rejects the import with
    "DTD validation failed. ID ts1 already defined". Appending a per-copy
    ``suffix`` to each definition id (and the refs that point at it within this
    same clip) restores document-wide uniqueness. Recurses, so captions nested
    inside a sync-clip's inner spine are covered too.
    """
    remap = {}
    for tsd in clip_el.iter("text-style-def"):
        old_id = tsd.get("id")
        if not old_id:
            continue
        new_id = f"{old_id}{suffix}"
        tsd.set("id", new_id)
        remap[old_id] = new_id
    if not remap:
        return
    for ts in clip_el.iter("text-style"):
        ref = ts.get("ref")
        if ref in remap:
            ts.set("ref", remap[ref])


def _build_copied_clip_node(
    parsed: ParsedFCPXML,
    select: Select,
    segment: SpineSegment,
    original_element: etree._Element,
    start_str: str,
    duration_str: str,
    offset_str: str,
    copy_seq: int = 0,
) -> etree._Element:
    """Emit a select by deep-copying its source spine clip and rewriting only
    the positioning attributes. Used for ``<sync-clip>`` and ``<asset-clip>``
    segments (mc-clips are rebuilt fresh — see :func:`_build_mc_clip_node`).

    Deep-copying keeps everything FCP needs to re-import the select exactly as
    the source played it: ``ref`` / ``format`` / ``conform-rate`` / audio role,
    a sync-clip's inner camera+audio ``<spine>``, an asset-clip's
    audio-channel config and filters. We rewrite ``offset`` (the select's slot
    on the new timeline), ``start`` (its source in-point), ``duration`` (its
    length) and ``name``.

    For sync-clips ``container_start`` is the source time into the chosen audio
    asset (angle offsets collapse to zero during parse). For asset-clips it is
    the in-point in the asset's own timeline. Either way it maps directly to the
    copied clip's ``start``. Any stale ``audioStart`` / ``audioDuration`` (a
    source J/L split) is dropped so the select's audio follows its new range
    instead of pointing at the original clip's audio window. Inherited timed
    children (markers, keyword / rating ranges) that fall outside the trimmed
    range are dropped or clamped — see :func:`_trim_nested_timing`.
    """
    new_clip = copy.deepcopy(original_element)
    new_clip.set("offset", offset_str)
    new_clip.set("start", start_str)
    new_clip.set("duration", duration_str)
    new_clip.set("name", scrub_xml_text(select.label) or "Select")
    for attr in ("audioStart", "audioDuration"):
        if attr in new_clip.attrib:
            del new_clip.attrib[attr]

    # The select keeps only its slice of the source clip, so prune inherited
    # annotations/effects that now sit outside that slice.
    sel_start = parse_rational(start_str)
    sel_end = sel_start + parse_rational(duration_str)
    _trim_nested_timing(
        new_clip, sel_start, sel_end, parsed.sequence_frame_duration,
    )

    # Connected captions (subtitles / lower-thirds) are anchored by ``offset``
    # in the clip's own source-time coordinate. A subtitled interview carries
    # hundreds; without pruning, every select drags the entire caption track.
    # Drop the ones that don't overlap the kept range (FCP would hide them
    # anyway — they fall outside the trimmed clip).
    for cap in list(new_clip.findall("caption")):
        cap_start = parse_rational(cap.get("offset"))
        cap_end = cap_start + parse_rational(cap.get("duration"))
        if cap_end <= sel_start or cap_start >= sel_end:
            new_clip.remove(cap)

    # FCP reuses one <text-style-def> across captions with identical styling:
    # the def lives in the FIRST such caption and later captions carry only
    # <text-style ref="tsN">. If pruning dropped the def-carrying caption but
    # kept a referencing one, the ref now dangles and FCP rejects the import
    # on IDREF validation. Rescue: copy each missing def from the ORIGINAL
    # element into the first kept caption before uniquification.
    kept_refs = {
        ts.get("ref") for ts in new_clip.iter("text-style") if ts.get("ref")
    }
    kept_defs = {
        d.get("id") for d in new_clip.iter("text-style-def") if d.get("id")
    }
    missing = kept_refs - kept_defs
    if missing:
        first_kept_caption = new_clip.find("caption")
        if first_kept_caption is not None:
            for d in original_element.iter("text-style-def"):
                if d.get("id") in missing:
                    first_kept_caption.append(copy.deepcopy(d))
                    missing.discard(d.get("id"))

    # Any caption/title styles that survived define <text-style-def id="tsN">
    # inline. Give this copy a private suffix so several selects cut from the
    # same captioned source clip can't all re-define ts1…tsN — the duplicate
    # ids FCP rejects with "DTD validation failed. ID ts1 already defined".
    _uniquify_text_style_defs(new_clip, f"_s{copy_seq}")

    note_text = scrub_xml_text(select.note)
    speaker = scrub_xml_text(select.speaker)
    if speaker:
        note_text = f"{note_text} — {speaker}" if note_text else speaker
    _set_clip_note(new_clip, note_text)

    return new_clip


def _safe_version(version: str) -> str:
    """parsed.version is spliced raw into the serialized root tag; a corrupt
    or hostile source attribute (quotes, angle brackets) would inject into
    our output. Only dotted-numeric versions pass; anything else falls back
    to a known-good value."""
    if version and re.fullmatch(r"\d+(\.\d+)*", version):
        return version
    return "1.11"


def _index_original_spine(parsed: ParsedFCPXML) -> List[etree._Element]:
    """Parse ``parsed.original_fcpxml_bytes`` and return the main-spine clip
    elements (``mc-clip`` / ``sync-clip`` / ``asset-clip``) in document order.

    Mirrors ``parsed.spine_segments`` one-for-one — the parser and this walker
    both visit direct children of ``<project>/<sequence>/<spine>`` whose tag is
    in :data:`SPINE_SEGMENT_TAGS`, so index N in the returned list corresponds
    to ``parsed.spine_segments[N]``.
    """
    root = etree.fromstring(parsed.original_fcpxml_bytes)
    # MUST anchor on the project's sequence: <resources> precedes <library>
    # in document order, so a compound clip stored as <media><sequence><spine>
    # would match a bare ".//sequence/spine" first and every select would
    # deep-copy clips from inside the compound instead of the real spine
    # (mirrors the parser's anchor at parse_fcpxml).
    spine = root.find(".//project/sequence/spine")
    if spine is None:
        spine = root.find(".//sequence/spine")
    if spine is None:
        return []
    return [c for c in spine if c.tag in SPINE_SEGMENT_TAGS]


def _build_selects_spine(
    parsed: ParsedFCPXML,
    selects: List[Select],
    skipped: Optional[List[Select]] = None,
) -> Tuple[etree._Element, Fraction]:
    spine = etree.Element("spine")
    timeline_cursor = Fraction(0)
    fd = parsed.sequence_frame_duration
    original_spine_clips = _index_original_spine(parsed)
    copy_seq = 0  # unique per deep-copied clip — drives text-style-def suffixing
    for s in selects:
        try:
            segment, container_start, container_end = _locate_select_range(parsed, s)
        except WriterError as e:
            if "falls outside" in str(e) and skipped is not None:
                skipped.append(s)
                continue
            raise
        seg_idx = parsed.spine_segments.index(segment)
        # Snap once: start/duration share the same snapped endpoints (so the edit
        # can't overshoot the source) and the cursor advances by the same snapped
        # duration (so the new spine stays frame-contiguous).
        start_str, dur_str, dur_frac = _snap_clip_times(container_start, container_end, fd)
        offset_str = seconds_to_rational(timeline_cursor, fd)
        if segment.kind == "mc-clip":
            node = _build_mc_clip_node(parsed, s, segment, start_str, dur_str, offset_str)
        else:
            # sync-clip and asset-clip both round-trip by deep-copying their
            # source spine element, so both need that original element on hand.
            if seg_idx >= len(original_spine_clips):
                if skipped is not None:
                    skipped.append(s)
                continue
            node = _build_copied_clip_node(
                parsed, s, segment, original_spine_clips[seg_idx],
                start_str, dur_str, offset_str, copy_seq,
            )
            copy_seq += 1
        spine.append(node)
        timeline_cursor += dur_frac
    return spine, timeline_cursor


def write_selects_as_new_project(
    parsed: ParsedFCPXML,
    selects: Iterable[Select],
    *,
    project_name: Optional[str] = None,
    event_name: Optional[str] = None,
    preserve_order: bool = False,
    skipped_out: Optional[List[Select]] = None,
) -> bytes:
    """Mode A — emit an FCPXML where the selects are a new project's spine.

    The original ``<resources>`` block is preserved byte-for-byte; only the
    ``<library>`` is rebuilt. Returns UTF-8 encoded FCPXML bytes.

    Pass ``preserve_order=True`` for Story Builder exports — keeps the build's
    custom clip order intact instead of chronologically sorting by source
    in-point.
    """
    selects = _iter_selects(selects, preserve_order=preserve_order)
    if not selects:
        raise WriterError("no selects provided")

    project_title = project_name or _format_suffix(parsed.project_name, "Doza Selects")
    event_title = event_name or (parsed.event_name or "Doza Selects")

    skipped: List[Select] = []
    spine_el, total_duration = _build_selects_spine(parsed, selects, skipped=skipped)
    if skipped_out is not None:
        # Surface partial drops to the caller — an editor exporting 12
        # selects and silently receiving 10 clips is how the lane-1-gap
        # bug shipped unnoticed.
        skipped_out.extend(skipped)

    if len(skipped) == len(selects):
        raise WriterError(
            "all selects fall outside the timeline segments — nothing to export"
        )

    # Total timeline duration is the sum of the per-clip snapped durations that
    # were actually placed — frame-aligned, so the sequence duration butts
    # exactly against the last clip's out point.
    fd = parsed.sequence_frame_duration

    sequence = etree.Element("sequence")
    synthesized_format = b""
    if parsed.sequence_format_id:
        sequence.set("format", parsed.sequence_format_id)
    else:
        # Resolve-exported FCPXML can omit the sequence format resource (the
        # parser warned and defaulted the frame duration). Referencing a
        # missing/empty id is an unresolved IDREF that FCP rejects — so
        # synthesize a format carrying the effective frame duration and
        # splice it in after the original resources.
        sequence.set("format", "dozaFmt1")
        synthesized_format = (
            f'<format id="dozaFmt1" frameDuration="{fd.numerator}/{fd.denominator}s"/>'
        ).encode("utf-8")
    sequence.set("duration", seconds_to_rational(total_duration, fd))
    sequence.set("tcStart", "0s")
    sequence.set("tcFormat", "NDF")
    sequence.set("audioLayout", "stereo")
    sequence.set("audioRate", "48k")
    sequence.append(spine_el)

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
    resources_xml = parsed.original_resources_xml
    if synthesized_format and b"</resources>" in resources_xml:
        resources_xml = resources_xml.replace(
            b"</resources>", b"    " + synthesized_format + b"\n</resources>", 1)

    buf = bytearray()
    buf += _XML_PROLOGUE
    buf += f'<fcpxml version="{_safe_version(parsed.version)}">\n    '.encode("utf-8")
    buf += resources_xml
    buf += b"\n    "
    buf += library_bytes
    buf += b"</fcpxml>\n"
    return bytes(buf)


# ---------- Mode B: markers on the existing timeline ------------------------

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
    note_text = scrub_xml_text(select.note)
    speaker = scrub_xml_text(select.speaker)
    if speaker:
        note_text = f"{note_text} — {speaker}" if note_text else speaker
    if note_text:
        marker.set("note", note_text)
    return marker


def write_markers_on_timeline(
    parsed: ParsedFCPXML,
    selects: Iterable[Select],
    *,
    project_name_suffix: str = "Doza Notes",
    skipped_out: Optional[List[Select]] = None,
) -> bytes:
    """Mode B — copy the original structure and inject markers at each select.

    The original resources block is preserved byte-for-byte; only the
    ``<library>`` structure is re-serialized with marker children added to the
    appropriate spine clips. Selects that fall outside any spine segment are
    silently dropped (that source audio is unused on the timeline).
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
        c for c in spine if c.tag in SPINE_SEGMENT_TAGS
    ]
    if len(spine_clip_elements) != len(parsed.spine_segments):
        raise WriterError(
            "spine structure changed since parse; cannot attach markers safely"
        )
    element_for_segment = {
        id(seg): el for seg, el in zip(parsed.spine_segments, spine_clip_elements)
    }

    for s in selects:
        segment, container_time = _locate_select_start(parsed, s.start_seconds)
        if segment is None or container_time is None:
            # Select falls in a gap — source audio is never on the timeline here.
            if skipped_out is not None:
                skipped_out.append(s)
            continue
        marker = _marker_element(parsed, s, container_time)
        target = element_for_segment[id(segment)]
        # FCPXML's content model puts marker items BEFORE audio-channel-source,
        # filters, and metadata; appending after them is DTD-invalid and real
        # FCP exports carry those children on most spine clips.
        _TRAILING_TAGS = (
            "audio-channel-source", "filter-video", "filter-video-mask",
            "filter-audio", "metadata",
        )
        insert_at = None
        for idx, child in enumerate(target):
            if child.tag in _TRAILING_TAGS:
                insert_at = idx
                break
        if insert_at is None:
            target.append(marker)
        else:
            target.insert(insert_at, marker)

    library_bytes = etree.tostring(library, pretty_print=True, encoding="utf-8")

    # Splice the original <resources> in verbatim (lxml would otherwise re-serialize
    # the bookmark blobs; in practice text content round-trips, but byte-splicing
    # costs nothing and removes any doubt from an FCP import reviewer's mind).
    buf = bytearray()
    buf += _XML_PROLOGUE
    buf += f'<fcpxml version="{_safe_version(parsed.version)}">\n    '.encode("utf-8")
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
