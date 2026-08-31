"""Event-level FCPXML import: synthesize the scratch sequence the editor skipped.

Final Cut can export XML straight from the browser: select clips in an event
and File > Export XML. The result has no ``<project>``/``<sequence>`` — the
clips sit directly under ``<event>`` (or, for a bare clip selection, directly
under ``<fcpxml>``; both shapes are in the v1.14 DTD's ``event_item``
production). ``parse_fcpxml`` anchors on ``.//project/sequence``, so these
files hard-fail at import today and editors work around it by building a
throwaway timeline just to hold the clip.

This module manufactures that throwaway timeline instead: it wraps ONE chosen
event-level clip in a minimal ``library > event > project > sequence > spine``
document, keeping the original ``<resources>`` block byte-for-byte (FCP's
bookmark validation is sensitive to it, same reason the writer preserves it).
The wrapper — not the raw event export — becomes the project's stored FCPXML,
so every downstream consumer (export-time re-parse, both writer modes, the
timeline-audio renderer) sees the exact shape a hand-made scratch sequence
produces and needs no changes.

V1 imports ``mc-clip`` items only. The other event_item kinds are counted so
the UI can say what was skipped, but their wrapper semantics differ (an
event-level ``asset-clip`` may omit ``duration`` entirely — DTD
``clip_attrs_with_optional_duration`` — and ``sync-clip``/``clip`` start
defaults are unproven on event exports), so they stay out until each kind's
attribute-default audit passes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

from lxml import etree

from .parser import SUPPORTED_VERSIONS, ParseError, _extract_resources_bytes
from .timecode import parse_rational

# Kinds the wrapper knows how to synthesize. Deliberately mc-clip-only in V1 —
# see module docstring before widening, and audit each kind's required/default
# attributes against the DTD when you do.
IMPORTABLE_KINDS = ("mc-clip",)

# Every event_item kind that is a clip (not a collection or nested project) —
# used to count what a V1 import skips so the chooser can be honest about it.
KNOWN_CLIP_KINDS = ("mc-clip", "sync-clip", "asset-clip", "clip", "ref-clip", "audition")

_XML_PROLOGUE = b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n\n'


@dataclass
class EventClip:
    """One importable event-level clip, in stable document order."""
    index: int              # index within the IMPORTABLE clips (the API contract)
    kind: str               # element tag ("mc-clip")
    name: str               # clip's name attribute, or "" when absent
    ref: str                # media resource id
    duration_seconds: float
    angle_count: int        # angles in the multicam resource (0 for non-mc kinds)
    event_name: Optional[str]  # enclosing <event> name, None for bare-fcpxml shape


def _hardened_root(fcpxml_bytes: bytes):
    """Parse bytes with the same DoS/network hardening as parse_fcpxml."""
    parser = etree.XMLParser(resolve_entities=False, huge_tree=False, no_network=True)
    try:
        return etree.fromstring(fcpxml_bytes, parser=parser)
    except etree.XMLSyntaxError as e:
        raise ParseError(f"invalid FCPXML: {e}") from e


def _event_only_root(fcpxml_bytes: bytes):
    """Root element for an event-only export this module should handle, else
    ``None`` (XML that doesn't parse, a non-fcpxml root, an unsupported
    version, or a document that HAS a ``project/sequence`` — those belong to
    the normal import path, whose original error must surface untouched)."""
    try:
        root = _hardened_root(fcpxml_bytes)
    except ParseError:
        return None
    if root.tag != "fcpxml" or root.get("version") not in SUPPORTED_VERSIONS:
        return None
    if root.find(".//project/sequence") is not None:
        return None
    return root


def _iter_event_level_clips(root):
    """Yield ``(event_el_or_None, clip_el)`` for both event-item shapes, in
    document order: children of each ``<event>``, then clips sitting directly
    under ``<fcpxml>`` (a bare browser-clip export)."""
    for event_el in root.findall("event"):
        for child in event_el:
            if isinstance(child.tag, str) and child.tag in KNOWN_CLIP_KINDS:
                yield event_el, child
    for child in root:
        if isinstance(child.tag, str) and child.tag in KNOWN_CLIP_KINDS:
            yield None, child


def _walk(root):
    """Single source of truth for what is importable, in what order.

    Returns ``(entries, skipped)`` where each entry is
    ``(event_el_or_None, clip_el, multicam_el, duration_seconds)`` for a clip
    that passes every gate, and ``skipped`` counts event-level clips that
    didn't (non-mc kinds, dangling/non-multicam refs, unusable durations).
    Both the enumeration and the synthesis index into this same list, so their
    indexes can never drift on a partially malformed file.
    """
    resources = root.find("resources")
    resource_by_id = {} if resources is None else {
        el.get("id"): el for el in resources if el.get("id")
    }
    entries = []
    skipped = 0
    for event_el, child in _iter_event_level_clips(root):
        if child.tag not in IMPORTABLE_KINDS:
            skipped += 1
            continue
        media_el = resource_by_id.get(child.get("ref") or "")
        multicam = media_el.find("multicam") if media_el is not None else None
        if multicam is None:
            # A dangling or non-multicam ref can't be wrapped; count it as
            # skipped rather than failing the whole enumeration.
            skipped += 1
            continue
        try:
            duration = float(parse_rational(child.get("duration")))
        except (ValueError, ZeroDivisionError):
            skipped += 1
            continue
        if duration <= 0:
            # DTD requires duration on mc-clip; a missing one parses as 0.
            skipped += 1
            continue
        entries.append((event_el, child, multicam, duration))
    return entries, skipped


def enumerate_event_clips(fcpxml_bytes: bytes) -> Tuple[List[EventClip], int]:
    """Return ``(importable_clips, skipped_other_clip_count)``.

    Returns ``([], 0)`` — never raises — for anything that is not an
    event-only export (see :func:`_event_only_root`).
    """
    root = _event_only_root(fcpxml_bytes)
    if root is None:
        return [], 0
    entries, skipped = _walk(root)
    clips = []
    for i, (event_el, child, multicam, duration) in enumerate(entries):
        angles = multicam.findall("mc-angle") or multicam.findall("angle")
        clips.append(EventClip(
            index=i,
            kind=child.tag,
            name=child.get("name") or "",
            ref=child.get("ref") or "",
            duration_seconds=duration,
            angle_count=len(angles),
            event_name=event_el.get("name") if event_el is not None else None,
        ))
    return clips, skipped


def synthesize_wrapper(fcpxml_bytes: bytes, index: int) -> Tuple[bytes, dict]:
    """Build the scratch-sequence wrapper for importable clip ``index``.

    Returns ``(wrapper_bytes, info)`` where ``info`` feeds the project's
    ``fcpxml_source.event_import`` block. Raises :class:`ParseError` on an
    out-of-range index or a clip whose multicam resource is unusable.
    """
    root = _event_only_root(fcpxml_bytes)
    if root is None:
        raise ParseError("not an event-only FCPXML export")
    entries, _skipped = _walk(root)
    if not entries:
        raise ParseError("no importable event-level clips in this FCPXML")
    if not (0 <= index < len(entries)):
        raise ParseError(
            f"event clip index {index} out of range — file has {len(entries)} importable clip(s)"
        )
    event_el, clip_el, multicam, _duration = entries[index]

    mcam_format = multicam.get("format")
    resources = root.find("resources")
    resource_ids = {el.get("id") for el in resources if el.get("id")}
    if not mcam_format or mcam_format not in resource_ids:
        # DTD marks multicam@format as a REQUIRED IDREF; a file violating that
        # would also fail FCP itself. Refuse rather than emit a dangling IDREF.
        raise ParseError(
            f"multicam resource {clip_el.get('ref')!r} has no resolvable format id"
        )

    spine_clip = copy.deepcopy(clip_el)
    spine_clip.set("offset", "0s")
    if spine_clip.get("start") is None:
        # An mc-clip's start lives in the multicam's timecode space. Absent
        # start would parse as 0, and on a jam-synced multicam (tcStart of
        # hours) a 0 window misses every angle file and degrades to the
        # single-file fallback — the 1.0.43 field bug shape. Pin the window
        # to the multicam's own origin explicitly.
        spine_clip.set("start", multicam.get("tcStart") or "0s")

    clip_name = clip_el.get("name") or ""
    event_name = event_el.get("name") if event_el is not None else None

    sequence = etree.Element("sequence")
    sequence.set("format", mcam_format)
    sequence.set("duration", clip_el.get("duration"))
    sequence.set("tcStart", "0s")
    sequence.set("tcFormat", "NDF")
    sequence.set("audioLayout", "stereo")
    sequence.set("audioRate", "48k")
    spine = etree.SubElement(sequence, "spine")
    spine.append(spine_clip)

    project = etree.Element("project")
    project.set("name", clip_name or "Imported Clip")
    project.append(sequence)

    event = etree.Element("event")
    event.set("name", event_name or "Imported Clips")
    event.append(project)

    library = etree.Element("library")
    library.append(event)
    library_bytes = etree.tostring(library, pretty_print=True, encoding="utf-8")

    version = root.get("version")
    buf = bytearray()
    buf += _XML_PROLOGUE
    buf += f'<fcpxml version="{version}">\n    '.encode("utf-8")
    buf += _extract_resources_bytes(fcpxml_bytes)
    buf += b"\n    "
    buf += library_bytes
    buf += b"</fcpxml>\n"

    info = {
        "clip_index": index,
        "clip_name": clip_name,
        "clip_kind": clip_el.tag,
        "total_clips": len(entries),
        "event_name": event_name,
        "angle_count": len(multicam.findall("mc-angle") or multicam.findall("angle")),
    }
    return bytes(buf), info
