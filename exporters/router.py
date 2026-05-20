"""
Maps platform strings to exporter instances.

Resolve switched from EDL (CMX 3600) to FCP7 XML in 0.8.x. FCP7 XML
preserves clip names, notes, and source media paths — Resolve's
ImportTimelineFromFile reconnects media automatically without a
separate Media Pool import. EDL is kept available via the
``resolve-edl`` platform key for the niche workflows that explicitly
need CMX 3600 (hardware deck handoff, legacy color systems).
"""

from .base import BaseExporter
from .fcpxml import FCPXMLExporter
from .premiere_xml import PremiereXMLExporter
from .edl import EDLExporter
from .resolve_xml import ResolveXMLExporter

PLATFORMS = ("fcp", "premiere", "resolve", "resolve-edl")
DEFAULT_PLATFORM = "fcp"

_REGISTRY = {
    "fcp": FCPXMLExporter,
    "premiere": PremiereXMLExporter,
    "resolve": ResolveXMLExporter,
    "resolve-edl": EDLExporter,
}


def get_exporter(platform: str) -> BaseExporter:
    if platform not in _REGISTRY:
        raise ValueError(f"Unknown editing platform: {platform!r}")
    return _REGISTRY[platform]()
