"""
Maps platform strings to exporter instances.

Resolve goes through the FCPXML exporter (Apple's modern .fcpxml format,
the same one FCP itself uses). Resolve has first-class support for
FCPXML import including automatic audio-channel routing — when we tried
FCP7 XML (the legacy xmeml v5 format) Resolve imported the video and
timeline structure cleanly but couldn't resolve the audio channel
mapping, leaving the timeline silent. FCPXML's structure declares
audio per-clip rather than relying on the source file's track table,
which is what Resolve's importer expects.

EDL is kept available via the ``resolve-edl`` platform key for the
niche workflows that explicitly need CMX 3600 (hardware deck handoff,
legacy color systems). FCP7 XML is available via ``resolve-xml`` for
the rare case where someone needs to round-trip through Premiere Pro
on the way to Resolve.
"""

from .base import BaseExporter
from .fcpxml import FCPXMLExporter
from .premiere_xml import PremiereXMLExporter
from .edl import EDLExporter
from .resolve_xml import ResolveFCPXMLExporter, ResolveXMLExporter

PLATFORMS = ("fcp", "premiere", "resolve", "resolve-edl", "resolve-xml")
DEFAULT_PLATFORM = "fcp"

_REGISTRY = {
    "fcp": FCPXMLExporter,
    "premiere": PremiereXMLExporter,
    "resolve": ResolveFCPXMLExporter,
    "resolve-edl": EDLExporter,
    "resolve-xml": ResolveXMLExporter,
}


def get_exporter(platform: str) -> BaseExporter:
    if platform not in _REGISTRY:
        raise ValueError(f"Unknown editing platform: {platform!r}")
    return _REGISTRY[platform]()
