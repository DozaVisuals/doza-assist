"""
Maps platform strings to exporter instances.
"""

from .base import BaseExporter
from .fcpxml import FCPXMLExporter
from .premiere_xml import PremiereXMLExporter
from .edl import EDLExporter

PLATFORMS = ("fcp", "premiere", "resolve")
DEFAULT_PLATFORM = "fcp"

_REGISTRY = {
    "fcp": FCPXMLExporter,
    "premiere": PremiereXMLExporter,
    "resolve": EDLExporter,
}


def get_exporter(platform: str) -> BaseExporter:
    if platform not in _REGISTRY:
        raise ValueError(f"Unknown editing platform: {platform!r}")
    return _REGISTRY[platform]()
