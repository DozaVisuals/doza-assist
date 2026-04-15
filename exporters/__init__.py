"""
NLE exporters for Doza Assist.

Each supported editing platform has an exporter that turns a list of marker
dicts (the same shape used everywhere else in the app) into a file that can
be imported by the target NLE. The router maps a platform string to the
right exporter instance.
"""

from .base import BaseExporter, ExportResult
from .router import get_exporter, PLATFORMS, DEFAULT_PLATFORM

__all__ = [
    "BaseExporter",
    "ExportResult",
    "get_exporter",
    "PLATFORMS",
    "DEFAULT_PLATFORM",
]
