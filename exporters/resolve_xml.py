"""DaVinci Resolve XML exporters.

Two flavors live here, picked via router.py:

  - ``ResolveFCPXMLExporter`` (registry key ``"resolve"``, the default):
    Apple's modern FCPXML format (.fcpxml). Resolve has first-class
    support for FCPXML import — clip names, source media via
    file:// URLs, AND per-clip audio routing all come through cleanly.
    We learned this the hard way: an earlier version of this module
    routed Resolve through FCP7 XML (the older xmeml v5 format), and
    while video and timeline structure imported fine, the audio was
    silent — Resolve couldn't bind the clipitems' <sourcetrack> entries
    to source-file audio tracks without extra schema that's brittle to
    emit. FCPXML side-steps that entirely.

  - ``ResolveXMLExporter`` (registry key ``"resolve-xml"``, opt-in):
    FCP7 XML / xmeml v5 — kept for the rare case where someone needs to
    round-trip through Premiere Pro on the way to Resolve, since
    Premiere only reads FCP7 XML, not FCPXML. The audio limitation is
    real for this path; users who need it usually re-link audio in
    Resolve manually.
"""

from .fcpxml import FCPXMLExporter
from .premiere_xml import PremiereXMLExporter


class ResolveFCPXMLExporter(FCPXMLExporter):
    """FCPXML output, tagged for Resolve so toasts/filenames are accurate.

    Schema is identical to the Final Cut Pro exporter — Resolve and
    FCP both consume the same FCPXML files. Only the metadata strings
    change so the route layer knows this was a Resolve-targeted export.
    """
    platform_name = "DaVinci Resolve"


class ResolveXMLExporter(PremiereXMLExporter):
    """FCP7 XML output (xmeml v5), tagged for Resolve.

    Opt-in via the ``resolve-xml`` platform key. See module docstring
    for why FCPXML is the default for Resolve.
    """
    format_name = "FCP7 XML for Resolve"
    file_extension = ".xml"
    platform_name = "DaVinci Resolve"
