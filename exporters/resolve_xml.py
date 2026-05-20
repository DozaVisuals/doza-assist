"""DaVinci Resolve timeline export — FCP7 XML (xmeml v5) format.

Why FCP7 XML and not EDL: EDL is CMX 3600 plain text. It truncates clip
names into `* CLIP NAME:` comments, has no field for editorial notes,
references source media only by an 8-char reel name, and forces the
editor to import source media into the Media Pool separately. The
resulting timeline lands offline until that manual import.

FCP7 XML carries:
  - clip names verbatim (no truncation)
  - per-clip notes via <comments><mastercomment*>
  - absolute media paths via <pathurl>file://localhost/...</pathurl>
  - rate/timebase/NTSC flags Resolve understands

When this file is imported via Resolve's MediaPool.ImportTimelineFromFile,
clips reconnect to media automatically without a separate Media Pool
import step — closer to FCP's one-click behavior.

EDL stays available as an opt-in (some workflows still need the bare
CMX 3600 format), but the Resolve registry now defaults to this.
"""

from .premiere_xml import PremiereXMLExporter


class ResolveXMLExporter(PremiereXMLExporter):
    """FCP7 XML output, branded as Resolve.

    Identical schema to the Premiere XML exporter (it IS FCP7 XML —
    Premiere doesn't have its own native interchange, just inherits
    the FCP7 schema). The only thing that changes is which app this
    file is destined for, so the wrapper just retags the metadata
    that downstream code uses to label toasts and pick filenames.
    """
    format_name = "FCP7 XML for Resolve"
    file_extension = ".xml"
    platform_name = "DaVinci Resolve"
