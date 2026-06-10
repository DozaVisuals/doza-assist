"""Shared text sanitation for XML-emitting exporters.

XML 1.0 forbids most C0 control characters even when entity-escaped; a
single stray byte (bad encodings, text pasted from PDFs) makes FCP and
Premiere reject an entire export — the string-template FCPXML generator
emitted them raw, and the Premiere builder crashed in its prettify
re-parse. Every exporter funnels content text through this scrub before
escaping/serialization.
"""

import re

# Everything XML 1.0 disallows below U+0020 (tab/LF/CR are legal).
_XML_ILLEGAL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def scrub_xml_text(text) -> str:
    """Drop XML-1.0-illegal control characters from ``text``."""
    if not text:
        return "" if text is None else str(text)
    return _XML_ILLEGAL_RE.sub("", str(text))
