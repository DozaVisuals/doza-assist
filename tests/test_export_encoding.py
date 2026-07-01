"""Regression: exports must be written UTF-8, not the process locale encoding.

Field bug (paid Norwegian user, direct channel):
    Export failed: 'ascii' codec can't encode character '\\xe5' ...
The export file writes used a bare ``open(path, "w")``, so Python picked the
locale's preferred encoding. The Flask backend is spawned by the Electron
shell with no ``LANG``/``LC_ALL`` set, so on a packaged Mac that resolves to
ASCII — and the first non-ASCII clip title (å, ·, —, …) raised
``UnicodeEncodeError`` mid-write, leaving a truncated file.

These tests SIMULATE that production condition: ``_ascii_default_open``
monkeypatches ``builtins.open`` so a text-mode write WITHOUT an explicit
``encoding=`` is forced to ASCII (exactly what an ASCII locale does), while
a call that passes ``encoding="utf-8"`` is honoured. So they FAIL on the
pre-fix bare opens and PASS once each export site passes ``encoding="utf-8"``
— independent of the host's own locale (which is UTF-8 in CI/dev and would
otherwise mask the bug, as the existing HOSTILE-marker tests show).
"""

import builtins
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from exporters.edl import EDLExporter
from exporters.fcpxml import FCPXMLExporter
from exporters.premiere_xml import PremiereXMLExporter
from exporters.resolve_xml import ResolveFCPXMLExporter, ResolveXMLExporter

# Norwegian clip titles + the exact reported breaking chars: å (\xe5),
# · (\xb7 / U+00B7), plus an em dash and ø for good measure.
NB_MARKERS = [
    {'start': 5, 'end': 9,
     'text': 'Håkon på Røa · første runde', 'note': 'Naboen — før midnatt',
     'category': 'Lyd'},
    {'start': 12, 'end': 20,
     'text': 'Jubel etter kampslutt · «vi er nr. 1»', 'note': 'fotball',
     'category': 'Klipp'},
]

# The FCPXML exporters now fail loudly when a named source file is missing
# at export time (instead of silently degrading to markers-only), so the
# non-ASCII source has to exist on disk. Content is never decoded.
@pytest.fixture
def common(tmp_path):
    src = tmp_path / 'Håkon.mp4'
    src.write_bytes(b'\x00')
    return dict(project_name='Fotballnetter · naboklage',
                source_path=str(src), media_duration=60.0,
                framerate=25.0, width=1920, height=1080)


@pytest.fixture
def ascii_locale(monkeypatch):
    """Force text-mode writes that omit ``encoding=`` to ASCII, mimicking the
    Electron-spawned backend's locale. Explicit ``encoding="utf-8"`` wins."""
    real_open = builtins.open

    def _patched(file, mode='r', *args, **kwargs):
        if 'b' not in mode and ('w' in mode or 'a' in mode) \
                and kwargs.get('encoding') is None and not args:
            kwargs['encoding'] = 'ascii'
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, 'open', _patched)


def _assert_utf8_roundtrip(file_path):
    """File exists and decodes cleanly as UTF-8 with the chars intact."""
    with open(file_path, 'rb') as fh:
        raw = fh.read()
    text = raw.decode('utf-8')  # raises if not valid UTF-8
    assert 'Håkon' in text and 'Røa' in text
    assert '·' in text  # the reported · (U+00B7)
    return text


_EXPORTERS = [
    ('edl', EDLExporter),
    ('fcpxml', FCPXMLExporter),
    ('premiere', PremiereXMLExporter),
    ('resolve-fcpxml', ResolveFCPXMLExporter),
    ('resolve-xml', ResolveXMLExporter),
]


@pytest.mark.parametrize('name,cls', _EXPORTERS)
def test_export_markers_utf8_under_ascii_locale(name, cls, ascii_locale, common, tmp_path):
    result = cls().export_markers(
        NB_MARKERS, export_type='labels', exports_dir=str(tmp_path),
        total_clips=len(NB_MARKERS), **common)
    _assert_utf8_roundtrip(result.file_path)


@pytest.mark.parametrize('name,cls', _EXPORTERS)
def test_export_story_utf8_under_ascii_locale(name, cls, ascii_locale, common, tmp_path):
    result = cls().export_story(
        NB_MARKERS, story_title='Naboen klager · runde 2',
        exports_dir=str(tmp_path), **common)
    _assert_utf8_roundtrip(result.file_path)
