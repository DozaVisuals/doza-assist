"""Producer-edge regressions from the 2026-08-23 certification sweep.

Three field-plausible shapes broke the round-trip; each test here is the
sweep's confirmed repro, kept as a permanent regression:

  1. Non-zero sequence tcStart (broadcast 01:00:00:00 / Resolve default
     timelines): FCP writes spine offsets in tcStart space; unnormalized
     offsets composed an all-silence timeline WAV and made every
     multi-source select unroutable.
  2. FCPXML v1.8 has no <media-rep> in Apple's DTD — assets carry ``src``.
     The parser rejected every genuine v1.8 file.
  3. A sequence format id that resolves to nothing was re-emitted by Mode A
     as a dangling IDREF, which Final Cut rejects on import.

DTD validation runs only where Final Cut Pro's own DTDs are installed
(developer machines); the numeric assertions carry the regression either way.
"""

import os
import subprocess
import sys
import tempfile
from fractions import Fraction

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from doza_assist.fcpxml import (  # noqa: E402
    Select,
    parse_fcpxml,
    write_selects_as_new_project,
)
from doza_assist.fcpxml.parser import ParseError  # noqa: E402
from doza_assist.fcpxml.writer import WriterError, re_parse  # noqa: E402
from doza_assist.fcpxml.timeline_audio import plan_render  # noqa: E402


_DTD_DIR = ("/Applications/Final Cut Pro.app/Contents/Frameworks/"
            "Interchange.framework/Versions/A/Resources")


def _dtd_validate(xml_bytes, version):
    """True/False when FCP's DTD is available, None when it isn't."""
    dtd = os.path.join(_DTD_DIR, f"FCPXMLv{version.replace('.', '_')}.dtd")
    if not os.path.isfile(dtd):
        return None
    # xmllint treats the DTD argument as a URI — copy to a space-free path.
    with tempfile.TemporaryDirectory() as td:
        local = os.path.join(td, "v.dtd")
        with open(dtd, "rb") as fin, open(local, "wb") as fout:
            fout.write(fin.read())
        doc = os.path.join(td, "doc.fcpxml")
        with open(doc, "wb") as f:
            f.write(xml_bytes)
        r = subprocess.run(["xmllint", "--noout", "--dtdvalid", local, doc],
                           capture_output=True, text=True, timeout=60)
        return r.returncode == 0


def _fmt(fid="r1"):
    return (f'<format id="{fid}" name="FFVideoFormat1080p25" frameDuration="100/2500s" '
            'width="1920" height="1080"/>')


def _asset(aid, name, dur="60s"):
    return (f'<asset id="{aid}" name="{name}" start="0s" duration="{dur}" hasVideo="1" '
            'hasAudio="1" audioSources="1" audioChannels="2" audioRate="48000" format="r1">'
            f'<media-rep kind="original-media" src="file:///Volumes/M/{name}.mov"/></asset>')


def _doc(resources, spine, version="1.13", seq_attrs='format="r1" duration="120s" tcStart="0s"'):
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n'
            f'<fcpxml version="{version}"><resources>{resources}</resources>'
            '<library location="file:///Users/t/Movies/T.fcpbundle/"><event name="E">'
            f'<project name="P"><sequence {seq_attrs} tcFormat="NDF" '
            'audioLayout="stereo" audioRate="48k">'
            f'<spine>{spine}</spine></sequence></project></event></library></fcpxml>')


def _parse(tmp_path, name, text):
    p = tmp_path / f"{name}.fcpxml"
    p.write_text(text)
    return parse_fcpxml(p)


# ── Bug 1: sequence tcStart normalization ─────────────────────────────

class TestSequenceTcStart:
    def _tc_doc(self, offsets=("3600s", "3660s")):
        res = _fmt() + _asset("r2", "camA") + _asset("r3", "camB")
        spine = (f'<asset-clip name="camA" ref="r2" offset="{offsets[0]}" duration="60s" start="0s"/>'
                 f'<asset-clip name="camB" ref="r3" offset="{offsets[1]}" duration="60s" start="0s"/>')
        return _doc(res, spine, seq_attrs='format="r1" duration="120s" tcStart="3600s"')

    def test_offsets_normalized_and_plan_playable(self, tmp_path):
        p = _parse(tmp_path, "tc", self._tc_doc())
        assert [float(s.offset_fraction) for s in p.spine_segments] == [0.0, 60.0]
        assert p.sequence_tc_start_fraction == Fraction(3600)
        plan = plan_render(p)
        # The confirmed field failure: delays were [3600000, 3660000] against
        # a 120s WAV base — pure silence.
        assert [it["timeline_offset_ms"] for it in plan] == [0, 60000]

    def test_selects_route_in_timeline_coordinates(self, tmp_path):
        p = _parse(tmp_path, "tc2", self._tc_doc())
        skipped = []
        out = write_selects_as_new_project(
            p, [Select(10, 20), Select(70, 80)], skipped_out=skipped)
        assert skipped == []
        rp = re_parse(out)
        assert [s.audio_source.path.rsplit("/", 1)[-1] for s in rp.spine_segments] \
            == ["camA.mov", "camB.mov"]
        ok = _dtd_validate(out, p.version)
        assert ok is not False

    def test_zero_based_offsets_with_tcstart_left_alone(self, tmp_path):
        # A producer that writes 0-based offsets despite tcStart: the guard
        # (all primary offsets >= tcStart) must leave them untouched.
        p = _parse(tmp_path, "tc3", self._tc_doc(offsets=("0s", "60s")))
        assert [float(s.offset_fraction) for s in p.spine_segments] == [0.0, 60.0]

    def test_tcstart_zero_unchanged(self, tmp_path):
        res = _fmt() + _asset("r2", "camA") + _asset("r3", "camB")
        spine = ('<asset-clip name="camA" ref="r2" offset="0s" duration="60s" start="0s"/>'
                 '<asset-clip name="camB" ref="r3" offset="60s" duration="60s" start="0s"/>')
        p = _parse(tmp_path, "tc4", _doc(res, spine))
        assert [it["timeline_offset_ms"] for it in plan_render(p)] == [0, 60000]
        assert p.sequence_tc_start_fraction == 0

    def test_segmented_multicam_with_tcstart(self, tmp_path):
        # The exposure that grew in 1.0.44: segmented angles are multi-source,
        # so the broadcast-start-TC shape must normalize for them too.
        angle = ('<asset-clip name="p1" ref="rf0" offset="0s" duration="100s" start="0s"/>'
                 '<gap name="G" offset="100s" start="3600s" duration="20s"/>'
                 '<asset-clip name="p2" ref="rf1" offset="120s" duration="100s" start="0s"/>')
        res = (_fmt() + _asset("rf0", "p1", "100s") + _asset("rf1", "p2", "100s")
               + '<media id="m1" name="MC"><multicam format="r1" tcStart="0s">'
                 f'<mc-angle name="A" angleID="a0">{angle}</mc-angle></multicam></media>')
        spine = ('<mc-clip ref="m1" offset="3600s" name="MC" duration="220s" start="0s">'
                 '<mc-source angleID="a0" srcEnable="all"/></mc-clip>')
        p = _parse(tmp_path, "tc5", _doc(res, spine,
                   seq_attrs='format="r1" duration="220s" tcStart="3600s"'))
        assert p.is_multi_source
        assert float(p.spine_segments[0].offset_fraction) == 0.0
        assert [it["timeline_offset_ms"] for it in plan_render(p)] == [0, 120000]


# ── Bug 2: FCPXML v1.8 asset@src form ─────────────────────────────────

class TestV18AssetSrc:
    V18 = ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n'
           '<fcpxml version="1.8"><resources>'
           '<format id="r1" name="FFVideoFormat1080p25" frameDuration="100/2500s" '
           'width="1920" height="1080"/>'
           '<asset id="r2" name="a" start="0s" duration="60s" hasVideo="1" hasAudio="1" '
           'src="file:///Volumes/M/a.mov" format="r1"/>'
           '</resources>'
           '<project name="R"><sequence format="r1" duration="60s" tcStart="0s" tcFormat="NDF">'
           '<spine><asset-clip name="a" ref="r2" offset="0s" duration="60s" start="0s"/>'
           '</spine></sequence></project></fcpxml>')

    def test_parses_via_src_attribute(self, tmp_path):
        p = _parse(tmp_path, "v18", self.V18)
        assert p.audio_file_path == "/Volumes/M/a.mov"
        assert p.nle_source == "resolve"

    def test_round_trip(self, tmp_path):
        p = _parse(tmp_path, "v18rt", self.V18)
        out = write_selects_as_new_project(p, [Select(5, 10)])
        rp = re_parse(out)
        assert rp.spine_segments[0].audio_source.path == "/Volumes/M/a.mov"
        assert _dtd_validate(out, "1.8") is not False

    def test_asset_without_media_rep_or_src_still_errors(self, tmp_path):
        broken = self.V18.replace(' src="file:///Volumes/M/a.mov"', "")
        with pytest.raises(ParseError, match="media-rep> or src"):
            _parse(tmp_path, "v18b", broken)


# ── Bug 3: unresolved sequence format synthesizes dozaFmt1 ────────────

class TestMissingSequenceFormat:
    DOC = ('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n'
           '<fcpxml version="1.9"><resources>'
           '<format id="r1" name="FFVideoFormat1080p25" frameDuration="100/2500s" '
           'width="1920" height="1080"/>'
           '<asset id="r2" name="a" start="0s" duration="60s" hasVideo="1" hasAudio="1" '
           'audioSources="1" audioChannels="2" audioRate="48000" format="r1">'
           '<media-rep kind="original-media" src="file:///Volumes/M/a.mov"/></asset>'
           '</resources>'
           '<project name="R"><sequence format="rMissing" duration="60s" tcStart="0s" tcFormat="NDF">'
           '<spine><asset-clip name="a" ref="r2" offset="0s" duration="60s" start="0s"/>'
           '</spine></sequence></project></fcpxml>')

    def test_format_id_blanked_and_synthesized(self, tmp_path):
        p = _parse(tmp_path, "fmt", self.DOC)
        assert p.sequence_format_id == ""
        assert abs(p.sequence_framerate - 23.976) < 0.001
        out = write_selects_as_new_project(p, [Select(5, 10)])
        assert b"dozaFmt1" in out
        assert b"rMissing" not in out.split(b"<library")[1]  # no dangling IDREF in the new project
        re_parse(out)
        assert _dtd_validate(out, "1.9") is not False

    def test_fcp_shape_still_errors(self, tmp_path):
        # An FCP-flavored document (library location) with a missing format
        # keeps the strict ParseError — only the Resolve default path relaxes.
        fcp = self.DOC.replace('<project name="R">',
                               '<library location="file:///Users/t/M.fcpbundle/">'
                               '<event name="E"><project name="R">') \
                      .replace('</project></fcpxml>', '</project></event></library></fcpxml>') \
                      .replace('<fcpxml version="1.9">', '<fcpxml version="1.13">')
        with pytest.raises(ParseError, match="missing format"):
            _parse(tmp_path, "fmt2", fcp)
