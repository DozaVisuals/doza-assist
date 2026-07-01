"""Cross-target export matrix + writer regressions from the 2026-06-10 audit.

Hostile text (quotes/&/unicode/control bytes) × framerates × TC shapes through
every generator family, validated for well-formedness; plus synthetic
fixtures for: compound-clip media in <resources> (wrong-spine indexing),
shared text-style defs across pruned captions, Mode B marker DTD ordering,
and Resolve sources missing a format resource."""

import os
import sys
import textwrap
import xml.dom.minidom as minidom

import pytest
from lxml import etree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fcpxml_export import generate_fcpxml, generate_story_fcpxml  # noqa: E402
from exporters.premiere_xml import PremiereXMLExporter  # noqa: E402
from exporters.edl import EDLExporter  # noqa: E402
from doza_assist.fcpxml.parser import parse_fcpxml  # noqa: E402
from doza_assist.fcpxml.writer import (  # noqa: E402
    Select, write_selects_as_new_project, write_markers_on_timeline,
)

HOSTILE = [
    {'start': 5, 'end': 9, 'text': '"We\'re #1 in R&D" — <Tracy\'s> intro', 'note': 'A & B\nmulti\nline', 'category': 'Sound&bite'},
    {'start': 12, 'end': 20, 'text': 'ünïcødé — ' + '"q" & \'a\' ' * 12, 'note': 'ctrl\x0bchar\x08note', 'category': 'C'},
    {'start': 25, 'end': 30, 'text': '"' * 16, 'note': '', 'category': 'M'},
]
RATES = [23.976, 25.0, 29.97, 48.0, 50.0, 59.94, 119.88]


def _touch(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(b"x")  # generator only checks existence, never decodes
    return str(p)


class TestFCPXMLMatrix:
    @pytest.mark.parametrize("rate", RATES)
    @pytest.mark.parametrize("mode", ["cuts", "markers"])
    def test_direct_generator(self, rate, mode, tmp_path):
        xml = generate_fcpxml(
            HOSTILE, project_name='P & "Q"', framerate=rate, mode=mode,
            source_path=_touch(tmp_path, "a & b's.mov"), media_duration=60.0,
            start_tc_frames=int(5.5 * 3600 * round(rate)), tc_format="DF")
        minidom.parseString(xml)
        assert "tcFormat" in xml

    @pytest.mark.parametrize("rate", [23.976, 29.97, 50.0])
    def test_story_generator(self, rate, tmp_path):
        xml = generate_story_fcpxml(
            HOSTILE, project_name="P", story_title="St'ory & <T>",
            framerate=rate, source_path=_touch(tmp_path, "x.wav"),
            media_duration=60.0)
        minidom.parseString(xml)

    def test_audio_only_has_no_video_flag(self, tmp_path):
        xml = generate_fcpxml(HOSTILE, source_path=_touch(tmp_path, "x.wav"),
                              media_duration=60.0)
        assert 'hasVideo="0"' in xml
        xml = generate_fcpxml(HOSTILE, source_path=_touch(tmp_path, "x.MOV"),
                              media_duration=60.0)
        assert 'hasVideo="1"' in xml  # case-insensitive video detection


class TestProbedAssetDecls:
    """hasVideo/hasAudio come from PROBED streams, not the file extension.

    Extension-only hasVideo declared a video component for audio-only
    .mp4/.mov sources (AAC podcast, multi-mono field recorder), which
    FCP/Resolve import offline/invalid — and let the connected-clip audio
    routing emit a <video> on a video-less asset. hasAudio="1" was
    hardcoded even for silent B-roll and for probe FAILURES (the silent
    pre-1.0.20 Resolve import shape). The extension heuristic survives
    only as a WARNED fallback when the probe can't answer.
    """

    MARKERS = [{"start": 1.0, "end": 5.0, "text": "a", "category": "x"}]

    def _patch(self, monkeypatch, *, video, layout, rate=48000, dialogue=None):
        from exporters import media_probe as mp
        monkeypatch.setattr(mp, "has_video_stream", lambda p: video)
        monkeypatch.setattr(mp, "get_audio_layout", lambda p: layout)
        monkeypatch.setattr(mp, "get_audio_sample_rate", lambda p: rate)
        monkeypatch.setattr(mp, "detect_dialogue_channels",
                            lambda p, *a, **k: dialogue)

    def test_audio_only_mp4_declares_no_video(self, monkeypatch, tmp_path):
        # Multi-mono audio-only container: must also NOT take the
        # connected-clip form (<clip><video>) despite the detected channel.
        self._patch(monkeypatch, video=False, layout=(4, 4), dialogue=[1])
        xml = generate_fcpxml(self.MARKERS,
                              source_path=_touch(tmp_path, "podcast.mp4"),
                              media_duration=60.0)
        assert 'hasVideo="0"' in xml
        assert '<video ' not in xml and '<clip ' not in xml
        assert '<asset-clip ' in xml

    def test_video_stream_wins_over_audio_extension(self, monkeypatch, tmp_path):
        self._patch(monkeypatch, video=True, layout=(1, 2))
        xml = generate_fcpxml(self.MARKERS,
                              source_path=_touch(tmp_path, "x.wav"),
                              media_duration=60.0)
        assert 'hasVideo="1"' in xml

    def test_video_probe_failure_falls_back_to_extension_with_warning(
            self, monkeypatch, tmp_path):
        self._patch(monkeypatch, video=None, layout=(1, 2))
        warnings = []
        xml = generate_fcpxml(self.MARKERS,
                              source_path=_touch(tmp_path, "x.mov"),
                              media_duration=60.0, warnings_out=warnings)
        assert 'hasVideo="1"' in xml
        assert any("video stream" in w for w in warnings)

    def test_no_audio_streams_declares_has_audio_zero(self, monkeypatch, tmp_path):
        # Silent B-roll / FX plate probes clean with zero audio streams:
        # the declaration must not contradict the media.
        self._patch(monkeypatch, video=True, layout=(0, 0), rate=None)
        warnings = []
        xml = generate_fcpxml(self.MARKERS,
                              source_path=_touch(tmp_path, "plate.mov"),
                              media_duration=60.0, warnings_out=warnings)
        assert 'hasAudio="0"' in xml
        assert 'audioSources=' not in xml and 'audioRole=' not in xml
        assert any("no audio streams" in w for w in warnings)

    def test_audio_probe_failure_fails_open_with_warning(self, monkeypatch, tmp_path):
        # ffprobe timeout on an audio-bearing MXF must keep the legacy
        # hasAudio="1" but WARN — a bare declaration with no layout attrs
        # is the exact shape that imported silent in Resolve pre-1.0.20.
        self._patch(monkeypatch, video=True, layout=None, rate=None)
        warnings = []
        xml = generate_fcpxml(self.MARKERS,
                              source_path=_touch(tmp_path, "big.mxf"),
                              media_duration=60.0, warnings_out=warnings)
        assert 'hasAudio="1"' in xml
        assert any("audio layout" in w for w in warnings)

    def test_story_generator_same_decls(self, monkeypatch, tmp_path):
        self._patch(monkeypatch, video=False, layout=(0, 0), rate=None)
        warnings = []
        xml = generate_story_fcpxml(
            self.MARKERS, project_name="P", story_title="S",
            source_path=_touch(tmp_path, "podcast.mp4"),
            media_duration=60.0, warnings_out=warnings)
        assert 'hasVideo="0"' in xml and 'hasAudio="0"' in xml
        assert any("no audio streams" in w for w in warnings)


class TestPremiereMatrix:
    @pytest.mark.parametrize("rate", [23.976, 25.0, 29.97, 119.88])
    @pytest.mark.parametrize("mode", ["cuts", "markers", "both"])
    def test_modes_parse(self, rate, mode, tmp_path):
        exporter = PremiereXMLExporter()
        result = exporter.export_markers(
            HOSTILE, project_name='P & "Q"', source_path="/tmp/v.mp4",
            media_duration=60.0, framerate=rate, width=1920, height=1080,
            export_type="labels", exports_dir=str(tmp_path), export_mode=mode)
        content = open(result.file_path).read()
        minidom.parseString(content)
        if mode in ("markers", "both"):
            assert "<marker>" in content  # mode honored, not silently cuts
        if mode == "both":
            assert "<clipitem" in content  # cuts present alongside markers


class TestEDLMatrix:
    def test_newlines_collapsed_and_tc_offset(self, tmp_path):
        exporter = EDLExporter()
        result = exporter.export_markers(
            HOSTILE, project_name="P\nQ", source_path="/tmp/a.mov",
            media_duration=60.0, framerate=23.976, width=1920, height=1080,
            export_type="labels", exports_dir=str(tmp_path),
            start_tc_frames=24 * 3600 * 5)  # 05:00:00:00 @ nominal 24
        content = open(result.file_path).read()
        for line in content.splitlines():
            assert "\r" not in line
            if line and not line.startswith(("TITLE:", "FCM:", "*")):
                assert line[:3].isdigit(), f"stray line: {line!r}"
        assert "05:00:05:" in content  # source TC carries embedded start
        assert "AA/V" in content

    def test_audio_only_channel_token(self, tmp_path):
        exporter = EDLExporter()
        result = exporter.export_markers(
            HOSTILE, project_name="P", source_path="/tmp/a.wav",
            media_duration=60.0, framerate=23.976, width=1920, height=1080,
            export_type="labels", exports_dir=str(tmp_path))
        content = open(result.file_path).read()
        assert "AA/V" not in content


def _roundtrip_fixture(tmp_path, *, compound_in_resources=False,
                       shared_caption_style=False, trailing_children=False,
                       missing_format=False):
    fmt = '' if missing_format else '<format id="r1" frameDuration="1001/24000s" width="1920" height="1080"/>'
    compound = ""
    if compound_in_resources:
        compound = textwrap.dedent("""\
            <media id="rc1" name="Compound">
                <sequence format="r1" duration="240240/24000s" tcStart="0s">
                    <spine>
                        <asset-clip name="INSIDE_COMPOUND" ref="r3" offset="0s" start="0s" duration="240240/24000s"/>
                    </spine>
                </sequence>
            </media>""")
    captions = ""
    if shared_caption_style:
        captions = textwrap.dedent("""\
            <caption name="c1" lane="1" offset="0s" start="0s" duration="48048/24000s">
                <text><text-style ref="ts1">Hello</text-style></text>
                <text-style-def id="ts1"><text-style font="Helvetica"/></text-style-def>
            </caption>
            <caption name="c2" lane="1" offset="192192/24000s" start="0s" duration="48048/24000s">
                <text><text-style ref="ts1">World</text-style></text>
            </caption>""")
    trailing = ""
    if trailing_children:
        trailing = ('<audio-channel-source srcCh="1, 2" role="dialogue"/>'
                    '<metadata><md key="k" value="v"/></metadata>')
    seq_format = '' if missing_format else ' format="r1"'
    fcpxml = tmp_path / "rt.fcpxml"
    fcpxml.write_text((f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE fcpxml>
        <fcpxml version="1.11">
            <resources>
                {fmt}
                <asset id="r2" name="main" start="0s" duration="600s" hasVideo="1" hasAudio="1">
                    <media-rep kind="original-media" src="file:///tmp/main.mov"/>
                </asset>
                <asset id="r3" name="other" start="0s" duration="600s" hasAudio="1">
                    <media-rep kind="original-media" src="file:///tmp/other.mov"/>
                </asset>
                {compound}
            </resources>
            <library>
                <event name="E">
                    <project name="P">
                        <sequence{seq_format} duration="480480/24000s" tcStart="0s" tcFormat="NDF">
                            <spine>
                                <asset-clip name="Main" ref="r2" offset="0s" start="0s" duration="480480/24000s"{seq_format}>
                                    {captions}
                                    {trailing}
                                </asset-clip>
                            </spine>
                        </sequence>
                    </project>
                </event>
            </library>
        </fcpxml>
    """).replace("\n        ", "\n").lstrip())
    return str(fcpxml)


class TestWriterRegressions:
    def test_compound_media_in_resources_copies_main_spine_clip(self, tmp_path):
        parsed = parse_fcpxml(_roundtrip_fixture(tmp_path, compound_in_resources=True))
        out = write_selects_as_new_project(
            parsed, [Select(start_seconds=1.0, end_seconds=2.0, label="S")])
        root = etree.fromstring(out)
        clips = root.findall(".//project/sequence/spine/asset-clip")
        assert len(clips) == 1
        assert clips[0].get("ref") == "r2"  # NOT the compound's r3

    def test_shared_text_style_def_rescued_on_prune(self, tmp_path):
        parsed = parse_fcpxml(_roundtrip_fixture(tmp_path, shared_caption_style=True))
        # Select overlaps ONLY caption c2 (8.008-10.01s) — c1 (with the def)
        # gets pruned; the def must be rescued into the kept caption.
        out = write_selects_as_new_project(
            parsed, [Select(start_seconds=8.2, end_seconds=9.0, label="S")])
        root = etree.fromstring(out)
        clip = root.find(".//project/sequence/spine/asset-clip")
        refs = {ts.get("ref") for ts in clip.iter("text-style") if ts.get("ref")}
        defs = {d.get("id") for d in clip.iter("text-style-def")}
        assert refs and refs <= defs, (refs, defs)

    def test_mode_b_marker_inserted_before_trailing_children(self, tmp_path):
        parsed = parse_fcpxml(_roundtrip_fixture(tmp_path, trailing_children=True))
        out = write_markers_on_timeline(
            parsed, [Select(start_seconds=3.0, end_seconds=4.0, label="M")])
        root = etree.fromstring(out)
        clip = root.find(".//project/sequence/spine/asset-clip")
        tags = [c.tag for c in clip]
        assert "marker" in tags
        assert tags.index("marker") < tags.index("audio-channel-source")
        assert tags.index("marker") < tags.index("metadata")

    def test_missing_format_synthesized(self, tmp_path):
        parsed = parse_fcpxml(_roundtrip_fixture(tmp_path, missing_format=True))
        out = write_selects_as_new_project(
            parsed, [Select(start_seconds=1.0, end_seconds=2.0, label="S")])
        root = etree.fromstring(out)
        seq = root.find(".//project/sequence")
        fmt_id = seq.get("format")
        assert fmt_id
        assert root.find(f".//resources/format[@id='{fmt_id}']") is not None
