"""End-to-end ingest tests for FCPXML input.

These drive the Flask app's ``/create`` endpoint through the test client,
verifying that dropping an FCPXML bundle (or loose .fcpxml file) creates a
project whose ``source_path`` points at the referenced audio, and that the
parsed FCPXML metadata is stashed under ``fcpxml_source``.

The Ella multicam test is skipped automatically if the edit drive referenced
in the bookmark is not mounted, so this suite still runs on any machine.
"""

import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module  # noqa: E402


ELLA_BUNDLE = Path("/Users/dozavisuals/Downloads/Ella Interview.fcpxmld")
ELLA_AUDIO = Path(
    "/Volumes/DOZA EDIT SSD/Trustees/Posey/Studio Visit/"
    "121525_133506/021926_075706_Tr2-esv2-83p-bg-10p.wav"
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Redirect the app's project storage into the tmp_path so tests don't
    # pollute the real projects/ directory.
    monkeypatch.setitem(app_module.app.config, "PROJECTS_DIR", str(tmp_path / "projects"))
    os.makedirs(app_module.app.config["PROJECTS_DIR"], exist_ok=True)
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as c:
        yield c


@pytest.mark.skipif(
    not (ELLA_BUNDLE.exists() and ELLA_AUDIO.exists()),
    reason="Ella fixture or edit drive not present",
)
class TestEllaBundleIngest:
    def test_create_project_from_fcpxmld_bundle(self, client):
        resp = client.post(
            "/create",
            json={"source_path": str(ELLA_BUNDLE), "project_name": "Ella Test"},
        )
        assert resp.status_code == 200, resp.data
        project_id = resp.get_json()["project_id"]

        projects_dir = app_module.app.config["PROJECTS_DIR"]
        meta_path = Path(projects_dir) / project_id / "meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())

        # The source path for transcription must be the underlying audio file,
        # not the FCPXML — transcription runs against this.
        assert meta["source_path"] == str(ELLA_AUDIO)
        assert meta["filepath"] == str(ELLA_AUDIO)

        # fcpxml_source metadata must be attached so the writer (pass B) can
        # round-trip selects back into the original timeline.
        fcpxml_source = meta["fcpxml_source"]
        assert fcpxml_source["container_type"] == "mc-clip"
        assert fcpxml_source["active_audio_angle_id"] == "qMugMvsqRpW4mCI2v5CgDA"
        assert fcpxml_source["audio_asset_id"] == "r4"
        assert fcpxml_source["version"] == "1.14"
        assert len(fcpxml_source["spine_segments"]) == 2

        # The Info.fcpxml should be copied into the project dir so later
        # exports don't depend on the bundle still being on disk.
        stored = Path(fcpxml_source["stored_fcpxml_path"])
        assert stored.exists()
        assert stored.parent == Path(projects_dir) / project_id


class TestMissingAudioError:
    """If the FCPXML references a path that isn't on disk (unmounted drive),
    the ingest must fail with a friendly, editor-facing error."""

    FIXTURE = textwrap.dedent("""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE fcpxml>
        <fcpxml version="1.14">
            <resources>
                <format id="r1" name="FFVideoFormat1080p2398" frameDuration="1001/24000s" width="1920" height="1080"/>
                <asset id="r2" name="missing" start="0s" duration="10s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                    <media-rep kind="original-media" src="file:///Volumes/NotMountedDrive/audio.wav"/>
                </asset>
            </resources>
            <library>
                <event name="E">
                    <project name="P">
                        <sequence format="r1" duration="10s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                            <spine>
                                <sync-clip offset="0s" duration="10s" name="S">
                                    <asset-clip ref="r2" offset="0s" duration="10s" audioRole="dialogue"/>
                                </sync-clip>
                            </spine>
                        </sequence>
                    </project>
                </event>
            </library>
        </fcpxml>
    """)

    def test_returns_400_with_drive_hint(self, client, tmp_path):
        fcpxml = tmp_path / "missing.fcpxml"
        fcpxml.write_text(self.FIXTURE)

        resp = client.post("/create", json={"source_path": str(fcpxml)})
        assert resp.status_code == 400
        body = resp.get_json()
        # Error message should name the missing path and hint at the drive.
        assert "NotMountedDrive" in body["error"]
        assert "audio" in body["error"].lower()


class TestSyncClipIngestWithRealAudio:
    """Confirms the happy path for sync-clip: parsing succeeds and the project
    points at a real audio file that transcription can pick up."""

    def test_create_project_from_sync_clip(self, client, tmp_path):
        # Ingest now dry-runs ffprobe on referenced media (decodability gate),
        # so the fixture must be a REAL minimal wav, not a 12-byte RIFF stub.
        import wave as _wave
        audio = tmp_path / "dialogue.wav"
        with _wave.open(str(audio), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 1600)  # 0.1s of silence

        fcpxml = tmp_path / "sync.fcpxml"
        fcpxml.write_text(textwrap.dedent(f"""\
            <?xml version="1.0" encoding="UTF-8"?>
            <!DOCTYPE fcpxml>
            <fcpxml version="1.14">
                <resources>
                    <format id="r1" name="FFVideoFormat1080p2398" frameDuration="1001/24000s" width="1920" height="1080"/>
                    <asset id="r2" name="dialogue" start="0s" duration="240000/24000s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                        <media-rep kind="original-media" src="file://{audio}"/>
                    </asset>
                </resources>
                <library>
                    <event name="E">
                        <project name="Sync Project">
                            <sequence format="r1" duration="240000/24000s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                                <spine>
                                    <sync-clip offset="0s" duration="240000/24000s" name="S">
                                        <asset-clip ref="r2" offset="0s" duration="240000/24000s" audioRole="dialogue"/>
                                    </sync-clip>
                                </spine>
                            </sequence>
                        </project>
                    </event>
                </library>
            </fcpxml>
        """))

        resp = client.post("/create", json={"source_path": str(fcpxml)})
        assert resp.status_code == 200, resp.data
        pid = resp.get_json()["project_id"]

        meta = json.loads(
            (Path(app_module.app.config["PROJECTS_DIR"]) / pid / "meta.json").read_text()
        )
        assert meta["source_path"] == str(audio)
        assert meta["fcpxml_source"]["container_type"] == "sync-clip"
        # Since no project_name was passed, the ingest should fall back to the
        # FCPXML's project name ("Sync Project"), not the filename.
        assert meta["name"] == "Sync Project"


def _make_wav(path, seconds=0.1):
    """A real minimal WAV — the ingest ffprobe gate and the timeline render
    both need decodable audio, not a RIFF stub."""
    import wave as _wave
    with _wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * int(16000 * seconds))


_HAS_FFMPEG = __import__("shutil").which("ffmpeg") is not None


# Interview (real audio on disk) + B-roll referencing an unmounted drive.
# ``{broll_asset_attrs}`` selects the muted flavor: hasAudio="0" (explicit) or
# no hasAudio at all (how FCP encodes video-only assets).
_BROLL_OFFLINE_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.14">
        <resources>
            <format id="r1" name="FF30" frameDuration="1/30s" width="1920" height="1080"/>
            <asset id="r2" name="interview" start="0s" duration="100s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                <media-rep kind="original-media" src="file://{interview}"/>
            </asset>
            <asset id="r3" name="broll" start="0s" duration="100s" {broll_asset_attrs}>
                <media-rep kind="original-media" src="file:///Volumes/EDITDRIVE/broll.mp4"/>
            </asset>
        </resources>
        <library location="file:///Users/x/Movies/X.fcpbundle/">
            <event name="E"><project name="Broll Offline">
                <sequence format="r1" duration="20s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                    <spine>
                        <asset-clip ref="r2" offset="0s" name="talker" start="0s" duration="10s" audioRole="dialogue"/>
                        <asset-clip ref="r3" offset="10s" name="broll" start="0s" duration="10s"/>
                    </spine>
                </sequence>
            </project></event>
        </library>
    </fcpxml>
""")


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not installed")
class TestMutedOfflineMediaDoesNotBlockIngest:
    """Muted / video-only segments are never rendered or transcribed, so
    their media being on an unmounted drive (or having no audio stream) must
    not fail project creation. Used to raise the 'Is the drive mounted?'
    error for a file whose audio is never touched."""

    def _create(self, client, tmp_path, broll_asset_attrs):
        interview = tmp_path / "interview.wav"
        _make_wav(interview)
        fcpxml = tmp_path / "broll_offline.fcpxml"
        fcpxml.write_text(_BROLL_OFFLINE_FIXTURE.format(
            interview=interview, broll_asset_attrs=broll_asset_attrs))
        return client.post("/create", json={"source_path": str(fcpxml)})

    def test_explicitly_muted_broll_offline_still_imports(self, client, tmp_path):
        resp = self._create(client, tmp_path, 'hasVideo="1" hasAudio="0" videoSources="1"')
        assert resp.status_code == 200, resp.data

    def test_video_only_broll_without_hasaudio_still_imports(self, client, tmp_path):
        # FCP omits hasAudio entirely for video-only assets.
        resp = self._create(client, tmp_path, 'hasVideo="1" videoSources="1"')
        assert resp.status_code == 200, resp.data

    def test_unmuted_offline_media_still_blocks(self, client, tmp_path):
        # The strict gate is intact for media that IS transcribed.
        resp = self._create(client, tmp_path,
                            'hasVideo="1" hasAudio="1" audioSources="1"')
        assert resp.status_code == 400
        assert "EDITDRIVE" in resp.get_json()["error"]


def _make_video_only_mp4(path, seconds=0.4):
    """A real MP4 with a video stream and ZERO audio streams — the ingest
    gate ffprobes actual streams, so a stub file won't do. The mpeg4
    encoder ships in every ffmpeg build (including LGPL-only ones)."""
    import subprocess
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"color=c=black:s=64x64:r=30:d={seconds}",
         "-c:v", "mpeg4", "-an", str(path)],
        check=True, capture_output=True,
    )


# One asset-clip whose asset DECLARES audio (hasAudio="1") over a file that
# has none — the mis-declared-asset shape the video-only probe pass targets.
_VIDEO_ONLY_SINGLE_CLIP_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.14">
        <resources>
            <format id="r1" name="FF30" frameDuration="1/30s" width="1920" height="1080"/>
            <asset id="r2" name="broll" start="0s" duration="100s" hasVideo="1" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                <media-rep kind="original-media" src="file://{media}"/>
            </asset>
        </resources>
        <library location="file:///Users/x/Movies/X.fcpbundle/">
            <event name="E"><project name="Video Only">
                <sequence format="r1" duration="10s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                    <spine>
                        <asset-clip ref="r2" offset="0s" name="broll" start="0s" duration="10s" audioRole="dialogue"/>
                    </spine>
                </sequence>
            </project></event>
        </library>
    </fcpxml>
""")

# Every clip muted/video-only by declaration (no hasAudio) — the muted
# exclusion leaves nothing to probe at all. Media needn't exist: nothing
# would ever be read from it.
_ALL_MUTED_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.14">
        <resources>
            <format id="r1" name="FF30" frameDuration="1/30s" width="1920" height="1080"/>
            <asset id="r2" name="broll" start="0s" duration="100s" hasVideo="1" videoSources="1">
                <media-rep kind="original-media" src="file:///Volumes/EDITDRIVE/broll.mp4"/>
            </asset>
        </resources>
        <library location="file:///Users/x/Movies/X.fcpbundle/">
            <event name="E"><project name="All Muted">
                <sequence format="r1" duration="10s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                    <spine>
                        <asset-clip ref="r2" offset="0s" name="broll" start="0s" duration="10s"/>
                    </spine>
                </sequence>
            </project></event>
        </library>
    </fcpxml>
""")


class TestAudiolessTimelineRejectedAtImport:
    """R4: a timeline whose EVERY source is video-only or muted can never
    transcribe. The video-only per-file pass rightly lets such files through
    individually (silence spans in a mixed timeline), but a project made of
    NOTHING else used to import and then die mid-transcription — the gate
    must say so at import time, with the project dir cleaned up."""

    MSG = "no audio to transcribe"

    @pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not installed")
    def test_single_video_only_clip_rejected(self, client, tmp_path):
        media = tmp_path / "broll.mp4"
        _make_video_only_mp4(media)
        fcpxml = tmp_path / "video_only.fcpxml"
        fcpxml.write_text(_VIDEO_ONLY_SINGLE_CLIP_FIXTURE.format(media=media))
        resp = client.post("/create", json={"source_path": str(fcpxml)})
        assert resp.status_code == 400, resp.data
        assert self.MSG in resp.get_json()["error"]
        # ValueError during ingest must clean up the half-created project.
        projects_dir = Path(app_module.app.config["PROJECTS_DIR"])
        assert list(projects_dir.iterdir()) == []

    def test_all_muted_timeline_rejected(self, client, tmp_path):
        fcpxml = tmp_path / "all_muted.fcpxml"
        fcpxml.write_text(_ALL_MUTED_FIXTURE)
        resp = client.post("/create", json={"source_path": str(fcpxml)})
        assert resp.status_code == 400, resp.data
        assert self.MSG in resp.get_json()["error"]


# Real dialogue WAV + an unsupported <ref-clip> the parser drops with a
# parse warning. Shared by the meta-storage, /create-response, and
# round-trip-export-payload surfacing tests.
_REF_CLIP_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.14">
        <resources>
            <format id="r1" name="FF30" frameDuration="1/30s" width="1920" height="1080"/>
            <asset id="r2" name="dialogue" start="0s" duration="100s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                <media-rep kind="original-media" src="file://{audio}"/>
            </asset>
        </resources>
        <library location="file:///Users/x/Movies/X.fcpbundle/">
            <event name="E"><project name="With Compound">
                <sequence format="r1" duration="30s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                    <spine>
                        <asset-clip ref="r2" offset="0s" name="talker" start="0s" duration="10s" audioRole="dialogue"/>
                        <ref-clip ref="rComp" offset="10s" name="Montage" duration="20s"/>
                    </spine>
                </sequence>
            </project></event>
        </library>
    </fcpxml>
""")


def _create_refclip_project(client, tmp_path):
    audio = tmp_path / "dialogue.wav"
    _make_wav(audio)
    fcpxml = tmp_path / "with_refclip.fcpxml"
    fcpxml.write_text(_REF_CLIP_FIXTURE.format(audio=audio))
    return client.post("/create", json={"source_path": str(fcpxml)})


class TestParseWarningsSurfacedInMeta:
    """Non-fatal parse warnings (dropped compound clips, retimes, conforms)
    must land in the project's fcpxml_source metadata so the UI can explain
    transcript holes."""

    def test_ref_clip_warning_stored(self, client, tmp_path):
        resp = _create_refclip_project(client, tmp_path)
        assert resp.status_code == 200, resp.data
        pid = resp.get_json()["project_id"]
        meta = json.loads(
            (Path(app_module.app.config["PROJECTS_DIR"]) / pid / "meta.json").read_text()
        )
        warnings = meta["fcpxml_source"]["parse_warnings"]
        assert len(warnings) == 1
        assert "Montage" in warnings[0] and "ref-clip" in warnings[0]

    def test_create_response_carries_warnings(self, client, tmp_path):
        # R5: stored-only warnings never reached the user — the /create
        # response must surface them so the import UI can toast them.
        resp = _create_refclip_project(client, tmp_path)
        assert resp.status_code == 200, resp.data
        warnings = resp.get_json()["warnings"]
        assert len(warnings) == 1
        assert "Montage" in warnings[0] and "ref-clip" in warnings[0]

    def test_create_response_has_no_warnings_key_when_clean(self, client, tmp_path):
        audio = tmp_path / "clean.wav"
        _make_wav(audio)
        fcpxml = tmp_path / "clean.fcpxml"
        fcpxml.write_text(_TWO_RECORDER_FIXTURE.format(rec_a=audio, rec_b=audio))
        resp = client.post("/create", json={"source_path": str(fcpxml)})
        assert resp.status_code == 200, resp.data
        assert "warnings" not in resp.get_json()


class TestRoundTripExportSurfacesParseWarnings:
    """R5: the stored FCPXML's parse warnings (retimed / rate-conformed
    clips, dropped ref-clip dialogue) must ride the round-trip export
    payload's ``warnings`` list — that export is exactly where the
    misalignment they describe would otherwise surface unexplained."""

    def test_multicam_route_payload_carries_parse_warnings(
            self, client, tmp_path, monkeypatch):
        resp = _create_refclip_project(client, tmp_path)
        assert resp.status_code == 200, resp.data
        pid = resp.get_json()["project_id"]

        meta_path = Path(app_module.app.config["PROJECTS_DIR"]) / pid / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["color_labels"] = {"green": "Best"}
        meta["labeled_sections"] = [
            {"start": 1.0, "end": 2.0, "color": "green", "text": "hook"},
        ]
        meta_path.write_text(json.dumps(meta))

        monkeypatch.setitem(app_module.app.config, "EXPORTS_DIR",
                            str(tmp_path / "exports"))
        monkeypatch.setattr(app_module, "_reveal_in_finder", lambda p: None)
        # The writer is not under test (and is churning in a parallel
        # branch) — parse_fcpxml stays REAL so the warnings come from the
        # stored file, not a stub.
        monkeypatch.setattr(
            app_module, "write_selects_as_new_project",
            lambda parsed, selects, preserve_order=False, skipped_out=None, **names:
                b'<fcpxml version="1.14"/>')

        resp = client.post(
            f"/project/{pid}/export/fcpxml-multicam",
            json={"mode": "selects_project", "source": "client_selects",
                  "deliver_to": "file"},
        )
        assert resp.status_code == 200, resp.data
        warnings = resp.get_json()["warnings"]
        assert any("Montage" in w and "ref-clip" in w for w in warnings)


# ---------- My Style (editorial_dna) staging --------------------------------

_TWO_RECORDER_FIXTURE = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE fcpxml>
    <fcpxml version="1.14">
        <resources>
            <format id="r1" name="FF30" frameDuration="1/30s" width="1920" height="1080"/>
            <asset id="r2" name="recA" start="0s" duration="100s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                <media-rep kind="original-media" src="file://{rec_a}"/>
            </asset>
            <asset id="r3" name="recB" start="0s" duration="100s" hasAudio="1" audioSources="1" audioChannels="1" audioRate="48000">
                <media-rep kind="original-media" src="file://{rec_b}"/>
            </asset>
        </resources>
        <library location="file:///Users/x/Movies/X.fcpbundle/">
            <event name="E"><project name="Two Recorders">
                <sequence format="r1" duration="20s" tcStart="0s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                    <spine>
                        <asset-clip ref="r2" offset="0s" name="A" start="0s" duration="10s" audioRole="dialogue"/>
                        <asset-clip ref="r3" offset="10s" name="B" start="0s" duration="10s" audioRole="dialogue"/>
                    </spine>
                </sequence>
            </project></event>
        </library>
    </fcpxml>
""")


class TestMyStylePartialMissingMedia:
    """stage_fcpxml promises non-blocking partial imports: an offline clip is
    skipped (silence) and reported via missing_media. It used to hard-fail on
    the renderer's all-sources-must-exist check, making the documented
    warning path unreachable."""

    def _stage(self, tmp_path, rec_a, rec_b):
        from editorial_dna.fcpxml_ingest import stage_fcpxml
        fcpxml = tmp_path / "style.fcpxml"
        fcpxml.write_text(_TWO_RECORDER_FIXTURE.format(rec_a=rec_a, rec_b=rec_b))
        return stage_fcpxml(str(fcpxml), str(tmp_path / "staging"))

    @pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not installed")
    def test_one_offline_recorder_is_skipped_and_reported(self, tmp_path):
        rec_a = tmp_path / "rec_a.wav"
        _make_wav(rec_a)
        rec_b = tmp_path / "offline_recorder.wav"  # never created

        result = self._stage(tmp_path, rec_a, rec_b)
        assert result.missing_media == [str(rec_b)]
        assert os.path.exists(result.audio_path)
        assert result.fcpxml_metadata["present_media_count"] == 1

    def test_everything_offline_still_fails(self, tmp_path):
        rec_a = tmp_path / "gone_a.wav"
        rec_b = tmp_path / "gone_b.wav"
        with pytest.raises(ValueError, match="none of the audio files"):
            self._stage(tmp_path, rec_a, rec_b)

    @pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not installed")
    def test_offline_muted_broll_not_reported_missing(self, tmp_path):
        # A muted segment never renders; its offline file is not a problem
        # worth a notice.
        from editorial_dna.fcpxml_ingest import stage_fcpxml
        rec_a = tmp_path / "rec_a.wav"
        _make_wav(rec_a)
        fcpxml = tmp_path / "style_muted.fcpxml"
        fcpxml.write_text(_BROLL_OFFLINE_FIXTURE.format(
            interview=rec_a,
            broll_asset_attrs='hasVideo="1" hasAudio="0" videoSources="1"'))
        result = stage_fcpxml(str(fcpxml), str(tmp_path / "staging"))
        assert result.missing_media == []


@pytest.mark.skipif(
    not (ELLA_BUNDLE.exists() and ELLA_AUDIO.exists()),
    reason="Ella fixture or edit drive not present",
)
class TestFCPXMLMulticamExportRoute:
    """Exercises the new /project/<id>/export/fcpxml-multicam route."""

    def _create_project_with_labels(self, client):
        # Ingest the Ella bundle, then seed some labeled_sections so there's
        # something to export.
        resp = client.post("/create", json={"source_path": str(ELLA_BUNDLE)})
        pid = resp.get_json()["project_id"]
        # Patch meta.json directly with some selects.
        meta_path = Path(app_module.app.config["PROJECTS_DIR"]) / pid / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["color_labels"] = {"green": "Best", "blue": "Supporting"}
        meta["labeled_sections"] = [
            {"start": 10.0, "end": 25.0, "color": "green", "text": "Opening hook"},
            {"start": 100.0, "end": 130.0, "color": "blue", "text": "Supporting beat"},
        ]
        meta_path.write_text(json.dumps(meta))
        return pid

    def test_selects_as_new_project_export(self, client):
        pid = self._create_project_with_labels(client)
        resp = client.post(
            f"/project/{pid}/export/fcpxml-multicam",
            json={"mode": "selects_project", "source": "client_selects"},
        )
        assert resp.status_code == 200, resp.data
        body = resp.data
        # Output is FCPXML with a new "{Project} – Selects 1" timeline (1.1
        # naming); the editor's own event name is kept and "Doza" never
        # appears in a timeline name.
        assert b"<fcpxml version=\"1.14\">" in body
        assert "– Selects 1".encode("utf-8") in body
        assert b"Doza Selects" not in body
        # Re-uses the multicam container (ref="r2") from the original resources.
        assert b'ref="r2"' in body

    def test_markers_on_timeline_export(self, client):
        pid = self._create_project_with_labels(client)
        resp = client.post(
            f"/project/{pid}/export/fcpxml-multicam",
            json={"mode": "markers_timeline", "source": "client_selects"},
        )
        assert resp.status_code == 200, resp.data
        assert "– Markers 1".encode("utf-8") in resp.data
        assert b"Doza Notes" not in resp.data
        assert b"<marker " in resp.data

    def test_rejects_when_project_has_no_fcpxml_source(self, client, tmp_path):
        # Create a plain (non-FCPXML) project and try to hit the multicam export.
        audio = tmp_path / "plain.wav"
        audio.write_bytes(b"RIFF____WAVE")
        resp = client.post("/create", json={"source_path": str(audio)})
        pid = resp.get_json()["project_id"]

        resp = client.post(
            f"/project/{pid}/export/fcpxml-multicam",
            json={"mode": "selects_project"},
        )
        assert resp.status_code == 400
        assert "not imported from an FCPXML" in resp.get_json()["error"]
