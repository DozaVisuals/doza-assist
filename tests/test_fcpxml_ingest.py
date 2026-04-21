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
        # Create a real (silent) audio file on disk so the existence check
        # passes. We don't need decodable content — ingest only checks os.path.exists.
        audio = tmp_path / "dialogue.wav"
        audio.write_bytes(b"RIFF____WAVE")

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
        # Output is FCPXML with a new Doza Selects project.
        assert b"<fcpxml version=\"1.14\">" in body
        assert b"Doza Selects" in body
        # Re-uses the multicam container (ref="r2") from the original resources.
        assert b'ref="r2"' in body

    def test_markers_on_timeline_export(self, client):
        pid = self._create_project_with_labels(client)
        resp = client.post(
            f"/project/{pid}/export/fcpxml-multicam",
            json={"mode": "markers_timeline", "source": "client_selects"},
        )
        assert resp.status_code == 200, resp.data
        assert b"Doza Notes" in resp.data
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
