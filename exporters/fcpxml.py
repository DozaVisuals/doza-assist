"""
FCPXML exporter — thin wrapper around the existing fcpxml_export module.

This file deliberately does NOT reimplement FCPXML generation. It calls the
existing generate_fcpxml / generate_story_fcpxml functions and adapts their
output into the BaseExporter contract. Existing FCPXML output is byte-for-byte
unchanged from before the refactor — this is the regression firewall.
"""

import os

from fcpxml_export import generate_fcpxml, generate_story_fcpxml
from .base import BaseExporter, ExportResult


def _markers_filename(project_name: str, export_type: str, marker_count: int, total_clips: int, ext: str) -> str:
    """Replicates the filename logic that previously lived in app.py:export_fcpxml."""
    name = project_name
    if export_type == "labels":
        if marker_count == 1:
            # Single-clip export uses the clip title
            return  # filled in by caller (needs marker text); see export_markers
        else:
            total = total_clips or marker_count
            suffix = "All Clips" if marker_count >= total else f"{marker_count} Clips"
    elif export_type == "social":
        suffix = "Social Clips"
    elif export_type == "story":
        suffix = "Story Beats"
    elif export_type == "soundbites":
        suffix = "Soundbites"
    elif export_type == "all":
        suffix = "Full Export"
    else:
        suffix = export_type

    return (
        f"{name.strip().rstrip('_')} - {suffix.strip().rstrip('_')}{ext}"
        .replace("/", "-")
        .replace("_", " ")
    )


def _compose_marker_filename(project_name: str, markers: list, export_type: str, total_clips: int, ext: str) -> str:
    """Wraps _markers_filename with the single-clip special case."""
    if export_type == "labels" and len(markers) == 1:
        clip_title = (markers[0].get("text") or "Clip")[:40].strip()
        return (
            f"{project_name.strip().rstrip('_')} - {clip_title.rstrip('_')}{ext}"
            .replace("/", "-")
            .replace("_", " ")
        )
    return _markers_filename(project_name, export_type, len(markers), total_clips, ext)


class FCPXMLExporter(BaseExporter):
    format_name = "FCPXML"
    file_extension = ".fcpxml"
    platform_name = "Final Cut Pro"

    def export_markers(
        self,
        markers,
        *,
        project_name,
        source_path,
        media_duration,
        framerate,
        width,
        height,
        export_type,
        exports_dir,
        export_mode="cuts",
        total_clips=0,
    ) -> ExportResult:
        content = generate_fcpxml(
            markers=markers,
            project_name=project_name,
            framerate=framerate,
            source_path=source_path,
            media_duration=media_duration,
            mode=export_mode,
            width=width,
            height=height,
        )

        filename = _compose_marker_filename(
            project_name, markers, export_type, total_clips, self.file_extension
        )
        file_path = os.path.join(exports_dir, filename)
        os.makedirs(exports_dir, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)

        return ExportResult(
            file_path=file_path,
            filename=filename,
            format_name=self.format_name,
            platform_name=self.platform_name,
            warnings=[],
        )

    def export_story(
        self,
        markers,
        *,
        project_name,
        story_title,
        source_path,
        media_duration,
        framerate,
        width,
        height,
        exports_dir,
    ) -> ExportResult:
        content = generate_story_fcpxml(
            markers=markers,
            project_name=project_name,
            story_title=story_title,
            framerate=framerate,
            source_path=source_path,
            media_duration=media_duration,
            width=width,
            height=height,
        )

        filename = (
            f"{project_name.strip()} - {story_title.strip()}{self.file_extension}"
            .replace("/", "-")
        )
        file_path = os.path.join(exports_dir, filename)
        os.makedirs(exports_dir, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)

        return ExportResult(
            file_path=file_path,
            filename=filename,
            format_name=self.format_name,
            platform_name=self.platform_name,
            warnings=[],
        )
