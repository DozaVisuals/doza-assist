"""
FCPXML exporter — thin wrapper around the existing fcpxml_export module.

This file deliberately does NOT reimplement FCPXML generation. It calls the
existing generate_fcpxml / generate_story_fcpxml functions and adapts their
output into the BaseExporter contract. Existing FCPXML output is byte-for-byte
unchanged from before the refactor — this is the regression firewall.

The wrapper owns export POLICY around that generation: it rejects exports
that would write a useless timeline (no markers; pre-cut mode with the
source media missing on disk), collects the generators' skipped-clip /
degraded-mode warnings into ExportResult.warnings, and writes the file
atomically so a concurrently-importing NLE never reads a truncated export.
"""

import os
import tempfile

from fcpxml_export import generate_fcpxml, generate_story_fcpxml
from .base import BaseExporter, ExportResult


def _write_atomic(file_path: str, content: str) -> None:
    """Write via a same-directory temp file + os.replace().

    Export filenames are deterministic and Flask runs threaded: a second
    send while FCP/Resolve is still parsing the previous file (scripted
    imports block up to ~180s) used to re-open the SAME path with a
    truncating open(), corrupting the read mid-import. os.replace() is
    atomic, so a concurrent reader keeps the complete old file and the
    name only ever points at a complete new one.
    """
    exports_dir = os.path.dirname(file_path)
    fd, tmp_path = tempfile.mkstemp(
        dir=exports_dir, prefix=".doza-export-",
        suffix=os.path.splitext(file_path)[1])
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        # mkstemp creates 0600 and os.replace carries that mode onto the
        # final file — every export would become owner-read-only (and a
        # re-export would DOWNGRADE an existing 0644 file), breaking any
        # other-UID consumer of the exports dir (second macOS account, SMB
        # share, backup daemon). Restore what plain open() gave in 1.0.26:
        # 0666 & ~umask (umask read via the standard set/restore round-trip).
        umask = os.umask(0o022)
        os.umask(umask)
        os.chmod(tmp_path, 0o666 & ~umask)
        os.replace(tmp_path, file_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _guard_exportable(markers, source_path, export_mode="cuts"):
    """Fail loudly on the two states that used to 'succeed' uselessly.

    Zero markers wrote an empty-spine timeline that auto-imported as a
    blank sequence (mirrors the multicam route's 'No selects available'
    guard); a set-but-missing source path silently degraded a pre-cut send
    into a markers-only gap timeline (unplugged drive / moved file) with a
    success toast. An EMPTY source path keeps the legacy markers-only
    fallback — that state means the project never had media to cut.
    """
    if not markers:
        raise ValueError(
            "No clips to export for the selected types - add clip labels "
            "or run analysis first.")
    if export_mode in ("cuts", "both") and source_path \
            and not os.path.exists(source_path):
        raise ValueError(
            f'Source media not found at "{source_path}" - reconnect the '
            'drive or relocate the file, or export markers only.')


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
        start_tc_frames=0,
        tc_format="NDF",
    ) -> ExportResult:
        _guard_exportable(markers, source_path, export_mode)

        warnings: list = []
        content = generate_fcpxml(
            markers=markers,
            project_name=project_name,
            framerate=framerate,
            source_path=source_path,
            media_duration=media_duration,
            mode=export_mode,
            width=width,
            height=height,
            start_tc_frames=start_tc_frames,
            tc_format=tc_format,
            warnings_out=warnings,
        )

        filename = _compose_marker_filename(
            project_name, markers, export_type, total_clips, self.file_extension
        )
        file_path = os.path.join(exports_dir, filename)
        os.makedirs(exports_dir, exist_ok=True)
        _write_atomic(file_path, content)

        return ExportResult(
            file_path=file_path,
            filename=filename,
            format_name=self.format_name,
            platform_name=self.platform_name,
            warnings=warnings,
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
        start_tc_frames=0,
        tc_format="NDF",
    ) -> ExportResult:
        # A story send is always a pre-cut timeline — same guards as markers.
        _guard_exportable(markers, source_path)

        warnings: list = []
        content = generate_story_fcpxml(
            markers=markers,
            project_name=project_name,
            story_title=story_title,
            framerate=framerate,
            source_path=source_path,
            media_duration=media_duration,
            width=width,
            height=height,
            start_tc_frames=start_tc_frames,
            tc_format=tc_format,
            warnings_out=warnings,
        )

        filename = (
            f"{project_name.strip()} - {story_title.strip()}{self.file_extension}"
            .replace("/", "-")
        )
        file_path = os.path.join(exports_dir, filename)
        os.makedirs(exports_dir, exist_ok=True)
        _write_atomic(file_path, content)

        return ExportResult(
            file_path=file_path,
            filename=filename,
            format_name=self.format_name,
            platform_name=self.platform_name,
            warnings=warnings,
        )
