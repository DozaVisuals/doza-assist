"""
Base exporter interface and shared types.

Every NLE exporter implements `export_markers` (clip/marker exports) and
`export_story` (assembled story builder sequences). Both consume a list of
marker dicts with this shape:

    {
        "start": float seconds,
        "end":   float seconds,
        "text":  str (clip title),
        "note":  str (editorial note / description),
        "color": str (optional, used by FCPXML),
        "category": str (optional, used by FCPXML keywords),
    }

Exporters are responsible for composing their own filename (using
self.file_extension) and writing the file under exports_dir. They return
an ExportResult that the route hands back to the user.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ExportResult:
    file_path: str            # absolute path written to disk
    filename: str             # basename used as the download name
    format_name: str          # "FCPXML" | "Premiere XML" | "EDL"
    platform_name: str        # "Final Cut Pro" | "Premiere Pro" | "DaVinci Resolve"
    warnings: list = field(default_factory=list)


class BaseExporter(ABC):
    # Subclasses set these as class attrs.
    format_name: str = ""
    file_extension: str = ""   # includes the dot, e.g. ".fcpxml"
    platform_name: str = ""

    @abstractmethod
    def export_markers(
        self,
        markers: list,
        *,
        project_name: str,
        source_path: str,
        media_duration,           # float | None
        framerate: float,
        width: int,
        height: int,
        export_type: str,         # "labels" | "social" | "story" | "soundbites" | "all"
        exports_dir: str,
        export_mode: str = "cuts",
        total_clips: int = 0,
    ) -> ExportResult:
        """Export a flat list of marker/clip dicts."""

    @abstractmethod
    def export_story(
        self,
        markers: list,
        *,
        project_name: str,
        story_title: str,
        source_path: str,
        media_duration,           # float | None
        framerate: float,
        width: int,
        height: int,
        exports_dir: str,
    ) -> ExportResult:
        """Export an assembled story builder sequence as a timeline."""
