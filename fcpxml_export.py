"""
FCPXML Export for Doza Assist.
Generates Final Cut Pro X compatible XML with actual cuts on the timeline.

Export modes:
  1. "cuts"    — Pre-cut timeline with each clip as an edit referencing the source media
  2. "markers" — Markers on a gap (legacy, for reference)
  3. "both"    — Cuts on timeline + markers for context

When imported into FCPX, the editor gets:
  - An Event with the source media
  - A Project with each clip placed on the timeline in order
  - Keyword ranges on the source for browser filtering
  - Chapter markers for quick navigation
"""

import os
import math
import uuid
from fractions import Fraction


def seconds_to_fcpxml_time(seconds, framerate=23.976):
    """
    Convert seconds to FCPXML rational time format.
    FCPXML uses rational numbers like '48048/24000s' for frame-accurate timing.
    """
    if framerate == 23.976:
        timebase = 24000
        frame_dur = 1001
    elif framerate == 29.97:
        timebase = 30000
        frame_dur = 1001
    elif framerate == 24.0:
        timebase = 24
        frame_dur = 1
    elif framerate == 25.0:
        timebase = 25
        frame_dur = 1
    elif framerate == 30.0:
        timebase = 30
        frame_dur = 1
    elif framerate == 59.94:
        timebase = 60000
        frame_dur = 1001
    elif framerate == 60.0:
        timebase = 60
        frame_dur = 1
    else:
        timebase = 24000
        frame_dur = 1001

    total_frames = round(seconds * timebase / frame_dur)
    rational_time = total_frames * frame_dur

    return f"{rational_time}/{timebase}s"


def get_frame_duration(framerate=23.976):
    """Get the frame duration string for FCPXML."""
    rates = {
        23.976: "1001/24000s",
        24.0: "100/2400s",
        25.0: "100/2500s",
        29.97: "1001/30000s",
        30.0: "100/3000s",
        59.94: "1001/60000s",
        60.0: "100/6000s",
    }
    return rates.get(framerate, "1001/24000s")


# FCPXML marker colors
MARKER_COLORS = {
    'blue': 'Blue',
    'green': 'Green',
    'purple': 'Purple',
    'red': 'Red',
    'orange': 'Orange',
    'yellow': 'Yellow',
    'cyan': 'Cyan',
    'pink': 'Pink',
}


def generate_fcpxml(markers, project_name="Interview", framerate=23.976,
                    source_path=None, media_duration=None, mode="cuts",
                    width=1920, height=1080):
    """
    Generate an FCPXML file.

    Args:
        markers: list of dicts with start, end, text, note, color, category
        project_name: str
        framerate: float
        source_path: str — path to the source media file (enables cut mode)
        media_duration: float — total duration of the source media in seconds
        mode: "cuts" | "markers" | "both"

    Returns:
        str: Complete FCPXML content
    """
    # If no source path, fall back to markers-only mode
    if not source_path or not os.path.exists(source_path):
        mode = "markers"

    if mode == "markers":
        return _generate_markers_only(markers, project_name, framerate, width, height)

    return _generate_cuts_timeline(markers, project_name, framerate,
                                   source_path, media_duration, mode, width, height)


def _generate_cuts_timeline(markers, project_name, framerate, source_path,
                            media_duration, mode, width=1920, height=1080):
    """Generate FCPXML with actual cuts on the timeline referencing source media."""
    frame_dur = get_frame_duration(framerate)
    safe_name = _escape_xml(project_name)
    uid = f"doza-{uuid.uuid4().hex[:8]}"

    # Sort markers by start time
    markers = sorted(markers, key=lambda m: m['start'])

    # Media duration fallback
    if not media_duration and markers:
        media_duration = max(m['end'] for m in markers) + 10.0
    elif not media_duration:
        media_duration = 60.0

    media_dur_str = seconds_to_fcpxml_time(media_duration, framerate)

    # Calculate total timeline duration (sum of all clips)
    total_timeline = sum(m['end'] - m['start'] for m in markers) if markers else 0
    if total_timeline <= 0:
        total_timeline = media_duration
    timeline_dur_str = seconds_to_fcpxml_time(total_timeline, framerate)

    # File reference — use file:// URL for the source media
    file_url = 'file://' + source_path.replace(' ', '%20')
    ext = os.path.splitext(source_path)[1].lower()

    # Determine if video or audio-only
    is_video = ext in ('.mp4', '.mov', '.mxf', '.avi', '.mkv')

    # Build the spine — each marker becomes an asset-clip on the timeline
    spine_clips = []
    timeline_offset = 0.0

    for i, m in enumerate(markers):
        clip_start = m['start']
        clip_end = m['end'] + 1.0  # Add 1s buffer — word timestamps end slightly early
        clip_dur = clip_end - clip_start
        if clip_dur <= 0:
            continue

        offset_str = seconds_to_fcpxml_time(timeline_offset, framerate)
        src_start_str = seconds_to_fcpxml_time(clip_start, framerate)
        dur_str = seconds_to_fcpxml_time(clip_dur, framerate)

        clip_name = _escape_xml(m.get('text', f'Clip {i+1}'))[:80]
        note = _escape_xml(m.get('note', ''))
        category = _escape_xml(m.get('category', 'Clip'))

        # Optional marker inside the clip
        marker_xml = ''
        if mode == "both":
            marker_xml = (
                f'\n                            <chapter-marker start="{src_start_str}" '
                f'duration="{frame_dur}" value="{clip_name}" '
                f'note="{note} [{category}]"/>'
            )

        # Keyword for the clip
        keyword_xml = (
            f'\n                            <keyword start="{src_start_str}" '
            f'duration="{dur_str}" value="{category}"/>'
        )

        spine_clips.append(
            f'                        <asset-clip name="{clip_name}" ref="r2" '
            f'offset="{offset_str}" duration="{dur_str}" start="{src_start_str}" '
            f'format="r1" tcFormat="NDF">'
            f'{keyword_xml}{marker_xml}'
            f'\n                        </asset-clip>'
        )

        timeline_offset += clip_dur

    spine_block = '\n'.join(spine_clips)

    # Also build keyword ranges on the source asset for browser filtering
    keyword_ranges = []
    for m in markers:
        kw_start = seconds_to_fcpxml_time(m['start'], framerate)
        kw_dur = seconds_to_fcpxml_time(m['end'] - m['start'], framerate)
        category = _escape_xml(m.get('category', m.get('text', 'Clip')))[:40]
        keyword_ranges.append(
            f'                <keyword start="{kw_start}" duration="{kw_dur}" value="{category}"/>'
        )
    keywords_block = '\n'.join(keyword_ranges)

    fcpxml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>

<fcpxml version="1.11">
    <resources>
        <format id="r1" name="{_format_name(width, height, framerate)}" frameDuration="{frame_dur}" width="{width}" height="{height}"/>
        <asset id="r2" name="{_escape_xml(os.path.basename(source_path))}" start="0/1s" duration="{media_dur_str}" hasVideo="{1 if is_video else 0}" hasAudio="1" format="r1">
            <media-rep kind="original-media" src="{file_url}"/>
        </asset>
    </resources>
    <library>
        <event name="{safe_name}">
            <asset-clip name="{_escape_xml(os.path.basename(source_path))}" ref="r2" duration="{media_dur_str}" format="r1" tcFormat="NDF">
{keywords_block}
            </asset-clip>
            <project name="{safe_name} - Selects" uid="{uid}">
                <sequence format="r1" duration="{timeline_dur_str}" tcStart="0/1s" tcFormat="NDF">
                    <spine>
{spine_block}
                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>"""

    return fcpxml


def generate_story_fcpxml(markers, project_name="Interview", story_title="Story",
                          framerate=23.976, source_path=None, media_duration=None,
                          width=1920, height=1080):
    """
    Generate FCPXML for a Story Builder sequence.
    Creates a single timeline with clips in narrative order as actual edits.
    """
    if not source_path or not os.path.exists(source_path):
        return _generate_markers_only(markers, f"{project_name} - {story_title}", framerate, width, height)

    frame_dur = get_frame_duration(framerate)
    safe_name = _escape_xml(project_name)
    safe_title = _escape_xml(story_title)
    uid = f"doza-story-{uuid.uuid4().hex[:8]}"

    markers = sorted(markers, key=lambda m: m.get('_order', markers.index(m)))

    if not media_duration and markers:
        media_duration = max(m['end'] for m in markers) + 10.0
    elif not media_duration:
        media_duration = 60.0

    media_dur_str = seconds_to_fcpxml_time(media_duration, framerate)

    total_timeline = sum(m['end'] - m['start'] for m in markers if m['end'] > m['start'])
    if total_timeline <= 0:
        total_timeline = media_duration
    timeline_dur_str = seconds_to_fcpxml_time(total_timeline, framerate)

    file_url = 'file://' + source_path.replace(' ', '%20')
    ext = os.path.splitext(source_path)[1].lower()
    is_video = ext in ('.mp4', '.mov', '.mxf', '.avi', '.mkv')

    spine_clips = []
    timeline_offset = 0.0

    for i, m in enumerate(markers):
        clip_start = m['start']
        clip_end = m['end'] + 1.0
        clip_dur = clip_end - clip_start
        if clip_dur <= 0:
            continue

        offset_str = seconds_to_fcpxml_time(timeline_offset, framerate)
        src_start_str = seconds_to_fcpxml_time(clip_start, framerate)
        dur_str = seconds_to_fcpxml_time(clip_dur, framerate)

        clip_name = _escape_xml(m.get('text', f'Clip {i+1}'))[:80]
        note = _escape_xml(m.get('note', ''))

        marker_xml = (
            f'\n                            <chapter-marker start="{src_start_str}" '
            f'duration="{frame_dur}" value="{clip_name}" '
            f'note="{note}"/>'
        )

        spine_clips.append(
            f'                        <asset-clip name="{clip_name}" ref="r2" '
            f'offset="{offset_str}" duration="{dur_str}" start="{src_start_str}" '
            f'format="r1" tcFormat="NDF">{marker_xml}'
            f'\n                        </asset-clip>'
        )

        timeline_offset += clip_dur

    spine_block = '\n'.join(spine_clips)

    fcpxml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>

<fcpxml version="1.11">
    <resources>
        <format id="r1" name="{_format_name(width, height, framerate)}" frameDuration="{frame_dur}" width="{width}" height="{height}"/>
        <asset id="r2" name="{_escape_xml(os.path.basename(source_path))}" start="0/1s" duration="{media_dur_str}" hasVideo="{1 if is_video else 0}" hasAudio="1" format="r1">
            <media-rep kind="original-media" src="{file_url}"/>
        </asset>
    </resources>
    <library>
        <event name="{safe_name}">
            <project name="{safe_title}" uid="{uid}">
                <sequence format="r1" duration="{timeline_dur_str}" tcStart="0/1s" tcFormat="NDF">
                    <spine>
{spine_block}
                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>"""

    return fcpxml


def _generate_markers_only(markers, project_name, framerate, width=1920, height=1080):
    """Legacy marker-only export (no source media reference)."""
    frame_dur = get_frame_duration(framerate)

    if markers:
        total_duration = max(m['end'] for m in markers) + 10.0
    else:
        total_duration = 60.0

    total_dur_str = seconds_to_fcpxml_time(total_duration, framerate)

    markers_xml = []
    for i, m in enumerate(markers):
        start_time = seconds_to_fcpxml_time(m['start'], framerate)
        duration = m['end'] - m['start']
        dur_str = seconds_to_fcpxml_time(max(duration, 1.0 / framerate), framerate)

        color = MARKER_COLORS.get(m.get('color', 'blue'), 'Blue')
        name = _escape_xml(m.get('text', f'Marker {i+1}'))
        note = _escape_xml(m.get('note', ''))
        category = m.get('category', 'Marker')

        display_name = name[:80] + '...' if len(name) > 80 else name

        markers_xml.append(
            f'                        <chapter-marker start="{start_time}" '
            f'duration="{dur_str}" value="{display_name}" '
            f'note="{note} [{category}]"/>'
        )

    markers_block = '\n'.join(markers_xml)

    fcpxml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>

<fcpxml version="1.11">
    <resources>
        <format id="r1" name="{_format_name(width, height, framerate)}" frameDuration="{frame_dur}" width="{width}" height="{height}"/>
    </resources>
    <library>
        <event name="{_escape_xml(project_name)} Markers">
            <project name="{_escape_xml(project_name)}" uid="doza-{project_name.replace(' ', '-').lower()}">
                <sequence format="r1" duration="{total_dur_str}" tcStart="0/1s" tcFormat="NDF">
                    <spine>
                        <gap name="Gap" offset="0/1s" duration="{total_dur_str}" start="0/1s">
{markers_block}
                        </gap>
                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>"""

    return fcpxml


def _format_name(width, height, framerate):
    """Get the exact FCPX format name string.

    FCPX uses specific format names like:
      FFVideoFormat1080p2398
      FFVideoFormat3840x2160p2398
      FFVideoFormat720p25

    For resolutions > 1080p, width is included.
    """
    rate = _framerate_label(framerate)
    if height <= 1080:
        return f"FFVideoFormat{height}p{rate}"
    else:
        return f"FFVideoFormat{width}x{height}p{rate}"


def _framerate_label(framerate):
    """Get FCPX format label for framerate."""
    labels = {
        23.976: "2398",
        24.0: "24",
        25.0: "25",
        29.97: "2997",
        30.0: "30",
        59.94: "5994",
        60.0: "60",
    }
    return labels.get(framerate, "2398")


def _escape_xml(text):
    """Escape special XML characters."""
    if not text:
        return ""
    return (str(text)
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&apos;'))
