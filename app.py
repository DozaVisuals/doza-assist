"""
Doza Assist
A local interview transcription, client review, and FCPX editing tool.
Built for Doza Visuals.
"""

import os
import json
import uuid
import time
import shutil
import subprocess
import threading
import hashlib
import re as _re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, Response, stream_with_context
from werkzeug.utils import secure_filename

from exporters import get_exporter, PLATFORMS, DEFAULT_PLATFORM
from exporters.media_probe import (
    get_video_resolution, get_video_framerate, get_video_start_timecode_frames,
)
from doza_assist.fcpxml import (
    parse_fcpxml, ParseError, Select, WriterError,
    write_selects_as_new_project, write_markers_on_timeline,
)
from doza_assist.fcpxml.timeline_audio import (
    render_timeline_audio, TimelineAudioError,
)
import preferences as prefs


def get_project_platform(project: dict) -> str:
    """Return the project's editing platform, falling back to the global default."""
    p = (project or {}).get('editing_platform')
    if p in PLATFORMS:
        return p
    return prefs.get_default_platform()


def _exporter_response(result, project, exporter):
    """Send an export file with X-Export-* headers for the frontend toast."""
    response = send_file(
        result.file_path,
        as_attachment=True,
        download_name=result.filename,
    )
    response.headers['X-Export-Format'] = result.format_name
    response.headers['X-Export-Platform'] = result.platform_name
    response.headers['X-Export-Extension'] = exporter.file_extension
    if result.warnings:
        # Headers must be ASCII-safe; join with " | ".
        response.headers['X-Export-Warnings'] = ' | '.join(result.warnings)
    return response


app = Flask(__name__)
# Data dir: honor DOZA_DATA_DIR if set (the packaged .app launcher points this
# at ~/Library/Application Support/DozaAssist). Fall back to the source tree
# for `python3 app.py` dev runs. Must never default to a path inside a signed
# .app bundle — those are read-only and os.makedirs below would EPERM.
_data_dir = os.environ.get('DOZA_DATA_DIR') or os.path.dirname(__file__)
app.config['PROJECTS_DIR'] = os.path.join(_data_dir, 'projects')
app.config['EXPORTS_DIR'] = os.path.join(_data_dir, 'exports')

# Small file drag-and-drop limit (500MB)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 * 1024  # 32 GB — My Style imports multiple large masters

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'mov', 'aac', 'm4a', 'flac', 'aif', 'aiff', 'mxf', 'fcpxml'}

os.makedirs(app.config['PROJECTS_DIR'], exist_ok=True)
os.makedirs(app.config['EXPORTS_DIR'], exist_ok=True)


# ── project_id safety (SEC-02) ────────────────────────────────────────────
# Every /project/<project_id>/... route turns an editor-supplied project_id
# into a path under PROJECTS_DIR. Real ids are uuid4 hex slices
# (str(uuid.uuid4())[:8]); folder/collection slugs use a separate <slug> param.
# We reject anything that isn't a plain id token so a crafted value (``../``,
# an absolute path, or a symlinked project dir pointing outside) can never
# escape PROJECTS_DIR — most importantly before it reaches the shutil.rmtree
# in delete_project. One shared helper, enforced once for every <project_id>
# route by the before_request guard below, with safe_project_dir() used at the
# filesystem chokepoints (get_project/save_project/delete) for defense in depth
# and to cover the few non-URL project_id sources (e.g. /export/send-to-nle).
_PROJECT_ID_RE = _re.compile(r'^[A-Za-z0-9_-]{1,64}$')


def validate_project_id(project_id):
    """True if ``project_id`` is a safe single path segment — no separators,
    no ``..``, not absolute, reasonable length."""
    return isinstance(project_id, str) and bool(_PROJECT_ID_RE.match(project_id))


def safe_project_dir(project_id):
    """Resolve ``project_id`` to its absolute project directory, or return
    ``None`` if the id is malformed or would resolve outside PROJECTS_DIR.

    The realpath + direct-child check also rejects a project_id whose dir is a
    symlink pointing outside PROJECTS_DIR. Single source of truth for turning a
    project_id into a filesystem path.
    """
    if not validate_project_id(project_id):
        return None
    base = os.path.realpath(app.config['PROJECTS_DIR'])
    candidate = os.path.realpath(os.path.join(base, project_id))
    # Must be a *direct* child of PROJECTS_DIR — no nesting, no escape.
    if os.path.dirname(candidate) != base:
        return None
    return candidate


@app.before_request
def _guard_project_id():
    """Reject any request whose matched route carries a malformed ``project_id``
    before the view (and its filesystem access) runs. One enforcement point for
    every /project/<project_id>/... route, including delete_project whose
    rmtree was the path-traversal sink (SEC-02).

    The URL <project_id> may be a single id OR a comma-separated list — several
    routes (multi-project chat, the multi-project workspace view) accept
    ``a,b,c`` and split it themselves. We mirror that split/strip and require
    *every* component to be a valid id, so ``a,b`` is allowed but ``a,../etc``
    is not. The single-id sinks (get_project/save_project/delete) re-validate
    each id strictly, so a comma-joined value can never reach a filesystem path
    intact.
    """
    raw = (request.view_args or {}).get('project_id')
    if raw is not None:
        parts = [p.strip() for p in raw.split(',') if p.strip()]
        if not parts or not all(validate_project_id(p) for p in parts):
            # Return a response (not abort()) so we short-circuit cleanly — the
            # app's global @errorhandler(Exception) would turn abort() into a 500.
            return jsonify({'error': 'Not found'}), 404


# ── Cross-origin write protection (SEC-01) ────────────────────────────────
# The API binds to 127.0.0.1 only, but loopback binding does NOT stop a web
# page the editor visits from issuing cross-origin *state-changing* requests to
# http://127.0.0.1:<port>/... (CSRF), nor a DNS-rebinding attack. There is no
# CORS policy and no auth, so state-changing requests must look same-origin to
# the loopback app:
#   - if an Origin header is present it must be loopback, AND
#   - the Host must be loopback (this second check defeats DNS rebinding, where
#     the page's Host is the attacker's domain that resolves to 127.0.0.1).
# Same-origin editor writes and the app's own fetches send a loopback Origin;
# non-browser callers (no Origin) are allowed only when the Host is loopback.
# Safe methods (GET/HEAD/OPTIONS) are never blocked, so page loads — including
# the shared client portal at GET /share/<id> — are unaffected.
#
# Client portal exception: the review/share portal is reached cross-origin
# through a Cloudflare tunnel, so its Origin is the tunnel URL (not loopback).
# A reviewing client may, cross-origin:
#   - add a comment and send chat messages (_PORTAL_WRITE_ENDPOINTS) — these
#     don't remove anything (clear_chat is a separate DELETE that stays blocked);
#   - ADD clips/selects but NOT remove, shrink, edit, or reorder existing ones
#     (_PORTAL_APPEND_ONLY) — save_selects and save_labels replace the whole
#     list, so for cross-origin callers we allow the write only when the
#     submitted list is a pure addition to what's stored (every stored item
#     still present, unchanged, in the same order, and the list is longer).
# Everything else stays blocked cross-origin: clip/select removal-via-shrink,
# story_update (PUT list-replace), clear_chat, delete/clear/retranscribe/rename/
# move/create/upload, provider + share settings, find-file, browse, export, etc.
# Same-origin editor requests are never subject to the append-only rule — the
# editor keeps full add/remove/edit/reorder.
_STATE_CHANGING_METHODS = {'POST', 'PUT', 'PATCH', 'DELETE'}
_LOOPBACK_HOSTS = {'127.0.0.1', 'localhost', '::1'}
# Endpoints the cross-origin client portal may call freely (no list-shrink risk).
_PORTAL_WRITE_ENDPOINTS = {'add_comment', 'chat', 'chat_stream'}
# Endpoints the portal may call cross-origin ONLY when the submitted list purely
# extends the stored list. Maps view function -> (stored project key, body key).
_PORTAL_APPEND_ONLY = {
    'save_selects': ('client_selects', 'selects'),
    'save_labels':  ('labeled_sections', 'labeled_sections'),
}


def _is_loopback_netloc(value):
    """True if a Host header or Origin URL points at the loopback interface."""
    if not value:
        return False
    netloc = value if '//' in value else '//' + value
    return urlparse(netloc).hostname in _LOOPBACK_HOSTS


def _request_is_cross_origin():
    """True if this request is NOT same-origin to the loopback app — i.e. it
    carries a non-loopback Origin, or its Host isn't loopback (rebinding)."""
    origin = request.headers.get('Origin')
    if origin is not None and not _is_loopback_netloc(origin):
        return True
    return not _is_loopback_netloc(request.host)


def _is_pure_addition(stored, submitted):
    """True iff ``submitted`` keeps every item of ``stored`` unchanged and in the
    same relative order (a subsequence) and is strictly longer — items were only
    added, never removed, edited, or reordered. Anything else returns False."""
    if not isinstance(stored, list) or not isinstance(submitted, list):
        return False
    if len(submitted) <= len(stored):
        return False
    i = 0
    for item in submitted:
        if i < len(stored) and item == stored[i]:
            i += 1
    return i == len(stored)


def _portal_append_only_ok(endpoint):
    """Append-only gate for cross-origin save_selects/save_labels: the submitted
    list must purely extend the stored list. For save_labels the color-label map
    must also be unchanged (no renaming labels cross-origin)."""
    stored_key, body_key = _PORTAL_APPEND_ONLY[endpoint]
    project_id = (request.view_args or {}).get('project_id')
    project = get_project(project_id) if project_id else None
    body = request.get_json(silent=True) or {}
    if not _is_pure_addition((project or {}).get(stored_key, []) or [],
                             body.get(body_key, [])):
        return False
    if endpoint == 'save_labels':
        # color_labels is the label-name map, not a clip list — it must not be
        # edited cross-origin (an omitted/changed map would wipe or rename it).
        if body.get('color_labels', {}) != (project or {}).get('color_labels', {}):
            return False
    return True


@app.before_request
def _guard_cross_origin_writes():
    """Block cross-origin state-changing requests (CSRF / DNS-rebinding) to the
    loopback API, with the narrow client-portal exceptions noted above (SEC-01)."""
    if request.method not in _STATE_CHANGING_METHODS:
        return None
    if not _request_is_cross_origin():
        return None  # same-origin editor — full add/remove/edit/reorder
    endpoint = request.endpoint
    if endpoint in _PORTAL_WRITE_ENDPOINTS:
        return None  # add comment / chat send — intentionally cross-origin
    if endpoint in _PORTAL_APPEND_ONLY:
        if _portal_append_only_ok(endpoint):
            return None  # pure addition of clips/selects — allowed
        return jsonify({'error': 'Cross-origin request may only add items, '
                                 'not remove, edit, or reorder them'}), 403
    return jsonify({'error': 'Cross-origin request blocked'}), 403


@app.context_processor
def inject_brand():
    """Make the user-visible app brand and logo configurable.

    Resolution order:
      1. Explicit env vars (``DOZA_BRAND``, ``DOZA_LOGO_URL``) win when set.
         The 0.7.0 Electron shell doesn't set these, so this branch is
         only used by legacy / dev overrides.
      2. Sibling ``../pro/`` directory present → Pro edition. The
         Electron bundle ships the Pro overlay at
         ``Contents/Resources/pro/`` and the OSS core at
         ``Contents/Resources/app/``; from this file's perspective that
         lands as ``../pro``. Auto-detect picks up the Pro overlay
         without requiring the launcher to know about branding.
      3. Default OSS branding.

    ``app_version`` falls back to the wrapper version when the launching
    shell exposes it via DOZA_WRAPPER_VERSION (set by the 0.7.x Electron
    shell). This eliminates the 0.7.0 confusion where the header showed
    the OSS Flask app version (3.3.0) instead of the wrapper version
    (0.7.0) that users actually downloaded.
    """
    from doza_assist import __version__ as _doza_version

    brand = os.environ.get('DOZA_BRAND')
    logo = os.environ.get('DOZA_LOGO_URL')

    if not brand or not logo:
        # __file__-relative detection — the legacy implementation looked
        # at the CWD which resolved against the run directory rather
        # than the source tree, missing the Pro overlay in the .app
        # bundle even though it was right next door.
        _here = os.path.dirname(os.path.abspath(__file__))
        _pro_sibling = os.path.join(_here, '..', 'pro')
        if os.path.isdir(_pro_sibling):
            if not brand:
                brand = 'Doza Assist'
            if not logo:
                # Pro overlay's collection blueprint serves the
                # branded logo at /collection/static/logo-pro.png.
                logo = '/collection/static/logo-pro.png'

    return {
        'brand': brand or 'Doza Assist',
        'logo_url': logo or '/static/logo.jpg',
        'app_version': os.environ.get('DOZA_WRAPPER_VERSION') or _doza_version,
    }


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _resolve_fcpxml_path(source_path: str) -> str:
    """Given either a .fcpxml file or a .fcpxmld bundle directory, return the
    path to the actual FCPXML document inside.

    FCP exports the bundle form by default (a directory with Info.fcpxml inside);
    editors who "Export XML" can get either form depending on FCP's dialog.
    """
    if os.path.isdir(source_path) and source_path.rstrip('/').endswith('.fcpxmld'):
        inner = os.path.join(source_path, 'Info.fcpxml')
        if not os.path.isfile(inner):
            raise ValueError(f'FCPXML bundle missing Info.fcpxml: {source_path}')
        return inner
    return source_path


def _is_fcpxml_input(path: str) -> bool:
    lower = path.lower().rstrip('/')
    return lower.endswith('.fcpxml') or lower.endswith('.fcpxmld')


def _ingest_fcpxml(fcpxml_path: str, project_dir: str) -> dict:
    """Parse an FCPXML, verify its audio source(s) are on disk, and return a
    dict of fields to merge into the project's meta.json.

    For single-source spines, the project's ``audio_path`` is the referenced
    source file directly. For multi-source spines (mixed mc-clip/sync-clip or
    multiple distinct audio assets), a composed timeline WAV is rendered into
    the project directory and used as ``audio_path`` — transcription then
    produces timestamps aligned to the sequence timeline.

    Raises :class:`ValueError` with an editor-friendly message if any audio
    source cannot be located — typically because the edit drive is not mounted.
    """
    inner_path = _resolve_fcpxml_path(fcpxml_path)

    try:
        parsed = parse_fcpxml(inner_path)
    except ParseError as e:
        raise ValueError(f'Could not read FCPXML: {e}') from e

    # Check every referenced source, not just the representative one — a
    # multi-source spine can reference several drives.
    unique_paths = []
    seen = set()
    for seg in parsed.spine_segments:
        if seg.audio_source is None:
            continue
        p = seg.audio_source.path
        if p in seen:
            continue
        seen.add(p)
        unique_paths.append(p)

    for p in unique_paths:
        if not os.path.exists(p):
            vol = ''
            if p.startswith('/Volumes/'):
                parts = p.split('/', 3)
                vol = parts[2] if len(parts) > 2 else ''
            hint = f' Is the "{vol}" drive mounted?' if vol else ''
            raise ValueError(
                f'FCPXML parsed OK, but the audio file it references was not found: '
                f'{p}.{hint}'
            )

    # Stash the original FCPXML inside the project directory so the writer
    # module can round-trip selects back out without needing the user to still
    # have the source file accessible.
    os.makedirs(project_dir, exist_ok=True)
    fcpxml_copy_name = os.path.basename(inner_path) or 'source.fcpxml'
    fcpxml_copy_path = os.path.join(project_dir, fcpxml_copy_name)
    if os.path.abspath(inner_path) != os.path.abspath(fcpxml_copy_path):
        shutil.copy2(inner_path, fcpxml_copy_path)

    if parsed.is_multi_source:
        # Compose the sequence's dialogue into one timeline-space WAV so the
        # transcription pipeline (which takes one audio file) produces
        # timeline-relative timestamps end-to-end.
        timeline_wav = os.path.join(project_dir, 'timeline_audio.wav')
        try:
            render_timeline_audio(parsed, timeline_wav)
        except TimelineAudioError as e:
            raise ValueError(f'Could not render timeline audio from FCPXML: {e}') from e
        audio_path = timeline_wav
    else:
        audio_path = parsed.audio_file_path

    return {
        'audio_path': audio_path,
        'fcpxml_source': {
            **parsed.to_metadata_dict(),
            'original_fcpxml_path': inner_path,
            'stored_fcpxml_path': fcpxml_copy_path,
            'timeline_audio_rendered': parsed.is_multi_source,
        },
    }


def get_project(project_id):
    """Load a project's metadata."""
    project_dir = safe_project_dir(project_id)
    if project_dir is None:
        return None
    meta_path = os.path.join(project_dir, 'meta.json')
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, 'r') as f:
        return json.load(f)


def load_segment_vectors(project_id):
    """Load structured segment vectors for a project, or [] if not yet generated."""
    path = os.path.join(app.config['PROJECTS_DIR'], project_id, 'segment_vectors.json')
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _paragraph_index_path(project_id):
    return os.path.join(app.config['PROJECTS_DIR'], project_id, 'paragraph_index.json')


def load_paragraph_index(project_id):
    """Load the TF-IDF paragraph retrieval index for a project, or None if
    not yet generated. Used by /chat to rank relevance via cosine similarity
    in addition to literal keyword matching.
    """
    from doza_assist.retrieval import load_index
    return load_index(_paragraph_index_path(project_id))


def save_project(project_id, data):
    """Save a project's metadata."""
    project_dir = safe_project_dir(project_id)
    if project_dir is None:
        raise ValueError(f'Invalid project_id: {project_id!r}')
    os.makedirs(project_dir, exist_ok=True)
    with open(os.path.join(project_dir, 'meta.json'), 'w') as f:
        json.dump(data, f, indent=2)


def list_projects():
    """List all projects sorted by date."""
    projects = []
    projects_dir = app.config['PROJECTS_DIR']
    if not os.path.exists(projects_dir):
        return projects
    for pid in os.listdir(projects_dir):
        meta_path = os.path.join(projects_dir, pid, 'meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                meta['id'] = pid
                projects.append(meta)
    projects.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return projects


def check_source_file(project):
    """Check if the source file still exists and is accessible."""
    filepath = project.get('source_path', project.get('filepath', ''))
    if not filepath:
        return False, 'No source file path recorded'
    if not os.path.exists(filepath):
        return False, f'Source file not found: {filepath}'
    return True, filepath


def format_file_size(size_bytes):
    """Format bytes into a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


# ── Activity log ─────────────────────────────────────────────────────────

def _activity_path(project_id):
    return os.path.join(app.config['PROJECTS_DIR'], project_id, 'activity.json')


def log_activity(project_id, event_type, description):
    """Append an event to the project's activity log (newest first, capped at 50)."""
    path = _activity_path(project_id)
    entries = []
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                entries = json.load(f)
        except (json.JSONDecodeError, OSError):
            entries = []
    entries.insert(0, {
        'ts': int(time.time()),
        'type': event_type,
        'description': description,
    })
    entries = entries[:50]
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(entries, f, indent=2)
    except OSError:
        pass


def get_recent_activity(project_id, limit=6):
    """Return the most recent activity entries with a pre-formatted relative timestamp."""
    path = _activity_path(project_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r') as f:
            entries = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    now = int(time.time())
    out = []
    for e in entries[:limit]:
        out.append({**e, 'relative': _relative_time(now - int(e.get('ts', now)))})
    return out


def _relative_time(seconds):
    if seconds < 60:
        return 'just now'
    if seconds < 3600:
        return f'{seconds // 60}m ago'
    if seconds < 86400:
        return f'{seconds // 3600}h ago'
    return f'{seconds // 86400}d ago'


# ── Routes ──────────────────────────────────────────────────────────────

@app.route('/')
def dashboard():
    """Main dashboard showing all projects grouped by folder."""
    projects = list_projects()

    # Group projects by folder
    folders = {}
    unfiled = []
    for p in projects:
        folder = p.get('folder', '')
        if folder:
            folders.setdefault(folder, []).append(p)
        else:
            unfiled.append(p)

    # Sort folder names
    sorted_folders = sorted(folders.items(), key=lambda x: x[0].lower())

    return render_template('dashboard.html',
                           projects=projects,
                           folders=sorted_folders,
                           unfiled=unfiled)


@app.route('/folder/create', methods=['POST'])
def create_folder():
    """Create a folder (just a name — projects reference it)."""
    name = (request.json or {}).get('name', '').strip()
    if not name:
        return jsonify({'error': 'Folder name required'}), 400
    return jsonify({'status': 'created', 'name': name})


@app.route('/project/<project_id>/move', methods=['POST'])
def move_project(project_id):
    """Move a project to a folder."""
    project = get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404
    folder = (request.json or {}).get('folder', '')
    project['folder'] = folder
    save_project(project_id, project)
    return jsonify({'status': 'moved', 'folder': folder})


@app.route('/folder/delete', methods=['POST'])
def delete_folder():
    """Delete EVERY project filed under a folder, and thus the folder itself.

    Folders are implicit — a folder exists only while a project's
    ``meta.folder`` names it — so removing every member project removes the
    folder from the dashboard. Destructive and irreversible: each matching
    project directory is removed with the same path-confined ``rmtree`` as
    :func:`delete_project` (``safe_project_dir`` rejects any id that escapes
    PROJECTS_DIR). The frontend gates this behind an explicit confirmation;
    original source media on the editor's drives is never touched (we only
    own the project directory, not the referenced footage).
    """
    name = (request.json or {}).get('folder', '')
    name = name.strip() if isinstance(name, str) else ''
    if not name:
        return jsonify({'error': 'Folder name required'}), 400

    deleted, skipped = 0, []
    for p in list_projects():
        if (p.get('folder') or '') != name:
            continue
        pid = p.get('id')
        project_dir = safe_project_dir(pid) if pid else None
        if project_dir is None:
            skipped.append(pid)
            continue
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir, ignore_errors=True)
        deleted += 1
    return jsonify({'status': 'deleted', 'folder': name, 'deleted': deleted, 'skipped': skipped})


# ── Editing platform (NLE) selection ───────────────────────────────

@app.route('/api/projects/<project_id>/editing_platform', methods=['PATCH'])
def update_editing_platform(project_id):
    """Set the project's editing platform and remember the choice as the new global default."""
    project = get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404
    data = request.json or {}
    platform = data.get('platform')
    if platform not in PLATFORMS:
        return jsonify({'error': f'Invalid platform: {platform!r}'}), 400
    project['editing_platform'] = platform
    save_project(project_id, project)
    prefs.set_default_platform(platform)
    return jsonify({
        'status': 'updated',
        'editing_platform': platform,
        'default_platform': prefs.get_default_platform(),
    })


@app.route('/api/preferences/default_platform', methods=['GET'])
def get_default_platform_pref():
    return jsonify({'platform': prefs.get_default_platform()})


@app.route('/api/preferences/default_platform', methods=['PATCH'])
def set_default_platform_pref():
    data = request.json or {}
    platform = data.get('platform')
    if platform not in PLATFORMS:
        return jsonify({'error': f'Invalid platform: {platform!r}'}), 400
    prefs.set_default_platform(platform)
    return jsonify({'platform': prefs.get_default_platform()})


@app.route('/api/ai-model/status', methods=['GET'])
def ai_model_status():
    """Return the current Gemma 4 variant plus per-variant speed/quality estimates.

    When a ``?project_id=...`` is supplied, the ``estimated_casual`` on each
    variant reflects the full analysis time for that project's transcript
    (chunked per 15-minute slice × 2 AI calls per chunk). Without a project
    context the estimate falls back to a single representative call, with the
    casual phrase ending in "per call" rather than "for this project" so the
    user isn't misled.
    """
    import model_config
    hw = model_config.detect_hardware_tier()
    current = model_config.get_gemma4_variant()

    project_id = (request.args.get('project_id') or '').strip()
    project_context = None
    total_seconds = None
    if project_id:
        p = get_project(project_id)
        if p and p.get('transcript'):
            segments = p['transcript'].get('segments', [])
            if segments:
                total_seconds = segments[-1].get('end', 0)
                project_context = {
                    'project_id': project_id,
                    'project_name': p.get('name', 'Project'),
                    'duration_seconds': total_seconds,
                }

    variants = model_config.get_variant_estimates(hw, total_seconds=total_seconds)

    # Resolve which model Ollama will ACTUALLY use on the next chat call.
    # When the selected variant isn't downloaded, _get_ollama_model silently
    # falls back to whatever IS downloaded — the UI uses this to warn the
    # user that "Use this" didn't actually change anything.
    try:
        from ai_analysis import get_effective_ollama_model
        effective_model, selected_variant, fallback_used = get_effective_ollama_model()
    except Exception as e:
        print(f"[ai-model/status] effective-model probe failed: {e}")
        effective_model, selected_variant, fallback_used = (
            current['variant'], current['variant'], False
        )

    return jsonify({
        'hardware': {
            'ram_gb': round(hw['ram_gb'], 1),
            'arch': hw['arch'],
            'arch_label': 'Apple Silicon' if hw['arch'].startswith('arm') else 'Intel',
            'disk_gb': round(hw['disk_gb'], 1),
        },
        'current': {
            'tier': current['tier'],
            'variant': current['variant'],
            'source': current['source'],
            'reason': current.get('reason', ''),
        },
        'effective': {
            'model': effective_model,
            'selected_variant': selected_variant,
            'fallback_used': fallback_used,
        },
        'variants': variants,
        'project_context': project_context,
    })


# ── BYO API key: error handler ──────────────────────────────────────
# Any AI feature endpoint that hits a missing-key, invalid-key, or
# rate-limit condition raises ``ProviderError`` from inside the provider
# layer. The errorhandler catches uncaught ones; routes with their own
# ``except Exception`` blocks should call ``_provider_error_response``
# from a more specific ``except ProviderError`` first.

def _provider_error_response(e):
    """Standardized JSON 400 for any ``ProviderError``.

    Includes ``settings_url`` so the frontend can render a clickable link
    (or button) that opens /settings — the user can fix the key without
    leaving the AI feature they were using.
    """
    return jsonify({
        'error': str(e),
        'code': getattr(e, 'code', '') or 'provider_error',
        'settings_url': '/settings',
    }), 400


@app.errorhandler(Exception)
def _handle_provider_error(e):
    from werkzeug.exceptions import HTTPException
    from ai_providers import ProviderError
    if isinstance(e, ProviderError):
        return _provider_error_response(e)
    # Let real HTTP errors keep their own status. Re-raising an HTTPException
    # from inside an errorhandler makes Flask emit a 500 + full traceback —
    # which is why a harmless poll of a non-existent route (e.g. the setup
    # assistant's /api/status hitting the main app after setup) was logging
    # scary 500 tracebacks for what is really a clean 404. HTTPException
    # instances are valid WSGI responses, so returning one preserves the
    # correct status (404/405/…) with no traceback.
    if isinstance(e, HTTPException):
        return e
    # Genuinely unexpected error → re-raise so Flask logs it and returns 500.
    raise e


# ── BYO API key: provider settings ──────────────────────────────────
# All AI calls route through ``ai_providers.get_active_provider()`` which
# reads ``provider_config.json`` from $DOZA_DATA_DIR. These three routes
# let the user select Local / Anthropic / OpenAI, paste an API key, and
# verify it before relying on it for analysis or chat.

@app.route('/settings/provider', methods=['GET'])
def get_provider_settings():
    """Return the current provider config with API keys masked for display."""
    from ai_providers import load_provider_config, masked_config, has_api_key
    cfg = load_provider_config()
    out = masked_config(cfg)
    # Source of truth is the Keychain (or the JSON fallback when Keychain is
    # unavailable) — both are wrapped by has_api_key. The cfg dict's api_key
    # field is hydrated the same way, so this is belt-and-braces, but it
    # keeps the API contract explicit at the route boundary.
    out['has_anthropic_key'] = has_api_key('anthropic')
    out['has_openai_key'] = has_api_key('openai')
    return jsonify(out)


@app.route('/settings/provider', methods=['POST'])
def save_provider_settings():
    """Save provider selection and/or API keys.

    Body fields are all optional; whichever ones are present get updated.
    An empty string clears that key. Returns the updated masked config.
    """
    from ai_providers import load_provider_config, save_provider_config, masked_config
    body = request.json or {}
    cfg = load_provider_config()

    active = body.get('active_provider')
    if active in ('ollama', 'anthropic', 'openai'):
        cfg['active_provider'] = active

    # ``api_key`` is provider-scoped via the body's ``provider`` field, OR
    # the saved active provider if the caller omits it. This keeps the
    # frontend simple — it just sends {provider, api_key} when the user
    # pastes a key into one of the provider panels.
    target = body.get('provider') or cfg.get('active_provider')
    if 'api_key' in body and target in ('anthropic', 'openai'):
        cfg.setdefault(target, {})['api_key'] = (body.get('api_key') or '').strip()

    if 'base_url' in body:
        # Empty -> resolve via OLLAMA_HOST at runtime (don't pin to :11434).
        cfg.setdefault('ollama', {})['base_url'] = (body['base_url'] or '').strip()

    try:
        save_provider_config(cfg)
    except Exception as e:
        return jsonify({'error': f'Save failed: {e}'}), 500
    return jsonify(masked_config(cfg))


@app.route('/settings/provider/test', methods=['POST'])
def test_provider_connection():
    """Test a provider with the supplied (or saved) credentials.

    Body: ``{"provider": "anthropic"|"openai"|"ollama", "api_key": "..."}``.
    If ``api_key`` is omitted, falls back to the saved key. Sends a tiny
    prompt and reports back ``{"success": bool, "error"|"response": "..."}``.
    """
    from ai_providers import get_provider, load_provider_config
    body = request.json or {}
    name = (body.get('provider') or '').strip().lower()
    if name not in ('ollama', 'anthropic', 'openai'):
        return jsonify({'success': False, 'error': f'Unknown provider: {name!r}'}), 400

    api_key = body.get('api_key') or ''
    if name in ('anthropic', 'openai') and not api_key:
        # Fall back to saved key.
        cfg = load_provider_config()
        api_key = (cfg.get(name) or {}).get('api_key') or ''
    if name in ('anthropic', 'openai') and not api_key:
        return jsonify({'success': False, 'error': 'No API key provided'}), 400

    try:
        if name == 'ollama':
            provider = get_provider('ollama', model_resolver=_get_active_ollama_model)
        else:
            provider = get_provider(name, api_key=api_key)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

    try:
        result = provider.test_connection()
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    return jsonify(result)


def _get_active_ollama_model():
    """Resolve the currently selected Ollama model tag.

    Wrapped in app.py (rather than imported at module scope) so the
    provider package stays decoupled from the OSS model_config helpers.
    """
    from ai_analysis import _get_ollama_model
    return _get_ollama_model()


@app.route('/api/ai-model', methods=['PATCH'])
def set_ai_model():
    """Switch the active Gemma 4 variant. Persists to model_config.json.

    Body: ``{"tier": "small" | "medium" | "large" | "xlarge"}``.
    """
    import model_config
    data = request.json or {}
    tier = data.get('tier', '').strip().lower()
    try:
        info = model_config.set_variant_manually(tier)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    # Evict any other models from VRAM. Ollama's default keep_alive=30m
    # keeps the previous variant resident, so on lower-RAM Macs you end
    # up with two Gemma weights in unified memory at once → Metal
    # compaction stalls.
    new_variant = info['variant']
    try:
        import requests as _req
        from ollama_url import ollama_base_url
        _base = ollama_base_url()
        ps = _req.get(f'{_base}/api/ps', timeout=2).json()
        for m in ps.get('models', []):
            name = m.get('name')
            if name and name != new_variant:
                _req.post(
                    f'{_base}/api/generate',
                    json={'model': name, 'keep_alive': 0, 'prompt': ''},
                    timeout=2,
                )
    except Exception as e:
        print(f"[ai-model] eviction failed: {e}")

    return jsonify({
        'tier': info['tier'],
        'variant': info['variant'],
        'description': info['description'],
        'source': info['source'],
    })


@app.route('/api/ai-model/pull', methods=['POST'])
def pull_ai_model():
    """Stream Ollama's download progress for a Gemma variant as NDJSON.

    Body: ``{"tier": "small" | "medium" | "large" | "xlarge"}``.

    Each line is a JSON object matching Ollama's ``/api/pull`` event schema,
    roughly:

    - ``{"status": "pulling manifest"}``
    - ``{"status": "downloading", "digest": "...", "total": N, "completed": M}``
    - ``{"status": "success"}``
    - ``{"status": "error", "message": "..."}``   (our synthetic terminus on failure)

    The frontend consumes these to render a progress bar in the AI Model
    settings modal and refresh the "downloaded" badges when the pull ends.
    """
    import model_config
    import requests as _requests
    data = request.json or {}
    tier = data.get('tier', '').strip().lower()
    if tier not in model_config.VALID_TIERS:
        return jsonify({'error': f'invalid tier {tier!r}'}), 400
    variant, _size, _desc = model_config.GEMMA4_VARIANTS[tier]

    def _err_event(message: str) -> bytes:
        return (json.dumps({'status': 'error', 'message': message}) + '\n').encode('utf-8')

    def generate():
        try:
            from ollama_url import ollama_base_url
            upstream = _requests.post(
                f'{ollama_base_url()}/api/pull',
                json={'name': variant, 'stream': True},
                stream=True,
                timeout=None,
            )
            if upstream.status_code != 200:
                yield _err_event(f'Ollama pull failed with HTTP {upstream.status_code}')
                return
            # iter_lines(decode_unicode=True) only actually decodes when the
            # upstream response exposes an encoding. Ollama's /api/pull
            # doesn't, so we stay in bytes-land end-to-end — mixing str and
            # bytes chunks in a Flask streaming response crashes with
            # "can't concat str to bytes".
            for line in upstream.iter_lines():
                if line:
                    yield line + b'\n'
        except _requests.exceptions.ConnectionError:
            yield _err_event('Could not reach the bundled Ollama. Is it running?')
        except Exception as e:
            yield _err_event(str(e))

    from flask import Response
    return Response(generate(), mimetype='application/x-ndjson')


def create_project_from_path(
    source_path,
    project_name=None,
    client_name='',
    interviewer_name='Interviewer',
    subject_name='Subject',
    num_speakers=2,
    language='en',
    project_id=None,
):
    """Create a new project from a file already on disk. Returns project_id.

    The source file is NOT copied or moved — meta.json points at it in
    place. Both the path-based ``/create`` route and CLI / scripted
    callers (e.g. batch importers) use this directly. The byte-stream
    ``/upload`` route saves the uploaded file into the project dir
    first, then calls this with ``project_id`` set so the same id is
    reused for the dir + meta (rather than generating a new one and
    leaving the saved bytes orphaned).

    Validation raises ``ValueError`` on a missing file, a non-file path,
    or an unsupported extension. FCPXML inputs are auto-ingested via the
    same ``_ingest_fcpxml`` path the routes use. Failure during ingest
    cleans up the half-created project directory before re-raising so a
    bad input never leaves a dangling empty project on disk.

    Args:
      source_path: absolute path to a media file or .fcpxml(d) on disk.
      project_name: display name. Defaults to the filename stem with
        underscores/hyphens turned into spaces.
      client_name, interviewer_name, subject_name, num_speakers,
        language: optional metadata fields persisted on meta.json.
      project_id: if provided, use this id instead of generating one.
        Lets ``/upload`` create the project dir and save bytes into it
        before the meta is written.
    """
    source_path = os.path.expanduser(source_path)

    is_fcpxml = _is_fcpxml_input(source_path)
    if not is_fcpxml and not os.path.isfile(source_path):
        raise ValueError(f'Path is not a file: {source_path}')
    if not is_fcpxml and not allowed_file(source_path):
        ext = source_path.rsplit('.', 1)[-1].lower() if '.' in source_path else 'unknown'
        raise ValueError(f'Unsupported file type: .{ext}')

    if project_id is None:
        project_id = str(uuid.uuid4())[:8]
    project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
    os.makedirs(project_dir, exist_ok=True)

    fcpxml_meta = None
    if is_fcpxml:
        try:
            ingest = _ingest_fcpxml(source_path, project_dir)
        except ValueError:
            shutil.rmtree(project_dir, ignore_errors=True)
            raise
        audio_path = ingest['audio_path']
        fcpxml_meta = ingest['fcpxml_source']
        if not project_name:
            project_name = fcpxml_meta.get('project_name') or Path(source_path).stem
        media_source_path = audio_path
    else:
        if not project_name:
            project_name = Path(source_path).stem.replace('_', ' ').replace('-', ' ')
        media_source_path = source_path

    file_size = os.path.getsize(media_source_path)

    meta = {
        'id': project_id,
        'name': project_name,
        'client_name': client_name,
        'interviewer_name': interviewer_name or 'Interviewer',
        'subject_name': subject_name or 'Subject',
        'num_speakers': num_speakers,
        'language': language or 'en',
        'filename': os.path.basename(media_source_path),
        'source_path': media_source_path,
        'filepath': media_source_path,
        'file_size': file_size,
        'file_size_formatted': format_file_size(file_size),
        'created_at': datetime.now().isoformat(),
        'status': 'uploaded',
        'transcript': None,
        'analysis': None,
        'client_selects': [],
        'social_clips': [],
        'editing_platform': prefs.get_default_platform(),
    }
    if fcpxml_meta is not None:
        meta['fcpxml_source'] = fcpxml_meta
    save_project(project_id, meta)

    return project_id


@app.route('/create', methods=['POST'])
def create_project():
    """Create a new project from a local file path."""
    data = request.json or {}
    source_path = data.get('source_path', '').strip()

    if not source_path:
        return jsonify({'error': 'No file path provided'}), 400

    # Expand user home directory before existence check so ~/foo works.
    expanded = os.path.expanduser(source_path)
    if not os.path.exists(expanded):
        return jsonify({'error': f'File not found: {expanded}'}), 400

    try:
        project_id = create_project_from_path(
            expanded,
            project_name=data.get('project_name', '').strip() or None,
            client_name=data.get('client_name', '').strip(),
            interviewer_name=data.get('interviewer_name', 'Interviewer').strip(),
            subject_name=data.get('subject_name', 'Subject').strip(),
            num_speakers=int(data.get('num_speakers', 2)),
            language=data.get('language', 'en').strip(),
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    return jsonify({'project_id': project_id, 'status': 'created'})


@app.route('/upload', methods=['POST'])
def upload():
    """Handle small file upload via drag-and-drop (under 500MB)."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    project_name = request.form.get('project_name', '').strip()
    client_name = request.form.get('client_name', '').strip()
    interviewer_name = request.form.get('interviewer_name', 'Interviewer').strip()
    subject_name = request.form.get('subject_name', 'Subject').strip()
    language = request.form.get('language', 'en').strip()

    if not project_name:
        project_name = file.filename.rsplit('.', 1)[0]

    # Generate the project id and dir up front so we have somewhere to
    # land the uploaded bytes. We then hand the saved path to
    # create_project_from_path with the same project_id, so the meta file
    # ends up in the same dir as the file (rather than the function
    # generating a fresh id and orphaning what we just saved).
    project_id = str(uuid.uuid4())[:8]
    project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
    os.makedirs(project_dir, exist_ok=True)

    filename = secure_filename(file.filename)
    filepath = os.path.join(project_dir, filename)
    file.save(filepath)

    try:
        create_project_from_path(
            filepath,
            project_name=project_name,
            client_name=client_name,
            interviewer_name=interviewer_name,
            subject_name=subject_name,
            language=language,
            project_id=project_id,
        )
    except ValueError as e:
        # File was already saved into project_dir; clean up so a bad
        # upload doesn't leave an orphan dir behind.
        shutil.rmtree(project_dir, ignore_errors=True)
        return jsonify({'error': str(e)}), 400

    return jsonify({'project_id': project_id, 'status': 'uploaded'})


@app.route('/find-file', methods=['POST'])
def find_file():
    """Find a file's full path by name and size (used when drag-and-dropping)."""
    data = request.json or {}
    filename = data.get('filename', '').strip()
    file_size = data.get('size', 0)

    if not filename:
        return jsonify({'error': 'No filename provided'}), 400

    home = str(Path.home())
    search_roots = ['/Volumes']
    for d in ['Desktop', 'Documents', 'Movies', 'Downloads', 'Music']:
        p = os.path.join(home, d)
        if os.path.exists(p):
            search_roots.append(p)

    matches = []
    seen = set()

    # FCP package bundles (e.g. .fcpxmld) are directories on disk, but the
    # browser drag-drop reports them as a single "file" with size 0. Match
    # on dirnames and skip size-checking when the target is a bundle.
    is_bundle = filename.lower().endswith(('.fcpxmld', '.fcpbundle'))

    for root_dir in search_roots:
        try:
            for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=True):
                # Match bundle before we prune — a bundle is a dir that happens
                # to match the target name.
                if is_bundle and filename in dirnames:
                    target_path = os.path.join(dirpath, filename)
                    real_path = os.path.realpath(target_path)
                    if real_path not in seen:
                        seen.add(real_path)
                        matches.append({
                            'path': target_path,
                            'size': 0,
                            'size_formatted': 'bundle',
                        })

                # Prune: skip hidden/system dirs, and don't descend into any
                # package bundle (otherwise we'd walk every .fcpxmld's internals).
                dirnames[:] = [
                    d for d in dirnames
                    if not d.startswith('.')
                    and d not in ('node_modules', '__pycache__', '.Trash')
                    and not d.lower().endswith(('.fcpxmld', '.fcpbundle'))
                ]

                if filename in filenames:
                    full_path = os.path.join(dirpath, filename)
                    real_path = os.path.realpath(full_path)

                    if real_path in seen:
                        continue
                    seen.add(real_path)

                    try:
                        stat = os.stat(full_path)
                        # Match by size if provided (within 1% tolerance for filesystem differences)
                        if file_size > 0:
                            size_diff = abs(stat.st_size - file_size)
                            tolerance = max(file_size * 0.01, 4096)
                            if size_diff <= tolerance:
                                matches.append({
                                    'path': full_path,
                                    'size': stat.st_size,
                                    'size_formatted': format_file_size(stat.st_size),
                                })
                        else:
                            matches.append({
                                'path': full_path,
                                'size': stat.st_size,
                                'size_formatted': format_file_size(stat.st_size),
                            })
                    except OSError:
                        continue

                # Stop deep recursion (max 6 levels deep)
                if dirpath.count(os.sep) - root_dir.count(os.sep) >= 6:
                    dirnames.clear()

        except (PermissionError, OSError):
            continue

    if len(matches) == 1:
        return jsonify({'status': 'found', 'path': matches[0]['path'], 'matches': matches})
    elif len(matches) > 1:
        return jsonify({'status': 'multiple', 'matches': matches})
    else:
        return jsonify({'status': 'not_found', 'filename': filename})


@app.route('/browse', methods=['GET'])
def browse_filesystem():
    """Browse the local filesystem for media files."""
    requested_path = request.args.get('path', '')

    # Default starting locations
    if not requested_path:
        home = str(Path.home())
        locations = []

        # Common starting points
        candidates = [
            ('/Volumes', 'External Drives'),
            (os.path.join(home, 'Desktop'), 'Desktop'),
            (os.path.join(home, 'Documents'), 'Documents'),
            (os.path.join(home, 'Movies'), 'Movies'),
            (os.path.join(home, 'Downloads'), 'Downloads'),
        ]

        for path, label in candidates:
            if os.path.exists(path):
                locations.append({
                    'name': label,
                    'path': path,
                    'type': 'directory',
                })

        return jsonify({'locations': locations, 'current_path': '', 'items': [], 'parent': None})

    # Expand and resolve the path
    requested_path = os.path.expanduser(requested_path)
    requested_path = os.path.realpath(requested_path)

    if not os.path.exists(requested_path):
        return jsonify({'error': 'Path does not exist'}), 404

    if not os.path.isdir(requested_path):
        return jsonify({'error': 'Path is not a directory'}), 400

    items = []
    try:
        entries = sorted(os.listdir(requested_path), key=lambda x: (not os.path.isdir(os.path.join(requested_path, x)), x.lower()))
        for entry in entries:
            # Skip hidden files
            if entry.startswith('.'):
                continue

            full_path = os.path.join(requested_path, entry)
            is_dir = os.path.isdir(full_path)

            if is_dir:
                items.append({
                    'name': entry,
                    'path': full_path,
                    'type': 'directory',
                })
            else:
                # Only show supported file types
                if allowed_file(entry):
                    try:
                        size = os.path.getsize(full_path)
                    except OSError:
                        size = 0
                    items.append({
                        'name': entry,
                        'path': full_path,
                        'type': 'file',
                        'size': size,
                        'size_formatted': format_file_size(size),
                    })
    except PermissionError:
        return jsonify({'error': 'Permission denied'}), 403

    parent = os.path.dirname(requested_path)
    if parent == requested_path:
        parent = None

    return jsonify({
        'current_path': requested_path,
        'items': items,
        'parent': parent,
    })


@app.route('/project/<project_id>/check-source')
def check_source(project_id):
    """Check if the source file is still accessible."""
    project = get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    exists, info = check_source_file(project)
    return jsonify({'exists': exists, 'info': info})


@app.route('/project/<project_id>/reveal', methods=['POST'])
def reveal_in_finder(project_id):
    """Open the source file location in Finder."""
    project = get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    source_path = project.get('source_path', project.get('filepath', ''))
    if not source_path or not os.path.exists(source_path):
        return jsonify({'error': 'Source file not found. It may have been moved or deleted.'}), 404

    try:
        subprocess.run(['open', '-R', source_path], check=True)
        return jsonify({'status': 'opened'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def group_into_paragraphs(segments):
    """
    Group transcript segments into readable paragraphs.

    Breaks on:
      - Speaker change (always)
      - Every ~4-6 sentences (keeps paragraphs short and readable)
      - Long pauses > 2 seconds (natural topic breaks)
    """
    if not segments:
        return []

    paragraphs = []
    current = {
        'speaker': segments[0].get('speaker', 'Speaker'),
        'start': segments[0]['start'],
        'start_formatted': segments[0].get('start_formatted', '00:00:00')[:8],
        'segments': [segments[0]],
    }
    sentence_count = 1

    for i in range(1, len(segments)):
        seg = segments[i]
        prev = segments[i - 1]
        speaker = seg.get('speaker', 'Speaker')
        gap = seg['start'] - prev['end']
        prev_text = prev.get('text', '').rstrip()
        ends_sentence = prev_text.endswith(('.', '!', '?'))

        new_para = False

        # Speaker change — always break
        if speaker != current['speaker']:
            new_para = True
        # Long pause — natural break
        elif gap >= 2.0:
            new_para = True
        # After ~5 sentences — keep paragraphs short
        elif sentence_count >= 5 and ends_sentence:
            new_para = True

        if new_para:
            paragraphs.append(current)
            current = {
                'speaker': speaker,
                'start': seg['start'],
                'start_formatted': seg.get('start_formatted', '00:00:00')[:8],
                'segments': [seg],
            }
            sentence_count = 1
        else:
            current['segments'].append(seg)
            if ends_sentence:
                sentence_count += 1

    paragraphs.append(current)
    return paragraphs


@app.route('/project/<project_id>')
def project_view(project_id):
    """View one or more projects. Accepts comma-separated IDs for multi-project workspace."""
    # Pre-warm the Ollama model on project open so the user's first chat
    # message starts with weights resident. Combined with the 30m
    # keep_alive on every generate call, repeat questions never pay
    # cold-load. Daemon thread so the page render isn't blocked.
    import threading
    from ai_analysis import warmup_ollama
    threading.Thread(target=warmup_ollama, daemon=True).start()

    project_ids = [pid.strip() for pid in project_id.split(',') if pid.strip()]

    # Load all requested projects
    projects = []
    for pid in project_ids:
        p = get_project(pid)
        if p:
            exists, _ = check_source_file(p)
            p['source_exists'] = exists
            # Canonicalize analysis field names on read so projects analyzed by
            # a small model (which may have emitted `beat_description`,
            # `start_time`, etc.) still render correctly without re-running
            # analysis. Idempotent — safe for already-normalized data.
            if p.get('analysis'):
                from ai_analysis import normalize_analysis
                p['analysis'] = normalize_analysis(p['analysis'])
            projects.append(p)

    if not projects:
        return redirect(url_for('dashboard'))

    # Primary project (first one) — used as fallback for single-project features
    project = projects[0]

    # All transcribed projects for the selector dropdown
    all_projects = [p for p in list_projects() if p.get('transcript')]

    # Assign a color index to each active project for visual distinction
    project_colors = ['accent', 'green', 'purple', 'orange', 'red']
    video_extensions = ('.mp4', '.mov', '.mxf', '.avi', '.mkv')
    projects_meta = []
    for i, p in enumerate(projects):
        src_ext = os.path.splitext(p.get('source_path', '') or '')[1].lower()
        projects_meta.append({
            'id': p['id'],
            'name': p.get('name', 'Untitled'),
            'color': project_colors[i % len(project_colors)],
            'is_video': src_ext in video_extensions,
        })

    # Build combined paragraphs across all projects
    paragraphs = []
    for i, p in enumerate(projects):
        if p.get('transcript') and p['transcript'].get('segments'):
            paras = group_into_paragraphs(p['transcript']['segments'])
            color = project_colors[i % len(project_colors)]
            for para in paras:
                para['project_id'] = p['id']
                para['project_name'] = p.get('name', 'Untitled')
                para['project_color'] = color
            paragraphs.extend(paras)

    is_multi = len(projects) > 1

    # Determine if any project has video (need video element if so)
    is_video = any(pm['is_video'] for pm in projects_meta)

    # Combine segment vectors across all active projects (used for clip badges
    # and as the menu Story Builder draws from).
    segment_vectors = []
    for p in projects:
        segment_vectors.extend(load_segment_vectors(p['id']))

    # Auto-detect framerate from primary project source for FCPXML export default
    detected_framerate = 23.976
    source_path = project.get('source_path', project.get('filepath', ''))
    if source_path and os.path.exists(source_path):
        ffprobe = shutil.which('ffprobe')
        if not ffprobe:
            for candidate in ['/opt/homebrew/bin/ffprobe', '/usr/local/bin/ffprobe']:
                if os.path.isfile(candidate):
                    ffprobe = candidate
                    break
        if ffprobe:
            try:
                result = subprocess.run([
                    ffprobe, '-v', 'quiet', '-select_streams', 'v:0',
                    '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0',
                    source_path
                ], capture_output=True, text=True, timeout=10)
                if result.returncode == 0 and result.stdout.strip():
                    num, den = result.stdout.strip().split('/')
                    fps = float(num) / float(den)
                    standards = [23.976, 24.0, 25.0, 29.97, 30.0, 59.94, 60.0]
                    detected_framerate = min(standards, key=lambda s: abs(s - fps))
            except Exception:
                pass

    project['editing_platform'] = get_project_platform(project)

    # Seed the client-feedback "pending" store for the template's clientPending
    # JS array. This is a render-only default (do NOT save_project here): after a
    # successful Pull, meta.json already carries client_pending written by
    # pro/share/cloud.py's _merge_client_feedback. This setdefault only covers the
    # pre-pull case so the template seed is always a list (missing key => []),
    # which is a zero-visual-change no-op for normal projects.
    project.setdefault('client_pending', [])

    # Recent activity across all active projects, merged and sorted by time.
    recent_activity = []
    for p in projects:
        for entry in get_recent_activity(p['id'], limit=20):
            entry['project_name'] = p.get('name', 'Untitled')
            recent_activity.append(entry)
    recent_activity.sort(key=lambda e: e.get('ts', 0), reverse=True)
    recent_activity = recent_activity[:6]

    return render_template('project.html',
                           project=project,
                           projects=projects,
                           projects_meta=projects_meta,
                           all_projects=all_projects,
                           active_ids=project_ids,
                           paragraphs=paragraphs,
                           is_multi=is_multi,
                           is_shared=False,
                           is_video=is_video,
                           segment_vectors=segment_vectors,
                           detected_framerate=detected_framerate,
                           editing_platform=project['editing_platform'],
                           recent_activity=recent_activity)


@app.route('/project/<project_id>/media')
def serve_media(project_id):
    """Stream the project's source media file (video or audio).

    Returns the file with HTTP Range support so the <video>/<audio>
    element can seek without re-downloading. Chromium handles decode and
    hardware-accel where available.
    """
    project = get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    source_path = project.get('source_path', project.get('filepath', ''))
    if source_path and os.path.exists(source_path):
        ext = os.path.splitext(source_path)[1].lower()
        mime = {
            '.mp4': 'video/mp4', '.mov': 'video/quicktime',
            '.m4v': 'video/x-m4v', '.webm': 'video/webm',
            '.mkv': 'video/x-matroska', '.avi': 'video/x-msvideo',
            '.mp3': 'audio/mpeg', '.m4a': 'audio/mp4',
            '.wav': 'audio/wav', '.aac': 'audio/aac',
        }.get(ext)
        resp = send_file(source_path, mimetype=mime, conditional=True)
        # Encourage the browser to reuse range responses across reloads/seeks.
        # Without this, every range comes back as a fresh 206 and the same file
        # streams thousands of times during a single playback session.
        resp.headers['Cache-Control'] = 'public, max-age=3600'
        resp.headers['Accept-Ranges'] = 'bytes'
        return resp

    project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
    audio_wav = os.path.join(project_dir, 'audio.wav')
    if os.path.exists(audio_wav):
        resp = send_file(audio_wav, mimetype='audio/wav', conditional=True)
        resp.headers['Cache-Control'] = 'public, max-age=3600'
        resp.headers['Accept-Ranges'] = 'bytes'
        return resp

    return jsonify({'error': 'Source file not found'}), 404


@app.route('/project/<project_id>/media/audio')
def serve_media_audio(project_id):
    """Serve audio-only for lightweight playback (avoids decoding heavy video).

    Returns the extracted 16kHz mono WAV already produced during transcription,
    or the timeline_audio.wav for multi-source FCPXML projects. Falls back to
    extracting audio on the fly if needed.
    """
    project = get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)

    # Prefer timeline WAV for FCPXML projects (already composed)
    timeline_wav = os.path.join(project_dir, 'timeline_audio.wav')
    if os.path.exists(timeline_wav):
        return send_file(timeline_wav, mimetype='audio/wav')

    # Standard extracted audio from transcription
    audio_wav = os.path.join(project_dir, 'audio.wav')
    if os.path.exists(audio_wav):
        return send_file(audio_wav, mimetype='audio/wav')

    # Extract on the fly if transcription hasn't run yet
    source_path = project.get('source_path', project.get('filepath', ''))
    if source_path and os.path.exists(source_path):
        from transcribe import extract_audio
        try:
            wav_path = extract_audio(source_path, project_dir=project_dir)
            mime = 'audio/wav' if wav_path.lower().endswith('.wav') else 'audio/mpeg'
            return send_file(wav_path, mimetype=mime)
        except Exception:
            pass

    return jsonify({'error': 'Audio not available'}), 404


# ── Optional non-English (Whisper) engine, installed on demand ─────────────
# Default install ships only Parakeet MLX (English). Whisper is the gateway
# to 99-language support but adds ~200MB of PyTorch+cmake to setup, so it's
# moved behind this on-demand install. The Retranscribe modal calls these
# endpoints when the user picks a non-English language and Whisper is missing.

import importlib.util as _impu
import sys as _sys

_whisper_install_state = {
    'status': 'idle',  # idle | running | done | error
    'detail': '',
    'started_at': None,
}
_whisper_install_lock = threading.Lock()


def _engine_available(name):
    """Return True if `name` is currently importable. Used over try/import so
    we don't pay the cost (or pollute sys.modules) of a real import every poll."""
    try:
        return _impu.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _install_whisper_worker():
    """Background install of openai-whisper into the running venv. Updates
    _whisper_install_state so the frontend can poll for progress."""
    global _whisper_install_state

    def set_state(status, detail):
        with _whisper_install_lock:
            _whisper_install_state['status'] = status
            _whisper_install_state['detail'] = detail

    try:
        # cmake is a transitive build-time dep of some whisper sub-packages on
        # certain Python versions. Best-effort install — if brew isn't present
        # or the install fails, pip will tell us when it actually needs cmake.
        set_state('running', 'Installing cmake (build dependency)...')
        for brew_path in ('/opt/homebrew/bin/brew', '/usr/local/bin/brew'):
            if os.path.isfile(brew_path):
                subprocess.run(
                    [brew_path, 'install', 'cmake'],
                    capture_output=True, timeout=600,
                )
                break

        # pip install into the same venv this Flask process is running from.
        # sys.executable resolves to venv/bin/python3 because launcher.sh
        # sources the venv before exec'ing app.py.
        set_state('running', 'Installing OpenAI Whisper (~200MB) — this takes a few minutes...')
        proc = subprocess.run(
            [_sys.executable, '-m', 'pip', 'install', 'openai-whisper'],
            capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or '')[-400:]
            set_state('error', f'pip install failed: {tail}')
            return

        # Force a re-scan of the import system so transcribe.py picks up the
        # new package without restarting Flask.
        import importlib
        importlib.invalidate_caches()

        if _engine_available('whisper'):
            set_state('done', 'Done — non-English transcription is now available.')
        else:
            set_state('error', 'Install completed but whisper module is not importable.')

    except subprocess.TimeoutExpired:
        set_state('error', 'Install timed out after 30 minutes. Check your internet connection.')
    except Exception as e:
        set_state('error', f'Install failed: {e}')


@app.route('/api/transcription-engines', methods=['GET'])
def transcription_engines():
    """Report which transcription engines are currently installed."""
    return jsonify({
        'parakeet': _engine_available('parakeet_mlx'),
        'whisper': _engine_available('whisper'),
    })


@app.route('/api/install-whisper', methods=['POST'])
def install_whisper():
    """Kick off a background install of openai-whisper. Idempotent — returns
    immediately if already installed or already running."""
    with _whisper_install_lock:
        if _engine_available('whisper'):
            _whisper_install_state['status'] = 'done'
            _whisper_install_state['detail'] = 'Already installed.'
            return jsonify({'status': 'done'})
        if _whisper_install_state['status'] == 'running':
            return jsonify({'status': 'running'})
        _whisper_install_state['status'] = 'running'
        _whisper_install_state['detail'] = 'Starting install...'
        _whisper_install_state['started_at'] = time.time()

    threading.Thread(target=_install_whisper_worker, daemon=True).start()
    return jsonify({'status': 'started'})


@app.route('/api/install-whisper/status', methods=['GET'])
def install_whisper_status():
    """Poll endpoint for the frontend install banner."""
    with _whisper_install_lock:
        state = dict(_whisper_install_state)
    state['available'] = _engine_available('whisper')
    return jsonify(state)


@app.route('/project/<project_id>/audio-duration', methods=['GET'])
def project_audio_duration(project_id):
    """Return the audio/video file's duration in seconds via ffprobe.

    Called by the transcribe-progress UI BEFORE starting transcription so
    the progress bar can pace itself against real audio length instead of
    a wildly-wrong file-size heuristic. 0.7.0 estimated transcription as
    ``fileSizeGB * 3 + 2`` seconds — for typical podcast mp3s that gives
    2-5 seconds, so the bar raced to 90% and stalled for the rest of
    actual transcription (which takes minutes). With real duration we
    estimate transcription as ~0.1x realtime — close to Parakeet MLX's
    measured pace on Apple Silicon, and within an order of magnitude
    for the Whisper / WhisperX fallback paths.

    Returns ``{"duration_seconds": <float>}`` on success, or
    ``{"duration_seconds": null}`` when ffprobe is unavailable / fails.
    Never raises — a probe failure shouldn't block transcription itself.
    """
    project = get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404
    source_path = project.get('source_path', project.get('filepath', ''))
    if not source_path or not os.path.exists(source_path):
        return jsonify({'duration_seconds': None}), 200
    # Locate ffprobe — same resolution order as transcribe.py uses for
    # ffmpeg, but probing a different binary. The 0.7.x+ bundle ships
    # ffprobe alongside ffmpeg under DOZA_FFMPEG_DIR.
    ffprobe = None
    bundled_dir = os.environ.get('DOZA_FFMPEG_DIR')
    if bundled_dir:
        candidate = os.path.join(bundled_dir, 'ffprobe')
        if os.path.isfile(candidate):
            ffprobe = candidate
    if not ffprobe:
        for candidate in ('/opt/homebrew/bin/ffprobe', '/usr/local/bin/ffprobe'):
            if os.path.isfile(candidate):
                ffprobe = candidate
                break
    if not ffprobe:
        return jsonify({'duration_seconds': None}), 200
    try:
        result = subprocess.run(
            [
                ffprobe, '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                source_path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            seconds = float(result.stdout.strip())
            return jsonify({'duration_seconds': seconds})
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass
    return jsonify({'duration_seconds': None}), 200


# ── Transcription job tracking ───────────────────────────────────────
# Background-thread transcription. POST /transcribe kicks off a thread
# and returns immediately; the frontend polls /transcribe/status for
# real progress (phase, pct, engine) emitted from transcribe.py via
# the progress_cb. Replaces a synchronous request that could block the
# UI for 15-25 min on slow paths (Whisper-CPU fallback) with no
# user-visible signal.

_transcribe_jobs: dict = {}
_transcribe_jobs_lock = threading.Lock()

# Process-global "only one transcription at a time" gate. Transcription
# engines share a single in-process model singleton (Parakeet/WhisperX/
# Whisper, cached behind transcribe._model_lock) and a single Metal GPU.
# Running two transcriptions concurrently means two threads driving the
# same MLX model object on the same GPU command queue — which deadlocks
# (the symptom users hit when adding a folder of clips via the import
# queue: every clip's background thread launches at once, piles onto the
# model load, and they all freeze at the "load_model" phase forever).
# Serializing here means a second job parks at phase="queued" until the
# first finishes, then runs cleanly. A single clip is unaffected.
_transcribe_run_lock = threading.Lock()


def _transcribe_status_path(project_id):
    return os.path.join(app.config['PROJECTS_DIR'], project_id, 'transcribe_status.json')


def _write_transcribe_status(project_id, data):
    path = _transcribe_status_path(project_id)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[transcribe] could not write status for {project_id}: {e}", flush=True)


def _make_transcribe_progress_writer(project_id):
    """Return a ``progress_cb(event)`` that persists transcription
    progress to disk for the frontend status poll to read.

    ``event`` is the dict emitted by ``transcribe.transcribe_file`` —
    ``{"phase": str, "pct": int, "engine": str, ...}``. We merge it
    into a session dict with started_at + updated_at timestamps so the
    UI can tell when transcription has stalled (no updates for >2 min
    likely means a hung model load).
    """
    started_at = datetime.now().isoformat()

    def writer(event):
        snapshot = {
            "started_at": started_at,
            "updated_at": datetime.now().isoformat(),
            "phase": event.get("phase", "transcribing"),
            "pct": int(event.get("pct", 0)),
            "engine": event.get("engine"),
            "slow_mode": bool(event.get("slow_mode")),
            "audio_sec": event.get("audio_sec"),
        }
        with _transcribe_jobs_lock:
            _transcribe_jobs[project_id] = snapshot
        _write_transcribe_status(project_id, snapshot)
    return writer


def _run_transcribe_job(project_id, source_path, num_speakers, language,
                       interviewer_name, subject_name):
    """Background worker that runs transcribe_file under a per-project
    progress writer. Final state (done / error) lands in
    transcribe_status.json so the frontend can stop polling."""
    from transcribe import transcribe_file
    project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
    progress_cb = _make_transcribe_progress_writer(project_id)

    # Serialize against any other in-flight transcription. If the gate is
    # already held (another clip is transcribing — e.g. a folder import
    # fired several jobs at once), park here at phase="queued" so the
    # frontend poll shows a clear "Queued…" state instead of a frozen
    # "Load_model…", then proceed once the engine is free.
    if not _transcribe_run_lock.acquire(blocking=False):
        progress_cb({"phase": "queued", "pct": 0})
        _transcribe_run_lock.acquire()
    try:
        result = transcribe_file(
            source_path,
            project_dir=project_dir,
            speaker_labels={
                'SPEAKER_00': interviewer_name,
                'SPEAKER_01': subject_name,
            },
            num_speakers=num_speakers,
            language=language,
            progress_cb=progress_cb,
        )
        # Persist to the project as before.
        project = get_project(project_id) or {}
        project['transcript'] = result
        project['status'] = 'transcribed'
        save_project(project_id, project)
        seg_count = len((result or {}).get('segments', []))
        log_activity(project_id, 'transcribed',
                     f"{project.get('name', 'Project')} transcribed · {seg_count} segments")
        # Auto-build the TF-IDF paragraph index (same as the synchronous
        # path used to do).
        try:
            idx_path = _paragraph_index_path(project_id)
            if not os.path.exists(idx_path):
                from doza_assist.retrieval import build_paragraph_index, save_index
                idx = build_paragraph_index(result)
                save_index(idx, idx_path)
                print(f"[transcribe] auto-built paragraph_index for {project_id}")
        except Exception as e:
            print(f"[transcribe] paragraph_index auto-build failed for {project_id}: {e}")
        # Auto-enqueue speaker diarization. The Pro extension installs an
        # @after_app_request hook on the /transcribe endpoint that used to
        # do this, but it only fires on a 200 response — and the new
        # background-thread architecture (0.8.10+) returns 202 from the
        # /transcribe route and finishes the actual work here in the
        # worker thread. The hook never sees a 200, so the enqueue
        # silently never ran. Doing it explicitly from the worker's
        # completion path is the canonical hook anyway: at this point we
        # know the transcript landed on disk and is ready to diarize.
        try:
            from diarization import get_worker as _get_diar_worker
            projects_dir = app.config['PROJECTS_DIR']
            _get_diar_worker(projects_dir).enqueue(project_id)
            print(f"[transcribe] diarization queued for {project_id}", flush=True)
        except ImportError:
            # OSS / no diarization extension on this build.
            pass
        except Exception as e:
            # Non-fatal; transcription still completes. Surface so the
            # support flow doesn't have to guess.
            print(f"[transcribe] diarization enqueue failed for {project_id}: {e}", flush=True)
        # Terminal status: done. Frontend stops polling when it sees this.
        final = {
            "started_at": _transcribe_jobs.get(project_id, {}).get("started_at"),
            "updated_at": datetime.now().isoformat(),
            "phase": "done",
            "pct": 100,
            "engine": _transcribe_jobs.get(project_id, {}).get("engine"),
            "segments": seg_count,
        }
        with _transcribe_jobs_lock:
            _transcribe_jobs[project_id] = final
        _write_transcribe_status(project_id, final)
    except Exception as e:
        import traceback
        traceback.print_exc()
        err_payload = {
            "started_at": _transcribe_jobs.get(project_id, {}).get("started_at"),
            "updated_at": datetime.now().isoformat(),
            "phase": "error",
            "pct": _transcribe_jobs.get(project_id, {}).get("pct", 0),
            "message": str(e),
        }
        with _transcribe_jobs_lock:
            _transcribe_jobs[project_id] = err_payload
        _write_transcribe_status(project_id, err_payload)
        try:
            project = get_project(project_id) or {}
            project['status'] = 'error'
            project['error'] = str(e)
            save_project(project_id, project)
        except Exception:
            pass
    finally:
        _transcribe_run_lock.release()


@app.route('/project/<project_id>/transcribe', methods=['POST'])
def transcribe(project_id):
    """Kick off a background transcription job. Returns immediately
    with the started phase; the frontend polls /transcribe/status."""
    project = get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    # Check that source file still exists
    source_path = project.get('source_path', project.get('filepath', ''))
    if not source_path or not os.path.exists(source_path):
        return jsonify({'error': 'Source file not found. It may have been moved or deleted.'}), 404

    # Guard non-English requests when only Parakeet (English-only) is installed.
    requested_language = project.get('language', 'en')
    if requested_language not in ('en', 'auto') and not _engine_available('whisper'):
        project['status'] = 'error'
        project['error'] = 'whisper_not_installed'
        save_project(project_id, project)
        return jsonify({
            'error': f'Non-English transcription ({requested_language}) requires the Whisper engine, which is not installed.',
            'needs_whisper_install': True,
            'requested_language': requested_language,
        }), 400

    # Refuse to start a second job for the same project while one is in
    # flight — protects against a double-click on the Transcribe button
    # spawning two concurrent jobs that overwrite each other's results.
    with _transcribe_jobs_lock:
        existing = _transcribe_jobs.get(project_id)
        if existing and existing.get("phase") not in ("done", "error", None):
            return jsonify({
                'status': 'transcribing',
                'message': 'Transcription already in progress.',
                'state': existing,
            }), 202

    project['status'] = 'transcribing'
    save_project(project_id, project)

    started = {
        "started_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "phase": "extract_audio",
        "pct": 0,
        "engine": None,
    }
    with _transcribe_jobs_lock:
        _transcribe_jobs[project_id] = started
    _write_transcribe_status(project_id, started)

    num_speakers = project.get('num_speakers', 2)
    language = project.get('language', 'en')
    interviewer_name = project.get('interviewer_name', 'Interviewer')
    subject_name = project.get('subject_name', 'Subject')

    thread = threading.Thread(
        target=_run_transcribe_job,
        args=(project_id, source_path, num_speakers, language, interviewer_name, subject_name),
        daemon=True,
    )
    thread.start()

    return jsonify({'status': 'started', 'state': started}), 202


@app.route('/project/<project_id>/transcribe/status', methods=['GET'])
def transcribe_status(project_id):
    """Poll endpoint for the background transcription job. Returns the
    in-memory snapshot if available, otherwise reads transcribe_status.json
    from disk (covers the case where the Flask process restarted while a
    transcription was running — the frontend sees the last known phase)."""
    with _transcribe_jobs_lock:
        snap = _transcribe_jobs.get(project_id)
    if snap is None:
        path = _transcribe_status_path(project_id)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    snap = json.load(f)
            except Exception:
                snap = None
    return jsonify({'state': snap or {"phase": "idle", "pct": 0}})


def _analyze_status_path(project_id):
    return os.path.join(app.config['PROJECTS_DIR'], project_id, 'analyze_status.json')


def _make_progress_writer(project_id):
    """Return a ``write(step, total, current)`` callable that persists per-step
    analyze progress to disk for the frontend status poll to read.

    Records ``started_at`` once (on creation) so the UI can compute elapsed
    and ETA from the latest snapshot. ``updated_at`` refreshes on every call,
    so the frontend can detect a stalled analyze (no updates for >2 min ⇒
    likely a hung Ollama call). Failures are swallowed — progress is
    advisory and must never break the actual analysis.
    """
    status_path = _analyze_status_path(project_id)
    started_at = datetime.now().isoformat()

    def _write(step, total, current):
        try:
            os.makedirs(os.path.dirname(status_path), exist_ok=True)
            payload = {
                'step': int(step),
                'total': int(total),
                'current': str(current),
                'started_at': started_at,
                'updated_at': datetime.now().isoformat(),
                'done': bool(int(step) >= int(total)),
            }
            with open(status_path, 'w') as f:
                json.dump(payload, f)
        except Exception:
            pass
    return _write


def _clear_analyze_status(project_id):
    """Remove the per-project analyze_status.json. Called on /analyze
    completion (success or failure) so the next polled status reads as
    'idle' for a fresh page load.
    """
    try:
        os.remove(_analyze_status_path(project_id))
    except FileNotFoundError:
        pass
    except Exception:
        pass


# Registry of background analysis threads keyed by project_id. Lets the
# /analyze handler reject duplicate requests on the same project and the
# UI detect that a re-rendered project page has work in flight that it
# should attach progress polling to. Module-global because Flask's
# in-process threaded server keeps everything in one Python process.
_analysis_threads = {}
_analysis_threads_lock = threading.Lock()

# Process-global "one analysis at a time" gate. The collection flow fires a
# /analyze for every interview; each detaches a worker thread, so without this
# they all hammer the single local Ollama at once — which doesn't go faster,
# it makes every call slower and pushes the big models (gemma4:26b/31b) past
# their HTTP timeout, so analyses error out and the collection dashboard comes
# back empty. Serializing means each analysis gets the full model and finishes
# well inside the (now model-aware) timeout. A queued job parks at
# current="queued" so the polling UI shows it waiting rather than stalled.
_analysis_run_lock = threading.Lock()


def _run_analysis_worker(project_id, analysis_type, transcript_hash, cache_snapshot):
    """Background-thread worker for /analyze.

    Mirrors what the inline /analyze handler used to do, but detached
    from the request lifecycle. The user-visible bug this fixes: editors
    who navigated away from a project mid-analysis came back to nothing
    — Flask's request thread was either getting torn down on disconnect
    or simply abandoned without persisting the partial result. Now the
    analysis runs in a daemon thread that survives navigation and writes
    the result to meta.json regardless of whether the editor is still
    looking. Per-step progress goes to analyze_status.json so a polling
    UI on any future page load can pick up where it left off.

    ``cache_snapshot`` is the analysis_cache dict captured at the time
    the request came in. We avoid re-reading the project's cache from
    disk inside the worker so we don't lose unrelated cache entries that
    a separate /analyze on a different transcript might have written
    concurrently.
    """
    progress = _make_progress_writer(project_id)
    # Serialize against any other in-flight analysis (see _analysis_run_lock).
    # If the gate is held, park at current="queued" so the UI shows the wait.
    if not _analysis_run_lock.acquire(blocking=False):
        try:
            progress(step=0, total=1, current="queued")
        except Exception:
            pass
        _analysis_run_lock.acquire()
    try:
        project = get_project(project_id)
        if not project or not project.get('transcript'):
            return
        from ai_analysis import analyze_transcript, generate_segment_vectors, expected_vector_chunks

        existing_vectors = load_segment_vectors(project_id)
        n_vector_chunks = expected_vector_chunks(project['transcript'])
        post_analyzer_steps = n_vector_chunks + 1  # +1 for paragraph index
        analyzer_state = {'step': 0, 'total': 1}

        def _from_analyzer(step, total, current):
            analyzer_state['step'] = step
            analyzer_state['total'] = total
            progress(step=step, total=total + post_analyzer_steps, current=current)

        result = analyze_transcript(
            project['transcript'],
            project_name=project['name'],
            analysis_type=analysis_type,
            segment_vectors=existing_vectors or None,
            progress_callback=_from_analyzer,
        )
        project['analysis'] = result
        analyzer_total = analyzer_state['total']
        global_total = analyzer_total + post_analyzer_steps

        def _from_vectors(chunk_idx=1, total_chunks=1, label="vectors"):
            progress(step=analyzer_total + int(chunk_idx),
                     total=global_total, current=label)

        segment_vectors = []
        try:
            segment_vectors = generate_segment_vectors(
                project['transcript'],
                project_name=project['name'],
                progress_callback=_from_vectors,
            )
        except Exception as ve:
            print(f"Segment vector generation failed: {ve}")

        if segment_vectors:
            project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
            vectors_path = os.path.join(project_dir, 'segment_vectors.json')
            with open(vectors_path, 'w') as f:
                json.dump(segment_vectors, f, indent=2)

        progress(step=global_total, total=global_total,
                 current="building paragraph index")
        try:
            from doza_assist.retrieval import build_paragraph_index, save_index
            paragraph_index = build_paragraph_index(project['transcript'])
            project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
            save_index(paragraph_index, _paragraph_index_path(project_id))
        except Exception as ie:
            print(f"Paragraph index build failed: {ie}")

        project['analysis_cache'] = {
            transcript_hash: {
                **(cache_snapshot.get(transcript_hash, {}) if isinstance(cache_snapshot, dict) else {}),
                analysis_type: {
                    'analysis': result,
                    'cached_at': datetime.now().isoformat(),
                },
            }
        }
        # Re-read on-disk meta right before save so we pick up any
        # changes that concurrent writers (most importantly the Pro
        # diarization worker) made to fields this worker does not own.
        #
        # Without this merge, the analyze worker's save_project clobbers
        # the SPEAKER_NN labels and the meta["diarization"] block that
        # diarization wrote while the LLM analysis was running. The
        # symptom: a freshly-built collection shows all segments with
        # the interviewer's name (e.g. "Chris") even though
        # diarization_segments.json on disk has the real per-speaker
        # boundaries and diarization_status.json says "done". (Verified
        # in production on 0.8.14 with a 2-project collection that
        # diarized correctly to 12 + 3 speakers respectively, but
        # meta.json segments stayed pinned at the OSS Parakeet
        # default-speaker label.)
        try:
            on_disk = get_project(project_id) or {}
            if on_disk.get('transcript'):
                # Take the LIVE transcript (with diarization's SPEAKER_NN
                # labels) over the stale one we loaded at the top of
                # this worker. Other transcript fields (text, timing,
                # words) are also LIVE; they only change when
                # transcribe re-runs, which would not happen during
                # the analyze window anyway.
                project['transcript'] = on_disk['transcript']
            if on_disk.get('diarization'):
                project['diarization'] = on_disk['diarization']
            if on_disk.get('speaker_names'):
                project['speaker_names'] = on_disk['speaker_names']
        except Exception as merge_err:
            # Non-fatal — analysis still completes. Surface it so the
            # support flow doesn't have to guess.
            print(
                f"[analyze worker] merge-before-save failed for "
                f"{project_id}: {merge_err}",
                flush=True,
            )
        save_project(project_id, project)
        title = result.get('suggested_title') or project.get('name', 'Project')
        log_activity(project_id, 'analyzed', f'AI analysis run · "{title}"')

        # Pro-only post-step: ask the LLM to identify each diarized speaker
        # by name from the first ~5 min of transcript context. The function
        # is loaded lazily so OSS installs (no Pro extension on sys.path)
        # silently skip this. The function itself is also defensive — it
        # short-circuits unless diarization is `done` and speaker_names is
        # empty, and it never raises. Wrapped here for belt-and-braces.
        try:
            from diarization import auto_name_speakers  # type: ignore
            auto_name_speakers(project_id, app.config['PROJECTS_DIR'])
        except ImportError:
            pass  # OSS / no Pro extension loaded
        except Exception as e:
            print(f"[analyze worker] auto_name_speakers failed for {project_id}: {e}")

        _clear_analyze_status(project_id)
    except Exception as e:
        print(f"[analyze worker] {project_id} failed: {e}")
        # Surface the error through the status file so the polling UI can
        # show it. We deliberately don't re-raise — the worker is detached.
        try:
            from ai_providers import ProviderError
            payload = {
                'done': True,
                'error': str(e),
                'updated_at': datetime.now().isoformat(),
            }
            if isinstance(e, ProviderError):
                payload['provider_error'] = True
                payload['code'] = e.code
            os.makedirs(os.path.dirname(_analyze_status_path(project_id)), exist_ok=True)
            with open(_analyze_status_path(project_id), 'w') as f:
                json.dump(payload, f)
        except Exception as inner:
            print(f"[analyze worker] could not persist error: {inner}")
    finally:
        _analysis_run_lock.release()
        with _analysis_threads_lock:
            _analysis_threads.pop(project_id, None)


def _transcript_hash(transcript):
    """Stable fingerprint for the transcript content. Caches AI analysis on
    this so re-running /analyze on an unchanged transcript returns instantly.

    Hashes only the segments list — not surrounding metadata like
    speaker_labels, since label edits don't change the analytic content the
    model would produce.
    """
    segments = (transcript or {}).get('segments', []) or []
    payload = json.dumps(segments, sort_keys=True, default=str).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


@app.route('/project/<project_id>/analyze', methods=['POST'])
def analyze(project_id):
    """Kick off AI analysis on the transcript.

    Cache hits return synchronously with ``status: 'cached'``. Real runs
    detach to a background thread (:func:`_run_analysis_worker`) and
    return immediately with ``status: 'started'`` — the editor can
    navigate away from the project page without aborting the run, and
    progress is published to ``analyze_status.json`` for the polling UI.
    """
    project = get_project(project_id)
    if not project or not project.get('transcript'):
        return jsonify({'error': 'No transcript available'}), 400

    analysis_type = request.json.get('type', 'all')  # 'story', 'social', 'all'
    force = bool(request.json.get('force'))

    # Cache lookup keyed by (transcript content, analysis_type). The model is
    # expensive on long interviews, and re-clicking Analyze used to redo it
    # from scratch. The user can still bust the cache by passing
    # ``{"force": true}`` (e.g. after a model or prompt upgrade).
    transcript_hash = _transcript_hash(project['transcript'])
    cache = project.get('analysis_cache') if isinstance(project.get('analysis_cache'), dict) else {}
    bucket = cache.get(transcript_hash) if isinstance(cache.get(transcript_hash), dict) else {}
    cached_entry = bucket.get(analysis_type) if isinstance(bucket.get(analysis_type), dict) else None
    if not force and cached_entry and isinstance(cached_entry.get('analysis'), dict):
        project['analysis'] = cached_entry['analysis']
        save_project(project_id, project)
        return jsonify({
            'status': 'cached',
            'analysis': cached_entry['analysis'],
            'segment_vectors': load_segment_vectors(project_id),
        })

    # Already running on this project? Don't kick off a duplicate worker;
    # let the existing one finish and the polling UI pick it up.
    with _analysis_threads_lock:
        existing = _analysis_threads.get(project_id)
        if existing and existing.is_alive():
            return jsonify({'status': 'running', 'analysis_type': analysis_type})

    # Seed the status file synchronously so the very first poll from the
    # frontend already sees the run as started — no race between thread
    # spawn and the UI noticing.
    _make_progress_writer(project_id)(step=0, total=1, current="starting")

    t = threading.Thread(
        target=_run_analysis_worker,
        args=(project_id, analysis_type, transcript_hash, cache),
        daemon=True,
        name=f'analyze-{project_id}',
    )
    with _analysis_threads_lock:
        _analysis_threads[project_id] = t
    t.start()

    return jsonify({'status': 'started', 'analysis_type': analysis_type})


@app.route('/project/<project_id>/analyze/status', methods=['GET'])
def analyze_status(project_id):
    """Return the current analyze progress for ``project_id``.

    Shape when an analyze is in flight:
        {step, total, current, started_at, updated_at, done}

    Shape when no analyze is running (file absent or read failure):
        {idle: true}

    Frontend polls this every ~1s while its /analyze fetch is in flight,
    computes elapsed and ETA, and stops polling on done==true (or when the
    fetch resolves, whichever comes first).
    """
    path = _analyze_status_path(project_id)
    if not os.path.exists(path):
        return jsonify({'idle': True})
    try:
        with open(path) as f:
            return jsonify(json.load(f))
    except (json.JSONDecodeError, OSError):
        return jsonify({'idle': True})


@app.route('/project/<project_id>/chat', methods=['POST'])
def chat(project_id):
    """Chat with AI about the transcript. Supports comma-separated IDs for multi-project."""
    pids = [pid.strip() for pid in project_id.split(',') if pid.strip()]
    projects_for_chat = []
    for pid in pids:
        p = get_project(pid)
        if p and p.get('transcript'):
            projects_for_chat.append(p)

    if not projects_for_chat:
        return jsonify({'error': 'No transcript available'}), 400

    data = request.json or {}
    message = data.get('message', '').strip()
    history = data.get('history', [])
    profile_id = data.get('profile_id')  # session-only override from the UI

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    try:
        from ai_analysis import chat_about_transcript

        if len(projects_for_chat) == 1:
            p = projects_for_chat[0]
            # segment_vectors and paragraph_index are both built by /analyze.
            # When present, they upgrade chat retrieval: theme_tags become
            # an extra search vocabulary, high-narrative-score windows get
            # prioritized as relevant excerpts, and TF-IDF cosine similarity
            # ranks paragraphs better than substring matching does. Without
            # them, behavior falls back to literal-keyword matching.
            reply = chat_about_transcript(
                transcript=p['transcript'],
                message=message,
                history=history,
                project_name=p.get('name', 'Interview'),
                analysis=p.get('analysis'),
                profile_id=profile_id,
                segment_vectors=load_segment_vectors(p['id']) or None,
                paragraph_index=load_paragraph_index(p['id']),
                labeled_sections=p.get('labeled_sections') or None,
                speaker_names=p.get('speaker_names') or None,
            )
        else:
            # Multi-project: combine transcripts with project labels.
            # No segment_vectors here — vectors are per-project and combining
            # them would require offsetting timecodes; not worth the
            # complexity until multi-project chat is in heavy use.
            combined_segments = []
            project_names = []
            for p in projects_for_chat:
                project_names.append(p.get('name', 'Untitled'))
                for seg in p['transcript'].get('segments', []):
                    seg_copy = dict(seg)
                    seg_copy['_project'] = p.get('name', 'Untitled')
                    combined_segments.append(seg_copy)

            combined_transcript = {
                'segments': combined_segments,
                'language': 'en',
            }
            reply = chat_about_transcript(
                transcript=combined_transcript,
                message=message,
                history=history,
                project_name=' + '.join(project_names),
                analysis=None,
                profile_id=profile_id,
            )

        # Persist chat history on single-project chats. Multi-project sessions
        # (comma-separated IDs) stay ephemeral — ownership is ambiguous and we
        # don't want to fork writes across multiple meta.json files.
        if len(projects_for_chat) == 1:
            p = projects_for_chat[0]
            pid = p['id']
            stored = get_project(pid) or p
            history_log = list(stored.get('chat_history') or [])
            now_iso = datetime.now().isoformat()
            history_log.append({'role': 'user', 'content': message, 'ts': now_iso})
            history_log.append({'role': 'assistant', 'content': reply, 'ts': now_iso})
            stored['chat_history'] = history_log
            save_project(pid, stored)

        return jsonify({'reply': reply})
    except Exception as e:
        from ai_providers import ProviderError
        if isinstance(e, ProviderError):
            return _provider_error_response(e)
        return jsonify({'error': str(e)}), 500


@app.route('/project/<project_id>/chat-stream', methods=['POST'])
def chat_stream(project_id):
    """Server-sent-events variant of /chat. Yields token chunks as the model
    produces them so the UI can render progressively instead of blocking on
    the whole reply. Falls back to /chat semantically: same payload, same
    persistence, same retrieval signals — just streamed.
    """
    pids = [pid.strip() for pid in project_id.split(',') if pid.strip()]
    projects_for_chat = []
    for pid in pids:
        p = get_project(pid)
        if p and p.get('transcript'):
            projects_for_chat.append(p)
    if not projects_for_chat:
        return jsonify({'error': 'No transcript available'}), 400

    data = request.json or {}
    message = data.get('message', '').strip()
    history = data.get('history', [])
    profile_id = data.get('profile_id')
    if not message:
        return jsonify({'error': 'No message provided'}), 400

    from ai_analysis import chat_about_transcript_stream

    if len(projects_for_chat) == 1:
        p = projects_for_chat[0]
        stream_kwargs = {
            'transcript': p['transcript'],
            'message': message,
            'history': history,
            'project_name': p.get('name', 'Interview'),
            'analysis': p.get('analysis'),
            'profile_id': profile_id,
            'segment_vectors': load_segment_vectors(p['id']) or None,
            'paragraph_index': load_paragraph_index(p['id']),
            'labeled_sections': p.get('labeled_sections') or None,
            'speaker_names': p.get('speaker_names') or None,
        }
        single_pid = p['id']
    else:
        combined_segments = []
        project_names = []
        for p in projects_for_chat:
            project_names.append(p.get('name', 'Untitled'))
            for seg in p['transcript'].get('segments', []):
                seg_copy = dict(seg)
                seg_copy['_project'] = p.get('name', 'Untitled')
                combined_segments.append(seg_copy)
        stream_kwargs = {
            'transcript': {'segments': combined_segments, 'language': 'en'},
            'message': message,
            'history': history,
            'project_name': ' + '.join(project_names),
            'analysis': None,
            'profile_id': profile_id,
        }
        single_pid = None

    def _generate():
        from ai_providers import ProviderError
        final_reply = ''
        # Emit a synthetic heartbeat as the very first SSE frame so the
        # browser knows the connection is alive before the model produces
        # any tokens. Some local models (Gemma 4 in particular) can sit in
        # a thinking phase for 10+ seconds; without this the renderer's
        # readable-stream pump only sees blocked I/O and the user thinks
        # the chat is stuck.
        yield f"data: {json.dumps({'event': 'heartbeat', 'data': 'connected'})}\n\n"
        try:
            for event_type, payload in chat_about_transcript_stream(**stream_kwargs):
                if event_type == 'done':
                    final_reply = payload or ''
                yield f"data: {json.dumps({'event': event_type, 'data': payload})}\n\n"
        except ProviderError as e:
            # Surface as a clean assistant reply rather than a generic
            # error so the conversation flow stays intact + the user sees
            # the message with a Settings link.
            final_reply = str(e)
            yield f"data: {json.dumps({'event': 'done', 'data': final_reply, 'provider_error': True, 'code': e.code, 'settings_url': '/settings'})}\n\n"
            return
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'data': str(e)})}\n\n"
            return

        # Persist on single-project chats only — same rule the non-streaming
        # endpoint enforces. Multi-project sessions stay ephemeral.
        if single_pid:
            try:
                stored = get_project(single_pid) or {}
                history_log = list(stored.get('chat_history') or [])
                now_iso = datetime.now().isoformat()
                history_log.append({'role': 'user', 'content': message, 'ts': now_iso})
                history_log.append({'role': 'assistant', 'content': final_reply, 'ts': now_iso})
                stored['chat_history'] = history_log
                save_project(single_pid, stored)
            except Exception as e:
                print(f"[chat-stream] history persist failed: {e}")

    return Response(stream_with_context(_generate()), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',  # disable nginx buffering if proxied
    })


@app.route('/project/<project_id>/chat', methods=['DELETE'])
def clear_chat(project_id):
    """Wipe persisted chat history for a project. Multi-project IDs clear only
    the first project (others weren't persisted in the first place)."""
    pid = project_id.split(',', 1)[0].strip()
    project = get_project(pid)
    if not project:
        return jsonify({'error': 'Project not found'}), 404
    project['chat_history'] = []
    save_project(pid, project)
    return jsonify({'status': 'cleared'})


@app.route('/project/<project_id>/selects', methods=['POST'])
def save_selects(project_id):
    """Save client selections from the review portal."""
    project = get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    selects = request.json.get('selects', [])
    project['client_selects'] = selects
    save_project(project_id, project)
    return jsonify({'status': 'saved', 'count': len(selects)})


@app.route('/project/<project_id>/labels', methods=['POST'])
def save_labels(project_id):
    """Save color label names and labeled transcript sections."""
    project = get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    data = request.json or {}
    prev_count = len(project.get('labeled_sections', []) or [])
    project['color_labels'] = data.get('color_labels', {})
    project['labeled_sections'] = data.get('labeled_sections', [])
    save_project(project_id, project)
    new_count = len(project['labeled_sections'])
    delta = new_count - prev_count
    if delta > 0:
        log_activity(project_id, 'clip_added',
                     f"{delta} clip{'s' if delta != 1 else ''} added · {new_count} total")
    elif delta < 0:
        log_activity(project_id, 'clip_removed',
                     f"{-delta} clip{'s' if -delta != 1 else ''} removed · {new_count} total")
    return jsonify({'status': 'saved', 'count': len(project['labeled_sections'])})


def _build_nle_export(project: dict, body: dict, force_platform: str | None = None):
    """Run the configured exporter for ``project`` against the selections in ``body``.

    Shared by ``/export/fcpxml`` (which streams the file back) and
    ``/export/send-to-nle`` (which writes the file then launches the NLE).
    Returns ``(result, exporter)``.
    """
    raw_types = body.get('types')
    if isinstance(raw_types, list) and raw_types:
        requested = {str(t).strip().lower() for t in raw_types if t}
    else:
        single = str(body.get('type', 'labels')).strip().lower()
        if single == 'all':
            requested = {'labels', 'social', 'story', 'soundbites'}
        elif single in ('selects', 'clips'):
            requested = {'labels'}
        else:
            requested = {single}

    export_type = 'all' if requested >= {'labels', 'social', 'story'} else next(iter(requested), 'labels')
    markers = []

    def _to_seconds(val):
        """Convert timecode string or number to float seconds."""
        if isinstance(val, (int, float)):
            return float(val)
        val = str(val).strip()
        if ':' in val:
            parts = val.split(':')
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    # Resolve a clip-style marker's speaker by looking up the first transcript
    # segment whose start time falls inside the marker's [start, end] range,
    # then applying the project's speaker_names rename map (Pro diarization).
    # Returns '' when the transcript carries no speaker labels or no segment
    # overlaps.
    _transcript_segments = (project.get('transcript') or {}).get('segments') or []
    _speaker_names_map = project.get('speaker_names') or {}

    def _speaker_at_range(start_s, end_s):
        if end_s <= start_s:
            return ''
        for seg in _transcript_segments:
            try:
                ss = _to_seconds(seg.get('start', 0))
            except Exception:
                continue
            if ss >= start_s and ss < end_s:
                raw = (seg.get('speaker') or '').strip()
                if not raw:
                    return ''
                resolved = _speaker_names_map.get(raw, raw)
                if isinstance(resolved, str):
                    return resolved.strip()
                return raw
        return ''

    if 'social' in requested:
        analysis = project.get('analysis', {})
        for clip in analysis.get('social_clips', []):
            cs = _to_seconds(clip.get('start', 0))
            ce = _to_seconds(clip.get('end', 0))
            markers.append({
                'start': cs,
                'end': ce,
                'text': clip.get('title', ''),
                'note': clip.get('platform', ''),
                'color': 'green',
                'category': 'Social Clip',
                'speaker': _speaker_at_range(cs, ce),
            })

    if 'story' in requested:
        analysis = project.get('analysis', {})
        for beat in analysis.get('story_beats', []):
            start = _to_seconds(beat.get('start', 0))
            end = _to_seconds(beat.get('end', start))
            if end <= start:
                end = start + 15
            markers.append({
                'start': start,
                'end': end,
                'text': beat.get('label', ''),
                'note': beat.get('description', ''),
                'color': 'purple',
                'category': 'Story Beat',
                'speaker': _speaker_at_range(start, end),
            })

    if 'soundbites' in requested:
        analysis = project.get('analysis', {})
        for sb in analysis.get('strongest_soundbites', []):
            start = _to_seconds(sb.get('start', 0))
            end = _to_seconds(sb.get('end', start))
            if end <= start:
                end = start + 15
            markers.append({
                'start': start,
                'end': end,
                'text': (sb.get('text', '') or '')[:80],
                'note': sb.get('why', ''),
                'color': 'orange',
                'category': 'Soundbite',
                'speaker': _speaker_at_range(start, end),
            })

    if 'labels' in requested:
        color_labels = project.get('color_labels', {})
        for sec in project.get('labeled_sections', []):
            label_name = color_labels.get(sec.get('color', ''), sec.get('color', ''))
            ls = _to_seconds(sec.get('start', 0))
            le = _to_seconds(sec.get('end', 0))
            markers.append({
                'start': ls,
                'end': le,
                'text': label_name,
                'note': sec.get('text', '')[:80],
                'color': sec.get('color', 'blue'),
                'category': label_name,
                'speaker': _speaker_at_range(ls, le),
            })

    if 'transcript' in requested:
        # Emit one marker per transcript segment so editors can navigate the
        # full interview in the NLE timeline. Kept separate from the other
        # categories because it can be noisy — user opts in explicitly.
        # The marker label is the speaker, resolved through the project's
        # ``speaker_names`` map so renamed speakers (e.g. SPEAKER_00 -> "Sarah
        # Chen" via the Pro diarization rename UI) surface as the display
        # name in the NLE timeline.
        transcript = project.get('transcript') or {}
        speaker_names_map = project.get('speaker_names') or {}
        for seg in transcript.get('segments', []):
            start = _to_seconds(seg.get('start', 0))
            end = _to_seconds(seg.get('end', start))
            if end <= start:
                end = start + 1
            raw_speaker = (seg.get('speaker') or '').strip()
            display_speaker = speaker_names_map.get(raw_speaker, raw_speaker) if raw_speaker else ''
            if isinstance(display_speaker, str):
                display_speaker = display_speaker.strip()
            text = (seg.get('text') or '').strip()
            label = display_speaker or raw_speaker or 'Transcript'
            markers.append({
                'start': start,
                'end': end,
                'text': label,
                'note': text[:160],
                'color': 'blue',
                'category': 'Transcript',
                'speaker': raw_speaker or '',
            })

    # Source file + media metadata
    source_path = project.get('source_path', project.get('filepath', ''))
    media_duration = None
    if project.get('transcript') and project['transcript'].get('duration'):
        media_duration = project['transcript']['duration']

    detected_fps = get_video_framerate(source_path)
    framerate = detected_fps or body.get('framerate', 23.976)
    export_mode = body.get('mode', 'cuts')  # 'cuts', 'markers', 'both'
    width, height = get_video_resolution(source_path)
    # Embedded start timecode (DJI/Sony stamp time-of-day TC); FCP rejects
    # 0-based edits exported against such media.
    start_tc_frames = get_video_start_timecode_frames(source_path, framerate)
    total_clips = body.get('total_clips', len(markers)) or len(markers)

    if force_platform and force_platform in PLATFORMS:
        platform = force_platform
    else:
        # Allow per-export platform override; otherwise use project's editing_platform.
        platform_override = body.get('platform')
        platform = platform_override if platform_override in PLATFORMS else get_project_platform(project)

    exporter = get_exporter(platform)
    result = exporter.export_markers(
        markers,
        project_name=project['name'],
        source_path=source_path,
        media_duration=media_duration,
        framerate=framerate,
        width=width,
        height=height,
        export_type=export_type,
        exports_dir=app.config['EXPORTS_DIR'],
        export_mode=export_mode,
        total_clips=total_clips,
        start_tc_frames=start_tc_frames,
    )
    return result, exporter


@app.route('/project/<project_id>/export/fcpxml', methods=['POST'])
def export_fcpxml(project_id):
    """Export selections to the project's selected NLE format (FCPXML/Premiere/EDL).

    Honors an optional ``deliver_to`` field on the body:

      - ``'nle'``: write the file then launch the chosen NLE (Final Cut Pro /
        Resolve auto-import; Premiere reveals in Finder). Returns JSON.
      - ``'file'``: write the file then reveal it in Finder. Returns JSON.
      - omitted: legacy stream-the-file-as-an-attachment behavior, preserved
        for any third-party integration that depends on it.
    """
    project = get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    # Accept either a list (``types``) or the legacy single ``type`` string.
    # The new checkbox UI sends ``types=['labels', 'social', ...]``; callers
    # that still send ``type='all'`` expand to the full set so existing tests
    # and any third-party integrations keep working.
    body = request.json or {}
    deliver_to = str(body.get('deliver_to') or '').strip().lower()
    nle = str(body.get('nle') or '').strip().lower()
    # When the caller wants NLE delivery and explicitly chose one, force the
    # exporter to that NLE's format (so toggling Resolve in the UI sends an
    # EDL even if the project was originally configured for FCP).
    force_platform = nle if (deliver_to == 'nle' and nle in NLE_DISPLAY_NAMES) else None

    try:
        result, exporter = _build_nle_export(project, body, force_platform=force_platform)
    except Exception as e:
        app.logger.error('Export failed: %s', e)
        return jsonify({'error': f'Export failed: {e}'}), 500

    if deliver_to == 'nle':
        if nle not in NLE_DISPLAY_NAMES:
            return jsonify({'error': f'Unknown NLE: {nle!r}'}), 400
        opened_in, info = _hand_file_to_nle(
            result.file_path, nle,
            source_media_path=project.get('source_path') or project.get('filepath'),
            project_name=project.get('name') or '',
            timeline_name=os.path.splitext(result.filename)[0],
        )
        if opened_in is None:
            return jsonify({'error': info.get('error', 'NLE delivery failed'),
                            'file': result.file_path}), 500
        payload = {
            'status': 'ok', 'delivery': 'nle', 'opened_in': opened_in,
            'nle': nle, 'nle_name': NLE_DISPLAY_NAMES[nle],
            'file': result.file_path, 'filename': result.filename,
            'format_name': result.format_name,
        }
        payload.update(info)
        return jsonify(payload)

    if deliver_to == 'file':
        _reveal_in_finder(result.file_path)
        return jsonify({
            'status': 'ok', 'delivery': 'file',
            'file': result.file_path, 'filename': result.filename,
            'format_name': result.format_name,
        })

    return _exporter_response(result, project, exporter)


# Display names paired with the platform keys used everywhere else
# (``fcp`` / ``premiere`` / ``resolve``). The user-visible toast and any
# "not installed" error draw from this map.
NLE_DISPLAY_NAMES = {
    'fcp': 'Final Cut Pro',
    'premiere': 'Premiere Pro',
    'resolve': 'DaVinci Resolve',
}


# CFBundleIdentifier for each NLE. Spotlight indexes apps by bundle ID
# regardless of install path, so mdfind is the canonical way to locate an
# NLE on macOS (vs. guessing /Applications/<Name>.app, which breaks for
# users who install to ~/Applications, Setapp, external volumes, or
# year-versioned Adobe directories).
_NLE_BUNDLE_IDS = {
    'fcp':      ('com.apple.FinalCut',),
    'premiere': ('com.adobe.PremierePro',),
    # Free and Studio variants of Resolve register different bundle IDs.
    'resolve':  ('com.blackmagic-design.DaVinciResolveStudio',
                 'com.blackmagic-design.DaVinciResolve'),
}

# Hardcoded fallback paths for the (rare) case where Spotlight is
# disabled on the volume the NLE lives on, or mdfind isn't on PATH.
_NLE_FALLBACK_PATHS = {
    'fcp': ('/Applications/Final Cut Pro.app',),
    'resolve': ('/Applications/DaVinci Resolve/DaVinci Resolve.app',
                '/Applications/DaVinci Resolve Studio/DaVinci Resolve Studio.app'),
    'premiere': (
        '/Applications/Adobe Premiere Pro 2026/Adobe Premiere Pro 2026.app',
        '/Applications/Adobe Premiere Pro 2025/Adobe Premiere Pro 2025.app',
        '/Applications/Adobe Premiere Pro 2024/Adobe Premiere Pro 2024.app',
        '/Applications/Adobe Premiere Pro.app',
    ),
}

# Per-process cache. Paths don't change while we're running.
_nle_path_cache: dict[str, str | None] = {}


def _mdfind_app_by_bundle_id(bundle_id: str) -> list[str]:
    """Return all .app paths whose CFBundleIdentifier matches ``bundle_id``.

    Empty list on no match or if mdfind fails (Spotlight off, sandbox, …).
    Filters out stale hits — Spotlight occasionally returns indexed paths
    that no longer exist on disk.
    """
    try:
        out = subprocess.check_output(
            ['mdfind', f"kMDItemCFBundleIdentifier == '{bundle_id}'"],
            text=True,
            timeout=3,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return []
    return [line for line in out.strip().splitlines() if line and os.path.isdir(line)]


def _rank_app_paths(paths: list[str]) -> list[str]:
    """Order discovered .app paths most-preferred first.

    Heuristic: a copy under /Applications beats one under ~/Applications,
    which beats anything else (Setapp subdirs, external volumes). Users
    who keep multiple copies typically want the system-wide one driven.
    """
    def rank(p: str) -> int:
        if p.startswith('/Applications/'):
            return 0
        if '/Applications/' in p and 'Setapp' not in p:
            return 1
        if 'Setapp' in p:
            return 3
        return 2
    return sorted(paths, key=rank)


def _find_nle_app_path(nle: str):
    """Return the on-disk .app path for the chosen NLE, or ``None`` if not installed.

    Locates the app by CFBundleIdentifier via mdfind so users who install
    FCP/Premiere/Resolve outside /Applications (~/Applications, Setapp,
    external volumes, year-versioned Adobe dirs) still get a working
    export. Falls back to a known-path list if Spotlight returns nothing.
    Result is cached per process — these paths don't move at runtime.
    """
    if nle in _nle_path_cache:
        return _nle_path_cache[nle]

    found: str | None = None
    for bundle_id in _NLE_BUNDLE_IDS.get(nle, ()):
        hits = _mdfind_app_by_bundle_id(bundle_id)
        if hits:
            found = _rank_app_paths(hits)[0]
            break

    if found is None:
        for candidate in _NLE_FALLBACK_PATHS.get(nle, ()):
            if os.path.isdir(candidate):
                found = candidate
                break

    if found is None:
        app.logger.warning(
            "NLE %r not found. bundle_ids=%s fallback_paths=%s",
            nle, _NLE_BUNDLE_IDS.get(nle, ()), _NLE_FALLBACK_PATHS.get(nle, ()),
        )

    _nle_path_cache[nle] = found
    return found


def _nle_bundle_id(nle: str) -> str | None:
    """Return the primary CFBundleIdentifier for ``nle``, or None if unknown."""
    ids = _NLE_BUNDLE_IDS.get(nle)
    return ids[0] if ids else None


def _hand_file_to_nle(file_path: str, nle: str, *,
                      source_media_path: str | None = None,
                      project_name: str = '',
                      timeline_name: str = ''):
    """Launch the chosen NLE with the export file, or reveal in Finder for NLEs
    that lack a file-association auto-import.

    Returns ``(opened_in, info)`` where ``info`` is always a dict. Keys:
      - ``error``: string if delivery failed; absent on success.
      - For Resolve fallback: ``setup_required: True`` plus ``reason``
        and ``hint`` strings the frontend uses to render the setup modal.
      - For Resolve scripted success: ``timeline_name`` (which timeline
        the user lands on inside Resolve).

    ``opened_in`` values:
      ``'app'``       — NLE auto-imports on Finder open (FCP only).
      ``'scripted'``  — Resolve auto-imported via the scripting API.
      ``'finder'``    — file revealed in Finder for manual drag-in.
      ``'finder+app'`` — file revealed AND app focused (Resolve fallback).
      ``None``        — failure (info['error'] is set).

    Resolve path: try scripting auto-import first (Phase 2). If the
    scripting API isn't reachable, fall back to Phase 1's reveal-in-Finder
    + app-launch, and tag the response with ``setup_required`` so the
    frontend can prompt the user to enable External Scripting.
    """
    app_path = _find_nle_app_path(nle)
    if not app_path:
        return None, {'error': f"{NLE_DISPLAY_NAMES[nle]} not found on this Mac. If it is installed, try moving it to /Applications and re-launching, or contact support."}
    try:
        if nle == 'premiere':
            subprocess.Popen(['open', '-R', file_path])
            return 'finder', {}
        if nle == 'resolve':
            return _hand_file_to_resolve(
                file_path,
                app_path=app_path,
                source_media_path=source_media_path,
                project_name=project_name,
                timeline_name=timeline_name,
            )
        # FCP: prefer launching by bundle ID so Launch Services resolves
        # the install location (works even if the user has FCP outside
        # /Applications, or has multiple copies). Falls back to -a <path>
        # if the bundle ID isn't mapped (shouldn't happen for fcp).
        bundle_id = _nle_bundle_id(nle)
        if bundle_id:
            subprocess.Popen(['open', '-b', bundle_id, file_path])
        else:
            subprocess.Popen(['open', '-a', app_path, file_path])
        return 'app', {}
    except Exception as e:
        return None, {'error': f'Could not launch {NLE_DISPLAY_NAMES[nle]}: {e}'}


def _hand_file_to_resolve(file_path: str, *,
                          app_path: str,
                          source_media_path: str | None,
                          project_name: str,
                          timeline_name: str):
    """Resolve-specific delivery. Tries scripting auto-import first; falls
    back to reveal-in-Finder + app-focus if scripting isn't available.

    Returns the same shape as _hand_file_to_nle.
    """
    # Lazy import — keeps the module from loading on machines without
    # Resolve, and avoids a startup cost.
    try:
        from exporters import resolve_import
    except Exception as e:
        # Module itself failed to import (shouldn't happen — it's pure
        # stdlib). Fall back to reveal+open.
        app.logger.warning('resolve_import unavailable: %s', e)
        subprocess.Popen(['open', '-R', file_path])
        subprocess.Popen(['open', '-a', app_path])
        return 'finder+app', {}

    result = resolve_import.import_timeline(
        file_path,
        source_media_path=source_media_path or None,
        project_name=project_name or 'Doza Assist Import',
        timeline_name=timeline_name or os.path.splitext(os.path.basename(file_path))[0],
    )

    if result.ok:
        # Bring Resolve to the front so the imported timeline is visible.
        # The scripting API does the import but doesn't activate the
        # window — the user clicked Export, they expect to see something.
        subprocess.Popen(['open', '-a', app_path])
        return 'scripted', {'timeline_name': result.timeline_name}

    # Scripting failed. Fall back to Phase-1 behavior so the file still
    # ends up somewhere visible. Surface the setup hint so the frontend
    # can show a remediation modal (especially for scripting_disabled).
    subprocess.Popen(['open', '-R', file_path])
    subprocess.Popen(['open', '-a', app_path])
    return 'finder+app', {
        'setup_required': True,
        'reason': result.reason,
        'hint': result.hint,
    }


def _reveal_in_finder(file_path: str):
    """Best-effort reveal of ``file_path`` in Finder. Failures are non-fatal."""
    try:
        subprocess.Popen(['open', '-R', file_path])
    except Exception as e:
        app.logger.error('Reveal in Finder failed: %s', e)


@app.route('/export/send-to-nle', methods=['POST'])
def send_to_nle():
    """Generate the export for ``project_id`` and hand it to the chosen NLE.

    Body: ``{"project_id": "...", "nle": "fcp"|"premiere"|"resolve",
              "export_type": "selects"|"story_builder", ...}``.

    For ``selects`` the remaining keys (``types``, ``framerate``, ``mode``, ...)
    are forwarded to the same pipeline ``/export/fcpxml`` uses, so the Export
    tab's current settings carry over. For ``story_builder`` the body must
    include ``clips`` and ``story_title``.

    The export format always matches ``nle`` (FCPXML for fcp, Premiere XML for
    premiere, EDL for resolve) — the user-selected NLE in the UI is the source
    of truth, not whatever the project is configured for.
    """
    body = request.json or {}
    project_id = body.get('project_id')
    # project_id arrives in the JSON body here (not the URL), so the
    # before_request guard doesn't see it — validate explicitly (SEC-02).
    if not validate_project_id(project_id):
        return jsonify({'error': 'Invalid project_id'}), 400
    export_type = str(body.get('export_type') or 'selects').strip().lower()

    # Multicam round-trip is FCP-specific (it preserves FCP's multicam /
    # sync-clip container) — force the target NLE to fcp regardless of what
    # the selector says, so the file lands in the only editor that can use it.
    if export_type == 'multicam':
        nle = 'fcp'
    else:
        nle = str(body.get('nle') or '').strip().lower()

    if nle not in NLE_DISPLAY_NAMES:
        return jsonify({'error': f'Unknown NLE: {nle!r}'}), 400

    project = get_project(project_id) if project_id else None
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    app_path = _find_nle_app_path(nle)
    if not app_path:
        return jsonify({
            'error': f"{NLE_DISPLAY_NAMES[nle]} not found on this Mac. If it is installed, try moving it to /Applications and re-launching, or contact support."
        }), 404

    file_path = None
    filename = None
    format_name = None
    try:
        if export_type == 'story_builder':
            if not body.get('clips'):
                return jsonify({'error': 'No clips in story builder'}), 400
            result, _ = _build_nle_story_export(project, body, force_platform=nle)
            file_path, filename, format_name = result.file_path, result.filename, result.format_name
        elif export_type == 'multicam':
            file_path, filename, _mode = _build_nle_multicam_export(project, body)
            format_name = 'FCPXML'
        else:
            result, _ = _build_nle_export(project, body, force_platform=nle)
            file_path, filename, format_name = result.file_path, result.filename, result.format_name
    except MulticamExportError as e:
        return jsonify({'error': str(e)}), e.status
    except Exception as e:
        app.logger.error('Send-to-NLE export failed: %s', e)
        return jsonify({'error': f'Export failed: {e}'}), 500

    # FCP and Resolve auto-import when launched with the file; Premiere's CLI
    # auto-import is unreliable, so we just reveal the file in Finder and let
    # the editor drag it into an open project.
    try:
        if nle == 'premiere':
            subprocess.Popen(['open', '-R', file_path])
            opened_in = 'finder'
        else:
            subprocess.Popen(['open', '-a', app_path, file_path])
            opened_in = 'app'
    except Exception as e:
        app.logger.error('Launching %s failed: %s', NLE_DISPLAY_NAMES[nle], e)
        return jsonify({
            'error': f'Could not launch {NLE_DISPLAY_NAMES[nle]}: {e}',
            'file': file_path,
        }), 500

    return jsonify({
        'status': 'ok',
        'nle': nle,
        'nle_name': NLE_DISPLAY_NAMES[nle],
        'opened_in': opened_in,
        'file': file_path,
        'filename': filename,
        'format_name': format_name,
    })


def _project_selects_for_fcpxml(project: dict, source, story_build_clips=None):
    """Build ``Select`` objects from a project, pulling from the requested buckets.

    ``source`` may be a single string or a list of strings. Recognised values:
      - ``'client_selects'``: editor-chosen labels (default)
      - ``'social'``: AI-identified social clips
      - ``'story'``: AI-identified story beats
      - ``'soundbites'``: AI-identified strongest soundbites
      - ``'story_build'``: a Story Builder build's ordered clips, supplied as
        ``story_build_clips`` (each ``{start_time, end_time, title,
        editorial_note, order}``). The round-trip Story Builder export uses
        this — clips come from the request body, not from project data.
      - ``'all'``: everything combined, in chronological order

    Speaker is resolved per-select via the transcript segments + the project's
    ``speaker_names`` rename map (so renamed pyannote labels surface in the
    round-trip FCPXML's ``<note>`` the same way they do in the standard
    direct-media export from Step 6).
    """
    if isinstance(source, (list, tuple)):
        sources = set(source)
    else:
        sources = {source}
    if 'all' in sources:
        sources = {'client_selects', 'social', 'story', 'soundbites'}
    def _to_seconds(val):
        if isinstance(val, (int, float)):
            return float(val)
        val = str(val or '').strip()
        if ':' in val:
            parts = val.split(':')
            try:
                if len(parts) == 3:
                    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                if len(parts) == 2:
                    return float(parts[0]) * 60 + float(parts[1])
            except ValueError:
                pass
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    def _kind_for_color(color: str) -> str:
        # Greenish/completion → strong; red → question; everything else → standard.
        c = (color or '').lower()
        if c in ('green', 'purple'):
            return 'strong'
        if c in ('red',):
            return 'question'
        return 'standard'

    # Speaker lookup helper — overlap a clip's time range against transcript
    # segments and resolve the first hit through the speaker_names map. Cheap
    # linear scan; transcript segment lists are typically ~150-1500 items and
    # we only do this once per select, not per segment.
    transcript_segments = (project.get('transcript') or {}).get('segments') or []
    speaker_names_map = project.get('speaker_names') or {}

    def _speaker_for_range(start_s: float, end_s: float) -> str:
        if end_s <= start_s:
            return ''
        for seg in transcript_segments:
            try:
                ss = _to_seconds(seg.get('start', 0))
            except Exception:
                continue
            if ss >= start_s and ss < end_s:
                raw = (seg.get('speaker') or '').strip()
                if not raw:
                    return ''
                resolved = speaker_names_map.get(raw, raw)
                return resolved.strip() if isinstance(resolved, str) else raw
        return ''

    selects: list[Select] = []

    if 'client_selects' in sources:
        for sec in project.get('labeled_sections') or []:
            color_labels = project.get('color_labels', {})
            label_name = color_labels.get(sec.get('color', ''), sec.get('color', '')) or 'Select'
            cs = _to_seconds(sec.get('start', 0))
            ce = _to_seconds(sec.get('end', 0))
            selects.append(Select(
                start_seconds=cs,
                end_seconds=ce,
                label=label_name,
                note=(sec.get('text') or '')[:80],
                kind=_kind_for_color(sec.get('color', '')),
                speaker=_speaker_for_range(cs, ce),
            ))

    if 'social' in sources:
        analysis = project.get('analysis') or {}
        for clip in analysis.get('social_clips') or []:
            cs = _to_seconds(clip.get('start', 0))
            ce = _to_seconds(clip.get('end', 0))
            selects.append(Select(
                start_seconds=cs,
                end_seconds=ce,
                label=clip.get('title') or 'Social Clip',
                note=clip.get('platform', ''),
                kind='strong',
                speaker=_speaker_for_range(cs, ce),
            ))

    if 'story' in sources:
        analysis = project.get('analysis') or {}
        for beat in analysis.get('story_beats') or []:
            start = _to_seconds(beat.get('start', 0))
            end = _to_seconds(beat.get('end', start))
            if end <= start:
                end = start + 15
            selects.append(Select(
                start_seconds=start, end_seconds=end,
                label=beat.get('label') or 'Story Beat',
                note=(beat.get('description') or '')[:120],
                kind='strong',
                speaker=_speaker_for_range(start, end),
            ))

    if 'soundbites' in sources:
        analysis = project.get('analysis') or {}
        for sb in analysis.get('strongest_soundbites') or []:
            start = _to_seconds(sb.get('start', 0))
            end = _to_seconds(sb.get('end', start))
            if end <= start:
                end = start + 15
            selects.append(Select(
                start_seconds=start, end_seconds=end,
                label=(sb.get('text') or 'Soundbite')[:60],
                note=(sb.get('why') or '')[:120],
                kind='strong',
                speaker=_speaker_for_range(start, end),
            ))

    if 'story_build' in sources:
        # Story Builder clips come from the request body, not from project
        # data — the build itself is stored under a build_id and the
        # frontend ships the full ordered clip list at export time. Preserve
        # order via the original 'order' field (or input index as fallback)
        # so the round-trip timeline reads narratively, not chronologically.
        if story_build_clips:
            ordered = list(enumerate(story_build_clips))
            ordered.sort(key=lambda iv: iv[1].get('order', iv[0]))
            for idx, clip in ordered:
                cs = _to_seconds(clip.get('start_time', clip.get('start', 0)))
                ce = _to_seconds(clip.get('end_time', clip.get('end', 0)))
                if ce <= cs:
                    continue
                selects.append(Select(
                    start_seconds=cs,
                    end_seconds=ce,
                    label=(clip.get('title') or f'Clip {idx + 1}')[:80],
                    note=(clip.get('editorial_note') or '')[:160],
                    kind='strong',
                    speaker=_speaker_for_range(cs, ce),
                ))

    return selects


class MulticamExportError(Exception):
    """Raised by ``_build_nle_multicam_export`` with an HTTP-status hint."""
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _build_nle_multicam_export(project, body):
    """Round-trip selects back into FCP-importable FCPXML.

    Returns ``(out_path, filename, mode)``. Always FCPXML — multicam round-trip
    is FCP-specific by design (it reuses the original multicam / sync-clip
    container), so there's no platform branch.
    """
    body = body or {}
    fcpxml_source = project.get('fcpxml_source')
    if not fcpxml_source:
        raise MulticamExportError(
            'This project was not imported from an FCPXML. '
            'Use the standard FCPX export instead.'
        )

    stored_path = fcpxml_source.get('stored_fcpxml_path')
    if not stored_path or not os.path.isfile(stored_path):
        raise MulticamExportError(
            'The original FCPXML file is no longer available in the project directory.'
        )

    mode = body.get('mode', 'selects_project')
    sources = body.get('sources') or [body.get('source', 'client_selects')]
    # Story Builder builds ship their clip list inline (the build's narrative
    # order isn't stored on the project — it lives in the request body).
    # When sources includes 'story_build', the frontend must also pass the
    # clip list. Preserve_order keeps that narrative ordering across the
    # round-trip; without it _iter_selects would sort by source in-point and
    # the story would read chronologically instead of the way the editor
    # arranged it.
    story_build_clips = body.get('story_build_clips') or body.get('clips')
    preserve_order = bool(body.get('preserve_order')) or (
        isinstance(sources, (list, tuple, set)) and 'story_build' in set(sources)
    )

    try:
        parsed = parse_fcpxml(stored_path)
    except ParseError as e:
        raise MulticamExportError(f'Could not re-read stored FCPXML: {e}', status=500)

    selects = _project_selects_for_fcpxml(
        project, sources, story_build_clips=story_build_clips,
    )
    if not selects:
        raise MulticamExportError(
            f'No selects available for sources={sources!r}. '
            'Pick a source with content, or add clip labels first.'
        )

    try:
        if mode == 'markers_timeline':
            output = write_markers_on_timeline(parsed, selects)
            suffix = 'Doza Notes'
        else:
            output = write_selects_as_new_project(
                parsed, selects, preserve_order=preserve_order,
            )
            suffix = 'Doza Selects'
    except WriterError as e:
        raise MulticamExportError(f'Export failed: {e}')

    # Story Builder exports get a more specific filename suffix.
    if preserve_order and story_build_clips:
        story_title = (body.get('story_title') or '').strip()
        if story_title:
            suffix = f"{story_title}"
    base = (project.get('name') or 'Project').strip().replace('/', '-')
    filename = f"{base} - {suffix}.fcpxml"
    exports_dir = app.config['EXPORTS_DIR']
    os.makedirs(exports_dir, exist_ok=True)
    out_path = os.path.join(exports_dir, filename)
    with open(out_path, 'wb') as fh:
        fh.write(output)

    return out_path, filename, mode


@app.route('/project/<project_id>/export/fcpxml-multicam', methods=['POST'])
def export_fcpxml_multicam(project_id):
    """Round-trip selects back into FCP-importable FCPXML via the multicam writer.

    Only available on projects that were ingested from an FCPXML (i.e. have a
    ``fcpxml_source`` metadata block). Two modes:

      - ``selects_project``: emits a new project whose spine is the selects
        laid end-to-end, each routed by its source segment kind — a fresh
        mc-clip for multicam, or a deep copy of the original sync-clip /
        asset-clip for synced / plain single-cam footage.
      - ``markers_timeline``: emits the original timeline with markers injected
        at each select's in-point.

    Honors ``deliver_to`` like ``/export/fcpxml``: ``'nle'`` always launches
    Final Cut Pro (the round-trip writer is FCP-only by construction);
    ``'file'`` reveals in Finder; omitted preserves the legacy attachment
    download.
    """
    project = get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    body = request.json or {}
    deliver_to = str(body.get('deliver_to') or '').strip().lower()

    try:
        out_path, filename, mode = _build_nle_multicam_export(project, body)
    except MulticamExportError as e:
        return jsonify({'error': str(e)}), e.status

    if deliver_to == 'nle':
        # Round-trip is FCP-only — ignore whatever NLE the user has selected
        # and always hand the file to Final Cut Pro.
        opened_in, info = _hand_file_to_nle(out_path, 'fcp')
        if opened_in is None:
            return jsonify({'error': info.get('error', 'NLE delivery failed'),
                            'file': out_path}), 500
        return jsonify({
            'status': 'ok', 'delivery': 'nle', 'opened_in': opened_in,
            'nle': 'fcp', 'nle_name': NLE_DISPLAY_NAMES['fcp'],
            'file': out_path, 'filename': filename,
            'format_name': 'FCPXML', 'mode': mode,
        })

    if deliver_to == 'file':
        _reveal_in_finder(out_path)
        return jsonify({
            'status': 'ok', 'delivery': 'file',
            'file': out_path, 'filename': filename,
            'format_name': 'FCPXML', 'mode': mode,
        })

    response = send_file(out_path, as_attachment=True, download_name=filename)
    response.headers['X-Export-Format'] = 'FCPXML'
    response.headers['X-Export-Platform'] = 'Final Cut Pro'
    response.headers['X-Export-Extension'] = '.fcpxml'
    response.headers['X-Export-Mode'] = mode
    return response


@app.route('/review/<project_id>')
def client_review(project_id):
    """Legacy review portal — redirect to shared view."""
    return redirect(f'/share/{project_id}')


@app.route('/project/<project_id>/share-settings', methods=['GET', 'POST'])
def save_share_settings(project_id):
    """Get or save which tabs are visible in the shared view."""
    project = get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    if request.method == 'GET':
        default_tabs = {'transcript': True, 'clips': True, 'analysis': True, 'chat': True, 'story': True, 'export': True}
        return jsonify({'shared_tabs': project.get('shared_tabs', default_tabs)})

    data = request.json or {}
    project['shared_tabs'] = data.get('shared_tabs', {})
    save_project(project_id, project)
    return jsonify({'status': 'saved'})


@app.route('/share/<project_id>')
def shared_view(project_id):
    """Shared project view — full project experience, read-only."""
    project = get_project(project_id)
    if not project or not project.get('transcript'):
        return render_template('review_unavailable.html')

    source_exists, _ = check_source_file(project)
    project['source_exists'] = source_exists

    # Normalize small-model field drift so the read-only view renders correctly.
    if project.get('analysis'):
        from ai_analysis import normalize_analysis
        project['analysis'] = normalize_analysis(project['analysis'])

    all_projects = [p for p in list_projects() if p.get('transcript')]
    project_ids = [project_id]
    projects_meta = [{'id': project['id'], 'name': project.get('name', 'Untitled'), 'color': 'accent'}]

    paragraphs = []
    if project.get('transcript') and project['transcript'].get('segments'):
        paragraphs = group_into_paragraphs(project['transcript']['segments'])
        for para in paragraphs:
            para['project_id'] = project['id']
            para['project_name'] = project.get('name', 'Untitled')
            para['project_color'] = 'accent'

    source_ext = os.path.splitext(project.get('source_path', '') or '')[1].lower()
    is_video = source_ext in ('.mp4', '.mov', '.mxf', '.avi', '.mkv')

    # Tab visibility — default all on
    default_tabs = {'transcript': True, 'clips': True, 'analysis': True, 'chat': True, 'story': True, 'export': True}
    shared_tabs = project.get('shared_tabs', default_tabs)

    return render_template('project.html',
                           project=project,
                           projects=[project],
                           projects_meta=projects_meta,
                           all_projects=all_projects,
                           active_ids=project_ids,
                           paragraphs=paragraphs,
                           is_multi=False,
                           is_shared=True,
                           is_video=is_video,
                           shared_tabs=shared_tabs,
                           segment_vectors=load_segment_vectors(project_id))


@app.route('/project/<project_id>/clear', methods=['POST'])
def clear_transcript(project_id):
    """Clear transcript and analysis but keep the project and source file."""
    project = get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    project['transcript'] = None
    project['analysis'] = None
    project['client_selects'] = []
    project['social_clips'] = []
    project['status'] = 'uploaded'
    project.pop('error', None)
    save_project(project_id, project)

    # Remove extracted audio (will be re-extracted on next transcribe)
    project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
    audio_wav = os.path.join(project_dir, 'audio.wav')
    if os.path.exists(audio_wav):
        os.remove(audio_wav)

    return jsonify({'status': 'cleared'})


@app.route('/project/<project_id>/retranscribe', methods=['POST'])
def retranscribe(project_id):
    """Update language and re-run transcription."""
    project = get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    data = request.get_json() or {}
    language = data.get('language', project.get('language', 'en')).strip()
    project['language'] = language

    # Clear existing transcript/analysis
    project['transcript'] = None
    project['analysis'] = None
    project['client_selects'] = []
    project['social_clips'] = []
    project['status'] = 'uploaded'
    project.pop('error', None)
    save_project(project_id, project)

    # Remove extracted audio so it gets re-extracted
    project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
    audio_wav = os.path.join(project_dir, 'audio.wav')
    if os.path.exists(audio_wav):
        os.remove(audio_wav)

    # Drop the cached paragraph_index + segment_vectors — they reference the
    # OLD transcript text. Letting them survive a retranscribe means the
    # /transcribe handler's idempotent skip would keep serving stale TF-IDF
    # results and the chat would look at the wrong paragraphs forever.
    for cache_name in ('paragraph_index.json', 'segment_vectors.json'):
        p = os.path.join(project_dir, cache_name)
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    return jsonify({'status': 'cleared', 'language': language})


@app.route('/project/<project_id>/delete', methods=['POST'])
def delete_project(project_id):
    """Delete a project and its files."""
    # Resolve + confine the path before rmtree: safe_project_dir rejects a
    # malformed id and any path that escapes PROJECTS_DIR (incl. via symlink),
    # so the rmtree below can only ever target a direct child of PROJECTS_DIR.
    # (The before_request guard already 404s a malformed URL id; this is the
    # belt-and-suspenders check at the destructive sink itself — SEC-02.)
    project_dir = safe_project_dir(project_id)
    if project_dir is None:
        return jsonify({'error': 'Project not found'}), 404
    if os.path.exists(project_dir):
        shutil.rmtree(project_dir)
    return jsonify({'status': 'deleted'})


@app.route('/project/<project_id>/rename', methods=['POST'])
def rename_project(project_id):
    """Rename a project."""
    project = get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    name = (request.json or {}).get('name', '').strip()
    if not name:
        return jsonify({'error': 'Name cannot be empty'}), 400

    old_name = project.get('name', 'Project')
    project['name'] = name
    save_project(project_id, project)
    if old_name != name:
        log_activity(project_id, 'renamed', f"Renamed \"{old_name}\" → \"{name}\"")
    return jsonify({'status': 'renamed', 'name': name})


@app.route('/project/<project_id>/update-speakers', methods=['POST'])
def update_speakers(project_id):
    """Update speaker label assignments after transcription (bulk rename).

    On projects that have been diarized, the canonical rename surface is
    the diarization speakers sidebar (writes to ``meta["speaker_names"]``,
    leaves the raw SPEAKER_NN labels on the segments untouched). The OSS
    rename collapses segments by raw label, which silently destroys the
    diarization assignment when multiple renames target the same string
    (we hit this in production: five SPEAKER_NN labels collapsed to one
    name across all 1167 segments). Defense-in-depth: refuse the write
    server-side so the bad state cannot be reached even if the
    front-end lockout races. Returns 409 with a message the renderer
    surfaces as a toast.
    """
    project = get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    diarization_status = (project.get('diarization') or {}).get('status')
    if diarization_status == 'done':
        return jsonify({
            'error': 'diarization_active',
            'message': (
                'This project has been diarized. Rename speakers from the '
                'Speakers sidebar (the rows under "Speakers") instead of the '
                'transcript labels.'
            ),
        }), 409

    mapping = request.json.get('mapping', {})
    transcript = project.get('transcript', {})
    segments = transcript.get('segments', [])

    for seg in segments:
        old_speaker = seg.get('speaker', '')
        if old_speaker in mapping:
            seg['speaker'] = mapping[old_speaker]

    project['transcript']['segments'] = segments
    save_project(project_id, project)
    return jsonify({'status': 'updated'})


@app.route('/project/<project_id>/update-speaker-range', methods=['POST'])
def update_speaker_range(project_id):
    """Update speaker for all segments in a time range (click-to-assign)."""
    project = get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    # Same reasoning as update_speakers above: on a diarized project the
    # raw SPEAKER_NN labels are the canonical assignment and must not be
    # overwritten by per-range click-to-assign.
    diarization_status = (project.get('diarization') or {}).get('status')
    if diarization_status == 'done':
        return jsonify({
            'error': 'diarization_active',
            'message': (
                'This project has been diarized. Rename speakers from the '
                'Speakers sidebar instead of clicking the transcript.'
            ),
        }), 409

    data = request.json or {}
    range_start = float(data.get('start', 0))
    range_end = float(data.get('end', 0))
    new_speaker = data.get('speaker', '')

    if not new_speaker:
        return jsonify({'error': 'No speaker specified'}), 400

    transcript = project.get('transcript', {})
    segments = transcript.get('segments', [])

    count = 0
    for seg in segments:
        # Segment overlaps with the range
        if seg['start'] >= range_start - 0.1 and seg['end'] <= range_end + 0.1:
            seg['speaker'] = new_speaker
            count += 1

    project['transcript']['segments'] = segments
    save_project(project_id, project)
    return jsonify({'status': 'updated'})


# ── Story Builder ──────────────────────────────────────────────────

@app.route('/project/<project_id>/story/build', methods=['POST'])
def story_build(project_id):
    """Build a narrative sequence from the transcript using AI."""
    project = get_project(project_id)
    if not project or not project.get('transcript'):
        return jsonify({'error': 'No transcript available'}), 400

    data = request.json or {}
    message = data.get('message', '').strip()
    profile_id = data.get('profile_id')  # session-only override from the UI
    if not message:
        return jsonify({'error': 'No story description provided'}), 400

    try:
        from ai_analysis import build_story, generate_segment_vectors
        # Prefer pre-generated segment vectors — much more consistent across
        # runs and the only reliable path on long (>15 min) interviews.
        segment_vectors = load_segment_vectors(project_id)
        project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
        if not segment_vectors:
            # Backfill the vector menu on demand. This used to be generated by
            # /analyze but the pre-chunking version silently failed on long
            # transcripts, leaving /story/build to fall through to a
            # raw-transcript path that also can't cope with 100-minute inputs.
            # Rather than push the user back to re-run analysis, regenerate
            # here and persist for reuse.
            try:
                segment_vectors = generate_segment_vectors(
                    project['transcript'],
                    project_name=project.get('name', 'Interview'),
                )
                if segment_vectors:
                    os.makedirs(project_dir, exist_ok=True)
                    vectors_path = os.path.join(project_dir, 'segment_vectors.json')
                    with open(vectors_path, 'w') as f:
                        json.dump(segment_vectors, f, indent=2)
            except Exception as ve:
                print(f"[story build] on-demand vector generation failed: {ve}")

        result = build_story(
            project['transcript'],
            message=message,
            project_name=project.get('name', 'Interview'),
            segment_vectors=segment_vectors or None,
            profile_id=profile_id,
        )

        clips = result.get('clips') or []
        if not clips:
            # The model returned a valid shell (or nothing) but zero clips —
            # signalling it couldn't commit to a narrative from what it saw.
            # Don't persist the empty build; surface a friendly error instead.
            return jsonify({
                'error': (
                    "The AI returned 0 clips for this prompt. This usually means "
                    "the transcript is long enough to overwhelm the model. Try a "
                    "shorter, more specific prompt, or switch to a larger Gemma "
                    "variant in AI Model settings."
                ),
            }), 500

        # Save the build to story_builds.json
        builds_path = os.path.join(project_dir, 'story_builds.json')

        builds = []
        if os.path.exists(builds_path):
            with open(builds_path, 'r') as f:
                builds = json.load(f)

        build_entry = {
            'id': str(uuid.uuid4())[:8],
            'prompt': message,
            'created_at': datetime.now().isoformat(),
            'story_title': result.get('story_title', 'Untitled'),
            'target_duration': result.get('target_duration', ''),
            'reasoning': result.get('reasoning', ''),
            'clips': clips,
        }
        builds.append(build_entry)

        with open(builds_path, 'w') as f:
            json.dump(builds, f, indent=2)

        clip_count = len(build_entry.get('clips', []))
        log_activity(
            project_id, 'story_built',
            f"Story \"{build_entry['story_title']}\" built · {clip_count} clip{'s' if clip_count != 1 else ''}",
        )
        return jsonify({'status': 'built', 'build': build_entry})
    except Exception as e:
        from ai_providers import ProviderError
        if isinstance(e, ProviderError):
            return _provider_error_response(e)
        return jsonify({'error': str(e)}), 500


@app.route('/project/<project_id>/story/builds')
def story_list(project_id):
    """List all story builds for a project."""
    project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
    builds_path = os.path.join(project_dir, 'story_builds.json')

    if not os.path.exists(builds_path):
        return jsonify({'builds': []})

    with open(builds_path, 'r') as f:
        builds = json.load(f)

    return jsonify({'builds': builds})


@app.route('/project/<project_id>/story/builds/<build_id>', methods=['PUT'])
def story_update(project_id, build_id):
    """Update or create a story build (upsert).

    If the build_id already exists, update it. If not, create a new entry.
    This supports both the Story Builder's edit-in-place flow and the Story
    Brief's "Send to Story Builder" flow which creates a build client-side
    and needs to persist it.
    """
    project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
    builds_path = os.path.join(project_dir, 'story_builds.json')

    builds = []
    if os.path.exists(builds_path):
        try:
            with open(builds_path, 'r') as f:
                builds = json.load(f)
        except (OSError, json.JSONDecodeError):
            builds = []

    data = request.json or {}

    # Try to find and update an existing build
    for i, b in enumerate(builds):
        if b['id'] == build_id:
            if 'clips' in data:
                builds[i]['clips'] = data['clips']
            if 'story_title' in data:
                builds[i]['story_title'] = data['story_title']
            with open(builds_path, 'w') as f:
                json.dump(builds, f, indent=2)
            return jsonify({'status': 'updated', 'build': builds[i]})

    # Build not found — create it (upsert)
    new_build = {
        'id': build_id,
        'story_title': data.get('story_title', 'Untitled'),
        'clips': data.get('clips', []),
        'created_at': datetime.now().isoformat(),
    }
    builds.append(new_build)
    with open(builds_path, 'w') as f:
        json.dump(builds, f, indent=2)
    return jsonify({'status': 'created', 'build': new_build}), 201


@app.route('/project/<project_id>/story/builds/<build_id>', methods=['DELETE'])
def story_delete(project_id, build_id):
    """Delete a story build."""
    project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
    builds_path = os.path.join(project_dir, 'story_builds.json')

    if not os.path.exists(builds_path):
        return jsonify({'error': 'No builds found'}), 404

    with open(builds_path, 'r') as f:
        builds = json.load(f)

    deleted_title = next((b.get('story_title', 'Story') for b in builds if b['id'] == build_id), 'Story')
    builds = [b for b in builds if b['id'] != build_id]

    with open(builds_path, 'w') as f:
        json.dump(builds, f, indent=2)

    log_activity(project_id, 'story_deleted', f"Story \"{deleted_title}\" deleted")
    return jsonify({'status': 'deleted'})


def _build_nle_story_export(project, body, force_platform=None):
    """Build a story-timeline export for ``project`` from a request body.

    Mirrors :func:`_build_nle_export` for assembled story sequences: the
    ``/story/export`` route streams the file back, ``/export/send-to-nle``
    hands it to the chosen NLE. Returns ``(result, exporter)``.
    """
    body = body or {}
    clips = body.get('clips') or []
    story_title = body.get('story_title', 'Story')

    def _to_seconds(val):
        if isinstance(val, (int, float)):
            return float(val)
        val = str(val).strip()
        if ':' in val:
            parts = val.split(':')
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    markers = []
    for i, clip in enumerate(clips):
        markers.append({
            'start': _to_seconds(clip.get('start_time', 0)),
            'end': _to_seconds(clip.get('end_time', 0)),
            'text': clip.get('title', 'Clip'),
            'note': clip.get('editorial_note', ''),
            '_order': clip.get('order', i),
        })

    source_path = project.get('source_path', project.get('filepath', ''))
    media_duration = None
    if project.get('transcript') and project['transcript'].get('duration'):
        media_duration = project['transcript']['duration']

    width, height = get_video_resolution(source_path)
    detected_fps = get_video_framerate(source_path)
    framerate = detected_fps or body.get('framerate', 23.976)
    start_tc_frames = get_video_start_timecode_frames(source_path, framerate)

    if force_platform and force_platform in PLATFORMS:
        platform = force_platform
    else:
        platform_override = body.get('platform')
        platform = platform_override if platform_override in PLATFORMS else get_project_platform(project)

    exporter = get_exporter(platform)
    result = exporter.export_story(
        markers,
        project_name=project['name'],
        story_title=story_title,
        source_path=source_path,
        media_duration=media_duration,
        framerate=framerate,
        width=width,
        height=height,
        exports_dir=app.config['EXPORTS_DIR'],
        start_tc_frames=start_tc_frames,
    )
    return result, exporter


@app.route('/project/<project_id>/story/export', methods=['POST'])
def story_export(project_id):
    """Export a story build as a timeline in the project's selected NLE format.

    Honors ``deliver_to`` like ``/project/<id>/export/fcpxml``: ``'nle'`` writes
    the file then launches the chosen NLE; ``'file'`` reveals the file in
    Finder; omitted preserves the legacy attachment-stream behavior.
    """
    project = get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    data = request.json or {}
    if not data.get('clips'):
        return jsonify({'error': 'No clips in sequence'}), 400

    deliver_to = str(data.get('deliver_to') or '').strip().lower()
    nle = str(data.get('nle') or '').strip().lower()
    force_platform = nle if (deliver_to == 'nle' and nle in NLE_DISPLAY_NAMES) else None

    try:
        result, exporter = _build_nle_story_export(project, data, force_platform=force_platform)
    except Exception as e:
        app.logger.error('Story export failed: %s', e)
        return jsonify({'error': f'Export failed: {e}'}), 500

    if deliver_to == 'nle':
        if nle not in NLE_DISPLAY_NAMES:
            return jsonify({'error': f'Unknown NLE: {nle!r}'}), 400
        opened_in, info = _hand_file_to_nle(
            result.file_path, nle,
            source_media_path=project.get('source_path') or project.get('filepath'),
            project_name=project.get('name') or '',
            timeline_name=os.path.splitext(result.filename)[0],
        )
        if opened_in is None:
            return jsonify({'error': info.get('error', 'NLE delivery failed'),
                            'file': result.file_path}), 500
        payload = {
            'status': 'ok', 'delivery': 'nle', 'opened_in': opened_in,
            'nle': nle, 'nle_name': NLE_DISPLAY_NAMES[nle],
            'file': result.file_path, 'filename': result.filename,
            'format_name': result.format_name,
        }
        payload.update(info)
        return jsonify(payload)

    if deliver_to == 'file':
        _reveal_in_finder(result.file_path)
        return jsonify({
            'status': 'ok', 'delivery': 'file',
            'file': result.file_path, 'filename': result.filename,
            'format_name': result.format_name,
        })

    return _exporter_response(result, project, exporter)


# ── Client Comments ────────────────────────────────────────────────

def _get_comments_path(project_id):
    return os.path.join(app.config['PROJECTS_DIR'], project_id, 'client_comments.json')


def _load_comments(project_id):
    path = _get_comments_path(project_id)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {'comments': []}


def _save_comments(project_id, data):
    path = _get_comments_path(project_id)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


@app.route('/project/<project_id>/comments', methods=['GET'])
def get_comments(project_id):
    """Get all comments for a project."""
    return jsonify(_load_comments(project_id))


@app.route('/project/<project_id>/comments', methods=['POST'])
def add_comment(project_id):
    """Add a new comment (from shared or editor view)."""
    data = request.json or {}
    comment = {
        'id': str(uuid.uuid4())[:8],
        'client_name': data.get('client_name', 'Anonymous').strip(),
        'comment_text': data.get('comment_text', '').strip(),
        'selected_text': data.get('selected_text', '').strip(),
        'start_time': data.get('start_time', ''),
        'end_time': data.get('end_time', ''),
        'created_at': datetime.now().isoformat(),
        'addressed': False,
    }

    if not comment['comment_text']:
        return jsonify({'error': 'Comment text required'}), 400

    comments_data = _load_comments(project_id)
    comments_data['comments'].append(comment)
    _save_comments(project_id, comments_data)

    return jsonify({'status': 'saved', 'comment': comment})


@app.route('/project/<project_id>/comments/<comment_id>/address', methods=['PUT'])
def address_comment(project_id, comment_id):
    """Mark a comment as addressed (editor only)."""
    comments_data = _load_comments(project_id)
    for c in comments_data['comments']:
        if c['id'] == comment_id:
            c['addressed'] = not c.get('addressed', False)
            _save_comments(project_id, comments_data)
            return jsonify({'status': 'updated', 'addressed': c['addressed']})
    return jsonify({'error': 'Comment not found'}), 404


@app.route('/project/<project_id>/comments/<comment_id>', methods=['DELETE'])
def delete_comment(project_id, comment_id):
    """Delete a comment (editor only)."""
    comments_data = _load_comments(project_id)
    comments_data['comments'] = [c for c in comments_data['comments'] if c['id'] != comment_id]
    _save_comments(project_id, comments_data)
    return jsonify({'status': 'deleted'})


# ── Cloudflare Tunnel for sharing ────────────────────────────────────

_tunnel_process = None
_tunnel_url = None


def _find_cloudflared():
    """Find cloudflared binary."""
    path = shutil.which('cloudflared')
    if path:
        return path
    for candidate in ['/opt/homebrew/bin/cloudflared', '/usr/local/bin/cloudflared']:
        if os.path.isfile(candidate):
            return candidate
    return None


# Tunnel disabled for beta. Re-enable with auth scoping before public release.
# The current implementation exposes the entire Flask API to the public
# internet via cloudflared with no token, no Cloudflare Access policy, and
# no per-route gating — anyone with the trycloudflare URL can reach every
# project, media file, and chat endpoint. Restore behind a per-session token
# (and ideally Cloudflare Access) before re-enabling the Share UI.
#
# @app.route('/tunnel/start', methods=['POST'])
# def start_tunnel():
#     """Start a Cloudflare quick tunnel and return the public URL."""
#     global _tunnel_process, _tunnel_url
#
#     # Already running?
#     if _tunnel_process and _tunnel_process.poll() is None and _tunnel_url:
#         return jsonify({'url': _tunnel_url, 'status': 'running'})
#
#     cloudflared = _find_cloudflared()
#     if not cloudflared:
#         return jsonify({'error': 'cloudflared not installed. Run: brew install cloudflared'}), 500
#
#     # Start tunnel in background
#     _tunnel_process = subprocess.Popen(
#         [cloudflared, 'tunnel', '--url', 'http://127.0.0.1:5050'],
#         stdout=subprocess.PIPE,
#         stderr=subprocess.STDOUT,
#         text=True,
#     )
#
#     # Read output in a thread to capture the URL
#     url_found = threading.Event()
#
#     def _read_output():
#         global _tunnel_url
#         for line in _tunnel_process.stdout:
#             # Cloudflare prints the URL like: https://xxxx-xxxx.trycloudflare.com
#             match = _re.search(r'(https://[a-zA-Z0-9-]+\.trycloudflare\.com)', line)
#             if match:
#                 _tunnel_url = match.group(1)
#                 url_found.set()
#
#     t = threading.Thread(target=_read_output, daemon=True)
#     t.start()
#
#     # Wait up to 15 seconds for the URL
#     url_found.wait(timeout=15)
#
#     if _tunnel_url:
#         return jsonify({'url': _tunnel_url, 'status': 'started'})
#     else:
#         return jsonify({'error': 'Tunnel started but URL not detected yet. Try again in a few seconds.'}), 500
#
#
# @app.route('/tunnel/stop', methods=['POST'])
# def stop_tunnel():
#     """Stop the Cloudflare tunnel."""
#     global _tunnel_process, _tunnel_url
#     if _tunnel_process:
#         _tunnel_process.terminate()
#         _tunnel_process = None
#     _tunnel_url = None
#     return jsonify({'status': 'stopped'})


@app.route('/tunnel/status')
def tunnel_status():
    """Check if tunnel is running."""
    global _tunnel_process, _tunnel_url
    if _tunnel_process and _tunnel_process.poll() is None and _tunnel_url:
        return jsonify({'url': _tunnel_url, 'status': 'running'})
    return jsonify({'url': None, 'status': 'stopped'})


# ---------------------------------------------------------------------------
# Editorial DNA v2.1 — multi-profile routes
# ---------------------------------------------------------------------------

from editorial_dna import profiles as edna_profiles
from editorial_dna import snapshots as edna_snapshots
from editorial_dna import analysis as edna_analysis


def _active_profile_for_page():
    """Return the currently-active profile or None, for rendering the page.

    Unlike the injector, this ignores the `active` toggle state so the
    dashboard can still render a profile the user has toggled off.
    """
    pid = edna_profiles.get_active_profile_id()
    if not pid:
        return None
    return edna_profiles.get_profile(pid)


@app.route('/my-style')
def my_style_page():
    # Allow ?profile_id=... to render any profile without changing the active
    # set — the dropdown uses this so picking a different profile to inspect
    # doesn't blow away a user's multi-active blend.
    requested = request.args.get('profile_id')
    profile = None
    if requested:
        profile = edna_profiles.get_profile(requested)
    if profile is None:
        profile = _active_profile_for_page()
    if profile is None:
        # Last-resort fallback to any profile that exists, so the page renders
        # something rather than the empty state when only inactive profiles
        # exist.
        all_profs = edna_profiles.list_profiles()
        if all_profs:
            profile = edna_profiles.get_profile(all_profs[0]['id'])
    all_profiles = edna_profiles.list_profiles()
    active_ids = edna_profiles.get_active_profile_ids()
    snapshots = edna_snapshots.list_snapshots(profile['id']) if profile else []
    # Surface a "regenerate this profile, the analyzer has been upgraded since
    # it was last analyzed" banner for stored summaries below ANALYSIS_VERSION.
    from editorial_dna.models import ANALYSIS_VERSION as CURRENT_ANALYSIS_VERSION
    needs_reanalyze = False
    if profile is not None:
        stored_v = (profile.get('summary') or {}).get('analysis_version') or 0
        try:
            needs_reanalyze = int(stored_v) < CURRENT_ANALYSIS_VERSION
        except (TypeError, ValueError):
            needs_reanalyze = True
    return render_template(
        'my_style.html',
        profile=profile,
        all_profiles=all_profiles,
        active_ids=active_ids,
        snapshots=snapshots,
        needs_reanalyze=needs_reanalyze,
        current_analysis_version=CURRENT_ANALYSIS_VERSION,
    )


# ── Profile CRUD ────────────────────────────────────────────────────────────

@app.route('/api/editorial_dna/profiles', methods=['GET'])
def edna_list_profiles():
    return jsonify({
        'profiles': edna_profiles.list_profiles(),
        'active_profile_id': edna_profiles.get_active_profile_id(),
        'active_profile_ids': edna_profiles.get_active_profile_ids(),
    })


@app.route('/api/editorial_dna/active-summary')
def edna_active_summary():
    """Lightweight endpoint for the global header pill. Returns just the names
    of currently-active profiles plus a count, so every page can render
    "Active: Doc Style, Social Cuts" without loading full summary blobs.
    """
    active_ids = set(edna_profiles.get_active_profile_ids() or [])
    names = []
    for entry in edna_profiles.list_profiles():
        if entry['id'] in active_ids:
            # Skip profiles whose master toggle is off — header should reflect
            # what's actually shaping AI suggestions, not what's merely selected.
            full = edna_profiles.get_profile(entry['id'])
            if full is None or not full.get('active', True):
                continue
            names.append(entry['name'])
    return jsonify({'count': len(names), 'names': names})


@app.route('/api/editorial_dna/profiles', methods=['POST'])
def edna_create_profile():
    data = request.get_json(force=True) or {}
    name = (data.get('name') or '').strip() or 'Untitled Style'
    description = (data.get('description') or '').strip()
    pid = edna_profiles.create_profile(name, description)
    return jsonify({'id': pid, 'name': name})


@app.route('/api/editorial_dna/profiles/<profile_id>', methods=['GET'])
def edna_get_profile(profile_id):
    profile = edna_profiles.get_profile(profile_id)
    if profile is None:
        return jsonify({'error': 'Profile not found'}), 404
    return jsonify(profile)


@app.route('/api/editorial_dna/profiles/<profile_id>', methods=['PATCH'])
def edna_patch_profile(profile_id):
    data = request.get_json(force=True) or {}
    if 'name' in data:
        edna_profiles.rename_profile(profile_id, data['name'])
    if 'description' in data:
        prof = edna_profiles.get_profile(profile_id)
        if prof is not None:
            metrics = {k: prof.get(k, {}) for k in (
                'speech_pacing', 'structural_rhythm', 'soundbite_craft',
                'story_shape', 'content_patterns', 'natural_language_summary'
            )}
            metrics['description'] = (data.get('description') or '').strip()[:500]
            edna_profiles.save_profile(profile_id, metrics)
    if 'active' in data:
        edna_profiles.set_profile_active_toggle(profile_id, bool(data['active']))
    if 'user_refinements' in data:
        profile = edna_profiles.get_profile(profile_id)
        if profile:
            summary = profile.get('summary') or {}
            summary['user_refinements'] = data.get('user_refinements') or ''
            edna_profiles.save_summary(profile_id, summary)
            # Rebuild the system prompt so the refinements take effect immediately
            try:
                metrics = {k: profile.get(k, {}) for k in (
                    'speech_pacing', 'structural_rhythm', 'soundbite_craft',
                    'story_shape', 'content_patterns', 'natural_language_summary'
                )}
                new_prompt = edna_analysis.generate_system_prompt(
                    profile_id, profile.get('name', 'My Style'), metrics, summary,
                )
                edna_profiles.save_system_prompt(profile_id, new_prompt)
            except Exception as e:
                print(f"[edna] prompt rebuild on refinement failed: {e}")
    return jsonify({'status': 'updated'})


@app.route('/api/editorial_dna/profiles/<profile_id>', methods=['DELETE'])
def edna_delete_profile(profile_id):
    edna_profiles.delete_profile(profile_id)
    return jsonify({'status': 'deleted'})


@app.route('/api/editorial_dna/profiles/<profile_id>/activate', methods=['POST'])
def edna_activate_profile(profile_id):
    """Single-select activation: replaces the active set with this one profile.

    The dropdown in the dashboard uses this — it preserves single-active
    behaviour for users who don't care about blending. For multi-active see
    the /active-set endpoint below.
    """
    ok = edna_profiles.set_active(profile_id)
    if not ok:
        return jsonify({'error': 'Profile not found'}), 404
    return jsonify({
        'active_profile_id': profile_id,
        'active_profile_ids': edna_profiles.get_active_profile_ids(),
    })


@app.route('/api/editorial_dna/profiles/<profile_id>/active-toggle', methods=['POST'])
def edna_toggle_in_active_set(profile_id):
    """Add or remove a profile from the multi-active set.

    Body: ``{"active": true}`` adds, ``{"active": false}`` removes. Used by the
    multi-toggle UI on the My Style page so the editor can blend Doc Style and
    Social Cuts without losing one when they pick the other.
    """
    data = request.get_json(force=True) or {}
    want = bool(data.get('active'))
    if want:
        ok = edna_profiles.add_active(profile_id)
    else:
        ok = edna_profiles.remove_active(profile_id)
    if not ok and want:
        return jsonify({'error': 'Profile not found'}), 404
    return jsonify({'active_profile_ids': edna_profiles.get_active_profile_ids()})


@app.route('/api/editorial_dna/active-set', methods=['POST'])
def edna_set_active_set():
    """Replace the entire active set in one call. Body: ``{"ids": [...]}``."""
    data = request.get_json(force=True) or {}
    ids = data.get('ids') or []
    edna_profiles.set_active_ids(ids)
    return jsonify({'active_profile_ids': edna_profiles.get_active_profile_ids()})


@app.route('/api/editorial_dna/profiles/<profile_id>/sources/<path:source_filename>',
           methods=['DELETE'])
def edna_delete_source(profile_id, source_filename):
    """Remove a single source from a profile (no auto-regenerate).

    The UI prompts the user to regenerate after the delete completes — we don't
    auto-regen because that's a long-running call and the user may want to
    delete several sources first.
    """
    ok = edna_profiles.delete_source(profile_id, source_filename)
    if not ok:
        return jsonify({'error': 'Source not found'}), 404
    return jsonify({'status': 'deleted'})


@app.route('/api/editorial_dna/profiles/<profile_id>/sources/<path:source_filename>/retry',
           methods=['POST'])
def edna_retry_source(profile_id, source_filename):
    """Retry a failed source by removing the failed entry; the UI then re-uploads.

    We don't keep the original upload around (it lives in tempfile-land during
    import), so retry from the server side just clears the failed entry. The
    front-end re-prompts the user to drop the file again.
    """
    ok = edna_profiles.delete_source(profile_id, source_filename)
    return jsonify({'status': 'cleared' if ok else 'not_found'})


# ── Regenerate / refine ─────────────────────────────────────────────────────

@app.route('/api/editorial_dna/profiles/<profile_id>/regenerate', methods=['POST'])
def edna_regenerate(profile_id):
    """Re-run the structured analysis pass on an existing profile.

    Uses whatever transcript text is stored in source_files.json (new imports
    capture this; migrated v1 profiles don't have it and will get
    placeholder narrative fields back with a flag indicating so).
    """
    profile = edna_profiles.get_profile(profile_id)
    if profile is None:
        return jsonify({'error': 'Profile not found'}), 404

    metrics = {k: profile.get(k, {}) for k in (
        'speech_pacing', 'structural_rhythm', 'soundbite_craft',
        'story_shape', 'content_patterns', 'natural_language_summary'
    )}
    source_files = profile.get('source_files') or []
    transcripts_text = '\n\n'.join(
        sf.get('transcript_text', '') for sf in source_files if sf.get('transcript_text')
    )

    new_summary = edna_analysis.generate_structured_summary(
        profile_id, profile.get('name', 'My Style'),
        metrics, source_files,
        transcripts_text=transcripts_text,
        existing_summary=profile.get('summary'),
    )
    edna_profiles.save_summary(profile_id, new_summary)

    # Also refresh the human-readable prose summary that shows on the dashboard
    try:
        from editorial_dna.summarizer import generate_summary as _gen_sum
        fresh_prose = _gen_sum(metrics, transcripts_text=transcripts_text)
        metrics['natural_language_summary'] = fresh_prose
        edna_profiles.save_profile(profile_id, metrics)
    except Exception as e:
        print(f"[edna] regenerate prose failed: {e}")

    new_prompt = edna_analysis.generate_system_prompt(
        profile_id, profile.get('name', 'My Style'), metrics, new_summary,
    )
    edna_profiles.save_system_prompt(profile_id, new_prompt)

    # Take a new snapshot so evolution tracking picks up the change
    edna_snapshots.create_snapshot(profile_id, note='Manual regeneration')

    return jsonify({
        'status': 'regenerated',
        'summary': new_summary,
        'had_transcripts': bool(transcripts_text.strip()),
    })


# ── Snapshots ───────────────────────────────────────────────────────────────

@app.route('/api/editorial_dna/profiles/<profile_id>/snapshots', methods=['GET'])
def edna_list_snapshots(profile_id):
    return jsonify({'snapshots': edna_snapshots.list_snapshots(profile_id)})


# ── Export / import ─────────────────────────────────────────────────────────

@app.route('/api/editorial_dna/export')
def edna_export_all():
    import io
    bundle = edna_profiles.export_all()
    buf = io.BytesIO(json.dumps(bundle, indent=2).encode('utf-8'))
    buf.seek(0)
    return send_file(buf, mimetype='application/json', as_attachment=True,
                     download_name='doza_editorial_dna_export.json')


@app.route('/api/editorial_dna/import', methods=['POST'])
def edna_import_bundle():
    # Accept either a file upload or a JSON body
    bundle = None
    if request.files.get('file'):
        try:
            bundle = json.load(request.files['file'].stream)
        except Exception as e:
            return jsonify({'error': f'Invalid JSON: {e}'}), 400
    else:
        bundle = request.get_json(force=True, silent=True)
    if not bundle:
        return jsonify({'error': 'No bundle provided'}), 400
    ids = edna_profiles.import_bundle(bundle)
    return jsonify({'imported_profile_ids': ids})


# ── Legacy /my-style/* aliases kept for backwards compat ───────────────────

@app.route('/my-style/profile')
def my_style_profile():
    profile = _active_profile_for_page()
    if profile is None:
        return jsonify({'error': 'No profile exists'}), 404
    return jsonify(profile)


@app.route('/my-style/status')
def my_style_status():
    profile = _active_profile_for_page()
    active = False
    if profile is not None:
        active = bool(profile.get('active', True))
    return jsonify({
        'active': active,
        'profile_exists': profile is not None,
    })


@app.route('/my-style/toggle', methods=['POST'])
def my_style_toggle():
    data = request.get_json(force=True)
    active = data.get('active', True)
    pid = edna_profiles.get_active_profile_id()
    if not pid:
        return jsonify({'error': 'No profile to toggle'}), 404
    edna_profiles.set_profile_active_toggle(pid, active)
    return jsonify({'active': active})


@app.route('/my-style/delete', methods=['POST'])
def my_style_delete_route():
    """Legacy: delete the active profile."""
    pid = edna_profiles.get_active_profile_id()
    if pid:
        edna_profiles.delete_profile(pid)
    return jsonify({'status': 'deleted'})


@app.route('/my-style/export')
def my_style_export_route():
    """Legacy: export the active profile only."""
    pid = edna_profiles.get_active_profile_id()
    profile = edna_profiles.get_profile(pid) if pid else None
    if profile is None:
        return jsonify({'error': 'No profile'}), 404
    import io
    buf = io.BytesIO(json.dumps(profile, indent=2, default=str).encode('utf-8'))
    buf.seek(0)
    return send_file(buf, mimetype='application/json', as_attachment=True,
                     download_name=f'{profile.get("name", "profile")}.json')


@app.route('/my-style/import', methods=['POST'])
def my_style_import():
    """
    Accept uploaded video/audio files, run the full pipeline per file
    (extract audio → transcribe → analyze), merge into existing profile,
    re-run classifiers and summarizer, save.

    Returns a streaming response with per-file progress.
    """
    import tempfile
    from transcribe import extract_audio, transcribe_file
    from editorial_dna.transcript_analyzer import analyze_transcript as edna_analyze, merge_metrics
    from editorial_dna.classifier import (
        classify_opening, classify_closing, classify_rhythm,
        classify_energy_arc, detect_callbacks, estimate_topic_count,
    )
    from editorial_dna.summarizer import generate_summary
    from editorial_dna import fcpxml_ingest as edna_fcpxml

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400

    # Which profile are we importing into? Optional query/form field; defaults
    # to the currently-active profile, creating a new "My Style" profile if
    # none exists yet.
    target_profile_id = request.form.get('profile_id') or edna_profiles.get_active_profile_id()
    if not target_profile_id:
        target_profile_id = edna_profiles.create_profile('My Style')
    # Make sure the target is in the active set so the page re-renders with it
    if target_profile_id not in (edna_profiles.get_active_profile_ids() or []):
        edna_profiles.add_active(target_profile_id)

    # Supported extensions for My Style import — video/audio go through the
    # standard transcription pipeline, .fcpxml/.fcpxmld go through the FCPXML
    # ingestion module which renders the dialogue timeline first.
    style_extensions = {'mp4', 'mov', 'm4v', 'mkv', 'mp3', 'wav', 'm4a', 'aac', 'flac', 'aif', 'aiff'}
    fcpxml_extensions = {'fcpxml', 'fcpxmld'}

    # IMPORTANT: Flask's request.files stream closes as soon as the response
    # generator starts yielding — so we must save every upload to disk BEFORE
    # entering the streaming generator. Otherwise we get "read of closed file".
    # FCPXML bundles (.fcpxmld) arrive as either a single .fcpxml drop or as
    # one or more files within the bundle — we accept both shapes here and let
    # the parser handle it downstream.
    staged = []  # list of (original_filename, tmp_path, tmp_dir)
    for i, f in enumerate(files):
        fname = f.filename or f'file_{i}'
        tmp_dir = tempfile.mkdtemp(prefix='doza_style_')
        tmp_path = os.path.join(tmp_dir, fname)
        try:
            f.save(tmp_path)
            staged.append((fname, tmp_path, tmp_dir))
        except Exception as e:
            print(f"[my-style import] failed to stage {fname}: {e}")
            staged.append((fname, None, tmp_dir))

    def generate():
        """Stream progress as newline-delimited JSON."""
        existing_profile = edna_profiles.get_profile(target_profile_id) or {}
        source_files = list(existing_profile.get('source_files') or [])
        # Build a v1-shaped metrics dict from existing profile (for merge_metrics)
        merged = None
        if existing_profile.get('speech_pacing'):
            merged = {
                'speech_pacing': existing_profile.get('speech_pacing', {}),
                'structural_rhythm': existing_profile.get('structural_rhythm', {}),
                'soundbite_craft': existing_profile.get('soundbite_craft', {}),
                'content_patterns': existing_profile.get('content_patterns', {}),
            }

        processed_count = 0
        total_files = len(staged)

        for i, (fname, tmp_path, tmp_dir) in enumerate(staged):
            ext = fname.rsplit('.', 1)[-1].lower() if '.' in fname else ''
            is_fcpxml = ext in fcpxml_extensions

            if ext not in style_extensions and not is_fcpxml:
                yield json.dumps({'file': fname, 'status': 'skipped', 'reason': 'unsupported format', 'progress': i + 1, 'total': total_files}) + '\n'
                continue

            if tmp_path is None:
                yield json.dumps({'file': fname, 'status': 'error', 'reason': 'failed to stage upload', 'progress': i + 1, 'total': total_files}) + '\n'
                continue

            yield json.dumps({'file': fname, 'status': 'processing', 'step': 'saving', 'progress': i + 1, 'total': total_files}) + '\n'

            try:
                fcpxml_metadata = None
                missing_media = []

                if is_fcpxml:
                    # FCPXML path: parse, render dialogue-timeline WAV, then
                    # transcribe the WAV. Missing media in the FCPXML is
                    # surfaced to the UI but doesn't block the rest.
                    yield json.dumps({'file': fname, 'status': 'processing', 'step': 'parsing FCPXML'}) + '\n'
                    ingest = edna_fcpxml.stage_fcpxml(tmp_path, tmp_dir)
                    audio_path = ingest.audio_path
                    fcpxml_metadata = ingest.fcpxml_metadata
                    missing_media = ingest.missing_media
                    file_duration_hint = ingest.duration_seconds
                    if missing_media:
                        yield json.dumps({
                            'file': fname,
                            'status': 'warning',
                            'reason': f'{len(missing_media)} clip(s) reference media not on disk; continuing with what was found',
                            'missing_media': missing_media[:10],
                        }) + '\n'
                else:
                    # Raw video/audio path: extract audio with ffmpeg, then
                    # transcribe. file_duration comes from the transcriber.
                    yield json.dumps({'file': fname, 'status': 'processing', 'step': 'extracting audio'}) + '\n'
                    audio_path = extract_audio(tmp_path, project_dir=tmp_dir)
                    file_duration_hint = None

                # Transcribe (same path for both source types)
                yield json.dumps({'file': fname, 'status': 'processing', 'step': 'transcribing'}) + '\n'
                transcript = transcribe_file(audio_path, project_dir=tmp_dir)

                # Analyze
                yield json.dumps({'file': fname, 'status': 'processing', 'step': 'analyzing'}) + '\n'
                metrics = edna_analyze(transcript)
                file_duration = transcript.get('duration') or file_duration_hint or 0

                # Merge with existing profile metrics
                if merged and 'speech_pacing' in merged:
                    merged_metrics = merge_metrics(merged, metrics, file_duration)
                else:
                    merged_metrics = {k: v for k, v in metrics.items() if not k.startswith('_')}

                merged_metrics['_raw'] = metrics.get('_raw', {})
                merged = merged_metrics

                # Capture the raw transcript text so the structured analysis
                # pass can run later. Stays local to the profile folder.
                transcript_text = ' '.join(
                    seg.get('text', '') for seg in transcript.get('segments', [])
                ).strip()

                source_entry = {
                    'filename': fname,
                    'source_type': 'fcpxml' if is_fcpxml else 'video',
                    'imported_at': datetime.now().isoformat(),
                    'duration_seconds': round(float(file_duration or 0), 2),
                    'transcribed_at': datetime.now().isoformat(),
                    'transcript_text': transcript_text,
                }
                if fcpxml_metadata is not None:
                    source_entry['fcpxml_metadata'] = fcpxml_metadata
                if missing_media:
                    source_entry['missing_media'] = missing_media

                source_files.append(source_entry)

                processed_count += 1
                yield json.dumps({'file': fname, 'status': 'done', 'progress': i + 1, 'total': total_files}) + '\n'

            except Exception as e:
                import traceback
                print(f"[my-style import] ERROR on {fname}: {e}")
                traceback.print_exc()
                yield json.dumps({'file': fname, 'status': 'error', 'reason': str(e)[:300], 'progress': i + 1, 'total': total_files}) + '\n'

            finally:
                # Clean up temp files
                try:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                except Exception:
                    pass

        if processed_count == 0:
            yield json.dumps({'status': 'complete', 'error': 'No files were processed successfully'}) + '\n'
            return

        # Run classifiers on merged metrics
        yield json.dumps({'status': 'classifying'}) + '\n'
        raw = merged.get('_raw', {})

        try:
            opening = classify_opening(raw.get('first_15s_text', ''))
            closing = classify_closing(raw.get('last_15s_text', ''))
            rhythm = classify_rhythm(merged.get('speech_pacing', {}))
            energy = classify_energy_arc(
                merged.get('structural_rhythm', {}).get('pacing_first_third_wpm', 0),
                merged.get('structural_rhythm', {}).get('pacing_middle_third_wpm', 0),
                merged.get('structural_rhythm', {}).get('pacing_last_third_wpm', 0),
            )
            callbacks = detect_callbacks(
                raw.get('opening_third_text', ''),
                raw.get('closing_third_text', ''),
            )
            topics = estimate_topic_count(
                raw.get('opening_third_text', '') + ' ' + raw.get('closing_third_text', '')
            )
        except Exception as e:
            # If classifiers fail, use defaults
            print(f"Classifier error: {e}")
            opening = 'other'
            closing = 'other'
            rhythm = 'conversational'
            energy = 'balanced'
            callbacks = False
            topics = 1

        merged['speech_pacing']['rhythm_descriptor'] = rhythm
        merged['structural_rhythm']['energy_arc'] = energy
        merged['content_patterns']['topic_count'] = topics

        story_shape = {
            'opening_style': opening,
            'closing_style': closing,
            'uses_callbacks': callbacks,
        }

        # Build full profile
        yield json.dumps({'status': 'generating summary'}) + '\n'

        metric_fields = {
            'speech_pacing': merged.get('speech_pacing', {}),
            'structural_rhythm': merged.get('structural_rhythm', {}),
            'soundbite_craft': merged.get('soundbite_craft', {}),
            'story_shape': story_shape,
            'content_patterns': merged.get('content_patterns', {}),
            'natural_language_summary': '',
        }

        try:
            # Build the concatenated transcript text for grounding the prose
            nl_transcripts = '\n\n'.join(
                sf.get('transcript_text', '') for sf in source_files
                if sf.get('transcript_text')
            )
            nl_summary = generate_summary(metric_fields, transcripts_text=nl_transcripts)
            metric_fields['natural_language_summary'] = nl_summary
        except Exception as e:
            import traceback
            print(f"Summary generation error: {e}")
            traceback.print_exc()
            metric_fields['natural_language_summary'] = 'Style profile generated but summary unavailable.'

        # Persist the metric fields + source files to the active v2.1 profile
        edna_profiles.save_profile(target_profile_id, metric_fields)
        edna_profiles.save_source_files(target_profile_id, source_files)

        # Run the structured analysis pass on the newly imported transcripts
        yield json.dumps({'status': 'generating structured summary'}) + '\n'
        try:
            transcripts_text = '\n\n'.join(
                sf.get('transcript_text', '') for sf in source_files
                if sf.get('transcript_text')
            )
            existing_summary_for_merge = (edna_profiles.get_profile(target_profile_id) or {}).get('summary')
            new_summary = edna_analysis.generate_structured_summary(
                target_profile_id,
                (edna_profiles.get_profile(target_profile_id) or {}).get('name', 'My Style'),
                metric_fields, source_files,
                transcripts_text=transcripts_text,
                existing_summary=existing_summary_for_merge,
            )
            edna_profiles.save_summary(target_profile_id, new_summary)

            new_prompt = edna_analysis.generate_system_prompt(
                target_profile_id,
                (edna_profiles.get_profile(target_profile_id) or {}).get('name', 'My Style'),
                metric_fields, new_summary,
            )
            edna_profiles.save_system_prompt(target_profile_id, new_prompt)
        except Exception as e:
            print(f"[edna] structured summary failed: {e}")

        # Take an evolution snapshot for this import
        try:
            edna_snapshots.create_snapshot(target_profile_id, note='Import')
        except Exception as e:
            print(f"[edna] snapshot after import failed: {e}")

        final_profile = edna_profiles.get_profile(target_profile_id)
        yield json.dumps({'status': 'complete', 'profile': final_profile}) + '\n'

    from flask import Response
    return Response(generate(), mimetype='application/x-ndjson')


def _load_extensions(flask_app):
    """Discover and register Flask blueprints from ``DOZA_EXTENSIONS_PATH``.

    A small extension system for community plugins. When the env var is
    set to a directory:

      1. The directory is added to ``sys.path`` (so its top-level
         subpackages are importable by name).
      2. Each top-level subdirectory is imported as a Python package.
      3. If the package exposes a ``blueprint`` attribute that is an
         instance of ``flask.Blueprint``, it's registered on the app.

    Failures are logged and swallowed — a misbehaving extension never
    blocks the app from starting. Subdirectories whose names start with
    ``.`` or ``_`` are skipped (so ``__pycache__`` and dotfiles don't
    get imported).

    Set ``DOZA_EXTENSIONS_PATH=/path/to/extensions`` to use this. The
    path can hold any number of independent extension packages.
    """
    import importlib
    import sys as _sys
    from flask import Blueprint as _Blueprint

    ext_path = os.environ.get('DOZA_EXTENSIONS_PATH')
    if not ext_path:
        return
    ext_path = os.path.expanduser(ext_path)
    if not os.path.isdir(ext_path):
        flask_app.logger.warning(
            'DOZA_EXTENSIONS_PATH=%s is not a directory; skipping extension load',
            ext_path,
        )
        return

    if ext_path not in _sys.path:
        _sys.path.insert(0, ext_path)

    try:
        entries = sorted(os.listdir(ext_path))
    except OSError as e:
        flask_app.logger.warning('Could not list %s: %s', ext_path, e)
        return

    for name in entries:
        if name.startswith('.') or name.startswith('_'):
            continue
        sub = os.path.join(ext_path, name)
        if not os.path.isdir(sub):
            continue
        try:
            module = importlib.import_module(name)
        except Exception as e:
            flask_app.logger.warning(
                'Skipping extension %r: import failed (%s)', name, e,
            )
            continue
        bp = getattr(module, 'blueprint', None)
        if bp is None:
            flask_app.logger.warning(
                'Skipping extension %r: no `blueprint` attribute exposed', name,
            )
            continue
        if not isinstance(bp, _Blueprint):
            flask_app.logger.warning(
                'Skipping extension %r: `blueprint` is not a flask.Blueprint',
                name,
            )
            continue
        try:
            flask_app.register_blueprint(bp)
            flask_app.logger.info('Loaded extension: %s', name)
        except Exception as e:
            flask_app.logger.warning(
                'Skipping extension %r: register_blueprint failed (%s)', name, e,
            )


# Load extensions at module import so any WSGI host (gunicorn, uwsgi)
# that does ``from app import app`` also picks them up — not just the
# ``python3 app.py`` direct-run path. Idempotent: Flask's
# register_blueprint raises on a second register, but the loader catches
# that as a logged warning rather than crashing the import.
_load_extensions(app)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '5050'))
    # Background pre-warm: load the configured Ollama model into RAM so
    # the first chat call doesn't pay the 4-12s cold-start. Fire-and-
    # forget thread; warmup_ollama swallows errors so a missing Ollama
    # daemon doesn't block Flask startup.
    import threading
    from ai_analysis import warmup_ollama
    threading.Thread(target=warmup_ollama, daemon=True).start()
    app.run(host='127.0.0.1', port=port, debug=False, threaded=True)
