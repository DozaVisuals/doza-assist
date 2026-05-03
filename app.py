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
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, Response, stream_with_context
from werkzeug.utils import secure_filename

from exporters import get_exporter, PLATFORMS, DEFAULT_PLATFORM
from exporters.media_probe import get_video_resolution, get_video_framerate
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


@app.context_processor
def inject_brand():
    """Make the user-visible app brand and logo configurable from the
    launching shell.

    Defaults: brand = "Doza Assist", logo_url = "/static/logo.jpg".
    An external launcher (a downstream shell that bundles this Flask
    backend) can override by setting DOZA_BRAND and/or DOZA_LOGO_URL in
    the environment before spawning python -- the shell is responsible
    for ensuring whatever URL it points at actually serves an image
    (drop the logo into static/ before launch).
    """
    return {
        'brand': os.environ.get('DOZA_BRAND', 'Doza Assist'),
        'logo_url': os.environ.get('DOZA_LOGO_URL', '/static/logo.jpg'),
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
    project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
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
    project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
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
    from ai_providers import ProviderError
    if not isinstance(e, ProviderError):
        raise e
    return _provider_error_response(e)


# ── BYO API key: provider settings ──────────────────────────────────
# All AI calls route through ``ai_providers.get_active_provider()`` which
# reads ``provider_config.json`` from $DOZA_DATA_DIR. These three routes
# let the user select Local / Anthropic / OpenAI, paste an API key, and
# verify it before relying on it for analysis or chat.

@app.route('/settings/provider', methods=['GET'])
def get_provider_settings():
    """Return the current provider config with API keys masked for display."""
    from ai_providers import load_provider_config, masked_config
    cfg = load_provider_config()
    out = masked_config(cfg)
    out['has_anthropic_key'] = bool((cfg.get('anthropic') or {}).get('api_key'))
    out['has_openai_key'] = bool((cfg.get('openai') or {}).get('api_key'))
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
        cfg.setdefault('ollama', {})['base_url'] = body['base_url'] or 'http://localhost:11434'

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
        ps = _req.get('http://127.0.0.1:11434/api/ps', timeout=2).json()
        for m in ps.get('models', []):
            name = m.get('name')
            if name and name != new_variant:
                _req.post(
                    'http://127.0.0.1:11434/api/generate',
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
            upstream = _requests.post(
                'http://localhost:11434/api/pull',
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
            yield _err_event('Could not reach Ollama at localhost:11434. Is the Ollama app running?')
        except Exception as e:
            yield _err_event(str(e))

    from flask import Response
    return Response(generate(), mimetype='application/x-ndjson')


@app.route('/create', methods=['POST'])
def create_project():
    """Create a new project from a local file path."""
    data = request.json or {}
    source_path = data.get('source_path', '').strip()

    if not source_path:
        return jsonify({'error': 'No file path provided'}), 400

    # Expand user home directory
    source_path = os.path.expanduser(source_path)

    if not os.path.exists(source_path):
        return jsonify({'error': f'File not found: {source_path}'}), 400

    is_fcpxml = _is_fcpxml_input(source_path)
    if not is_fcpxml and not os.path.isfile(source_path):
        return jsonify({'error': 'Path is not a file'}), 400

    if not is_fcpxml and not allowed_file(source_path):
        ext = source_path.rsplit('.', 1)[-1].lower() if '.' in source_path else 'unknown'
        return jsonify({'error': f'Unsupported file type: .{ext}'}), 400

    project_name = data.get('project_name', '').strip()
    client_name = data.get('client_name', '').strip()
    interviewer_name = data.get('interviewer_name', 'Interviewer').strip()
    subject_name = data.get('subject_name', 'Subject').strip()
    num_speakers = int(data.get('num_speakers', 2))
    language = data.get('language', 'en').strip()

    project_id = str(uuid.uuid4())[:8]
    project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
    os.makedirs(project_dir, exist_ok=True)

    fcpxml_meta = None
    if is_fcpxml:
        try:
            ingest = _ingest_fcpxml(source_path, project_dir)
        except ValueError as e:
            shutil.rmtree(project_dir, ignore_errors=True)
            return jsonify({'error': str(e)}), 400
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
        'interviewer_name': interviewer_name,
        'subject_name': subject_name,
        'num_speakers': num_speakers,
        'language': language,
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

    project_id = str(uuid.uuid4())[:8]
    project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
    os.makedirs(project_dir, exist_ok=True)

    filename = secure_filename(file.filename)
    filepath = os.path.join(project_dir, filename)
    file.save(filepath)

    fcpxml_meta = None
    if filename.lower().endswith('.fcpxml'):
        try:
            ingest = _ingest_fcpxml(filepath, project_dir)
        except ValueError as e:
            shutil.rmtree(project_dir, ignore_errors=True)
            return jsonify({'error': str(e)}), 400
        media_path = ingest['audio_path']
        fcpxml_meta = ingest['fcpxml_source']
        media_filename = os.path.basename(media_path)
    else:
        media_path = filepath
        media_filename = filename

    file_size = os.path.getsize(media_path)

    meta = {
        'id': project_id,
        'name': project_name,
        'client_name': client_name,
        'interviewer_name': interviewer_name,
        'subject_name': subject_name,
        'language': language,
        'filename': media_filename,
        'source_path': media_path,
        'filepath': media_path,
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
            return send_file(wav_path, mimetype='audio/wav')
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


@app.route('/project/<project_id>/transcribe', methods=['POST'])
def transcribe(project_id):
    """Run transcription on the source file."""
    project = get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    # Check that source file still exists
    source_path = project.get('source_path', project.get('filepath', ''))
    if not source_path or not os.path.exists(source_path):
        return jsonify({'error': 'Source file not found. It may have been moved or deleted.'}), 404

    # Guard non-English requests when only Parakeet (English-only) is installed.
    # Without this we'd kick off the full transcribe pipeline only to fail
    # with a generic "no engine" error after the audio extraction. The frontend
    # uses needs_whisper_install to render an inline install prompt.
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

    project['status'] = 'transcribing'
    save_project(project_id, project)

    try:
        from transcribe import transcribe_file

        # Pass both source path and project directory for audio extraction
        project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
        num_speakers = project.get('num_speakers', 2)
        language = project.get('language', 'en')
        result = transcribe_file(
            source_path,
            project_dir=project_dir,
            speaker_labels={
                'SPEAKER_00': project.get('interviewer_name', 'Interviewer'),
                'SPEAKER_01': project.get('subject_name', 'Subject'),
            },
            num_speakers=num_speakers,
            language=language,
        )
        project['transcript'] = result
        project['status'] = 'transcribed'
        save_project(project_id, project)
        seg_count = len((result or {}).get('segments', []))
        log_activity(project_id, 'transcribed',
                     f"{project.get('name', 'Project')} transcribed · {seg_count} segments")

        # Auto-build the TF-IDF paragraph index so chat retrieval immediately
        # works against semantic ranking instead of substring matching only.
        # This is a pure-Python pass (no LLM call), runs in seconds, and
        # transforms chat output quality on a fresh project — without it the
        # chat falls back to literal-keyword matching and produces vague or
        # un-clip-cited responses (this was the laptop bug).
        # Idempotent: skip if already on disk so re-transcription doesn't
        # rebuild needlessly. The full /analyze endpoint still rebuilds it
        # alongside segment_vectors when the user runs AI Analysis.
        try:
            idx_path = _paragraph_index_path(project_id)
            if not os.path.exists(idx_path):
                from doza_assist.retrieval import build_paragraph_index, save_index
                idx = build_paragraph_index(result)
                save_index(idx, idx_path)
                print(f"[transcribe] auto-built paragraph_index for {project_id}")
        except Exception as e:
            # Non-fatal — chat will fall back to substring matching. Log and
            # let the response succeed; a missing index never breaks chat,
            # just degrades retrieval.
            print(f"[transcribe] paragraph_index auto-build failed for {project_id}: {e}")

        return jsonify({'status': 'transcribed', 'transcript': result})

    except Exception as e:
        project['status'] = 'error'
        project['error'] = str(e)
        save_project(project_id, project)
        return jsonify({'error': str(e)}), 500


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
    """Run AI analysis on the transcript."""
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

    progress = _make_progress_writer(project_id)
    try:
        from ai_analysis import analyze_transcript, generate_segment_vectors, expected_vector_chunks
        # Reuse vectors from a previous run if they're on disk so the cap+rerank
        # pass can score by narrative_score instead of falling back to length.
        # First-ever analysis runs without this signal — that's expected; the
        # caller still gets the ≤7 cap, just less curated.
        existing_vectors = load_segment_vectors(project_id)

        # Each chunk of vector generation is its own LLM call, so it has to
        # count as its own progress step or the bar races to ~94% and stalls
        # while the vector phase grinds on for several minutes. Pre-compute
        # how many vector chunks the analyzer will run so we can size the
        # global step total correctly.
        n_vector_chunks = expected_vector_chunks(project['transcript'])
        post_analyzer_steps = n_vector_chunks + 1  # +1 for paragraph index

        # Wrap the analyzer's progress callback so its step count rolls into
        # a global total that includes the vector + index phases. The
        # analyzer reports e.g. step=5, total=16; we publish step=5 of 24.
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

        # Vector phase: each chunk = one progress step. With this the bar
        # advances at a real per-LLM-call cadence through the second half of
        # the run instead of jumping from "94%" to "100%".
        def _from_vectors(chunk_idx=1, total_chunks=1, label="vectors"):
            progress(step=analyzer_total + int(chunk_idx),
                     total=global_total,
                     current=label)

        segment_vectors = []
        try:
            segment_vectors = generate_segment_vectors(
                project['transcript'],
                project_name=project['name'],
                progress_callback=_from_vectors,
            )
        except Exception as ve:
            # Don't fail the whole analysis if vector generation hiccups —
            # the human-readable analysis is still useful on its own.
            print(f"Segment vector generation failed: {ve}")

        if segment_vectors:
            project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
            vectors_path = os.path.join(project_dir, 'segment_vectors.json')
            with open(vectors_path, 'w') as f:
                json.dump(segment_vectors, f, indent=2)

        # Final step: paragraph index build (instant, but worth a tick so
        # the user sees the bar finish properly).
        progress(step=global_total, total=global_total,
                 current="building paragraph index")
        try:
            from doza_assist.retrieval import build_paragraph_index, save_index
            paragraph_index = build_paragraph_index(project['transcript'])
            project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
            save_index(paragraph_index, _paragraph_index_path(project_id))
        except Exception as ie:
            print(f"Paragraph index build failed: {ie}")

        # Update cache. Keyed by (transcript_hash, analysis_type) so the same
        # transcript can hold separate entries for 'story', 'social', and
        # 'all'. Drop stale buckets — anything keyed off a different hash is
        # for a transcript content that no longer matches.
        project['analysis_cache'] = {
            transcript_hash: {
                **(cache.get(transcript_hash, {}) if isinstance(cache, dict) else {}),
                analysis_type: {
                    'analysis': result,
                    'cached_at': datetime.now().isoformat(),
                },
            }
        }

        save_project(project_id, project)
        title = result.get('suggested_title') or project.get('name', 'Project')
        log_activity(project_id, 'analyzed', f"AI analysis run · \"{title}\"")
        _clear_analyze_status(project_id)
        return jsonify({
            'status': 'analyzed',
            'analysis': result,
            'segment_vectors': segment_vectors,
        })

    except Exception as e:
        _clear_analyze_status(project_id)
        from ai_providers import ProviderError
        if isinstance(e, ProviderError):
            return _provider_error_response(e)
        return jsonify({'error': str(e)}), 500


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

    if 'social' in requested:
        analysis = project.get('analysis', {})
        for clip in analysis.get('social_clips', []):
            markers.append({
                'start': _to_seconds(clip.get('start', 0)),
                'end': _to_seconds(clip.get('end', 0)),
                'text': clip.get('title', ''),
                'note': clip.get('platform', ''),
                'color': 'green',
                'category': 'Social Clip',
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
            })

    if 'labels' in requested:
        color_labels = project.get('color_labels', {})
        for sec in project.get('labeled_sections', []):
            label_name = color_labels.get(sec.get('color', ''), sec.get('color', ''))
            markers.append({
                'start': _to_seconds(sec.get('start', 0)),
                'end': _to_seconds(sec.get('end', 0)),
                'text': label_name,
                'note': sec.get('text', '')[:80],
                'color': sec.get('color', 'blue'),
                'category': label_name,
            })

    if 'transcript' in requested:
        # Emit one marker per transcript segment so editors can navigate the
        # full interview in the NLE timeline. Kept separate from the other
        # categories because it can be noisy — user opts in explicitly.
        transcript = project.get('transcript') or {}
        for seg in transcript.get('segments', []):
            start = _to_seconds(seg.get('start', 0))
            end = _to_seconds(seg.get('end', start))
            if end <= start:
                end = start + 1
            speaker = (seg.get('speaker') or '').strip()
            text = (seg.get('text') or '').strip()
            label = speaker or 'Transcript'
            markers.append({
                'start': start,
                'end': end,
                'text': label,
                'note': text[:160],
                'color': 'blue',
                'category': 'Transcript',
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
        opened_in, err = _hand_file_to_nle(result.file_path, nle)
        if err:
            return jsonify({'error': err, 'file': result.file_path}), 500
        return jsonify({
            'status': 'ok', 'delivery': 'nle', 'opened_in': opened_in,
            'nle': nle, 'nle_name': NLE_DISPLAY_NAMES[nle],
            'file': result.file_path, 'filename': result.filename,
            'format_name': result.format_name,
        })

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


def _find_nle_app_path(nle: str):
    """Return the on-disk .app path for the chosen NLE, or ``None`` if not installed.

    Premiere ships under a year-suffixed directory (``Adobe Premiere Pro 2026/``,
    ``2025/``, etc.) so we probe the recent versions in newest-first order and
    fall back to the unversioned bundle.
    """
    if nle == 'fcp':
        path = '/Applications/Final Cut Pro.app'
        return path if os.path.isdir(path) else None
    if nle == 'resolve':
        path = '/Applications/DaVinci Resolve/DaVinci Resolve.app'
        return path if os.path.isdir(path) else None
    if nle == 'premiere':
        for candidate in (
            '/Applications/Adobe Premiere Pro 2026/Adobe Premiere Pro 2026.app',
            '/Applications/Adobe Premiere Pro 2025/Adobe Premiere Pro 2025.app',
            '/Applications/Adobe Premiere Pro 2024/Adobe Premiere Pro 2024.app',
            '/Applications/Adobe Premiere Pro.app',
        ):
            if os.path.isdir(candidate):
                return candidate
        return None
    return None


def _hand_file_to_nle(file_path: str, nle: str):
    """Launch the chosen NLE with the export file, or reveal in Finder for Premiere.

    Returns ``(opened_in, error)`` where ``opened_in`` is ``'app'`` (NLE auto-imports
    the file), ``'finder'`` (we revealed the file in Finder for the user to drag in),
    or ``None`` on failure.
    """
    app_path = _find_nle_app_path(nle)
    if not app_path:
        return None, f'{NLE_DISPLAY_NAMES[nle]} not found at expected location'
    try:
        if nle == 'premiere':
            subprocess.Popen(['open', '-R', file_path])
            return 'finder', None
        subprocess.Popen(['open', '-a', app_path, file_path])
        return 'app', None
    except Exception as e:
        return None, f'Could not launch {NLE_DISPLAY_NAMES[nle]}: {e}'


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
            'error': f'{NLE_DISPLAY_NAMES[nle]} not found at expected location'
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


def _project_selects_for_fcpxml(project: dict, source: str):
    """Build ``Select`` objects from a project, pulling from the requested bucket.

    ``source`` is one of:
      - ``'client_selects'``: editor-chosen labels (default)
      - ``'social'``: AI-identified social clips
      - ``'story'``: AI-identified story beats
      - ``'soundbites'``: AI-identified strongest soundbites
      - ``'all'``: everything combined, in chronological order
    """
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

    selects: list[Select] = []

    if source in ('client_selects', 'all'):
        for sec in project.get('labeled_sections') or []:
            color_labels = project.get('color_labels', {})
            label_name = color_labels.get(sec.get('color', ''), sec.get('color', '')) or 'Select'
            selects.append(Select(
                start_seconds=_to_seconds(sec.get('start', 0)),
                end_seconds=_to_seconds(sec.get('end', 0)),
                label=label_name,
                note=(sec.get('text') or '')[:80],
                kind=_kind_for_color(sec.get('color', '')),
            ))

    if source in ('social', 'all'):
        analysis = project.get('analysis') or {}
        for clip in analysis.get('social_clips') or []:
            selects.append(Select(
                start_seconds=_to_seconds(clip.get('start', 0)),
                end_seconds=_to_seconds(clip.get('end', 0)),
                label=clip.get('title') or 'Social Clip',
                note=clip.get('platform', ''),
                kind='strong',
            ))

    if source in ('story', 'all'):
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
            ))

    if source in ('soundbites', 'all'):
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
    source = body.get('source', 'client_selects')

    try:
        parsed = parse_fcpxml(stored_path)
    except ParseError as e:
        raise MulticamExportError(f'Could not re-read stored FCPXML: {e}', status=500)

    selects = _project_selects_for_fcpxml(project, source)
    if not selects:
        raise MulticamExportError(
            f'No selects available for source={source!r}. '
            'Pick a source with content, or add clip labels first.'
        )

    try:
        if mode == 'markers_timeline':
            output = write_markers_on_timeline(parsed, selects)
            suffix = 'Doza Notes'
        else:
            output = write_selects_as_new_project(parsed, selects)
            suffix = 'Doza Selects'
    except WriterError as e:
        raise MulticamExportError(f'Export failed: {e}')

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
        laid end-to-end as mc-clips (multicam) or asset-clips (sync-clip).
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
        opened_in, err = _hand_file_to_nle(out_path, 'fcp')
        if err:
            return jsonify({'error': err, 'file': out_path}), 500
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
    import shutil
    project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
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
    """Update speaker label assignments after transcription (bulk rename)."""
    project = get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

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
    """Update a story build (reorder clips, remove clips, etc.)."""
    project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
    builds_path = os.path.join(project_dir, 'story_builds.json')

    if not os.path.exists(builds_path):
        return jsonify({'error': 'No builds found'}), 404

    with open(builds_path, 'r') as f:
        builds = json.load(f)

    data = request.json or {}
    for i, b in enumerate(builds):
        if b['id'] == build_id:
            if 'clips' in data:
                builds[i]['clips'] = data['clips']
            if 'story_title' in data:
                builds[i]['story_title'] = data['story_title']
            with open(builds_path, 'w') as f:
                json.dump(builds, f, indent=2)
            return jsonify({'status': 'updated', 'build': builds[i]})

    return jsonify({'error': 'Build not found'}), 404


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
        opened_in, err = _hand_file_to_nle(result.file_path, nle)
        if err:
            return jsonify({'error': err, 'file': result.file_path}), 500
        return jsonify({
            'status': 'ok', 'delivery': 'nle', 'opened_in': opened_in,
            'nle': nle, 'nle_name': NLE_DISPLAY_NAMES[nle],
            'file': result.file_path, 'filename': result.filename,
            'format_name': result.format_name,
        })

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


@app.route('/tunnel/start', methods=['POST'])
def start_tunnel():
    """Start a Cloudflare quick tunnel and return the public URL."""
    global _tunnel_process, _tunnel_url

    # Already running?
    if _tunnel_process and _tunnel_process.poll() is None and _tunnel_url:
        return jsonify({'url': _tunnel_url, 'status': 'running'})

    cloudflared = _find_cloudflared()
    if not cloudflared:
        return jsonify({'error': 'cloudflared not installed. Run: brew install cloudflared'}), 500

    # Start tunnel in background
    _tunnel_process = subprocess.Popen(
        [cloudflared, 'tunnel', '--url', 'http://127.0.0.1:5050'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Read output in a thread to capture the URL
    url_found = threading.Event()

    def _read_output():
        global _tunnel_url
        for line in _tunnel_process.stdout:
            # Cloudflare prints the URL like: https://xxxx-xxxx.trycloudflare.com
            match = _re.search(r'(https://[a-zA-Z0-9-]+\.trycloudflare\.com)', line)
            if match:
                _tunnel_url = match.group(1)
                url_found.set()

    t = threading.Thread(target=_read_output, daemon=True)
    t.start()

    # Wait up to 15 seconds for the URL
    url_found.wait(timeout=15)

    if _tunnel_url:
        return jsonify({'url': _tunnel_url, 'status': 'started'})
    else:
        return jsonify({'error': 'Tunnel started but URL not detected yet. Try again in a few seconds.'}), 500


@app.route('/tunnel/stop', methods=['POST'])
def stop_tunnel():
    """Stop the Cloudflare tunnel."""
    global _tunnel_process, _tunnel_url
    if _tunnel_process:
        _tunnel_process.terminate()
        _tunnel_process = None
    _tunnel_url = None
    return jsonify({'status': 'stopped'})


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
