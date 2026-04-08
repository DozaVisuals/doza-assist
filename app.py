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
import re as _re
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['PROJECTS_DIR'] = os.path.join(os.path.dirname(__file__), 'projects')
app.config['EXPORTS_DIR'] = os.path.join(os.path.dirname(__file__), 'exports')

# Small file drag-and-drop limit (500MB)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'mp4', 'mov', 'aac', 'm4a', 'flac', 'aif', 'aiff', 'mxf'}

os.makedirs(app.config['PROJECTS_DIR'], exist_ok=True)
os.makedirs(app.config['EXPORTS_DIR'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_project(project_id):
    """Load a project's metadata."""
    project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
    meta_path = os.path.join(project_dir, 'meta.json')
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, 'r') as f:
        return json.load(f)


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

    if not os.path.isfile(source_path):
        return jsonify({'error': 'Path is not a file'}), 400

    if not allowed_file(source_path):
        ext = source_path.rsplit('.', 1)[-1].lower() if '.' in source_path else 'unknown'
        return jsonify({'error': f'Unsupported file type: .{ext}'}), 400

    project_name = data.get('project_name', '').strip()
    client_name = data.get('client_name', '').strip()
    interviewer_name = data.get('interviewer_name', 'Interviewer').strip()
    subject_name = data.get('subject_name', 'Subject').strip()
    num_speakers = int(data.get('num_speakers', 2))

    if not project_name:
        project_name = Path(source_path).stem.replace('_', ' ').replace('-', ' ')

    file_size = os.path.getsize(source_path)

    project_id = str(uuid.uuid4())[:8]
    project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
    os.makedirs(project_dir, exist_ok=True)

    meta = {
        'id': project_id,
        'name': project_name,
        'client_name': client_name,
        'interviewer_name': interviewer_name,
        'subject_name': subject_name,
        'num_speakers': num_speakers,
        'filename': os.path.basename(source_path),
        'source_path': source_path,
        'filepath': source_path,
        'file_size': file_size,
        'file_size_formatted': format_file_size(file_size),
        'created_at': datetime.now().isoformat(),
        'status': 'uploaded',
        'transcript': None,
        'analysis': None,
        'client_selects': [],
        'social_clips': [],
    }
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

    if not project_name:
        project_name = file.filename.rsplit('.', 1)[0]

    project_id = str(uuid.uuid4())[:8]
    project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
    os.makedirs(project_dir, exist_ok=True)

    filename = secure_filename(file.filename)
    filepath = os.path.join(project_dir, filename)
    file.save(filepath)

    file_size = os.path.getsize(filepath)

    meta = {
        'id': project_id,
        'name': project_name,
        'client_name': client_name,
        'interviewer_name': interviewer_name,
        'subject_name': subject_name,
        'filename': filename,
        'source_path': filepath,
        'filepath': filepath,
        'file_size': file_size,
        'file_size_formatted': format_file_size(file_size),
        'created_at': datetime.now().isoformat(),
        'status': 'uploaded',
        'transcript': None,
        'analysis': None,
        'client_selects': [],
        'social_clips': [],
    }
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

    for root_dir in search_roots:
        try:
            for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=True):
                # Skip hidden directories and system dirs
                dirnames[:] = [d for d in dirnames if not d.startswith('.') and d not in ('node_modules', '__pycache__', '.Trash')]

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
    project_ids = [pid.strip() for pid in project_id.split(',') if pid.strip()]

    # Load all requested projects
    projects = []
    for pid in project_ids:
        p = get_project(pid)
        if p:
            exists, _ = check_source_file(p)
            p['source_exists'] = exists
            projects.append(p)

    if not projects:
        return redirect(url_for('dashboard'))

    # Primary project (first one) — used as fallback for single-project features
    project = projects[0]

    # All transcribed projects for the selector dropdown
    all_projects = [p for p in list_projects() if p.get('transcript')]

    # Assign a color index to each active project for visual distinction
    project_colors = ['accent', 'green', 'purple', 'orange', 'red']
    projects_meta = []
    for i, p in enumerate(projects):
        projects_meta.append({
            'id': p['id'],
            'name': p.get('name', 'Untitled'),
            'color': project_colors[i % len(project_colors)],
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

    # Determine if source is video for the player
    source_ext = os.path.splitext(project.get('source_path', '') or '')[1].lower()
    is_video = source_ext in ('.mp4', '.mov', '.mxf', '.avi', '.mkv')

    return render_template('project.html',
                           project=project,
                           projects=projects,
                           projects_meta=projects_meta,
                           all_projects=all_projects,
                           active_ids=project_ids,
                           paragraphs=paragraphs,
                           is_multi=is_multi,
                           is_shared=False,
                           is_video=is_video)


@app.route('/project/<project_id>/media')
def serve_media(project_id):
    """Stream the project's source media file (video or audio)."""
    project = get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    # Always serve the original source file for video playback
    source_path = project.get('source_path', project.get('filepath', ''))
    if source_path and os.path.exists(source_path):
        return send_file(source_path)

    # Fall back to extracted audio
    project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
    audio_wav = os.path.join(project_dir, 'audio.wav')
    if os.path.exists(audio_wav):
        return send_file(audio_wav, mimetype='audio/wav')

    return jsonify({'error': 'Source file not found'}), 404


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

    project['status'] = 'transcribing'
    save_project(project_id, project)

    try:
        from transcribe import transcribe_file

        # Pass both source path and project directory for audio extraction
        project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
        num_speakers = project.get('num_speakers', 2)
        result = transcribe_file(
            source_path,
            project_dir=project_dir,
            speaker_labels={
                'SPEAKER_00': project.get('interviewer_name', 'Interviewer'),
                'SPEAKER_01': project.get('subject_name', 'Subject'),
            },
            num_speakers=num_speakers,
        )
        project['transcript'] = result
        project['status'] = 'transcribed'
        save_project(project_id, project)
        return jsonify({'status': 'transcribed', 'transcript': result})

    except Exception as e:
        project['status'] = 'error'
        project['error'] = str(e)
        save_project(project_id, project)
        return jsonify({'error': str(e)}), 500


@app.route('/project/<project_id>/analyze', methods=['POST'])
def analyze(project_id):
    """Run AI analysis on the transcript."""
    project = get_project(project_id)
    if not project or not project.get('transcript'):
        return jsonify({'error': 'No transcript available'}), 400

    analysis_type = request.json.get('type', 'all')  # 'story', 'social', 'all'

    try:
        from ai_analysis import analyze_transcript
        result = analyze_transcript(
            project['transcript'],
            project_name=project['name'],
            analysis_type=analysis_type
        )
        project['analysis'] = result
        save_project(project_id, project)
        return jsonify({'status': 'analyzed', 'analysis': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


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

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    try:
        from ai_analysis import chat_about_transcript

        if len(projects_for_chat) == 1:
            p = projects_for_chat[0]
            reply = chat_about_transcript(
                transcript=p['transcript'],
                message=message,
                history=history,
                project_name=p.get('name', 'Interview'),
                analysis=p.get('analysis'),
            )
        else:
            # Multi-project: combine transcripts with project labels
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
            )

        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
    project['color_labels'] = data.get('color_labels', {})
    project['labeled_sections'] = data.get('labeled_sections', [])
    save_project(project_id, project)
    return jsonify({'status': 'saved', 'count': len(project['labeled_sections'])})


@app.route('/project/<project_id>/export/fcpxml', methods=['POST'])
def export_fcpxml(project_id):
    """Export selections as FCPXML markers."""
    project = get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    from fcpxml_export import generate_fcpxml

    export_type = request.json.get('type', 'selects')  # 'selects', 'social', 'all'
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

    if export_type in ('social', 'all'):
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

    if export_type in ('story', 'all'):
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

    if export_type in ('labels', 'all'):
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

    framerate = request.json.get('framerate', 23.976)
    export_mode = request.json.get('mode', 'cuts')  # 'cuts', 'markers', 'both'

    # Get source file path and media duration for cut-based export
    source_path = project.get('source_path', project.get('filepath', ''))
    media_duration = None
    if project.get('transcript') and project['transcript'].get('duration'):
        media_duration = project['transcript']['duration']

    fcpxml_content = generate_fcpxml(
        markers=markers,
        project_name=project['name'],
        framerate=framerate,
        source_path=source_path,
        media_duration=media_duration,
        mode=export_mode,
    )

    export_filename = f"{project['name'].replace(' ', '_')}_{export_type}.fcpxml"
    export_path = os.path.join(app.config['EXPORTS_DIR'], export_filename)
    with open(export_path, 'w') as f:
        f.write(fcpxml_content)

    return send_file(export_path, as_attachment=True, download_name=export_filename)


@app.route('/review/<project_id>')
def client_review(project_id):
    """Legacy review portal — redirect to shared view."""
    return redirect(f'/share/{project_id}')


@app.route('/share/<project_id>')
def shared_view(project_id):
    """Shared project view — full project experience, read-only."""
    project = get_project(project_id)
    if not project or not project.get('transcript'):
        return render_template('review_unavailable.html')

    source_exists, _ = check_source_file(project)
    project['source_exists'] = source_exists

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

    return render_template('project.html',
                           project=project,
                           projects=[project],
                           projects_meta=projects_meta,
                           all_projects=all_projects,
                           active_ids=project_ids,
                           paragraphs=paragraphs,
                           is_multi=False,
                           is_shared=True,
                           is_video=is_video)


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

    project['name'] = name
    save_project(project_id, project)
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


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5050, debug=True)
