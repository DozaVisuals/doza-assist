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
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 * 1024  # 32 GB — My Style imports multiple large masters

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
    language = data.get('language', 'en').strip()

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
        'language': language,
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
    language = request.form.get('language', 'en').strip()

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
        'language': language,
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
                           detected_framerate=detected_framerate)


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
        from ai_analysis import analyze_transcript, generate_segment_vectors
        result = analyze_transcript(
            project['transcript'],
            project_name=project['name'],
            analysis_type=analysis_type
        )
        project['analysis'] = result

        # Also generate structured segment vectors and persist them alongside meta.
        # This is the source of truth Story Builder + Clips badges read from.
        segment_vectors = []
        try:
            segment_vectors = generate_segment_vectors(
                project['transcript'],
                project_name=project['name'],
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

        save_project(project_id, project)
        return jsonify({
            'status': 'analyzed',
            'analysis': result,
            'segment_vectors': segment_vectors,
        })

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
    profile_id = data.get('profile_id')  # session-only override from the UI

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
                profile_id=profile_id,
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
                profile_id=profile_id,
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

    def _get_video_resolution(path):
        """Detect video width and height using ffprobe."""
        if not path or not os.path.exists(path):
            return 1920, 1080
        ffprobe = shutil.which('ffprobe')
        if not ffprobe:
            for candidate in ['/opt/homebrew/bin/ffprobe', '/usr/local/bin/ffprobe']:
                if os.path.isfile(candidate):
                    ffprobe = candidate
                    break
        if not ffprobe:
            return 1920, 1080
        try:
            result = subprocess.run([
                ffprobe, '-v', 'quiet',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=p=0',
                path
            ], capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(',')
                if len(parts) >= 2:
                    return int(parts[0]), int(parts[1])
        except Exception:
            pass
        return 1920, 1080

    def _get_video_framerate(path):
        """Detect video framerate using ffprobe. Returns nearest standard rate."""
        if not path or not os.path.exists(path):
            return None
        ffprobe = shutil.which('ffprobe')
        if not ffprobe:
            for candidate in ['/opt/homebrew/bin/ffprobe', '/usr/local/bin/ffprobe']:
                if os.path.isfile(candidate):
                    ffprobe = candidate
                    break
        if not ffprobe:
            return None
        try:
            result = subprocess.run([
                ffprobe, '-v', 'quiet',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate',
                '-of', 'csv=p=0',
                path
            ], capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                num, den = result.stdout.strip().split('/')
                fps = float(num) / float(den)
                # Snap to nearest standard framerate
                standards = [23.976, 24.0, 25.0, 29.97, 30.0, 59.94, 60.0]
                return min(standards, key=lambda s: abs(s - fps))
        except Exception:
            pass
        return None

    # Get source file path and media duration for cut-based export
    source_path = project.get('source_path', project.get('filepath', ''))
    media_duration = None
    if project.get('transcript') and project['transcript'].get('duration'):
        media_duration = project['transcript']['duration']

    detected_fps = _get_video_framerate(source_path)
    framerate = detected_fps or request.json.get('framerate', 23.976)
    export_mode = request.json.get('mode', 'cuts')  # 'cuts', 'markers', 'both'

    # Detect video resolution from source file
    width, height = _get_video_resolution(source_path)

    fcpxml_content = generate_fcpxml(
        markers=markers,
        project_name=project['name'],
        framerate=framerate,
        source_path=source_path,
        media_duration=media_duration,
        mode=export_mode,
        width=width,
        height=height,
    )

    # Build clean filename
    name = project['name']
    if export_type == 'labels':
        count = len(markers)
        if count == 1:
            clip_title = markers[0].get('text', 'Clip')[:40].strip()
            suffix = clip_title
        else:
            total = request.json.get('total_clips', count)
            suffix = 'All Clips' if count >= total else f'{count} Clips'
    elif export_type == 'social':
        suffix = 'Social Clips'
    elif export_type == 'story':
        suffix = 'Story Beats'
    elif export_type == 'all':
        suffix = 'Full Export'
    else:
        suffix = export_type

    export_filename = f"{name.strip().rstrip('_')} - {suffix.strip().rstrip('_')}.fcpxml".replace('/', '-').replace('_', ' ')
    export_path = os.path.join(app.config['EXPORTS_DIR'], export_filename)

    try:
        os.makedirs(app.config['EXPORTS_DIR'], exist_ok=True)
        with open(export_path, 'w') as f:
            f.write(fcpxml_content)
    except Exception as e:
        app.logger.error('Export failed writing %s: %s', export_path, e)
        return jsonify({'error': f'Export failed: {e}'}), 500

    return send_file(export_path, as_attachment=True, download_name=export_filename)


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
        from ai_analysis import build_story
        # Prefer pre-generated segment vectors — much more consistent across runs.
        segment_vectors = load_segment_vectors(project_id)
        result = build_story(
            project['transcript'],
            message=message,
            project_name=project.get('name', 'Interview'),
            segment_vectors=segment_vectors or None,
            profile_id=profile_id,
        )

        # Save the build to story_builds.json
        project_dir = os.path.join(app.config['PROJECTS_DIR'], project_id)
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
            'clips': result.get('clips', []),
        }
        builds.append(build_entry)

        with open(builds_path, 'w') as f:
            json.dump(builds, f, indent=2)

        return jsonify({'status': 'built', 'build': build_entry})
    except Exception as e:
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

    builds = [b for b in builds if b['id'] != build_id]

    with open(builds_path, 'w') as f:
        json.dump(builds, f, indent=2)

    return jsonify({'status': 'deleted'})


@app.route('/project/<project_id>/story/export', methods=['POST'])
def story_export(project_id):
    """Export a story build as FCPXML timeline."""
    project = get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found'}), 404

    from fcpxml_export import generate_story_fcpxml

    data = request.json or {}
    clips = data.get('clips', [])
    story_title = data.get('story_title', 'Story')

    if not clips:
        return jsonify({'error': 'No clips in sequence'}), 400

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

    # Convert clips to markers format
    markers = []
    for clip in clips:
        markers.append({
            'start': _to_seconds(clip.get('start_time', 0)),
            'end': _to_seconds(clip.get('end_time', 0)),
            'text': clip.get('title', 'Clip'),
            'note': clip.get('editorial_note', ''),
        })

    source_path = project.get('source_path', project.get('filepath', ''))
    media_duration = None
    if project.get('transcript') and project['transcript'].get('duration'):
        media_duration = project['transcript']['duration']

    # Detect resolution and framerate
    source_ext = os.path.splitext(source_path or '')[1].lower()
    is_video = source_ext in ('.mp4', '.mov', '.mxf', '.avi', '.mkv')
    width, height = 1920, 1080
    detected_fps = None
    if is_video and source_path and os.path.exists(source_path):
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
                    '-show_entries', 'stream=width,height,r_frame_rate',
                    '-of', 'csv=p=0', source_path
                ], capture_output=True, text=True, timeout=10)
                if result.returncode == 0 and result.stdout.strip():
                    parts = result.stdout.strip().split(',')
                    if len(parts) >= 2:
                        width, height = int(parts[0]), int(parts[1])
                    if len(parts) >= 3:
                        try:
                            num, den = parts[2].split('/')
                            fps = float(num) / float(den)
                            standards = [23.976, 24.0, 25.0, 29.97, 30.0, 59.94, 60.0]
                            detected_fps = min(standards, key=lambda s: abs(s - fps))
                        except Exception:
                            pass
            except Exception:
                pass

    framerate = detected_fps or data.get('framerate', 23.976)

    fcpxml_content = generate_story_fcpxml(
        markers=markers,
        project_name=project['name'],
        story_title=story_title,
        framerate=framerate,
        source_path=source_path,
        media_duration=media_duration,
        width=width,
        height=height,
    )

    export_filename = f"{project['name'].strip()} - {story_title.strip()}.fcpxml".replace('/', '-')
    export_path = os.path.join(app.config['EXPORTS_DIR'], export_filename)

    try:
        os.makedirs(app.config['EXPORTS_DIR'], exist_ok=True)
        with open(export_path, 'w') as f:
            f.write(fcpxml_content)
    except Exception as e:
        app.logger.error('Story export failed writing %s: %s', export_path, e)
        return jsonify({'error': f'Export failed: {e}'}), 500

    return send_file(export_path, as_attachment=True, download_name=export_filename)


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
    profile = _active_profile_for_page()
    all_profiles = edna_profiles.list_profiles()
    snapshots = edna_snapshots.list_snapshots(profile['id']) if profile else []
    return render_template(
        'my_style.html',
        profile=profile,
        all_profiles=all_profiles,
        snapshots=snapshots,
    )


# ── Profile CRUD ────────────────────────────────────────────────────────────

@app.route('/api/editorial_dna/profiles', methods=['GET'])
def edna_list_profiles():
    return jsonify({
        'profiles': edna_profiles.list_profiles(),
        'active_profile_id': edna_profiles.get_active_profile_id(),
    })


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
    ok = edna_profiles.set_active(profile_id)
    if not ok:
        return jsonify({'error': 'Profile not found'}), 404
    return jsonify({'active_profile_id': profile_id})


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

    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400

    # Which profile are we importing into? Optional query/form field; defaults
    # to the currently-active profile, creating a new "My Style" profile if
    # none exists yet.
    target_profile_id = request.form.get('profile_id') or edna_profiles.get_active_profile_id()
    if not target_profile_id:
        target_profile_id = edna_profiles.create_profile('My Style')
    # Make sure the target is active so the import page re-renders with it
    edna_profiles.set_active(target_profile_id)

    # Supported extensions for My Style import
    style_extensions = {'mp4', 'mov', 'm4v', 'mkv', 'mp3', 'wav', 'm4a', 'aac', 'flac', 'aif', 'aiff'}

    # IMPORTANT: Flask's request.files stream closes as soon as the response
    # generator starts yielding — so we must save every upload to disk BEFORE
    # entering the streaming generator. Otherwise we get "read of closed file".
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

            if ext not in style_extensions:
                yield json.dumps({'file': fname, 'status': 'skipped', 'reason': 'unsupported format', 'progress': i + 1, 'total': total_files}) + '\n'
                continue

            if tmp_path is None:
                yield json.dumps({'file': fname, 'status': 'error', 'reason': 'failed to stage upload', 'progress': i + 1, 'total': total_files}) + '\n'
                continue

            yield json.dumps({'file': fname, 'status': 'processing', 'step': 'saving', 'progress': i + 1, 'total': total_files}) + '\n'

            try:
                # Extract audio
                yield json.dumps({'file': fname, 'status': 'processing', 'step': 'extracting audio'}) + '\n'
                audio_path = extract_audio(tmp_path, project_dir=tmp_dir)

                # Transcribe
                yield json.dumps({'file': fname, 'status': 'processing', 'step': 'transcribing'}) + '\n'
                transcript = transcribe_file(audio_path, project_dir=tmp_dir)

                # Analyze
                yield json.dumps({'file': fname, 'status': 'processing', 'step': 'analyzing'}) + '\n'
                metrics = edna_analyze(transcript)
                file_duration = transcript.get('duration', 0)

                # Merge with existing
                if merged and 'speech_pacing' in merged:
                    merged_metrics = merge_metrics(merged, metrics, file_duration)
                else:
                    merged_metrics = {k: v for k, v in metrics.items() if not k.startswith('_')}

                # Keep _raw from latest for classifier (will re-classify on merged)
                merged_metrics['_raw'] = metrics.get('_raw', {})
                merged = merged_metrics

                # Capture the raw transcript text so the v2.1 structured
                # analysis pass can run later. We keep this local to the
                # profile folder — it never leaves the machine.
                transcript_text = ' '.join(
                    seg.get('text', '') for seg in transcript.get('segments', [])
                ).strip()

                source_files.append({
                    'filename': fname,
                    'imported_at': datetime.now().isoformat(),
                    'duration_seconds': round(file_duration, 2),
                    'transcribed_at': datetime.now().isoformat(),
                    'transcript_text': transcript_text,
                })

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
    app.run(host='127.0.0.1', port=5050, debug=True)
