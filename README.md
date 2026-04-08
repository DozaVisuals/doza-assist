# Doza Assist

A local-first AI-powered interview transcription and editing tool for documentary filmmakers.

Built for any workflow where you need to find the best moments in recorded interviews, conversations, or presentations. Documentary films, corporate video, podcasts, news, legal depositions, customer testimonials, training content. If someone is talking on camera and you need to find what matters, this tool does the heavy lifting.

Transcribe interviews, discover clips with AI, highlight and organize selects, and export pre-cut timelines directly to Final Cut Pro — all running locally on your machine.

<!-- Add your demo video here:
https://github.com/user-attachments/assets/YOUR_VIDEO_ID
or embed a GIF: ![Demo](demo.gif) -->

## Why I Built This

I'm a documentary filmmaker and I needed a way to find story beats and soundbites across hours of interview footage without uploading client material to cloud services. Existing tools were either too expensive, too slow, or required sending sensitive footage to third-party servers. So I built something that runs entirely on my Mac, uses AI locally, and exports directly to my Final Cut Pro timeline.

---

## Features

**Transcription**
- Drag and drop video/audio files (MP4, MOV, WAV, MP3, MXF, etc.)
- Transcribes locally with OpenAI Whisper — no cloud uploads
- Word-level timestamps for precise sync
- Click speaker names to assign who said what

**Transcript Viewer**
- Clean paragraph layout grouped by speaker
- Video player synced to transcript with word-level highlighting
- Click any word to jump to that moment
- Color highlighter — drag across words to create clips (like highlighting in a document)
- 5 renamable color labels for organizing selects

**Clip Library**
- All highlights collected in a visual grid
- Each clip has play/pause, scrub bar, duration, and transcript excerpt
- Checkbox select for batch export
- Add clips from transcript, AI analysis, or AI chat

**AI Analysis** (powered by Ollama — free, local)
- Story structure with beats (hook, context, rising action, climax, resolution)
- Social media clip suggestions with platform recommendations
- Strongest soundbites identified
- Every item has play/scrub controls and one-click "Add to Clips"

**AI Chat**
- Conversational AI that knows your transcript
- Ask for clips, themes, story angles, soundbites
- AI suggests clips with timecodes — play them instantly or add to your library
- Follow-up questions maintain context

**FCPX Export**
- Pre-cut timeline — each clip becomes an actual edit referencing your source media
- Import the .fcpxml and your selects are ready to review in Final Cut Pro
- Keyword ranges on source clip for browser filtering
- Also exports SRT subtitles, plain text, and JSON

**Client Sharing**
- One-click Cloudflare Tunnel generates a public URL
- Clients see the full project: transcript, player, highlighting tools
- No destructive controls exposed — clients can highlight and listen
- No accounts or signups needed

**Project Organization**
- Folder system for organizing by client
- Rename, move, clear, delete projects
- Multi-project workspace — combine interviews in one view

**Dark / Light Theme**
- Toggle between dark and light mode
- Persists across sessions

---

## Quick Start

### Prerequisites
- macOS (tested on Mac Studio M2)
- Python 3.11+
- ffmpeg (`brew install ffmpeg`)
- Ollama for AI features (`https://ollama.com`)

### Install

```bash
git clone https://github.com/DozaVisuals/doza-assist.git
cd doza-assist

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install ffmpeg if you don't have it
brew install ffmpeg
```

### Run

```bash
./start.sh
```

Open **http://localhost:5050**

### AI Setup (optional but recommended)

**Local with Ollama (free, private — recommended):**
```bash
# Install and start Ollama
brew install ollama
ollama serve

# Pull a model (gemma4 recommended)
ollama pull gemma4
```

**Cloud with Claude API (higher quality, optional):**
If you want better AI analysis quality, you can use Anthropic's Claude API as an alternative backend. Transcription still runs locally — only the AI analysis and chat use the API.
```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
# Add to ~/.zshrc to persist
```
The app automatically tries Ollama first and falls back to Claude if configured.

---

## How It Works

1. **Add a file** — Paste a path, browse, or drag a video/audio file
2. **Transcribe** — Click "Transcribe" to process locally with Whisper
3. **Assign speakers** — Click speaker names in the transcript to toggle between speakers
4. **Highlight clips** — Select a color and drag across words to mark selects
5. **Discover with AI** — Run AI Analysis or ask the Chat for clips and story structure
6. **Export to FCPX** — Export pre-cut timeline with your clips as edits on the timeline
7. **Share with clients** — Click Share to generate a public link for client review

---

## Tech Stack

- **Backend:** Python / Flask
- **Frontend:** Vanilla JS, CSS custom properties
- **Transcription:** OpenAI Whisper (local, runs on CPU)
- **AI:** Ollama with Gemma 4 (local, free) or Claude API (optional)
- **Audio:** ffmpeg for extraction
- **Sharing:** Cloudflare Tunnel (free, no account needed)
- **Export:** FCPXML 1.11 with asset references
- **Storage:** JSON files per project (no database)

---

## Project Structure

```
doza-assist/
├── app.py              # Flask server + all routes
├── transcribe.py       # Whisper transcription engine
├── ai_analysis.py      # AI analysis + chat (Ollama/Claude)
├── fcpxml_export.py    # FCPXML generation with pre-cut timelines
├── start.sh            # Launch script
├── install.sh          # First-time setup
├── requirements.txt    # Python dependencies
├── static/
│   └── style.css       # All styles (dark + light themes)
├── templates/
│   ├── dashboard.html   # Projects page with folders
│   ├── project.html     # Main project view (all tabs)
│   └── ...
├── projects/            # User data (gitignored)
└── exports/             # FCPXML exports (gitignored)
```

---

## Privacy

- All transcription runs locally on your machine
- AI analysis uses Ollama (local) by default — nothing leaves your computer
- Audio/video files are never uploaded anywhere
- Client sharing uses a temporary tunnel URL that stops when you quit the app
- All project data stored as local JSON files

---

## License

MIT

---

Built by [Doza Visuals](https://dozavisuals.com)
