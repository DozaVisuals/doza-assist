#!/bin/bash
# ── Doza Assist Installer ──
# Run this once from the project root.
# Usage:  bash install.sh          — normal install
#         bash install.sh --clean  — wipe everything and start fresh

cd "$(dirname "$0")"

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
RESET='\033[0m'

LOG_FILE="$(pwd)/install_log.txt"

# ── Logging: write to file and terminal ──
log() {
    local msg="$1"
    echo "$msg" | tee -a "$LOG_FILE"
}

log_raw() {
    # No tee — used for color output that we also want stripped in the log
    local msg="$1"
    local plain
    plain=$(echo -e "$msg" | sed 's/\x1b\[[0-9;]*m//g')
    echo -e "$msg"
    echo "$plain" >> "$LOG_FILE"
}

fail() {
    local msg="$1"
    local hint="$2"
    log_raw ""
    log_raw "${RED}${BOLD}ERROR:${RESET} $msg"
    if [ -n "$hint" ]; then
        log_raw "${YELLOW}What to do:${RESET} $hint"
    fi
    log_raw ""
    log_raw "Full install log saved to: $LOG_FILE"
    log_raw "Share this file when reporting issues."
    echo "" >> "$LOG_FILE"
    exit 1
}

# ── Ensure Homebrew on PATH ──
ensure_homebrew_on_path() {
    if [ -d "/opt/homebrew/bin" ]; then
        export PATH="/opt/homebrew/bin:$PATH"
    elif [ -d "/usr/local/bin" ]; then
        export PATH="/usr/local/bin:$PATH"
    fi
}

ensure_homebrew_on_path

# ── Header ──
echo "" | tee "$LOG_FILE"   # reset log on each run
log_raw "  ╔═══════════════════════════════════╗"
log_raw "  ║       Doza Assist Setup           ║"
log_raw "  ╚═══════════════════════════════════╝"
log_raw ""
log "  Install started: $(date)"
log_raw ""

# ── --clean flag: wipe and reinstall ──
if [[ "$1" == "--clean" ]]; then
    log_raw "${YELLOW}--clean flag detected. Running uninstaller before reinstalling...${RESET}"
    log_raw ""
    if [ -f "./uninstall.sh" ]; then
        bash ./uninstall.sh
        echo ""
        log_raw "${GREEN}Clean complete. Starting fresh install...${RESET}"
        log_raw ""
    else
        log_raw "${YELLOW}uninstall.sh not found — continuing with fresh install anyway.${RESET}"
        # Manual cleanup of venv at minimum
        rm -rf venv
    fi
fi

# ── Step 0: Xcode Command Line Tools ──
log_raw "${BOLD}Checking for Xcode Command Line Tools...${RESET}"

if ! xcode-select -p &>/dev/null; then
    log_raw ""
    log_raw "${YELLOW}macOS needs to install Command Line Tools first.${RESET}"
    log_raw ""
    log_raw "A dialog box should appear on your screen."
    log_raw "Click ${BOLD}Install${RESET}, wait for it to finish, then press Enter here to continue."
    log_raw ""

    xcode-select --install 2>/dev/null || true

    read -p "Press Enter once the Command Line Tools install is complete... " _

    if ! xcode-select -p &>/dev/null; then
        fail \
            "Xcode Command Line Tools are still not installed." \
            "Run 'xcode-select --install' in Terminal, wait for the installer to finish, then run 'bash install.sh' again."
    fi
fi

log_raw "${GREEN}✓ Xcode Command Line Tools found: $(xcode-select -p)${RESET}"

# ── Step 1: Python 3.11+ ──
log_raw ""
log_raw "${BOLD}Checking for Python 3.11+...${RESET}"

find_python() {
    for candidate in python3.13 python3.12 python3.11 python3; do
        local py
        py=$(command -v "$candidate" 2>/dev/null || true)
        if [ -n "$py" ]; then
            local ver
            ver=$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
            if [ -n "$ver" ]; then
                local major minor
                major=$(echo "$ver" | cut -d. -f1)
                minor=$(echo "$ver" | cut -d. -f2)
                if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
                    echo "$py"
                    return 0
                fi
            fi
        fi
    done
    return 1
}

PYTHON=$(find_python 2>/dev/null || true)

if [ -z "$PYTHON" ]; then
    log_raw "${YELLOW}Python 3.11+ not found. Installing via Homebrew...${RESET}"

    if ! command -v brew &>/dev/null; then
        fail \
            "Homebrew is not installed and Python 3.11+ was not found." \
            "Install Homebrew first: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\" — then run 'bash install.sh' again."
    fi

    brew install python@3.12 >> "$LOG_FILE" 2>&1 || \
        fail \
            "Python installation via Homebrew failed." \
            "This usually means Xcode Command Line Tools aren't fully installed. Try running 'xcode-select --install', then 'bash install.sh --clean'."

    ensure_homebrew_on_path
    PYTHON=$(find_python 2>/dev/null || true)

    if [ -z "$PYTHON" ]; then
        fail \
            "Python was installed but couldn't be found on PATH." \
            "Close this Terminal window, open a new one, and run 'bash install.sh' again."
    fi
fi

log_raw "${GREEN}✓ Python found: $PYTHON ($($PYTHON --version 2>&1))${RESET}"

# ── Step 2: Virtual Environment ──
log_raw ""
log_raw "${BOLD}Setting up Python virtual environment...${RESET}"

if [ ! -d "venv" ]; then
    $PYTHON -m venv venv >> "$LOG_FILE" 2>&1 || \
        fail \
            "Failed to create the Python virtual environment." \
            "Try running 'bash install.sh --clean' to start fresh. If it still fails, make sure Python is working: run '$PYTHON --version'."
fi

# shellcheck source=/dev/null
source venv/bin/activate || \
    fail \
        "Failed to activate the virtual environment." \
        "Try deleting the venv folder and running 'bash install.sh' again."

log_raw "${GREEN}✓ Virtual environment ready${RESET}"

# ── Step 3: Core pip packages ──
log_raw ""
log_raw "${BOLD}Installing core dependencies...${RESET}"

pip install --upgrade pip >> "$LOG_FILE" 2>&1 || \
    fail \
        "Failed to upgrade pip." \
        "Try running 'bash install.sh --clean' to start fresh."

pip install flask werkzeug requests >> "$LOG_FILE" 2>&1 || \
    fail \
        "Failed to install Flask and core packages." \
        "Check your internet connection and try again. If the problem persists, run 'bash install.sh --clean'."

log_raw "${GREEN}✓ Core packages installed${RESET}"

# ── Step 4: ffmpeg ──
log_raw ""
log_raw "${BOLD}Checking for ffmpeg...${RESET}"

if ! command -v ffmpeg &>/dev/null; then
    log_raw "${YELLOW}ffmpeg not found.${RESET} Trying to install via Homebrew..."

    if command -v brew &>/dev/null; then
        brew install ffmpeg >> "$LOG_FILE" 2>&1 && \
            log_raw "${GREEN}✓ ffmpeg installed${RESET}" || \
            log_raw "${YELLOW}⚠  ffmpeg install failed. You can install it later with: brew install ffmpeg${RESET}"
    else
        log_raw "${YELLOW}⚠  Homebrew not found. Install ffmpeg manually: brew install ffmpeg${RESET}"
        log_raw "   ffmpeg is required for video file audio extraction."
    fi
else
    log_raw "${GREEN}✓ ffmpeg found: $(command -v ffmpeg)${RESET}"
fi

# ── Step 5: Transcription Engine ──
log_raw ""
log_raw "${BOLD}Choose your transcription engine:${RESET}"
echo ""
echo "  1) Parakeet TDT via MLX  (RECOMMENDED — fast, Apple Silicon native)"
echo "  2) WhisperX              (transcription + speaker diarization)"
echo "  3) Lightning Whisper MLX (fastest, no speaker labels)"
echo "  4) Standard Whisper      (basic, most compatible)"
echo "  5) Skip                  (install later)"
echo ""
read -p "Enter choice [1]: " engine_choice
engine_choice=${engine_choice:-1}

case $engine_choice in
    1)
        log_raw "${BOLD}Installing Parakeet TDT (MLX)...${RESET}"
        pip install mlx-audio >> "$LOG_FILE" 2>&1 && \
            log_raw "${GREEN}✓ Parakeet / MLX installed${RESET}" || \
            log_raw "${YELLOW}⚠  Parakeet install failed. You can install it later: pip install mlx-audio${RESET}"
        ;;
    2)
        log_raw "${BOLD}Installing WhisperX...${RESET}"
        pip install whisperx >> "$LOG_FILE" 2>&1 && \
            log_raw "${GREEN}✓ WhisperX installed${RESET}" || \
            fail \
                "WhisperX installation failed." \
                "Check the log at $LOG_FILE. You can try 'pip install whisperx' manually after running 'source venv/bin/activate'."
        log_raw ""
        log_raw "${YELLOW}Speaker diarization needs a free HuggingFace token:${RESET}"
        log_raw "  1. Create an account at huggingface.co"
        log_raw "  2. Accept terms at: huggingface.co/pyannote/speaker-diarization-3.1"
        log_raw "  3. Get your token at: huggingface.co/settings/tokens"
        log_raw "  4. Add to your shell: export HF_TOKEN=your_token_here"
        ;;
    3)
        log_raw "${BOLD}Installing Lightning Whisper MLX...${RESET}"
        pip install lightning-whisper-mlx >> "$LOG_FILE" 2>&1 && \
            log_raw "${GREEN}✓ Lightning Whisper MLX installed${RESET}" || \
            log_raw "${YELLOW}⚠  Install failed. Try manually: pip install lightning-whisper-mlx${RESET}"
        ;;
    4)
        log_raw "${BOLD}Installing OpenAI Whisper...${RESET}"
        pip install openai-whisper >> "$LOG_FILE" 2>&1 && \
            log_raw "${GREEN}✓ OpenAI Whisper installed${RESET}" || \
            log_raw "${YELLOW}⚠  Install failed. Try manually: pip install openai-whisper${RESET}"
        ;;
    5)
        log_raw "${YELLOW}Skipping transcription engine.${RESET}"
        log_raw "Install later with: source venv/bin/activate && pip install mlx-audio"
        ;;
    *)
        log_raw "${YELLOW}Unrecognized choice — skipping transcription engine.${RESET}"
        ;;
esac

# ── Done ──
log_raw ""
log_raw "══════════════════════════════════════════"
log_raw "${GREEN}${BOLD}  ✅ Setup complete!${RESET}"
log_raw ""
log_raw "  Start the app:   ${BOLD}./start.sh${RESET}"
log_raw "  Then open:       ${BOLD}http://localhost:5050${RESET}"
log_raw ""
log_raw "  Optional — AI analysis (free, local):"
log_raw "    brew install ollama"
log_raw "    ollama serve"
log_raw "    ollama pull gemma4"
log_raw ""
log_raw "  Optional — Cloud AI (higher quality):"
log_raw "    export ANTHROPIC_API_KEY=sk-ant-..."
log_raw "══════════════════════════════════════════"
log_raw ""
log "  Install log saved to: $LOG_FILE"
log_raw ""
