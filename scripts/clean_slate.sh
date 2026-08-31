#!/bin/bash
# ── Doza Assist Clean Slate ──
# Factory-reset a testing machine: removes EVERYTHING Doza Assist installs
# so the next install starts truly fresh.
#
# Usage:
#   bash scripts/clean_slate.sh               — interactive (one confirmation)
#   bash scripts/clean_slate.sh --yes         — no prompts, wipe it all
#   bash scripts/clean_slate.sh --keep-ollama — leave Ollama + models alone
#   bash scripts/clean_slate.sh --with-deps   — also remove Homebrew ffmpeg
#                                               and python@3.12 (shared tools!)
#
# Unlike uninstall.sh (the gentle user-facing uninstaller, which preserves
# projects and transcripts), this script removes app data too. It is meant
# for test machines only.

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
RESET='\033[0m'

if [[ "$(uname)" != "Darwin" ]]; then
    echo "This script only supports macOS (all Doza Assist install paths are Mac paths)."
    exit 1
fi

YES=0
KEEP_OLLAMA=0
WITH_DEPS=0
for arg in "$@"; do
    case "$arg" in
        --yes|-y)       YES=1 ;;
        --keep-ollama)  KEEP_OLLAMA=1 ;;
        --with-deps)    WITH_DEPS=1 ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

echo ""
echo "  ╔═══════════════════════════════════════╗"
echo "  ║     Doza Assist — Clean Slate         ║"
echo "  ╚═══════════════════════════════════════╝"
echo ""
echo -e "${RED}${BOLD}This wipes EVERYTHING Doza Assist put on this machine:${RESET}"
echo ""
echo "    • Doza Assist app bundles (/Applications, ~/Applications, Desktop)"
echo "    • All app data — including projects, exports, and transcripts"
echo "    • Preferences, setup state, logs, editorial DNA profiles"
if [ "$KEEP_OLLAMA" -eq 0 ]; then
    echo "    • Ollama (app/binary/service) and ALL its models, Gemma included"
fi
echo "    • Downloaded transcription models (Parakeet, Whisper, pyannote)"
echo "    • Source-tree venv / install logs (if run from a checkout)"
if [ "$WITH_DEPS" -eq 1 ]; then
    echo "    • Homebrew ffmpeg and python@3.12 (--with-deps)"
fi
echo ""
echo "  Doza Assist Core stores no license keys, so there are none to remove."
echo "  A leftover sweep at the end reports anything Doza-named it finds."
echo ""

if [ "$YES" -eq 0 ]; then
    if [ ! -t 0 ]; then
        echo "Non-interactive shell detected. Re-run with --yes to confirm the wipe."
        exit 1
    fi
    read -rp "Wipe this machine clean? (y/n): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo ""
        echo "Cancelled. Nothing was removed."
        echo ""
        exit 0
    fi
fi

echo ""

REMOVED=()
SKIPPED=()

zap() {
    # zap <path> <label> — rm -rf a file/dir if it exists and record it
    local path="$1" label="$2"
    if [ -e "$path" ] || [ -L "$path" ]; then
        echo "  Removing ${label}..."
        rm -rf "$path" && REMOVED+=("$label") || SKIPPED+=("$label (removal failed — try sudo)")
    fi
}

zap_glob() {
    # zap_glob <label> <glob...> — expand globs, zap every match
    local label="$1"; shift
    local matched=0 p
    for p in "$@"; do
        [ -e "$p" ] || [ -L "$p" ] || continue
        rm -rf "$p" && matched=1
    done
    [ "$matched" -eq 1 ] && REMOVED+=("$label")
}

# ── 1. Stop running processes ──
echo -e "${BOLD}Stopping Doza Assist...${RESET}"
osascript -e 'tell application "Doza Assist" to quit' >/dev/null 2>&1 || true
pkill -f "Doza Assist.app" 2>/dev/null || true
# The Flask server the launcher spawns (port 5050)
PORT_PIDS=$(lsof -ti tcp:5050 2>/dev/null || true)
if [ -n "$PORT_PIDS" ]; then
    echo "  Stopping app server on port 5050..."
    kill $PORT_PIDS 2>/dev/null || true
    REMOVED+=("Running app server (port 5050, stopped)")
fi

# ── 2. App bundles ──
echo -e "${BOLD}Removing app bundles...${RESET}"
for dir in "/Applications" "$HOME/Applications" "$HOME/Desktop"; do
    zap "$dir/Doza Assist.app"     "Doza Assist.app ($dir)"
    zap "$dir/Doza Assist Pro.app" "Doza Assist Pro.app ($dir)"
done

# ── 3. App data, preferences, state ──
echo -e "${BOLD}Removing app data and preferences...${RESET}"
zap "$HOME/Library/Application Support/DozaAssist"   "App data (venv, setup state, projects, exports, logs)"
zap "$HOME/Library/Application Support/Doza Assist"  "Preferences (preferences.json)"
zap "$HOME/.doza-assist"                             "Editorial DNA profiles (~/.doza-assist)"
defaults delete com.dozavisuals.transcribe >/dev/null 2>&1 && REMOVED+=("Defaults (com.dozavisuals.transcribe)") || true
zap_glob "Preference plists (com.dozavisuals.*)" "$HOME/Library/Preferences/com.dozavisuals."*.plist
zap_glob "Saved application state"               "$HOME/Library/Saved Application State/com.dozavisuals."*
zap_glob "Caches (com.dozavisuals.*)"            "$HOME/Library/Caches/com.dozavisuals."* "$HOME/Library/Caches/Doza Assist"*
zap_glob "Logs (Doza Assist)"                    "$HOME/Library/Logs/Doza Assist"* "$HOME/Library/Logs/DozaAssist"*

# ── 4. Source-tree leftovers (when run from a checkout) ──
ROOT="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"
if [ -n "$ROOT" ] && [ -f "$ROOT/app.py" ]; then
    echo -e "${BOLD}Cleaning source checkout at $ROOT...${RESET}"
    zap "$ROOT/venv"             "Source-tree venv"
    zap "$ROOT/install_log.txt"  "Install log"
    zap "$ROOT/projects"         "Source-tree projects (dev runs)"
    zap "$ROOT/exports"          "Source-tree exports (dev runs)"
fi

# ── 5. Ollama + models (Gemma lives in ~/.ollama/models) ──
if [ "$KEEP_OLLAMA" -eq 0 ]; then
    echo -e "${BOLD}Removing Ollama and all models...${RESET}"
    if command -v brew &>/dev/null; then
        brew services stop ollama >/dev/null 2>&1 || true
    fi
    osascript -e 'tell application "Ollama" to quit' >/dev/null 2>&1 || true
    pkill -x ollama 2>/dev/null && REMOVED+=("Ollama process (stopped)") || true
    sleep 1
    zap "$HOME/.ollama"                                 "Ollama models and data (~/.ollama — includes Gemma)"
    zap "/Applications/Ollama.app"                      "Ollama.app"
    zap "$HOME/Library/Application Support/Ollama"      "Ollama app data"
    zap_glob "Ollama preferences/state/caches" \
        "$HOME/Library/Preferences/com.electron.ollama"* \
        "$HOME/Library/Saved Application State/com.electron.ollama"* \
        "$HOME/Library/Caches/com.electron.ollama"* \
        "$HOME/Library/Caches/ollama"* \
        "$HOME/Library/LaunchAgents/com.ollama."*
    if command -v brew &>/dev/null && brew list ollama &>/dev/null 2>&1; then
        echo "  Uninstalling Ollama via Homebrew..."
        brew uninstall ollama >/dev/null 2>&1 && REMOVED+=("Ollama (Homebrew)") || SKIPPED+=("Ollama (brew uninstall failed)")
    fi
    zap "/usr/local/bin/ollama"    "Ollama binary (/usr/local/bin)"
    zap "/opt/homebrew/bin/ollama" "Ollama binary (/opt/homebrew/bin)"
else
    SKIPPED+=("Ollama (kept — --keep-ollama)")
fi

# ── 6. Downloaded transcription models ──
echo -e "${BOLD}Removing transcription model caches...${RESET}"
HF_HUB="${XDG_CACHE_HOME:-$HOME/.cache}/huggingface/hub"
zap "$HF_HUB/models--mlx-community--parakeet-tdt-0.6b-v2" "Parakeet TDT model"
zap_glob "Whisper/pyannote models (HuggingFace cache)" \
    "$HF_HUB/models--"*whisper* \
    "$HF_HUB/models--Systran--"* \
    "$HF_HUB/models--pyannote--"* \
    "$HF_HUB/models--mlx-community--"*
zap "${XDG_CACHE_HOME:-$HOME/.cache}/whisper" "Whisper weights cache"

# ── 7. Optional: shared Homebrew dependencies ──
if [ "$WITH_DEPS" -eq 1 ] && command -v brew &>/dev/null; then
    echo -e "${BOLD}Removing Homebrew dependencies (--with-deps)...${RESET}"
    for pkg in ffmpeg python@3.12; do
        if brew list "$pkg" &>/dev/null 2>&1; then
            echo "  Uninstalling $pkg..."
            brew uninstall "$pkg" >/dev/null 2>&1 && REMOVED+=("$pkg (Homebrew)") || SKIPPED+=("$pkg (brew uninstall failed — other packages may depend on it)")
        fi
    done
fi

# ── 8. Leftover sweep ──
echo -e "${BOLD}Sweeping for leftovers...${RESET}"
LEFTOVERS=$(find \
    "$HOME/Library/Application Support" \
    "$HOME/Library/Preferences" \
    "$HOME/Library/Caches" \
    "$HOME/Library/Logs" \
    "$HOME/Library/Saved Application State" \
    "/Applications" \
    -maxdepth 1 -iname "*doza*" 2>/dev/null || true)

# ── Summary ──
echo ""
echo "  ─────────────────────────────────────"
echo -e "  ${GREEN}${BOLD}Clean slate! Removed:${RESET}"
echo ""
for item in "${REMOVED[@]}"; do
    echo -e "  ${GREEN}✓${RESET} $item"
done
if [ ${#REMOVED[@]} -eq 0 ]; then
    echo "  (nothing found — machine was already clean)"
fi

if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo ""
    echo -e "  ${YELLOW}Skipped:${RESET}"
    for item in "${SKIPPED[@]}"; do
        echo "    - $item"
    done
fi

if [ -n "$LEFTOVERS" ]; then
    echo ""
    echo -e "  ${YELLOW}Still present (not created by this repo — remove by hand if unwanted):${RESET}"
    echo "$LEFTOVERS" | sed 's/^/    /'
fi

echo ""
echo "  Note: this script does not delete the source checkout it runs from."
echo "  For a 100% fresh test, delete that folder too and re-clone."
echo ""
echo -e "  ${BOLD}Fresh install:${RESET} git clone the repo, then: bash install.sh"
echo ""
