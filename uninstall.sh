#!/bin/bash
# ── Doza Assist Uninstaller ──

cd "$(dirname "$0")"

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
RESET='\033[0m'

echo ""
echo "  ╔═══════════════════════════════════════╗"
echo "  ║      Doza Assist Uninstaller          ║"
echo "  ╚═══════════════════════════════════════╝"
echo ""
echo -e "${YELLOW}This will remove Doza Assist and its setup files.${RESET}"
echo ""
echo "  What WILL be removed:"
echo "    • Python virtual environment (venv)"
echo "    • Setup state and install logs"
echo "    • Doza Assist config files"
echo ""
echo "  What will NOT be removed:"
echo "    • Your projects, media files, and transcripts"
echo "    • Homebrew, Python, or ffmpeg"
echo "    • Any other apps on your Mac"
echo ""

read -p "Continue? (y/n): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo ""
    echo "Cancelled. Nothing was removed."
    echo ""
    exit 0
fi

echo ""

REMOVED=()
SKIPPED=()

# ── Remove venv (project root) ──
if [ -d "venv" ]; then
    echo -e "  Removing virtual environment..."
    rm -rf venv
    REMOVED+=("Python virtual environment (venv)")
else
    SKIPPED+=("venv (not found)")
fi

# ── Remove venv (app support dir, used by .app bundle) ──
SUPPORT_DIR="$HOME/Library/Application Support/DozaAssist"
if [ -d "$SUPPORT_DIR/venv" ]; then
    echo -e "  Removing app support virtual environment..."
    rm -rf "$SUPPORT_DIR/venv"
    REMOVED+=("App support virtual environment")
fi

# ── Remove setup state / config ──
if [ -f "$SUPPORT_DIR/setup.json" ]; then
    echo -e "  Removing setup state..."
    rm -f "$SUPPORT_DIR/setup.json"
    REMOVED+=("Setup state (setup.json)")
fi

if [ -f "$SUPPORT_DIR/setup.log" ]; then
    rm -f "$SUPPORT_DIR/setup.log"
    REMOVED+=("Setup log")
fi

# Remove empty support dir if nothing else is in it
if [ -d "$SUPPORT_DIR" ] && [ -z "$(ls -A "$SUPPORT_DIR" 2>/dev/null)" ]; then
    rmdir "$SUPPORT_DIR"
fi

# ── Remove preferences (separate location with space in name) ──
PREFS_DIR="$HOME/Library/Application Support/Doza Assist"
if [ -d "$PREFS_DIR" ]; then
    echo -e "  Removing preferences..."
    rm -rf "$PREFS_DIR"
    REMOVED+=("Preferences (Doza Assist)")
fi

# ── Remove editorial DNA profiles (legacy location) ──
EDITORIAL_DIR="$HOME/.doza-assist"
if [ -d "$EDITORIAL_DIR" ]; then
    echo -e "  Removing editorial DNA profiles..."
    rm -rf "$EDITORIAL_DIR"
    REMOVED+=("Editorial DNA profiles (~/.doza-assist)")
fi

# ── Remove local install log ──
if [ -f "install_log.txt" ]; then
    echo -e "  Removing install log..."
    rm -f install_log.txt
    REMOVED+=("Install log (install_log.txt)")
fi

# ── Stop Ollama if running ──
if pgrep -x "ollama" > /dev/null 2>&1; then
    echo -e "  Stopping Ollama..."
    pkill -x ollama 2>/dev/null || true
    sleep 1
    REMOVED+=("Ollama process (stopped)")
elif command -v ollama &>/dev/null; then
    # Try graceful stop even if pgrep didn't find it
    ollama stop 2>/dev/null || true
fi

# ── Optional: remove Ollama + models ──
echo ""
echo -e "${YELLOW}Ollama is the AI engine that powers analysis and chat features.${RESET}"
echo "  Removing it frees up disk space (~3–10 GB) but will also affect"
echo "  any other apps on your Mac that use Ollama."
echo ""
read -p "Remove Ollama and its downloaded AI models? (y/n): " remove_ollama

if [[ "$remove_ollama" == "y" || "$remove_ollama" == "Y" ]]; then
    # Remove models directory
    if [ -d "$HOME/.ollama" ]; then
        echo -e "  Removing Ollama models and data..."
        rm -rf "$HOME/.ollama"
        REMOVED+=("Ollama models (~/.ollama)")
    fi

    # Remove Ollama app / binary
    if [ -d "/Applications/Ollama.app" ]; then
        echo -e "  Removing Ollama.app..."
        rm -rf "/Applications/Ollama.app"
        REMOVED+=("Ollama.app")
    fi

    if command -v brew &>/dev/null && brew list ollama &>/dev/null 2>&1; then
        echo -e "  Uninstalling Ollama via Homebrew..."
        brew uninstall ollama 2>/dev/null && REMOVED+=("Ollama (Homebrew)") || true
    elif [ -f "/usr/local/bin/ollama" ]; then
        rm -f "/usr/local/bin/ollama"
        REMOVED+=("Ollama binary (/usr/local/bin/ollama)")
    elif [ -f "/opt/homebrew/bin/ollama" ]; then
        rm -f "/opt/homebrew/bin/ollama"
        REMOVED+=("Ollama binary (/opt/homebrew/bin/ollama)")
    fi
else
    SKIPPED+=("Ollama (kept)")
fi

# ── Summary ──
echo ""
echo "  ─────────────────────────────────────"
echo -e "  ${GREEN}${BOLD}Done! Here's what was removed:${RESET}"
echo ""

for item in "${REMOVED[@]}"; do
    echo -e "  ${GREEN}✓${RESET} $item"
done

if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo ""
    echo -e "  ${YELLOW}Skipped (not found or kept):${RESET}"
    for item in "${SKIPPED[@]}"; do
        echo "    - $item"
    done
fi

echo ""
echo "  Your projects, media, and transcripts are untouched."
echo ""
echo -e "  ${BOLD}To reinstall, run:${RESET} bash install.sh"
echo ""
