#!/bin/bash
#
# Doza Assist — Quick Dependency Check
# Returns 0 if all dependencies are present, 1 if setup is needed.
# Prints missing dependency names to stdout (one per line).
#

SUPPORT_DIR="$HOME/Library/Application Support/DozaAssist"
VENV_DIR="$SUPPORT_DIR/venv"
SETUP_JSON="$SUPPORT_DIR/setup.json"

# Ensure Homebrew on PATH
if [ -d "/opt/homebrew/bin" ]; then
    export PATH="/opt/homebrew/bin:$PATH"
elif [ -d "/usr/local/bin" ]; then
    export PATH="/usr/local/bin:$PATH"
fi

missing=0

# Check setup.json exists
if [ ! -f "$SETUP_JSON" ]; then
    echo "setup_state"
    missing=1
fi

# Check Python in venv
if [ ! -x "$VENV_DIR/bin/python3" ]; then
    echo "venv"
    missing=1
fi

# Check ffmpeg
if ! command -v ffmpeg &>/dev/null; then
    echo "ffmpeg"
    missing=1
fi

# Check ollama
if ! command -v ollama &>/dev/null; then
    echo "ollama"
    missing=1
fi

# Check Flask is installed in venv
if [ -x "$VENV_DIR/bin/python3" ]; then
    if ! "$VENV_DIR/bin/python3" -c "import flask" 2>/dev/null; then
        echo "pip_packages"
        missing=1
    fi
fi

exit $missing
