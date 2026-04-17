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

# Check ollama binary
if ! command -v ollama &>/dev/null; then
    echo "ollama"
    missing=1
else
    # Daemon listening on 11434? Binary alone is not enough — AI features
    # hang at use-time if the daemon isn't running. The launcher will try
    # to start it automatically when it sees this.
    if ! /usr/bin/curl -sf --max-time 2 "http://127.0.0.1:11434/api/tags" > /dev/null 2>&1; then
        echo "ollama_daemon"
        missing=1
    fi
fi

# Check the core runtime deps in venv. A bare `import flask` let shipped
# bundles through where werkzeug/requests/certifi were broken, surfacing
# only when the user tried to transcribe. Check the full set used at
# app.py import time.
if [ -x "$VENV_DIR/bin/python3" ]; then
    if ! "$VENV_DIR/bin/python3" -c "import flask, werkzeug, requests, certifi" 2>/dev/null; then
        echo "pip_packages"
        missing=1
    fi
fi

exit $missing
