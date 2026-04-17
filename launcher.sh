#!/bin/bash
#
# Doza Assist Launcher
# Installed at: Doza Assist.app/Contents/MacOS/launch
# Handles first-launch setup, dependency checking, and starting the Flask server.
#

# ── Resolve paths ──
BUNDLE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
APP_SRC="${BUNDLE_DIR}/Contents/Resources/app"
SUPPORT_DIR="$HOME/Library/Application Support/DozaAssist"
VENV_DIR="$SUPPORT_DIR/venv"
SETUP_JSON="$SUPPORT_DIR/setup.json"
LOG_FILE="$SUPPORT_DIR/launch.log"

FLASK_PORT=5050
SETUP_PORT=5051
FLASK_URL="http://127.0.0.1:${FLASK_PORT}"

mkdir -p "$SUPPORT_DIR"
mkdir -p "$SUPPORT_DIR/projects"
mkdir -p "$SUPPORT_DIR/exports"

# ── Symlink data directories so project data persists outside the .app bundle ──
if [ ! -e "${APP_SRC}/projects" ]; then
    ln -s "$SUPPORT_DIR/projects" "${APP_SRC}/projects"
fi
if [ ! -e "${APP_SRC}/exports" ]; then
    ln -s "$SUPPORT_DIR/exports" "${APP_SRC}/exports"
fi

# ── Logging ──
log() {
    echo "[$(date '+%H:%M:%S')] $1" >> "$LOG_FILE"
}

# ── Helpers ──

# Show an error dialog that includes the last 15 lines of a log file, so the
# user can diagnose without having to open Application Support themselves.
show_error_dialog() {
    local message="$1"
    local log_path="${2:-}"

    /usr/bin/osascript <<APPLESCRIPT 2>/dev/null || true
set dialogText to "$message"
set logPath to "$log_path"
if logPath is not "" then
    try
        set logTail to do shell script "/usr/bin/tail -15 " & quoted form of logPath & " 2>/dev/null || true"
        if logTail is not "" then
            set dialogText to dialogText & "

Last lines of log:

" & logTail & "

Full log: " & logPath
        end if
    end try
end if
display dialog dialogText with title "Doza Assist" buttons {"OK"} default button "OK" with icon stop
APPLESCRIPT
}

# PID of whatever is listening on the given TCP port, or empty if free.
port_holder_pid() {
    /usr/sbin/lsof -iTCP:"$1" -sTCP:LISTEN -t -n -P 2>/dev/null | head -1
}

# Short name of a process (e.g. "Python", "node", "nginx").
pid_name() {
    /bin/ps -p "$1" -o comm= 2>/dev/null | awk -F/ '{print $NF}'
}

# ── Ensure Homebrew on PATH ──
if [ -d "/opt/homebrew/bin" ]; then
    export PATH="/opt/homebrew/bin:$PATH"
elif [ -d "/usr/local/bin" ]; then
    export PATH="/usr/local/bin:$PATH"
fi

# ── Detect Apple Silicon and set arch prefix ──
ARCH_PREFIX=""
if /usr/sbin/sysctl -n hw.optional.arm64 2>/dev/null | grep -q "1"; then
    ARCH_PREFIX="arch -arm64"
fi

# ── Port conflict check / already-running detection ──
# If the port is already serving HTTP, assume it's our own running server
# and hand off to the browser. If the port is bound but not serving HTTP,
# it's a conflict — fail fast with the offending PID instead of letting
# Flask silently fail to bind and then timing out 30 seconds later.
if /usr/bin/curl -sf --max-time 2 "${FLASK_URL}" > /dev/null 2>&1; then
    log "Server already running on ${FLASK_PORT}, opening browser."
    /usr/bin/open "${FLASK_URL}"
    exit 0
fi

HOLDER_PID=$(port_holder_pid "${FLASK_PORT}")
if [ -n "$HOLDER_PID" ]; then
    HOLDER_NAME=$(pid_name "$HOLDER_PID")
    log "ERROR: Port ${FLASK_PORT} held by PID $HOLDER_PID ($HOLDER_NAME), not serving HTTP."
    /usr/bin/osascript <<APPLESCRIPT 2>/dev/null || true
display dialog "Doza Assist cannot start — port ${FLASK_PORT} is in use by another process (PID $HOLDER_PID — $HOLDER_NAME).

To free the port, open Terminal and run:
    kill $HOLDER_PID

Then relaunch Doza Assist." with title "Doza Assist — Port Conflict" buttons {"OK"} default button "OK" with icon stop
APPLESCRIPT
    exit 1
fi

# ── Quick dependency check ──
log "Running dependency check..."
MISSING=$( bash "${APP_SRC}/dep_check.sh" 2>/dev/null ) || true

# ── Auto-start ollama daemon if the binary is installed but not running ──
# Most "ollama installed but not running" cases are benign: Homebrew users who
# didn't enable the service, or someone who quit Ollama.app. Start it silently
# instead of forcing the user through the full setup flow.
if echo "$MISSING" | grep -q "^ollama_daemon$"; then
    log "Ollama daemon not running — starting in background."
    nohup ollama serve > "$SUPPORT_DIR/ollama.log" 2>&1 &
    for _ in {1..10}; do
        if /usr/bin/curl -sf --max-time 1 "http://127.0.0.1:11434/api/tags" > /dev/null 2>&1; then
            log "Ollama daemon started."
            break
        fi
        sleep 0.5
    done
    MISSING=$(echo "$MISSING" | grep -v "^ollama_daemon$" || true)
fi

if [ -z "$MISSING" ]; then
    # ── All dependencies present — launch directly ──
    log "All dependencies present. Starting Flask server."

    export DOZA_APP_DIR="${APP_SRC}"
    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"
    cd "${APP_SRC}" || exit 1

    $ARCH_PREFIX python3 app.py >> "$SUPPORT_DIR/server.log" 2>&1 &
    SERVER_PID=$!
    log "Flask server PID: $SERVER_PID"
    echo "$SERVER_PID" > "$SUPPORT_DIR/server.pid"

    # Wait for server to be ready (up to 30 seconds)
    for _ in {1..60}; do
        if /usr/bin/curl -sf "${FLASK_URL}" > /dev/null 2>&1; then
            log "Server ready."
            /usr/bin/open "${FLASK_URL}"
            exit 0
        fi
        sleep 0.5
    done

    log "ERROR: Server failed to start within 30 seconds."
    show_error_dialog "Doza Assist failed to start within 30 seconds." "$SUPPORT_DIR/server.log"
    exit 1
fi

# ── Setup needed ──
log "Setup needed. Missing: $(echo "$MISSING" | tr '\n' ',')"

# ── Phase 1: Pre-Python bootstrap (Xcode CLT, Homebrew, Python) ──
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

PYTHON_PATH=$(find_python 2>/dev/null || true)

if [ -z "$PYTHON_PATH" ]; then
    log "No suitable Python found. Running Phase 1 bootstrap."

    osascript -e 'display dialog "Doza Assist needs to install some tools for first-time setup.\n\nA Terminal window will open to install:\n• Developer tools\n• Homebrew package manager\n• Python\n\nYou may be asked for your Mac password." with title "Doza Assist — First Launch" buttons {"Continue"} default button "Continue" with icon note' 2>/dev/null

    PHASE1_RESULT="$SUPPORT_DIR/phase1_result.txt"
    rm -f "$PHASE1_RESULT"

    PHASE1_WRAPPER="$SUPPORT_DIR/phase1_wrapper.sh"
    cat > "$PHASE1_WRAPPER" << WRAPPER_EOF
#!/bin/bash
echo ""
echo "  ╔═══════════════════════════════════╗"
echo "  ║    Doza Assist — First Launch     ║"
echo "  ╚═══════════════════════════════════╝"
echo ""
# setup_runner.sh: log messages go to stderr (visible in Terminal), python path goes to stdout
PYTHON_PATH=\$(bash "${APP_SRC}/setup_runner.sh")
echo "\$PYTHON_PATH" > "${PHASE1_RESULT}"
if [ -n "\$PYTHON_PATH" ] && [ -x "\$PYTHON_PATH" ]; then
    echo ""
    echo "  ✅ Basic tools installed. Continuing setup in your browser..."
    sleep 2
else
    echo ""
    echo "  ❌ Setup encountered an error. Check the log at:"
    echo "     $SUPPORT_DIR/setup.log"
    echo ""
    echo "  Press any key to close..."
    read -rn 1
fi
WRAPPER_EOF
    chmod +x "$PHASE1_WRAPPER"

    /usr/bin/open -a Terminal "$PHASE1_WRAPPER"

    # Wait for the result file to appear (up to 20 minutes)
    for _ in {1..240}; do
        if [ -f "$PHASE1_RESULT" ]; then
            break
        fi
        sleep 5
    done

    if [ -f "$PHASE1_RESULT" ]; then
        PYTHON_PATH=$(tr -d '[:space:]' < "$PHASE1_RESULT")
    fi

    rm -f "$PHASE1_WRAPPER" "$PHASE1_RESULT"

    if [ -z "$PYTHON_PATH" ] || [ ! -x "$PYTHON_PATH" ]; then
        log "ERROR: Phase 1 failed to install Python."
        osascript -e 'display dialog "Setup failed to install Python.\n\nPlease check the log at:\n'"$SUPPORT_DIR/setup.log"'" with title "Doza Assist" buttons {"OK"} default button "OK" with icon stop' 2>/dev/null
        exit 1
    fi
fi

log "Using Python: $PYTHON_PATH"

# ── Phase 2: Python-based setup assistant ──
log "Starting Phase 2 setup assistant..."

export DOZA_APP_DIR="${APP_SRC}"
$ARCH_PREFIX "$PYTHON_PATH" "${APP_SRC}/setup_assistant.py" >> "$SUPPORT_DIR/setup.log" 2>&1 &
SETUP_PID=$!

# Wait for setup server to be ready
for _ in {1..20}; do
    if /usr/bin/curl -sf "http://127.0.0.1:${SETUP_PORT}" > /dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

wait $SETUP_PID 2>/dev/null || true
log "Setup assistant finished."

if [ ! -f "$SETUP_JSON" ]; then
    log "ERROR: Setup did not complete successfully."
    osascript -e 'display dialog "Setup did not complete.\n\nPlease relaunch Doza Assist to try again." with title "Doza Assist" buttons {"OK"} default button "OK" with icon stop' 2>/dev/null
    exit 1
fi

# ── Launch Flask server ──
log "Starting Flask server..."
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
cd "${APP_SRC}" || exit 1

$ARCH_PREFIX python3 app.py >> "$SUPPORT_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$SUPPORT_DIR/server.pid"

for _ in {1..60}; do
    if /usr/bin/curl -sf "${FLASK_URL}" > /dev/null 2>&1; then
        log "Server ready after setup. PID: $SERVER_PID"
        /usr/bin/open "${FLASK_URL}"
        exit 0
    fi
    sleep 0.5
done

log "ERROR: Server failed to start after setup."
show_error_dialog "Doza Assist server failed to start after setup." "$SUPPORT_DIR/server.log"
exit 1
