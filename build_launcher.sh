#!/bin/bash
#
# Build the Doza Assist macOS app (.app bundle + .dmg)
# Run this from the doza-transcribe directory:
#   bash build_launcher.sh
#
# Creates "Doza Assist.app" on your Desktop and optionally a .dmg for distribution.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="Doza Assist"
APP_DIR="$HOME/Desktop/${APP_NAME}.app"
CONTENTS_DIR="${APP_DIR}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"
APP_SRC_DIR="${RESOURCES_DIR}/app"
ICONSET_DIR="/tmp/DozaAssist.iconset"

echo "Building ${APP_NAME}..."
echo "Source: ${SCRIPT_DIR}"
echo ""

# ── Step 1: Generate icon PNGs ──
echo "1. Generating icon images..."
cd "${SCRIPT_DIR}"
python3 make_icon.py

# ── Step 2: Build .icns from PNGs ──
echo ""
echo "2. Building .icns icon file..."
rm -rf "${ICONSET_DIR}"
mkdir -p "${ICONSET_DIR}"

ICON_BUILD="${SCRIPT_DIR}/icon_build"

cp "${ICON_BUILD}/icon_16x16.png"    "${ICONSET_DIR}/icon_16x16.png"
cp "${ICON_BUILD}/icon_32x32.png"    "${ICONSET_DIR}/icon_16x16@2x.png"
cp "${ICON_BUILD}/icon_32x32.png"    "${ICONSET_DIR}/icon_32x32.png"
cp "${ICON_BUILD}/icon_64x64.png"    "${ICONSET_DIR}/icon_32x32@2x.png"
cp "${ICON_BUILD}/icon_128x128.png"  "${ICONSET_DIR}/icon_128x128.png"
cp "${ICON_BUILD}/icon_256x256.png"  "${ICONSET_DIR}/icon_128x128@2x.png"
cp "${ICON_BUILD}/icon_256x256.png"  "${ICONSET_DIR}/icon_256x256.png"
cp "${ICON_BUILD}/icon_512x512.png"  "${ICONSET_DIR}/icon_256x256@2x.png"
cp "${ICON_BUILD}/icon_512x512.png"  "${ICONSET_DIR}/icon_512x512.png"
cp "${ICON_BUILD}/icon_1024x1024.png" "${ICONSET_DIR}/icon_512x512@2x.png"

iconutil -c icns "${ICONSET_DIR}" -o "/tmp/DozaAssist.icns"
echo "   Icon created."

# ── Step 3: Create .app bundle ──
echo ""
echo "3. Creating .app bundle..."

rm -rf "${APP_DIR}"
mkdir -p "${MACOS_DIR}"
mkdir -p "${RESOURCES_DIR}"
mkdir -p "${APP_SRC_DIR}"

# Copy icon
cp "/tmp/DozaAssist.icns" "${RESOURCES_DIR}/AppIcon.icns"

# ── Step 4: Bundle app source files ──
echo ""
echo "4. Bundling application files..."

# Core Python files
for f in app.py transcribe.py ai_analysis.py fcpxml_export.py; do
    cp "${SCRIPT_DIR}/${f}" "${APP_SRC_DIR}/"
done

# Setup system files
cp "${SCRIPT_DIR}/setup_assistant.py" "${APP_SRC_DIR}/"
cp "${SCRIPT_DIR}/setup_runner.sh"    "${APP_SRC_DIR}/"
cp "${SCRIPT_DIR}/dep_check.sh"       "${APP_SRC_DIR}/"
cp "${SCRIPT_DIR}/requirements.txt"   "${APP_SRC_DIR}/"

# Templates and static assets
cp -R "${SCRIPT_DIR}/templates" "${APP_SRC_DIR}/"
cp -R "${SCRIPT_DIR}/static"    "${APP_SRC_DIR}/"

# Make scripts executable
chmod +x "${APP_SRC_DIR}/setup_runner.sh"
chmod +x "${APP_SRC_DIR}/dep_check.sh"

echo "   Bundled $(find "${APP_SRC_DIR}" -type f | wc -l | tr -d ' ') files."

# ── Step 5: Create Info.plist ──
cat > "${CONTENTS_DIR}/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Doza Assist</string>
    <key>CFBundleDisplayName</key>
    <string>Doza Assist</string>
    <key>CFBundleIdentifier</key>
    <string>com.dozavisuals.transcribe</string>
    <key>CFBundleVersion</key>
    <string>2.0</string>
    <key>CFBundleShortVersionString</key>
    <string>2.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundleExecutable</key>
    <string>launch</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>LSUIElement</key>
    <false/>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST

# ── Step 6: Create the launch script ──
echo ""
echo "5. Creating launcher..."

cat > "${MACOS_DIR}/launch" << 'LAUNCHER_EOF'
#!/bin/bash
#
# Doza Assist Launcher
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

# ── Ensure Homebrew on PATH ──
if [ -d "/opt/homebrew/bin" ]; then
    export PATH="/opt/homebrew/bin:$PATH"
elif [ -d "/usr/local/bin" ]; then
    export PATH="/usr/local/bin:$PATH"
fi

# ── If Flask server already running, just open browser ──
if /usr/bin/curl -sf "${FLASK_URL}" > /dev/null 2>&1; then
    log "Server already running, opening browser."
    /usr/bin/open "${FLASK_URL}"
    exit 0
fi

# ── Quick dependency check ──
log "Running dependency check..."
MISSING=$( bash "${APP_SRC}/dep_check.sh" 2>/dev/null ) || true

if [ -z "$MISSING" ]; then
    # ── All dependencies present — launch directly ──
    log "All dependencies present. Starting Flask server."

    # Activate venv and start the server in the background
    export DOZA_APP_DIR="${APP_SRC}"
    source "${VENV_DIR}/bin/activate"
    cd "${APP_SRC}"

    # Start Flask server
    python3 app.py >> "$SUPPORT_DIR/server.log" 2>&1 &
    SERVER_PID=$!
    log "Flask server PID: $SERVER_PID"

    # Save PID for later cleanup
    echo "$SERVER_PID" > "$SUPPORT_DIR/server.pid"

    # Wait for server to be ready (up to 30 seconds)
    for i in $(seq 1 60); do
        if /usr/bin/curl -sf "${FLASK_URL}" > /dev/null 2>&1; then
            log "Server ready."
            /usr/bin/open "${FLASK_URL}"
            exit 0
        fi
        sleep 0.5
    done

    log "ERROR: Server failed to start within 30 seconds."
    osascript -e 'display dialog "Doza Assist failed to start.\n\nCheck the log at:\n'"$SUPPORT_DIR/server.log"'" with title "Doza Assist" buttons {"OK"} default button "OK" with icon stop' 2>/dev/null
    exit 1
fi

# ── Setup needed ──
log "Setup needed. Missing: $(echo $MISSING | tr '\n' ', ')"

# ── Phase 1: Pre-Python bootstrap (Xcode CLT, Homebrew, Python) ──
# Check if we need Phase 1 (Python-related deps missing)
NEED_PHASE1=false

# Find a usable Python 3.11+
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
    NEED_PHASE1=true
    log "No suitable Python found. Running Phase 1 bootstrap."

    # Phase 1 runs in Terminal so the user can see progress and enter passwords
    osascript -e 'display dialog "Doza Assist needs to install some tools for first-time setup.\n\nA Terminal window will open to install:\n• Developer tools\n• Homebrew package manager\n• Python\n\nYou may be asked for your Mac password." with title "Doza Assist — First Launch" buttons {"Continue"} default button "Continue" with icon note' 2>/dev/null

    # Run Phase 1 in Terminal and capture the Python path
    PHASE1_RESULT="$SUPPORT_DIR/phase1_result.txt"
    rm -f "$PHASE1_RESULT"

    # Create a wrapper script that runs Phase 1 and saves the result
    PHASE1_WRAPPER="$SUPPORT_DIR/phase1_wrapper.sh"
    cat > "$PHASE1_WRAPPER" << WRAPPER_EOF
#!/bin/bash
echo ""
echo "  ╔═══════════════════════════════════╗"
echo "  ║    Doza Assist — First Launch     ║"
echo "  ╚═══════════════════════════════════╝"
echo ""
PYTHON_PATH=\$(bash "${APP_SRC}/setup_runner.sh" 2>&1 | tee /dev/stderr | tail -1)
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
    read -n 1
fi
WRAPPER_EOF
    chmod +x "$PHASE1_WRAPPER"

    # Open in Terminal and wait
    /usr/bin/open -a Terminal -W "$PHASE1_WRAPPER"

    # Read the result
    if [ -f "$PHASE1_RESULT" ]; then
        PYTHON_PATH=$(cat "$PHASE1_RESULT" | tr -d '[:space:]')
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
"$PYTHON_PATH" "${APP_SRC}/setup_assistant.py" >> "$SUPPORT_DIR/setup.log" 2>&1 &
SETUP_PID=$!

# Wait for setup server to be ready
for i in $(seq 1 20); do
    if /usr/bin/curl -sf "http://127.0.0.1:${SETUP_PORT}" > /dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

# Wait for setup to complete (setup_assistant.py exits when done)
wait $SETUP_PID 2>/dev/null || true
log "Setup assistant finished."

# Verify setup completed
if [ ! -f "$SETUP_JSON" ]; then
    log "ERROR: Setup did not complete successfully."
    osascript -e 'display dialog "Setup did not complete.\n\nPlease relaunch Doza Assist to try again." with title "Doza Assist" buttons {"OK"} default button "OK" with icon stop' 2>/dev/null
    exit 1
fi

# ── Launch Flask server ──
log "Starting Flask server..."
source "${VENV_DIR}/bin/activate"
cd "${APP_SRC}"

python3 app.py >> "$SUPPORT_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$SUPPORT_DIR/server.pid"

# Wait for server to be ready
for i in $(seq 1 60); do
    if /usr/bin/curl -sf "${FLASK_URL}" > /dev/null 2>&1; then
        log "Server ready after setup. PID: $SERVER_PID"
        /usr/bin/open "${FLASK_URL}"
        exit 0
    fi
    sleep 0.5
done

log "ERROR: Server failed to start after setup."
osascript -e 'display dialog "Doza Assist server failed to start after setup.\n\nCheck the log at:\n'"$SUPPORT_DIR/server.log"'" with title "Doza Assist" buttons {"OK"} default button "OK" with icon stop' 2>/dev/null
exit 1
LAUNCHER_EOF

chmod +x "${MACOS_DIR}/launch"

# ── Step 7: Clean up build artifacts ──
echo ""
echo "6. Cleaning up..."
rm -rf "${ICON_BUILD}"
rm -rf "${ICONSET_DIR}"
rm -f "/tmp/DozaAssist.icns"

# ── Step 8: Create .dmg ──
echo ""
echo "7. Creating .dmg for distribution..."

DMG_NAME="${APP_NAME}"
DMG_DIR="/tmp/DozaAssist_dmg"
DMG_PATH="$HOME/Desktop/${DMG_NAME}.dmg"

# Clean up any previous DMG build
rm -rf "${DMG_DIR}"
rm -f "${DMG_PATH}"

mkdir -p "${DMG_DIR}"
cp -R "${APP_DIR}" "${DMG_DIR}/"

# Create a symlink to /Applications for drag-to-install
ln -s /Applications "${DMG_DIR}/Applications"

# Create the DMG
hdiutil create -volname "${DMG_NAME}" \
    -srcfolder "${DMG_DIR}" \
    -ov -format UDZO \
    "${DMG_PATH}" \
    > /dev/null 2>&1

rm -rf "${DMG_DIR}"

echo ""
echo "================================================"
echo "  Build complete!"
echo "================================================"
echo ""
echo "  App:  ~/Desktop/Doza Assist.app"
echo "  DMG:  ~/Desktop/Doza Assist.dmg"
echo ""
echo "  The .app is self-contained — drag it to"
echo "  Applications or double-click to launch."
echo ""
echo "  Share the .dmg file for distribution."
echo "  On first launch, it will automatically"
echo "  install all required dependencies."
echo ""
