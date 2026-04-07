#!/bin/bash
#
# Build the Doza Assist macOS launcher (.app bundle)
# Run this once from the doza-transcribe directory:
#   bash build_launcher.sh
#
# It will create "Doza Assist.app" on your Desktop.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="Doza Assist"
APP_DIR="$HOME/Desktop/${APP_NAME}.app"
CONTENTS_DIR="${APP_DIR}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"
ICONSET_DIR="/tmp/DozaAssist.iconset"

echo "Building ${APP_NAME} launcher..."
echo "App location: ${SCRIPT_DIR}"
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

# macOS iconset requires specific naming
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

# Remove old version if it exists
rm -rf "${APP_DIR}"

mkdir -p "${MACOS_DIR}"
mkdir -p "${RESOURCES_DIR}"

# Copy icon
cp "/tmp/DozaAssist.icns" "${RESOURCES_DIR}/AppIcon.icns"

# Create Info.plist
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
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
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

# Create the launch script
# NOTE: macOS sandboxes .app bundles and blocks direct file access.
# So we launch start.sh via /usr/bin/open which runs it in Terminal
# with full user permissions -- bypassing the sandbox restriction.
cat > "${MACOS_DIR}/launch" << LAUNCHER_EOF
#!/bin/bash
#
# Doza Assist Launcher
# Opens the start script in Terminal, then opens the browser.
#

APP_DIR="${SCRIPT_DIR}"
PORT=5050
URL="http://127.0.0.1:\${PORT}"

# If already running, just open browser
if /usr/bin/curl -s "\${URL}" > /dev/null 2>&1; then
    /usr/bin/open "\${URL}"
    exit 0
fi

# Launch start.sh in Terminal (has full user permissions)
/usr/bin/open -a Terminal "\${APP_DIR}/start.sh"

# Wait for server to be ready (up to 20 seconds)
for i in \$(seq 1 40); do
    if /usr/bin/curl -s "\${URL}" > /dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

# Open in default browser
/usr/bin/open "\${URL}"
LAUNCHER_EOF

chmod +x "${MACOS_DIR}/launch"

# ── Step 4: Clean up build artifacts ──
rm -rf "${ICON_BUILD}"
rm -rf "${ICONSET_DIR}"
rm -f "/tmp/DozaAssist.icns"

echo ""
echo "================================================"
echo "  Doza Assist.app created on your Desktop!"
echo "================================================"
echo ""
echo "You can:"
echo "  - Double-click it to launch the app"
echo "  - Drag it to your Dock for quick access"
echo "  - Move it to /Applications if you prefer"
echo ""
echo "The app starts the server at localhost:5050"
echo "and opens your browser automatically."
echo ""
