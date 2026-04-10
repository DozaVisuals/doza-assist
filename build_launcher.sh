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
APP_DIR="${OUTPUT_DIR:-$HOME/Desktop}/${APP_NAME}.app"
CONTENTS_DIR="${APP_DIR}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"
APP_SRC_DIR="${RESOURCES_DIR}/app"
ICONSET_DIR="/tmp/DozaAssist.iconset"

# Clean up temp files on exit (success or failure)
trap 'rm -rf "${ICONSET_DIR}" "/tmp/DozaAssist.icns" "/tmp/DozaAssist_dmg"' EXIT

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

cp "${SCRIPT_DIR}/launcher.sh" "${MACOS_DIR}/launch"

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
DMG_PATH="${OUTPUT_DIR:-$HOME/Desktop}/${DMG_NAME}.dmg"

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
    "${DMG_PATH}"

rm -rf "${DMG_DIR}"

echo ""
echo "================================================"
echo "  Build complete!"
echo "================================================"
echo ""
echo "  App:  ${APP_DIR}"
echo "  DMG:  ${DMG_PATH}"
echo ""
echo "  The .app is self-contained — drag it to"
echo "  Applications or double-click to launch."
echo ""
echo "  Share the .dmg file for distribution."
echo "  On first launch, it will automatically"
echo "  install all required dependencies."
echo ""
