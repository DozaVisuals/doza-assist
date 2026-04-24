#!/bin/bash
#
# scripts/sign-and-notarize.sh
#
# Sign, notarize, staple, and verify a built Doza Assist .app bundle, and
# produce a distribution-ready .dmg. Idempotent — safe to re-run.
#
# Usage:
#   scripts/sign-and-notarize.sh "/path/to/Doza Assist.app"
#
# Preconditions:
#   - $APPLE_TEAM_ID is exported (stored in ~/.zshrc for this machine)
#   - A "Developer ID Application" certificate for that team is in the login keychain
#   - A notarytool keychain profile named "AC_PASSWORD" has been created:
#         xcrun notarytool store-credentials AC_PASSWORD --team-id "$APPLE_TEAM_ID"
#
# Notes:
#   - No entitlements.plist is used. The current OSS bundle's CFBundleExecutable
#     is a shell script (Contents/MacOS/launch). Hardened-runtime / entitlements
#     are Mach-O load-command features and are semantically inert on a shell
#     script. The bundle contains no Mach-O binaries (Python/ffmpeg are
#     user-provisioned post-launch via Homebrew). The .app signature + DMG
#     signature + stapled notarization ticket is what Gatekeeper actually checks
#     on first quarantine-open.
#   - If a future build adds nested Mach-O binaries (dylibs, .so, compiled
#     helpers), this script will find and sign them innermost-first. At that
#     point re-evaluate whether entitlements.plist is needed.
#

set -euo pipefail

# ── Output helpers ──
BOLD="\033[1m"
BLUE="\033[34m"
GREEN="\033[32m"
RED="\033[31m"
YELLOW="\033[33m"
DIM="\033[2m"
RESET="\033[0m"

phase() { printf "\n${BOLD}${BLUE}▶ %s${RESET}\n" "$1"; }
ok()    { printf "${GREEN}  ✓ %s${RESET}\n" "$1"; }
info()  { printf "${DIM}    %s${RESET}\n" "$1"; }
warn()  { printf "${YELLOW}  ⚠ %s${RESET}\n" "$1"; }
fail()  { printf "${RED}  ✗ %s${RESET}\n" "$1" >&2; exit 1; }

# ── Argument + environment validation ──
phase "Validating inputs"

APP_PATH="${1:-}"
if [ -z "$APP_PATH" ]; then
    fail "Usage: $0 <path/to/App.app>"
fi
if [ ! -d "$APP_PATH" ]; then
    fail "Not a directory: $APP_PATH"
fi
if [[ "$APP_PATH" != *.app ]]; then
    fail "Expected a .app bundle path, got: $APP_PATH"
fi
APP_PATH="$(cd "$APP_PATH" && pwd)"
APP_NAME="$(basename "$APP_PATH" .app)"
ok "App bundle: $APP_PATH"

if [ -z "${APPLE_TEAM_ID:-}" ]; then
    fail "APPLE_TEAM_ID is not set. Export it (e.g. in ~/.zshrc) before running."
fi
ok "Team ID: $APPLE_TEAM_ID"

# Locate the Developer ID Application identity for this team.
IDENTITY="$(security find-identity -v -p codesigning 2>/dev/null \
    | grep "Developer ID Application" \
    | grep "($APPLE_TEAM_ID)" \
    | head -1 \
    | sed -E 's/.*"(Developer ID Application: [^"]+)".*/\1/')"

if [ -z "$IDENTITY" ]; then
    fail "No 'Developer ID Application' certificate for team $APPLE_TEAM_ID in any keychain."
fi
ok "Signing identity: $IDENTITY"

# Verify the notarytool keychain profile resolves. `history` is a cheap
# round-trip that validates credentials without submitting anything.
if ! xcrun notarytool history --keychain-profile AC_PASSWORD >/dev/null 2>&1; then
    printf "${RED}  ✗ Keychain profile 'AC_PASSWORD' not found or invalid.${RESET}\n" >&2
    printf "${DIM}    Create it with:${RESET}\n" >&2
    printf "${DIM}      xcrun notarytool store-credentials AC_PASSWORD --team-id %s${RESET}\n" "$APPLE_TEAM_ID" >&2
    exit 1
fi
ok "Notarization credentials: AC_PASSWORD (validated)"

# ── Strip stale extended attributes so codesign has a clean slate ──
phase "Preparing bundle"
xattr -cr "$APP_PATH"
ok "Extended attributes cleared."

# ── Sign nested Mach-O binaries innermost-first ──
#
# find -depth gives bottom-up traversal: contents of a directory are yielded
# before the directory itself, which is what "innermost-first" requires for
# frameworks and nested bundles. `file --brief` on regular files detects
# Mach-O binaries (executables, dylibs, bundles, .so files) regardless of
# extension. Symlinks are excluded by -type f.
phase "Signing nested Mach-O binaries"

SIGN_FLAGS=(
    --force
    --timestamp
    --options runtime
    --sign "$IDENTITY"
)

MACH_COUNT=0
while IFS= read -r -d '' f; do
    kind="$(/usr/bin/file --brief "$f" 2>/dev/null || true)"
    if [[ "$kind" == Mach-O* ]]; then
        info "signing: ${f#$APP_PATH/}"
        codesign "${SIGN_FLAGS[@]}" "$f"
        MACH_COUNT=$((MACH_COUNT + 1))
    fi
done < <(find "$APP_PATH" -depth -type f -print0)

if [ "$MACH_COUNT" -eq 0 ]; then
    ok "No Mach-O binaries found inside bundle (expected for shell-script launcher)."
else
    ok "Signed $MACH_COUNT nested Mach-O binary/binaries."
fi

# ── Sign the outer .app bundle ──
phase "Signing .app bundle"
codesign "${SIGN_FLAGS[@]}" "$APP_PATH"
ok "Bundle signed."

# ── Verify bundle signature ──
phase "Verifying .app signature"
codesign --verify --deep --strict --verbose=2 "$APP_PATH"
ok "Signature valid."

# ── Build DMG (matches build_launcher.sh convention: hdiutil UDZO + /Applications symlink) ──
phase "Building DMG"

OUT_DIR="$(dirname "$APP_PATH")"
DMG_PATH="${OUT_DIR}/${APP_NAME}.dmg"
DMG_STAGE="$(mktemp -d -t dozaassist-dmg)"
trap 'rm -rf "$DMG_STAGE"' EXIT

# Idempotent: drop any prior DMG at the target path.
rm -f "$DMG_PATH"

cp -R "$APP_PATH" "$DMG_STAGE/"
ln -s /Applications "$DMG_STAGE/Applications"

hdiutil create \
    -volname "$APP_NAME" \
    -srcfolder "$DMG_STAGE" \
    -ov -format UDZO \
    "$DMG_PATH" >/dev/null
ok "DMG: $DMG_PATH"

# ── Sign the DMG itself ──
# --options runtime is omitted here: hardened runtime is a Mach-O attribute
# and a DMG is a disk image, not an executable. Timestamp + Developer ID
# signature is what notarization validates on the container.
phase "Signing DMG"
codesign --force --timestamp --sign "$IDENTITY" "$DMG_PATH"
ok "DMG signed."

# ── Submit for notarization ──
phase "Submitting DMG for notarization (typically 1–5 minutes)"

NOTARY_OUT="$(mktemp -t dozaassist-notary)"
set +e
xcrun notarytool submit "$DMG_PATH" \
    --keychain-profile AC_PASSWORD \
    --wait \
    --output-format plist \
    > "$NOTARY_OUT"
NOTARY_EXIT=$?
set -e

SUBMISSION_ID="$(/usr/libexec/PlistBuddy -c 'Print :id' "$NOTARY_OUT" 2>/dev/null || true)"
STATUS="$(/usr/libexec/PlistBuddy -c 'Print :status' "$NOTARY_OUT" 2>/dev/null || true)"

info "submission id: ${SUBMISSION_ID:-<unknown>}"
info "status:        ${STATUS:-<unknown>}"

if [ "$NOTARY_EXIT" -ne 0 ] || [ "$STATUS" != "Accepted" ]; then
    printf "\n${RED}${BOLD}  ✗ Notarization failed (status: %s).${RESET}\n" "${STATUS:-unknown}" >&2
    if [ -n "$SUBMISSION_ID" ]; then
        printf "\n${YELLOW}  Fetching full notarization log:${RESET}\n\n"
        xcrun notarytool log "$SUBMISSION_ID" --keychain-profile AC_PASSWORD || true
    else
        printf "\n${YELLOW}  No submission id — raw notarytool output:${RESET}\n\n"
        cat "$NOTARY_OUT"
    fi
    rm -f "$NOTARY_OUT"
    exit 1
fi
rm -f "$NOTARY_OUT"
ok "Notarization accepted."

# ── Staple the ticket to the DMG ──
phase "Stapling notarization ticket"
xcrun stapler staple "$DMG_PATH"
ok "Ticket stapled."

# ── Final Gatekeeper assessment ──
# For a signed+stapled DMG, --type open with primary-signature context is the
# check Gatekeeper applies at first-open from a quarantine download. A pass
# here ("accepted") means end users on a fresh Mac will not see the
# "developer not trusted" dialog.
phase "Gatekeeper assessment"
SPCTL_OUT="$(spctl --assess --type open --context context:primary-signature -vv "$DMG_PATH" 2>&1 || true)"
printf "${DIM}%s${RESET}\n" "$SPCTL_OUT"
if echo "$SPCTL_OUT" | grep -q "accepted"; then
    ok "Gatekeeper will accept this DMG on a fresh Mac."
else
    fail "spctl did not report 'accepted' — inspect output above."
fi

# ── Done ──
printf "\n${BOLD}${GREEN}══════════════════════════════════════════════${RESET}\n"
printf "${BOLD}${GREEN}  Signed · Notarized · Stapled · Verified${RESET}\n"
printf "${BOLD}${GREEN}══════════════════════════════════════════════${RESET}\n\n"
printf "  DMG: ${BOLD}%s${RESET}\n\n" "$DMG_PATH"
