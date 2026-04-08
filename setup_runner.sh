#!/bin/bash
#
# Doza Assist — Phase 1 Bootstrap
# Installs Xcode CLT, Homebrew, and Python before the Python-based setup assistant takes over.
# Uses native macOS dialogs (osascript) since Python may not be available yet.
#

set -e

SUPPORT_DIR="$HOME/Library/Application Support/DozaAssist"
SETUP_JSON="$SUPPORT_DIR/setup.json"
LOG_FILE="$SUPPORT_DIR/setup.log"

mkdir -p "$SUPPORT_DIR"

# ── Logging ──
log() {
    echo "[$(date '+%H:%M:%S')] $1" >> "$LOG_FILE"
    echo "$1"
}

# ── Detect Homebrew path ──
homebrew_prefix() {
    if [ -d "/opt/homebrew" ]; then
        echo "/opt/homebrew"
    elif [ -d "/usr/local/Homebrew" ]; then
        echo "/usr/local"
    else
        echo ""
    fi
}

ensure_homebrew_on_path() {
    local prefix
    prefix=$(homebrew_prefix)
    if [ -n "$prefix" ] && [[ ":$PATH:" != *":$prefix/bin:"* ]]; then
        export PATH="$prefix/bin:$PATH"
    fi
}

# ── Find usable Python 3.11+ ──
find_python() {
    ensure_homebrew_on_path
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

# ── Native macOS dialog ──
show_dialog() {
    local message="$1"
    local title="${2:-Doza Assist Setup}"
    osascript -e "display dialog \"$message\" with title \"$title\" buttons {\"OK\"} default button \"OK\" with icon note" 2>/dev/null || true
}

show_password_dialog() {
    local message="$1"
    osascript -e "display dialog \"$message\" with title \"Doza Assist Setup\" buttons {\"OK\"} default button \"OK\" with icon caution" 2>/dev/null || true
}

# ── Step 1: Xcode Command Line Tools ──
install_xcode_clt() {
    if xcode-select -p &>/dev/null; then
        log "Xcode CLT already installed."
        return 0
    fi

    log "Installing Xcode Command Line Tools..."
    show_dialog "Doza Assist needs to install Apple's developer tools.\n\nA system dialog will appear — click \"Install\" to continue.\n\nThis may take a few minutes."

    # Trigger the CLT install dialog
    xcode-select --install 2>/dev/null || true

    # Wait for installation to complete (up to 20 minutes)
    log "Waiting for Xcode CLT installation..."
    local elapsed=0
    while ! xcode-select -p &>/dev/null; do
        sleep 10
        elapsed=$((elapsed + 10))
        if [ $elapsed -ge 1200 ]; then
            log "ERROR: Xcode CLT installation timed out."
            show_dialog "Xcode Command Line Tools installation timed out.\n\nPlease install manually by running:\nxcode-select --install\n\nThen relaunch Doza Assist." "Setup Error"
            return 1
        fi
    done

    log "Xcode CLT installed successfully."
    return 0
}

# ── Step 2: Homebrew ──
install_homebrew() {
    ensure_homebrew_on_path
    if command -v brew &>/dev/null; then
        log "Homebrew already installed."
        return 0
    fi

    log "Installing Homebrew..."
    show_password_dialog "Doza Assist needs to install Homebrew (a package manager for macOS).\n\nYou may be asked for your Mac password in the Terminal window that appears.\n\nThis is normal and required for installation."

    # Run Homebrew installer non-interactively
    NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" >> "$LOG_FILE" 2>&1

    # Add Homebrew to PATH for this session
    ensure_homebrew_on_path

    if ! command -v brew &>/dev/null; then
        log "ERROR: Homebrew installation failed."
        show_dialog "Homebrew installation failed.\n\nCheck the log at:\n$LOG_FILE\n\nThen relaunch Doza Assist." "Setup Error"
        return 1
    fi

    log "Homebrew installed successfully."
    return 0
}

# ── Step 3: Python 3.11+ ──
install_python() {
    local py
    py=$(find_python 2>/dev/null || true)
    if [ -n "$py" ]; then
        log "Python 3.11+ already available: $py"
        echo "$py"
        return 0
    fi

    log "Installing Python via Homebrew..."
    ensure_homebrew_on_path
    brew install python@3.12 >> "$LOG_FILE" 2>&1

    py=$(find_python 2>/dev/null || true)
    if [ -n "$py" ]; then
        log "Python installed: $py"
        echo "$py"
        return 0
    fi

    log "ERROR: Python installation failed."
    show_dialog "Python installation failed.\n\nCheck the log at:\n$LOG_FILE\n\nThen relaunch Doza Assist." "Setup Error"
    return 1
}

# ── Main ──
main() {
    log "=== Doza Assist Phase 1 Bootstrap ==="
    log "Architecture: $(uname -m)"

    # Step 1: Xcode CLT
    install_xcode_clt || exit 1

    # Step 2: Homebrew
    install_homebrew || exit 1

    # Step 3: Python
    local python_path
    python_path=$(install_python) || exit 1

    log "Phase 1 complete. Python at: $python_path"
    echo "$python_path"
}

main "$@"
