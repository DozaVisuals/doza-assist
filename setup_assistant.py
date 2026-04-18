#!/usr/bin/env python3
"""
Doza Assist — Setup Assistant (Phase 2)
Lightweight HTTP server using only stdlib that shows setup progress in the browser.
Installs: ffmpeg, ollama, venv, pip packages, ollama model.
"""

import http.server
import json
import os
import re
import shutil
import signal
import socketserver
import subprocess
import sys
import threading
import time

# ── Paths ──
SUPPORT_DIR = os.path.expanduser("~/Library/Application Support/DozaAssist")
VENV_DIR = os.path.join(SUPPORT_DIR, "venv")
SETUP_JSON = os.path.join(SUPPORT_DIR, "setup.json")
LOG_FILE = os.path.join(SUPPORT_DIR, "setup.log")

# The app source directory (set by launcher, defaults to script directory)
APP_DIR = os.environ.get("DOZA_APP_DIR", os.path.dirname(os.path.abspath(__file__)))
REQUIREMENTS_FILE = os.path.join(APP_DIR, "requirements.txt")

SETUP_PORT = 5051
FLASK_PORT = 5050

# ── State ──
setup_state = {
    "status": "running",  # running, complete, error
    "current_step": 0,
    "steps": [
        {"id": "xcode",     "name": "Xcode Command Line Tools",           "status": "pending", "detail": ""},
        {"id": "homebrew",  "name": "Install Homebrew (package manager)",  "status": "pending", "detail": ""},
        {"id": "ffmpeg",    "name": "Install ffmpeg (audio processing)",   "status": "pending", "detail": ""},
        {"id": "ollama",    "name": "Install Ollama (AI engine)",          "status": "pending", "detail": ""},
        {"id": "venv",      "name": "Create Python environment",          "status": "pending", "detail": ""},
        {"id": "pip",       "name": "Install Python packages",            "status": "pending", "detail": ""},
        {"id": "transcribe","name": "Install transcription engine (English)", "status": "pending", "detail": ""},
        {"id": "transcribe_model","name": "Download transcription model (~636MB)", "status": "pending", "detail": ""},
        {"id": "model",     "name": "Download AI model (gemma4 or gemma3)", "status": "pending", "detail": ""},
    ],
    "error": None,
}
state_lock = threading.Lock()


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def homebrew_prefix():
    if os.path.isdir("/opt/homebrew"):
        return "/opt/homebrew"
    elif os.path.isdir("/usr/local/Homebrew"):
        return "/usr/local"
    return ""


def ensure_path():
    """Ensure Homebrew and essential dirs are on PATH.
    macOS .app bundles launch with a minimal PATH (/usr/bin:/bin:/usr/sbin:/sbin)
    so we must explicitly add Homebrew."""
    prefix = homebrew_prefix()
    current = os.environ.get("PATH", "")
    additions = []
    if prefix:
        for d in [os.path.join(prefix, "bin"), os.path.join(prefix, "sbin")]:
            if d not in current:
                additions.append(d)
    # Also ensure standard paths
    for d in ["/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]:
        if d not in current and d not in additions:
            additions.append(d)
    if additions:
        os.environ["PATH"] = ":".join(additions) + ":" + current
    log(f"PATH: {os.environ['PATH']}")


def update_step(index, status, detail=""):
    with state_lock:
        setup_state["steps"][index]["status"] = status
        setup_state["steps"][index]["detail"] = detail
        if status == "running":
            setup_state["current_step"] = index


def is_apple_silicon():
    """Check if the hardware is Apple Silicon (even if running under Rosetta)."""
    try:
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "hw.optional.arm64"],
            capture_output=True, text=True
        )
        return result.stdout.strip() == "1"
    except Exception:
        return os.path.isdir("/opt/homebrew")


def brew_cmd(args):
    """Build a brew command string, using arch -arm64 on Apple Silicon
    to avoid Rosetta/x86 issues when Python itself is an Intel binary."""
    ensure_path()
    brew_path = shutil.which("brew")
    if not brew_path:
        # Fallback: check common locations directly
        for p in ["/opt/homebrew/bin/brew", "/usr/local/bin/brew"]:
            if os.path.isfile(p):
                brew_path = p
                break
    if not brew_path:
        return None

    if is_apple_silicon() and "/opt/homebrew" in brew_path:
        return f'arch -arm64 "{brew_path}" {args}'
    return f'"{brew_path}" {args}'


def run_cmd(cmd, capture=True, env=None):
    """Run a shell command and return (returncode, stdout, stderr)."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=capture, text=True,
            env=merged_env, timeout=1800  # 30min max per command
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)


# ── Step indices ──
STEP_XCODE      = 0
STEP_HOMEBREW   = 1
STEP_FFMPEG     = 2
STEP_OLLAMA     = 3
STEP_VENV       = 4
STEP_PIP        = 5
STEP_TRANSCRIBE       = 6
STEP_TRANSCRIBE_MODEL = 7
STEP_MODEL            = 8


# ── Installation Steps ──

def install_xcode_clt():
    """Install Xcode Command Line Tools if missing."""
    # Check if already installed
    rc, out, _ = run_cmd("xcode-select -p")
    if rc == 0:
        log("Xcode CLT already installed.")
        return True

    log("Installing Xcode Command Line Tools...")
    update_step(STEP_XCODE, "running", "Installing Xcode Command Line Tools...")

    # Trigger the install
    run_cmd("xcode-select --install")

    # Wait for installation (up to 20 minutes)
    update_step(STEP_XCODE, "running", "Waiting for Xcode tools to install (check for a system dialog)...")
    for i in range(120):  # 120 * 10s = 20 minutes
        time.sleep(10)
        rc, _, _ = run_cmd("xcode-select -p")
        if rc == 0:
            log("Xcode CLT installed.")
            return True
        if i % 6 == 0:  # Update detail every minute
            mins = (i * 10) // 60
            update_step(STEP_XCODE, "running", f"Installing Xcode tools... ({mins}m elapsed, check for system dialog)")

    log("ERROR: Xcode CLT installation timed out.")
    update_step(STEP_XCODE, "error", "Timed out. Look for an Apple system dialog asking to install developer tools.")
    return False


def install_homebrew():
    """Install Homebrew if missing. Opens Terminal for password entry."""
    ensure_path()
    if shutil.which("brew"):
        log("Homebrew already installed.")
        return True

    log("Installing Homebrew (requires Terminal for password)...")
    update_step(STEP_HOMEBREW, "running", "Opening Terminal to install Homebrew — enter your Mac password when asked...")

    # Homebrew installer needs sudo, which needs a TTY for the password prompt.
    # We write a small script, open it in Terminal, and wait for it to finish.
    marker_file = os.path.join(SUPPORT_DIR, "brew_install_done")
    try:
        os.remove(marker_file)
    except FileNotFoundError:
        pass

    brew_script = os.path.join(SUPPORT_DIR, "install_brew.sh")
    with open(brew_script, "w") as f:
        f.write(f"""#!/bin/bash
echo ""
echo "  ╔═══════════════════════════════════════════════╗"
echo "  ║  Doza Assist — Installing Homebrew            ║"
echo "  ║  Enter your Mac password when prompted below  ║"
echo "  ╚═══════════════════════════════════════════════╝"
echo ""
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
RESULT=$?
if [ $RESULT -eq 0 ]; then
    echo "SUCCESS" > "{marker_file}"
    echo ""
    echo "  ✅ Homebrew installed! You can close this window."
    echo ""
else
    echo "FAILED" > "{marker_file}"
    echo ""
    echo "  ❌ Homebrew installation failed."
    echo "  Press any key to close..."
    read -n 1
fi
""")
    os.chmod(brew_script, 0o755)

    # Open the script in Terminal
    subprocess.Popen(["open", "-a", "Terminal", brew_script])

    # Wait for the marker file (up to 15 minutes)
    for i in range(180):  # 180 * 5s = 15 minutes
        time.sleep(5)
        if os.path.isfile(marker_file):
            break
        if i % 12 == 0 and i > 0:
            mins = (i * 5) // 60
            update_step(STEP_HOMEBREW, "running",
                f"Waiting for Homebrew install in Terminal... ({mins}m elapsed)")
    else:
        log("ERROR: Homebrew install timed out.")
        update_step(STEP_HOMEBREW, "error", "Homebrew install timed out. Please try again.")
        return False

    # Check result
    try:
        result = open(marker_file).read().strip()
    except Exception:
        result = "FAILED"

    # Clean up
    try:
        os.remove(brew_script)
        os.remove(marker_file)
    except Exception:
        pass

    if result != "SUCCESS":
        log("ERROR: Homebrew installation failed in Terminal.")
        update_step(STEP_HOMEBREW, "error", "Homebrew installation failed. Please try again.")
        return False

    # Refresh PATH
    ensure_path()

    if not shutil.which("brew"):
        log("ERROR: Homebrew installed but not found on PATH.")
        update_step(STEP_HOMEBREW, "error", "Homebrew installed but not found on PATH. Please restart your Mac and relaunch.")
        return False

    log("Homebrew installed.")
    return True


def install_ffmpeg():
    ensure_path()
    if shutil.which("ffmpeg"):
        log("ffmpeg already installed.")
        return True

    cmd = brew_cmd("install ffmpeg")
    if not cmd:
        log("ERROR: brew not found. Homebrew may not be installed.")
        update_step(STEP_FFMPEG, "error", "Homebrew not found. Please relaunch to retry setup.")
        return False

    log(f"Installing ffmpeg: {cmd}")
    update_step(STEP_FFMPEG, "running", "Installing ffmpeg via Homebrew (this may take a few minutes)...")
    rc, out, err = run_cmd(cmd)
    if rc != 0:
        log(f"ffmpeg install failed (rc={rc}).")
        log(f"stdout: {out[-500:] if out else 'none'}")
        log(f"stderr: {err[-500:] if err else 'none'}")
        update_step(STEP_FFMPEG, "error", f"brew install ffmpeg failed: {(err or out or 'unknown error')[-200:]}")
        return False
    ensure_path()
    if not shutil.which("ffmpeg"):
        log("WARNING: brew install succeeded but ffmpeg not found on PATH.")
        return False
    log("ffmpeg installed.")
    return True


def install_ollama():
    ensure_path()
    if shutil.which("ollama"):
        log("Ollama already installed.")
        return True

    cmd = brew_cmd("install ollama")
    if not cmd:
        log("ERROR: brew not found on PATH.")
        update_step(STEP_OLLAMA, "error", "Homebrew not found. Please relaunch to retry setup.")
        return False

    log(f"Installing Ollama: {cmd}")
    update_step(STEP_OLLAMA, "running", "Installing Ollama via Homebrew...")
    rc, out, err = run_cmd(cmd)
    if rc != 0:
        log(f"Ollama install failed (rc={rc}).")
        log(f"stdout: {out[-500:] if out else 'none'}")
        log(f"stderr: {err[-500:] if err else 'none'}")
        update_step(STEP_OLLAMA, "error", f"brew install ollama failed: {(err or out or 'unknown error')[-200:]}")
        return False
    ensure_path()
    log("Ollama installed.")
    return True


def create_venv():
    if os.path.isfile(os.path.join(VENV_DIR, "bin", "python3")):
        log("Virtual environment already exists.")
        return True
    log("Creating virtual environment...")
    update_step(STEP_VENV, "running", "Creating Python virtual environment...")
    python_path = sys.executable
    log(f"Using Python: {python_path}")
    rc, out, err = run_cmd(f'"{python_path}" -m venv "{VENV_DIR}"')
    if rc != 0:
        log(f"venv creation failed (rc={rc}): {err}")
        update_step(STEP_VENV, "error", f"Failed to create venv: {(err or 'unknown error')[-200:]}")
        return False
    if not os.path.isfile(os.path.join(VENV_DIR, "bin", "python3")):
        log("ERROR: venv created but python3 not found inside it.")
        return False
    log("Virtual environment created.")
    return True


def install_pip_packages():
    pip_path = os.path.join(VENV_DIR, "bin", "pip")
    if not os.path.isfile(pip_path):
        log("ERROR: pip not found in venv.")
        return False

    log("Installing pip packages...")
    update_step(STEP_PIP, "running", "Upgrading pip...")
    run_cmd(f'"{pip_path}" install --upgrade pip')

    if not os.path.isfile(REQUIREMENTS_FILE):
        log(f"WARNING: requirements.txt not found at {REQUIREMENTS_FILE}")
        # Install minimum packages
        update_step(STEP_PIP, "running", "Installing core packages...")
        rc, out, err = run_cmd(f'"{pip_path}" install flask werkzeug requests certifi')
        return rc == 0

    update_step(STEP_PIP, "running", "Installing packages from requirements.txt...")
    rc, out, err = run_cmd(f'"{pip_path}" install -r "{REQUIREMENTS_FILE}"')
    if rc != 0:
        log(f"pip install failed: {err}")
        # Try installing packages one at a time to identify the failure
        update_step(STEP_PIP, "running", "Retrying packages individually...")
        with open(REQUIREMENTS_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    pkg = line.split("#")[0].strip()
                    if pkg:
                        update_step(STEP_PIP, "running", f"Installing {pkg}...")
                        rc2, _, err2 = run_cmd(f'"{pip_path}" install {pkg}')
                        if rc2 != 0:
                            log(f"WARNING: Failed to install {pkg}: {err2}")
        # Verify core packages at minimum
        rc, _, _ = run_cmd(f'"{pip_path}" show flask')
        if rc != 0:
            return False

    log("Pip packages installed.")
    return True


def install_transcription_engine():
    """Install Parakeet MLX (the default English transcription engine).

    OpenAI Whisper (the non-English / 99-language fallback) is intentionally
    NOT installed at first launch — it pulls PyTorch (~200MB) plus cmake and
    historically dominated the setup wall-time, dragging total install from
    ~3 minutes to 10-15 minutes. Parakeet alone covers the English-language
    primary use case. Non-English support is added on demand from the app's
    settings (see /install-whisper endpoint, TODO).
    """
    pip_path = os.path.join(VENV_DIR, "bin", "pip")
    python_path = os.path.join(VENV_DIR, "bin", "python3")
    if not os.path.isfile(pip_path):
        log("ERROR: pip not found in venv.")
        return False

    # On Apple Silicon, MLX requires running under arm64 (not Rosetta/x86_64).
    # The .app bundle may launch Python under Rosetta, so we force arch -arm64.
    arm_prefix = "arch -arm64 " if is_apple_silicon() else ""

    update_step(STEP_TRANSCRIBE, "running", "Installing Parakeet MLX (Apple Silicon transcription)...")
    # Use --force-reinstall to resolve conflicts from previous failed installs
    rc, out, err = run_cmd(f'{arm_prefix}"{pip_path}" install --force-reinstall parakeet-mlx')
    if rc != 0:
        log(f"ERROR: parakeet-mlx install failed: {err}")
        update_step(STEP_TRANSCRIBE, "error",
                    "Parakeet MLX install failed. Check your internet connection and retry. "
                    f"Error: {err[:200]}")
        return False

    log("Parakeet MLX installed successfully.")
    # Verify it imports (must use arm64 for MLX)
    rc2, _, err2 = run_cmd(f'{arm_prefix}"{python_path}" -c "import parakeet_mlx"')
    if rc2 != 0:
        log(f"ERROR: parakeet-mlx installed but import failed: {err2}")
        update_step(STEP_TRANSCRIBE, "error",
                    f"Parakeet installed but cannot be imported. {err2[:200]}")
        return False

    log("Parakeet MLX verified.")
    return True


def _dir_size_bytes(path):
    """Sum size of all files under path. Returns 0 if path doesn't exist."""
    total = 0
    try:
        for root, _, files in os.walk(path):
            for fn in files:
                try:
                    total += os.path.getsize(os.path.join(root, fn))
                except OSError:
                    continue
    except OSError:
        pass
    return total


# Approx total bytes after Parakeet TDT 0.6b v2 finishes downloading. Used to
# render a percentage in the setup UI. Off-by-a-few-MB is fine — we cap display
# at 99% until the subprocess actually exits.
PARAKEET_MODEL_BYTES = 636 * 1024 * 1024


def download_parakeet_model():
    """Pre-download the Parakeet model during setup so first-transcribe is instant.

    Without this step the model fetch (~636MB, ~5-15min) happens silently the
    first time the user clicks Transcribe, with no progress indication — they
    see a generic "Transcribing..." spinner for 10+ minutes and assume the app
    is broken. Doing it here surfaces real progress and front-loads the wait
    into the setup phase where users expect things to take a while.
    """
    python_path = os.path.join(VENV_DIR, "bin", "python3")
    if not os.path.isfile(python_path):
        log("ERROR: venv python not found, cannot pre-download model.")
        update_step(STEP_TRANSCRIBE_MODEL, "error", "Python venv missing.")
        return False

    arm_prefix = ["arch", "-arm64"] if is_apple_silicon() else []
    cache_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v2"
    )

    # If the model is already on disk (idempotent setup re-run), skip.
    if _dir_size_bytes(cache_dir) > 0.95 * PARAKEET_MODEL_BYTES:
        log("Parakeet model already present, skipping download.")
        update_step(STEP_TRANSCRIBE_MODEL, "running", "Model already present.")
        return True

    update_step(STEP_TRANSCRIBE_MODEL, "running",
                "Connecting to HuggingFace... (first transcription will be instant after this)")

    # from_pretrained triggers the same download path the app uses at runtime,
    # so we get exact same files in exact same cache layout.
    fetch_cmd = arm_prefix + [
        python_path, "-c",
        "from parakeet_mlx import from_pretrained; "
        "from_pretrained('mlx-community/parakeet-tdt-0.6b-v2'); "
        "print('OK')"
    ]
    log(f"Spawning model download: {' '.join(fetch_cmd)}")
    # Inherit parent stdout/stderr (which the launcher already redirects to
    # setup.log). Piping risks deadlock if huggingface_hub's tqdm fills the
    # pipe buffer faster than we drain it.
    proc = subprocess.Popen(fetch_cmd, env=os.environ.copy())

    # Poll the cache directory size and report progress until the subprocess
    # exits. ~636MB target — we report bytes downloaded so the user sees the
    # number climb, which is the signal that "something is happening".
    last_logged_pct = -10
    while proc.poll() is None:
        time.sleep(1.5)
        downloaded = _dir_size_bytes(cache_dir)
        mb = downloaded / (1024 * 1024)
        pct = min(99, int((downloaded / PARAKEET_MODEL_BYTES) * 100))
        update_step(STEP_TRANSCRIBE_MODEL, "running",
                    f"Downloading transcription model: {mb:.0f} / 636 MB ({pct}%)")
        if pct >= last_logged_pct + 10:
            log(f"Model download progress: {mb:.0f} MB ({pct}%)")
            last_logged_pct = pct

    if proc.returncode != 0:
        log(f"ERROR: Parakeet model download failed (exit {proc.returncode})")
        update_step(STEP_TRANSCRIBE_MODEL, "error",
                    "Model download failed. Check your internet connection and retry. "
                    "(The app will still try to download on first transcribe.)")
        return False

    final_mb = _dir_size_bytes(cache_dir) / (1024 * 1024)
    log(f"Parakeet model downloaded successfully ({final_mb:.0f} MB).")
    update_step(STEP_TRANSCRIBE_MODEL, "running", f"Done — {final_mb:.0f} MB cached.")
    return True


def native_cmd(args_list):
    """Wrap a command list with arch -arm64 on Apple Silicon to avoid Rosetta issues."""
    if is_apple_silicon():
        return ["arch", "-arm64"] + args_list
    return args_list


def _do_pull(ollama_path, model, attempt, max_attempts):
    """Stream one ollama pull attempt. Returns True on success, False on failure."""
    if attempt == 1:
        update_step(STEP_MODEL, "running",
                    f"Downloading {model} — this can take 5‑15 min depending on your connection...")
    else:
        update_step(STEP_MODEL, "running",
                    f"Retrying download of {model} (attempt {attempt}/{max_attempts})...")

    last_lines = []
    try:
        proc = subprocess.Popen(
            native_cmd([ollama_path, "pull", model]),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for raw in proc.stdout:
            line = raw.strip()
            if not line:
                continue
            last_lines.append(line)
            if len(last_lines) > 8:
                last_lines.pop(0)

            pct_match = re.search(r'(\d+)%', line)
            if pct_match:
                pct = pct_match.group(1)
                size_match = re.search(r'([\d.]+\s*[GMKT]B)\s*/\s*([\d.]+\s*[GMKT]B)', line)
                if size_match:
                    detail = f"Downloading {model}... {pct}% ({size_match.group(1)} / {size_match.group(2)})"
                else:
                    detail = f"Downloading {model}... {pct}%"
                update_step(STEP_MODEL, "running", detail)
            elif "success" in line.lower():
                update_step(STEP_MODEL, "running", f"{model} downloaded!")
            log(f"ollama pull {model}: {line}")

        proc.wait()

        if proc.returncode == 0:
            log(f"{model} pulled successfully.")
            return True

        # Surface the actual error to the browser
        error_detail = " | ".join(last_lines[-3:]) if last_lines else "unknown error"
        log(f"ERROR: ollama pull {model} failed (rc={proc.returncode}): {error_detail}")
        update_step(STEP_MODEL, "running", f"Download failed: {error_detail[:160]}")
        return False

    except Exception as e:
        log(f"ERROR: ollama pull {model} exception: {e}")
        update_step(STEP_MODEL, "running", f"Download error: {e}")
        return False


def pull_ollama_model():
    ensure_path()
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        for p in ["/opt/homebrew/bin/ollama", "/usr/local/bin/ollama"]:
            if os.path.isfile(p):
                ollama_path = p
                break
    if not ollama_path:
        log("ERROR: ollama not found on PATH.")
        update_step(STEP_MODEL, "error", "Ollama not found. The install step may have failed — click Retry.")
        return False

    # Check if any usable model is already present
    rc, out, _ = run_cmd(f'arch -arm64 "{ollama_path}" list' if is_apple_silicon() else f'"{ollama_path}" list')
    for existing in ("gemma4", "gemma3"):
        if rc == 0 and existing in out:
            log(f"{existing} model already available.")
            return True

    # Ensure Ollama service is running
    rc, _, _ = run_cmd("curl -sf http://127.0.0.1:11434/api/version")
    if rc != 0:
        log("Starting Ollama service...")
        update_step(STEP_MODEL, "running", "Starting Ollama service...")
        subprocess.Popen(
            native_cmd([ollama_path, "serve"]),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        for _ in range(30):
            time.sleep(1)
            rc, _, _ = run_cmd("curl -sf http://127.0.0.1:11434/api/version")
            if rc == 0:
                break
        else:
            log("ERROR: Ollama service failed to start within 30 seconds.")
            update_step(STEP_MODEL, "error",
                        "Ollama service didn't start. Try running 'ollama serve' in Terminal, then click Retry.")
            return False

    # Try preferred models in order, each with up to 2 attempts
    models_to_try = ["gemma4", "gemma3"]
    max_attempts = 2

    for model in models_to_try:
        log(f"Trying to pull {model}...")
        for attempt in range(1, max_attempts + 1):
            if _do_pull(ollama_path, model, attempt, max_attempts):
                return True
            if attempt < max_attempts:
                wait = attempt * 15
                log(f"Waiting {wait}s before retry...")
                update_step(STEP_MODEL, "running", f"Waiting {wait}s before retry...")
                time.sleep(wait)
        log(f"All attempts for {model} failed. Trying next model...")

    # All models failed — give a clear, actionable message
    update_step(STEP_MODEL, "error",
                "Model download failed after retries. Check your internet connection, then click Retry. "
                "You can also skip this and set up Ollama manually: run 'ollama pull gemma3' in Terminal.")
    return False


def save_setup_state():
    """Write setup.json marking setup as complete."""
    ensure_path()
    state = {
        "setup_complete": True,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.executable,
        "venv": VENV_DIR,
        "ffmpeg": shutil.which("ffmpeg") or "",
        "ollama": shutil.which("ollama") or "",
    }
    os.makedirs(SUPPORT_DIR, exist_ok=True)
    with open(SETUP_JSON, "w") as f:
        json.dump(state, f, indent=2)
    log(f"Setup state saved to {SETUP_JSON}")


# ── Setup Runner Thread ──

def run_setup():
    """Execute all setup steps sequentially."""
    steps = [
        (STEP_XCODE,            "xcode",            install_xcode_clt),
        (STEP_HOMEBREW,         "homebrew",         install_homebrew),
        (STEP_FFMPEG,           "ffmpeg",           install_ffmpeg),
        (STEP_OLLAMA,           "ollama",           install_ollama),
        (STEP_VENV,             "venv",             create_venv),
        (STEP_PIP,              "pip",              install_pip_packages),
        (STEP_TRANSCRIBE,       "transcribe",       install_transcription_engine),
        (STEP_TRANSCRIBE_MODEL, "transcribe_model", download_parakeet_model),
        (STEP_MODEL,            "model",            pull_ollama_model),
    ]

    for idx, name, func in steps:
        update_step(idx, "running")
        try:
            success = func()
        except Exception as e:
            log(f"Step {name} raised exception: {e}")
            success = False

        if success:
            update_step(idx, "done", "")
        else:
            update_step(idx, "error", setup_state["steps"][idx].get("detail", f"Failed to install {name}"))
            with state_lock:
                setup_state["status"] = "error"
                setup_state["error"] = f"Failed at step: {setup_state['steps'][idx]['name']}"
            return

    # All steps done
    save_setup_state()
    with state_lock:
        setup_state["status"] = "complete"

    log("=== Setup complete! ===")

    # Auto-shutdown the setup server after a delay to let the browser redirect
    def shutdown():
        time.sleep(3)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=shutdown, daemon=True).start()


# ── HTTP Server ──

SETUP_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Doza Assist — Setup</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', 'Helvetica Neue', sans-serif;
    background: #0a0a0c;
    color: #e8e8ec;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    padding: 2rem;
  }
  .container {
    max-width: 520px;
    width: 100%;
  }
  .logo {
    width: 72px; height: 72px;
    background: #4a9eff;
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 32px; font-weight: 700; color: #0a0a0c;
    margin: 0 auto 1.5rem;
  }
  h1 {
    text-align: center;
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
  }
  .subtitle {
    text-align: center;
    color: #888;
    font-size: 0.9rem;
    margin-bottom: 2rem;
  }
  .steps {
    list-style: none;
    margin-bottom: 2rem;
  }
  .step {
    display: flex;
    align-items: flex-start;
    padding: 0.75rem 0;
    border-bottom: 1px solid #1a1a1e;
    gap: 0.75rem;
  }
  .step:last-child { border-bottom: none; }
  .step-icon {
    width: 24px; height: 24px;
    flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    border-radius: 50%;
    font-size: 14px;
    margin-top: 1px;
  }
  .step-icon.pending { border: 2px solid #333; color: #333; }
  .step-icon.running { border: 2px solid #4a9eff; color: #4a9eff; animation: pulse 1.5s infinite; }
  .step-icon.done { background: #2ea043; border: none; color: #fff; }
  .step-icon.error { background: #d73a49; border: none; color: #fff; }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }
  .step-content { flex: 1; }
  .step-name {
    font-size: 0.95rem;
    font-weight: 500;
    color: #e8e8ec;
  }
  .step.pending .step-name { color: #666; }
  .step-detail {
    font-size: 0.8rem;
    color: #4a9eff;
    margin-top: 2px;
  }
  .step.error .step-detail { color: #f85149; }
  .message {
    text-align: center;
    padding: 1rem;
    border-radius: 8px;
    font-size: 0.9rem;
  }
  .message.complete {
    background: rgba(46, 160, 67, 0.15);
    color: #2ea043;
  }
  .message.error {
    background: rgba(215, 58, 73, 0.15);
    color: #f85149;
  }
  .retry-btn {
    display: block;
    margin: 1rem auto 0;
    padding: 0.6rem 2rem;
    background: #4a9eff;
    color: #0a0a0c;
    border: none;
    border-radius: 8px;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
  }
  .retry-btn:hover { background: #3d8be0; }
  .spinner {
    display: inline-block;
    width: 12px; height: 12px;
    border: 2px solid #4a9eff;
    border-top-color: transparent;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="container">
  <div class="logo">D</div>
  <h1>Setting up Doza Assist</h1>
  <p class="subtitle">One-time setup — this won't happen again</p>

  <ul class="steps" id="steps"></ul>
  <div id="message"></div>
</div>

<script>
const FLASK_PORT = """ + str(FLASK_PORT) + r""";

function iconHTML(status) {
  switch(status) {
    case 'pending': return '';
    case 'running': return '<div class="spinner"></div>';
    case 'done':    return '✓';
    case 'error':   return '✗';
  }
}

function render(state) {
  const stepsEl = document.getElementById('steps');
  const msgEl = document.getElementById('message');

  stepsEl.innerHTML = state.steps.map(s => `
    <li class="step ${s.status}">
      <div class="step-icon ${s.status}">${iconHTML(s.status)}</div>
      <div class="step-content">
        <div class="step-name">${s.name}</div>
        ${s.detail ? `<div class="step-detail">${s.detail}</div>` : ''}
      </div>
    </li>
  `).join('');

  if (state.status === 'complete') {
    msgEl.innerHTML = '<div class="message complete">Setup complete! Launching Doza Assist...</div>';
    setTimeout(() => {
      window.location.href = 'http://127.0.0.1:' + FLASK_PORT;
    }, 2000);
  } else if (state.status === 'error') {
    msgEl.innerHTML = `
      <div class="message error">${state.error || 'An error occurred during setup.'}</div>
      <button class="retry-btn" onclick="retry()">Retry</button>
    `;
  } else {
    msgEl.innerHTML = '';
  }
}

function poll() {
  fetch('/api/status')
    .then(r => r.json())
    .then(state => {
      render(state);
      if (state.status === 'running') {
        setTimeout(poll, 1000);
      }
    })
    .catch(() => {
      setTimeout(poll, 2000);
    });
}

function retry() {
  fetch('/api/retry', { method: 'POST' })
    .then(() => {
      document.getElementById('message').innerHTML = '';
      poll();
    });
}

// Also restart polling on error state so retry works
function pollAfterError() {
  setTimeout(poll, 1000);
}

poll();
</script>
</body>
</html>"""


retry_event = threading.Event()


class SetupHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default HTTP logging

    def do_GET(self):
        if self.path == "/" or self.path == "/setup":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(SETUP_HTML.encode())
        elif self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            with state_lock:
                self.wfile.write(json.dumps(setup_state).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/retry":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
            # Reset error state and re-run
            with state_lock:
                setup_state["status"] = "running"
                setup_state["error"] = None
                for step in setup_state["steps"]:
                    if step["status"] == "error":
                        step["status"] = "pending"
                        step["detail"] = ""
            retry_event.set()
        else:
            self.send_error(404)


def kill_stale_setup_server():
    """Kill any previous setup assistant still holding port 5051."""
    try:
        result = subprocess.run(
            f"lsof -ti tcp:{SETUP_PORT}", shell=True,
            capture_output=True, text=True
        )
        if result.stdout.strip():
            for pid in result.stdout.strip().split("\n"):
                pid = pid.strip()
                if pid and pid != str(os.getpid()):
                    log(f"Killing stale setup process on port {SETUP_PORT}: PID {pid}")
                    os.kill(int(pid), signal.SIGTERM)
            time.sleep(1)
    except Exception as e:
        log(f"Warning: could not check for stale processes: {e}")


def run_server():
    """Start the HTTP server."""
    kill_stale_setup_server()
    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("127.0.0.1", SETUP_PORT), SetupHandler) as httpd:
            log(f"Setup server running at http://127.0.0.1:{SETUP_PORT}")
            httpd.serve_forever()
    except OSError as e:
        log(f"ERROR: Could not start setup server: {e}")
        # Try once more after a brief wait
        time.sleep(2)
        kill_stale_setup_server()
        time.sleep(1)
        with socketserver.TCPServer(("127.0.0.1", SETUP_PORT), SetupHandler) as httpd:
            log(f"Setup server running at http://127.0.0.1:{SETUP_PORT} (retry)")
            httpd.serve_forever()


def setup_loop():
    """Run setup, handle retries."""
    while True:
        run_setup()
        with state_lock:
            if setup_state["status"] == "complete":
                break
        # Wait for retry signal
        retry_event.wait()
        retry_event.clear()


def main():
    os.makedirs(SUPPORT_DIR, exist_ok=True)

    # Set up PATH immediately so all steps can find brew, etc.
    ensure_path()

    log("=== Doza Assist Phase 2 Setup Assistant ===")
    log(f"App directory: {APP_DIR}")
    log(f"Support directory: {SUPPORT_DIR}")
    log(f"PATH: {os.environ.get('PATH', '')}")
    log(f"Python: {sys.executable}")
    log(f"brew: {shutil.which('brew') or 'NOT FOUND'}")

    # Start HTTP server in background
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Open browser to setup page
    subprocess.Popen(["open", f"http://127.0.0.1:{SETUP_PORT}"])

    # Run setup in foreground
    setup_loop()


if __name__ == "__main__":
    main()
