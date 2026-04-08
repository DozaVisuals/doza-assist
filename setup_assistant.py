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
        {"id": "ffmpeg",    "name": "Install ffmpeg (audio processing)",   "status": "pending", "detail": ""},
        {"id": "ollama",    "name": "Install Ollama (AI engine)",          "status": "pending", "detail": ""},
        {"id": "venv",      "name": "Create Python environment",          "status": "pending", "detail": ""},
        {"id": "pip",       "name": "Install Python packages",            "status": "pending", "detail": ""},
        {"id": "model",     "name": "Download AI model (gemma4)",         "status": "pending", "detail": ""},
    ],
    "error": None,
}
state_lock = threading.Lock()


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
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
    prefix = homebrew_prefix()
    if prefix:
        bin_dir = os.path.join(prefix, "bin")
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + ":" + os.environ.get("PATH", "")


def update_step(index, status, detail=""):
    with state_lock:
        setup_state["steps"][index]["status"] = status
        setup_state["steps"][index]["detail"] = detail
        if status == "running":
            setup_state["current_step"] = index


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


# ── Installation Steps ──

def install_ffmpeg():
    ensure_path()
    if shutil.which("ffmpeg"):
        log("ffmpeg already installed.")
        return True
    log("Installing ffmpeg via Homebrew...")
    update_step(0, "running", "Installing ffmpeg via Homebrew...")
    rc, out, err = run_cmd("brew install ffmpeg")
    if rc != 0:
        log(f"ffmpeg install failed: {err}")
        return False
    log("ffmpeg installed.")
    return True


def install_ollama():
    ensure_path()
    if shutil.which("ollama"):
        log("Ollama already installed.")
        return True
    log("Installing Ollama via Homebrew...")
    update_step(1, "running", "Installing Ollama via Homebrew...")
    rc, out, err = run_cmd("brew install ollama")
    if rc != 0:
        log(f"Ollama install failed: {err}")
        return False
    log("Ollama installed.")
    return True


def create_venv():
    if os.path.isfile(os.path.join(VENV_DIR, "bin", "python3")):
        log("Virtual environment already exists.")
        return True
    log("Creating virtual environment...")
    update_step(2, "running", "Creating Python virtual environment...")
    python_path = sys.executable
    rc, out, err = run_cmd(f'"{python_path}" -m venv "{VENV_DIR}"')
    if rc != 0:
        log(f"venv creation failed: {err}")
        return False
    log("Virtual environment created.")
    return True


def install_pip_packages():
    pip_path = os.path.join(VENV_DIR, "bin", "pip")
    if not os.path.isfile(pip_path):
        log("ERROR: pip not found in venv.")
        return False

    log("Installing pip packages...")
    update_step(3, "running", "Upgrading pip...")
    run_cmd(f'"{pip_path}" install --upgrade pip')

    if not os.path.isfile(REQUIREMENTS_FILE):
        log(f"WARNING: requirements.txt not found at {REQUIREMENTS_FILE}")
        # Install minimum packages
        update_step(3, "running", "Installing core packages...")
        rc, out, err = run_cmd(f'"{pip_path}" install flask werkzeug requests certifi')
        return rc == 0

    update_step(3, "running", "Installing packages from requirements.txt...")
    rc, out, err = run_cmd(f'"{pip_path}" install -r "{REQUIREMENTS_FILE}"')
    if rc != 0:
        log(f"pip install failed: {err}")
        # Try installing packages one at a time to identify the failure
        update_step(3, "running", "Retrying packages individually...")
        with open(REQUIREMENTS_FILE) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    pkg = line.split("#")[0].strip()
                    if pkg:
                        update_step(3, "running", f"Installing {pkg}...")
                        rc2, _, err2 = run_cmd(f'"{pip_path}" install {pkg}')
                        if rc2 != 0:
                            log(f"WARNING: Failed to install {pkg}: {err2}")
        # Verify core packages at minimum
        rc, _, _ = run_cmd(f'"{pip_path}" show flask')
        if rc != 0:
            return False

    log("Pip packages installed.")
    return True


def pull_ollama_model():
    ensure_path()
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        log("ERROR: ollama not found on PATH.")
        return False

    # Check if model already exists
    rc, out, _ = run_cmd(f'"{ollama_path}" list')
    if rc == 0 and "gemma4" in out:
        log("gemma3 model already available.")
        return True

    # Start Ollama if not running
    rc, _, _ = run_cmd("curl -sf http://127.0.0.1:11434/api/version")
    if rc != 0:
        log("Starting Ollama service...")
        update_step(4, "running", "Starting Ollama service...")
        subprocess.Popen(
            [ollama_path, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        # Wait for Ollama to be ready
        for _ in range(30):
            time.sleep(1)
            rc, _, _ = run_cmd("curl -sf http://127.0.0.1:11434/api/version")
            if rc == 0:
                break
        else:
            log("ERROR: Ollama failed to start.")
            return False

    # Pull the model with progress tracking
    log("Pulling gemma3 model...")
    update_step(4, "running", "Downloading AI model (this may take several minutes)...")

    try:
        proc = subprocess.Popen(
            [ollama_path, "pull", "gemma4"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            line = line.strip()
            if line:
                # Parse progress from ollama output
                # Typical: "pulling abc123... 45% ▕██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░▏ 1.2 GB/3.5 GB"
                pct_match = re.search(r'(\d+)%', line)
                if pct_match:
                    pct = pct_match.group(1)
                    # Try to extract size info
                    size_match = re.search(r'([\d.]+\s*[GMKT]B)\s*/\s*([\d.]+\s*[GMKT]B)', line)
                    if size_match:
                        detail = f"Downloading AI model... {pct}% ({size_match.group(1)} / {size_match.group(2)})"
                    else:
                        detail = f"Downloading AI model... {pct}%"
                    update_step(4, "running", detail)
                elif "success" in line.lower():
                    update_step(4, "running", "Model downloaded successfully!")
                log(f"ollama pull: {line}")

        proc.wait()
        if proc.returncode != 0:
            log("ERROR: ollama pull failed.")
            return False
    except Exception as e:
        log(f"ERROR: ollama pull exception: {e}")
        return False

    log("gemma3 model pulled successfully.")
    return True


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
        (0, "ffmpeg",  install_ffmpeg),
        (1, "ollama",  install_ollama),
        (2, "venv",    create_venv),
        (3, "pip",     install_pip_packages),
        (4, "model",   pull_ollama_model),
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
      setTimeout(poll, 500);
    });
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


def run_server():
    """Start the HTTP server."""
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", SETUP_PORT), SetupHandler) as httpd:
        log(f"Setup server running at http://127.0.0.1:{SETUP_PORT}")
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

    log("=== Doza Assist Phase 2 Setup Assistant ===")
    log(f"App directory: {APP_DIR}")
    log(f"Support directory: {SUPPORT_DIR}")

    # Start HTTP server in background
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Open browser to setup page
    subprocess.Popen(["open", f"http://127.0.0.1:{SETUP_PORT}"])

    # Run setup in foreground
    setup_loop()


if __name__ == "__main__":
    main()
