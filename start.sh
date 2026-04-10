#!/bin/bash
# ── Start Doza Assist ──

cd "$(dirname "$0")" || exit 1

# Ensure Homebrew binaries (ffmpeg, etc.) are on PATH
if [ -d "/opt/homebrew/bin" ]; then
    export PATH="/opt/homebrew/bin:$PATH"
elif [ -d "/usr/local/bin" ]; then
    export PATH="/usr/local/bin:$PATH"
fi

# shellcheck source=/dev/null
source venv/bin/activate

echo ""
echo "  Doza Assist"
echo "  http://localhost:5050"
echo ""
echo "  Client review links: http://localhost:5050/review/<project_id>"
echo "  (Use ngrok or Cloudflare Tunnel to share externally)"
echo ""
echo "  Press Ctrl+C to stop"
echo ""

python3 app.py
