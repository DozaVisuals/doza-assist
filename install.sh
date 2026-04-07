#!/bin/bash
# ── Doza Assist Installer ──
# Run this once on your Mac Studio

set -e

echo ""
echo "  ╔═══════════════════════════════════╗"
echo "  ║       Doza Assist Setup       ║"
echo "  ╚═══════════════════════════════════╝"
echo ""

cd "$(dirname "$0")"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Install from python.org"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install core dependencies
echo ""
echo "Installing core dependencies..."
pip install --upgrade pip -q
pip install flask werkzeug requests -q

# Install ffmpeg check
if ! command -v ffmpeg &> /dev/null; then
    echo ""
    echo "⚠️  ffmpeg not found. Install with: brew install ffmpeg"
    echo "   (Required for video file audio extraction)"
fi

# Install transcription engine
echo ""
echo "Choose your transcription engine:"
echo "  1) WhisperX (RECOMMENDED - transcription + speaker diarization)"
echo "  2) Lightning Whisper MLX (fastest, no diarization)"
echo "  3) Standard Whisper (basic, word timestamps)"
echo "  4) Skip (install later)"
echo ""
read -p "Enter choice [1]: " engine_choice
engine_choice=${engine_choice:-1}

case $engine_choice in
    1)
        echo "Installing WhisperX..."
        pip install whisperx -q
        echo ""
        echo "⚠️  For speaker diarization, you need a HuggingFace token:"
        echo "   1. Create account at huggingface.co"
        echo "   2. Accept terms at: https://huggingface.co/pyannote/speaker-diarization-3.1"
        echo "   3. Get token at: https://huggingface.co/settings/tokens"
        echo "   4. Set: export HF_TOKEN=your_token_here (add to ~/.zshrc)"
        ;;
    2)
        echo "Installing Lightning Whisper MLX..."
        pip install lightning-whisper-mlx -q
        ;;
    3)
        echo "Installing OpenAI Whisper..."
        pip install openai-whisper -q
        ;;
    4)
        echo "Skipping transcription engine. Install later with:"
        echo "  source venv/bin/activate && pip install whisperx"
        ;;
esac

echo ""
echo "══════════════════════════════════════"
echo "  ✅ Installation complete!"
echo ""
echo "  Start the app:  ./start.sh"
echo "  Then open:       http://localhost:5050"
echo ""
echo "  Optional: Set up AI analysis"
echo "  Local (Ollama):  ollama serve"
echo "  Cloud (Claude):  export ANTHROPIC_API_KEY=sk-..."
echo "══════════════════════════════════════"
echo ""
