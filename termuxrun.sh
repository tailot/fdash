#!/usr/bin/env bash
# Copyright (c) 2026 Vincenzo Tilotta
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo " 📈 Financial Analysis Master Dashboard - Termux  "
echo "=================================================="

# Check if running inside Termux
if [ -n "$TERMUX_VERSION" ] || [ -d "/data/data/com.termux" ]; then
    echo "[+] Termux environment detected."

    # Install required system packages if python or git are missing
    if ! command -v python3 &> /dev/null || ! command -v git &> /dev/null; then
        echo "[*] Installing system dependencies via pkg..."
        pkg update -y
        pkg install -y python git clang make libcrypt libffi openssl
    fi
else
    echo "[!] Termux environment not explicitly detected; proceeding with system environment."
fi

# Ensure python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "[!] Error: python3 is not installed or not in PATH."
    exit 1
fi

# Setup Python virtual environment
VENV_DIR="$SCRIPT_DIR/.venv_termux"

if [ ! -d "$VENV_DIR" ]; then
    echo "[*] Creating Python virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

echo "[*] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "[*] Upgrading pip and wheel..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1 || true

echo "[*] Installing Python dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "[!] Warning: requirements.txt not found!"
fi

# Check for optional --timesfm flag
if [[ "$*" == *"--timesfm"* ]]; then
    if [ -f "requirements-timesfm.txt" ]; then
        echo "[*] Installing optional TimesFM dependencies..."
        pip install -r requirements-timesfm.txt
    fi
fi

PORT=${PORT:-8501}

echo ""
echo "=================================================="
echo " 🚀 Launching Streamlit Dashboard..."
echo " 🌐 Open in browser: http://localhost:${PORT}"
echo "=================================================="
echo ""

exec streamlit run app.py --server.port "$PORT" --server.address 0.0.0.0
