#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/backend"
BACKEND_VENV="$BACKEND_DIR/.venv"
PYTHON_BIN="$BACKEND_VENV/bin/python"
PIP_BIN="$BACKEND_VENV/bin/pip"
UVICORN_BIN="$BACKEND_VENV/bin/uvicorn"
HEALTH_URL="http://127.0.0.1:8000/health"
EXTENSION_DIR="$ROOT_DIR/extension"

print_usage() {
  cat <<EOF
Usage: ./run-dev.sh
Starts the A-Eye backend for local extension development, verifies the health
endpoint, and allows the extension path to load in Chrome.
EOF
}

if [[ "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but was not found on PATH." >&2
  exit 1
fi

mkdir -p "$BACKEND_DIR"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Creating backend virtual environment..."
  python3 -m venv "$BACKEND_VENV"
fi
if ! "$PYTHON_BIN" -c "import fastapi, uvicorn" >/dev/null 2>&1; then
  echo "Installing backend dependencies..."
  "$PIP_BIN" install --upgrade pip
  "$PIP_BIN" install -r "$BACKEND_DIR/requirements.txt"
fi
cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

echo "Starting A-Eye backend on http://127.0.0.1:8000 ..."
(
  cd "$BACKEND_DIR"
  exec "$UVICORN_BIN" app.main:app --reload --host 127.0.0.1 --port 8000
) &
SERVER_PID=$!

echo "Waiting for backend health check."
for _ in {1..30}; do
  if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
from urllib.request import urlopen
urlopen("http://127.0.0.1:8000/health", timeout=1)
PY
  then
    break
  fi
  sleep 1
done

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
from urllib.request import urlopen
urlopen("http://127.0.0.1:8000/health", timeout=1)
PY
then
  echo "Backend did not become healthy at $HEALTH_URL" >&2
  exit 1
fi
cat <<EOF
A-Eye backend is running.

Backend health:
  $HEALTH_URL

INSTRUCTIONS:
Load the browser extension in Chrome:
  1. Open chrome://extensions
  2. Turn on Developer mode
  3. Click "Load unpacked"
  4. Select this folder:
     $EXTENSION_DIR
  5. Open any page with a remote image, right click it, and choose:
     "Analyze image with A-Eye"

The script will keep the backend running until you press Ctrl+C.
EOF

wait "$SERVER_PID"
