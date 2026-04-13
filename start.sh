#!/usr/bin/env bash
# start.sh — Start backend + frontend, stream logs to terminal
# Usage: ./start.sh
# Logs:  logs/backend.log  logs/frontend.log
# Stop:  Ctrl-C (kills both processes)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"
BACKEND_PID_FILE="$LOG_DIR/backend.pid"
FRONTEND_PID_FILE="$LOG_DIR/frontend.pid"

cleanup_stale_pidfile() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(<"$pid_file")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
  fi
}

kill_port_if_busy() {
  local port="$1"
  local pids
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "Port $port is in use. Stopping existing process(es)..."
    kill $pids 2>/dev/null || true
    sleep 1
    pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
      kill -9 $pids 2>/dev/null || true
    fi
  fi
}

# Truncate previous logs
> "$BACKEND_LOG"
> "$FRONTEND_LOG"

cleanup() {
  echo ""
  echo "Shutting down..."
  [[ -n "${BACKEND_PID:-}" ]] && kill "$BACKEND_PID" 2>/dev/null || true
  [[ -n "${FRONTEND_PID:-}" ]] && kill "$FRONTEND_PID" 2>/dev/null || true
  [[ -n "${BACKEND_PID:-}" ]] && wait "$BACKEND_PID" 2>/dev/null || true
  [[ -n "${FRONTEND_PID:-}" ]] && wait "$FRONTEND_PID" 2>/dev/null || true
  rm -f "$BACKEND_PID_FILE" "$FRONTEND_PID_FILE"
  echo "Stopped."
}
trap cleanup EXIT INT TERM

# Ensure clean restart state.
cleanup_stale_pidfile "$BACKEND_PID_FILE"
cleanup_stale_pidfile "$FRONTEND_PID_FILE"
kill_port_if_busy 8000
kill_port_if_busy 3000

# Prefer python, fallback to python3.
if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "Error: python/python3 not found in PATH."
  exit 1
fi

# --- Backend (FastAPI) ---------------------------------------------------
echo "Starting backend on http://localhost:8000 ..."
cd "$ROOT_DIR"
"$PYTHON_BIN" -m uvicorn bridge_server:app --reload --port 8000 \
  > "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!
echo "$BACKEND_PID" > "$BACKEND_PID_FILE"

# --- Frontend (React) ----------------------------------------------------
echo "Starting frontend on http://localhost:3000 ..."
cd "$ROOT_DIR/text-to-sql-agent-frontend"
npm start > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
echo "$FRONTEND_PID" > "$FRONTEND_PID_FILE"

# --- Stream both logs with prefixed labels --------------------------------
echo ""
echo "Both servers starting. Streaming logs (Ctrl-C to stop)..."
echo "Log files: $BACKEND_LOG | $FRONTEND_LOG"
echo "-----------------------------------------------------------"
tail -f "$BACKEND_LOG" | sed 's/^/[backend]  /' &
tail -f "$FRONTEND_LOG" | sed 's/^/[frontend] /' &

wait
