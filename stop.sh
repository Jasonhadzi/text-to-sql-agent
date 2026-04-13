#!/usr/bin/env bash
# stop.sh — Stop the backend and frontend servers
# Usage: ./stop.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"
BACKEND_PID_FILE="$LOG_DIR/backend.pid"
FRONTEND_PID_FILE="$LOG_DIR/frontend.pid"

stop_pidfile() {
  local name="$1"
  local pid_file="$2"

  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(<"$pid_file")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "Stopping $name (pid $pid)..."
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      echo "  $name stopped."
    else
      echo "  $name PID file found, but process not running."
    fi
    rm -f "$pid_file"
  else
    echo "  No PID file for $name."
  fi
}

kill_port_if_busy() {
  local name="$1"
  local port="$2"
  local pids
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "Stopping $name listener(s) on port $port..."
    kill $pids 2>/dev/null || true
    sleep 1
    pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
      kill -9 $pids 2>/dev/null || true
    fi
    echo "  Port $port is clear."
  else
    echo "  Port $port already clear."
  fi
}

stop_pidfile "backend" "$BACKEND_PID_FILE"
stop_pidfile "frontend" "$FRONTEND_PID_FILE"

echo "Stopping backend by name (fallback)..."
pkill -f "uvicorn bridge_server:app" 2>/dev/null && echo "  Backend stopped by name." || echo "  No backend process matched."

echo "Stopping frontend by name (fallback)..."
pkill -f "react-scripts start" 2>/dev/null && echo "  Frontend stopped by name." || echo "  No frontend process matched."

kill_port_if_busy "backend" 8000
kill_port_if_busy "frontend" 3000

echo "Done."
