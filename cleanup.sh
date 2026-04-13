#!/usr/bin/env bash
# cleanup.sh — Clear all run outputs and logs
# Usage: ./cleanup.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Clearing outputs/runs/..."
rm -rf "$ROOT_DIR/outputs/runs/"*
echo "  Done."

echo "Clearing logs/..."
rm -rf "$ROOT_DIR/logs/"*
echo "  Done."

echo "Cleanup complete."
