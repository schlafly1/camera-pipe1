#!/usr/bin/env bash
# stop_thor.sh — stop the native pipeline/query server (and chromadb with
# --all). TensorRT engine cache is a plain file on disk either way, so
# stopping never loses it.

set -euo pipefail
cd "$(dirname "$0")"

echo "==> Stopping pipeline and query server..."
pkill -f 'python3? .*pipeline_multi.py' 2>/dev/null || true
pkill -f 'python3? .*query_server.py'   2>/dev/null || true

if [ "${1:-}" = "--all" ]; then
    echo "==> Stopping chromadb..."
    pkill -f '.venv/bin/chroma run' 2>/dev/null || true
fi
echo "==> Done."
