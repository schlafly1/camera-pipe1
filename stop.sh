#!/usr/bin/env bash
# stop.sh — stop the pipeline and query server (containers keep running,
# preserving the TensorRT engine cache). Add --containers to stop those too.

set -euo pipefail
cd "$(dirname "$0")"

COMPOSE="docker compose -f cam_multi.yml"
DS=$($COMPOSE ps -q deepstream || true)

if [ -n "$DS" ]; then
    echo "==> Stopping pipeline and query server..."
    docker exec "$DS" pkill -f 'python3? .*pipeline_multi' 2>/dev/null || true
    docker exec "$DS" pkill -f 'python3? .*query_server'   2>/dev/null || true
else
    echo "    (deepstream container not running)"
fi

if [ "${1:-}" = "--containers" ]; then
    echo "==> Stopping containers..."
    $COMPOSE stop
fi
echo "==> Done."
