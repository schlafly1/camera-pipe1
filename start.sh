#!/usr/bin/env bash
# start.sh — bring up the whole camera pipeline with one command.
#
#   ./start.sh              start (or restart) everything, then run monitor.py
#   ./start.sh --no-monitor start everything and return to the shell
#   ./stop.sh               stop the python processes (containers keep running)
#
# What it does:
#   1. docker compose up -d   (chromadb + deepstream containers)
#   2. kills any old pipeline_multi.py / query_server.py inside the container
#   3. relaunches both detached; output goes to logs/*.log
#   4. runs monitor.py in this terminal (Ctrl+C exits monitor only —
#      the pipeline keeps running)

set -euo pipefail
cd "$(dirname "$0")"

COMPOSE="docker compose -f cam_multi.yml"

echo "==> Starting containers..."
$COMPOSE up -d

DS=$($COMPOSE ps -q deepstream)
if [ -z "$DS" ]; then
    echo "ERROR: deepstream container not found" >&2
    exit 1
fi

# Allow container X11 access if a display pipeline is configured (harmless if not)
if [ -n "${DISPLAY:-}" ] && command -v xhost >/dev/null 2>&1; then
    xhost +local: >/dev/null 2>&1 || true
fi

echo "==> Stopping any old pipeline/query processes..."
docker exec "$DS" pkill -f 'python3? .*pipeline_multi' 2>/dev/null || true
docker exec "$DS" pkill -f 'python3? .*query_server'   2>/dev/null || true
sleep 1

echo "==> Launching pipeline_multi.py (logs/pipeline.log, console: logs/pipeline-console.log)..."
docker exec -d "$DS" bash -c \
    'cd /workspace && exec python3 -u pipeline_multi.py > logs/pipeline-console.log 2>&1'

echo "==> Launching query_server.py (logs/query_server.log)..."
docker exec -d "$DS" bash -c \
    'cd /workspace && exec python3 -u query_server.py > logs/query_server.log 2>&1'

sleep 3
ok=1
for proc in pipeline_multi query_server; do
    if docker exec "$DS" pgrep -f "python3? .*$proc" >/dev/null; then
        echo "    $proc.py running"
    else
        ok=0
        echo "ERROR: $proc.py is NOT running — last log lines:" >&2
        docker exec "$DS" bash -c \
            "tail -n 20 logs/${proc/pipeline_multi/pipeline-console}.log 2>/dev/null" >&2 || true
    fi
done
[ "$ok" = 1 ] || exit 1

echo "==> All up.  Search UI: http://localhost:8001   Logs: ./logs/"

if [ "${1:-}" = "--no-monitor" ]; then
    echo "    Run 'python3 monitor.py' any time to watch stats."
else
    echo "==> Starting monitor (Ctrl+C exits monitor only; pipeline keeps running)"
    exec python3 monitor.py
fi
