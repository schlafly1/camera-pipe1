#!/usr/bin/env bash
# boot_thor.sh — unattended start of the whole Thor stack at boot.
# Run by the systemd *user* unit deploy/camera-pipe1.service (roger has
# linger enabled, so the user manager starts at boot without a login).
#
#   - Chroma: the Docker container (restart=unless-stopped) comes up on its
#     own; start_thor.sh is told to WAIT for :8000 and never start a native
#     chroma on the same chroma_data/ (CHROMA_NO_NATIVE=1).
#   - pipeline_multi.py + query_server.py: via start_thor.sh --no-monitor
#     (also reapplies the GPU DVFS smoothing, which resets on reboot).
#   - monitor.py: nohup'd to logs/monitor.log (previous log rotated to .1,
#     since monitor.log is screen-redraw output and is never rotated).
#
# Idempotent: if pipeline_multi.py is already running it does nothing, so a
# manual `systemctl --user start camera-pipe1` can't create a second stack.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
exec >>logs/boot.log 2>&1 </dev/null
echo "==== boot_thor.sh $(date '+%Y-%m-%d %H:%M:%S %Z')"

if pgrep -f '^[^ ]*python3?( -u)? pipeline_multi\.py' >/dev/null; then
    echo "pipeline_multi.py already running; not starting another stack"
    exit 0
fi

export CHROMA_NO_NATIVE=1
export CHROMA_WAIT_S="${CHROMA_WAIT_S:-180}"
if ! ./start_thor.sh --no-monitor; then
    echo "start_thor.sh failed"
    exit 1
fi

if ! pgrep -f '^[^ ]*python3?( -u)? monitor\.py' >/dev/null; then
    [ -f logs/monitor.log ] && mv -f logs/monitor.log logs/monitor.log.1
    nohup .venv/bin/python3 -u monitor.py > logs/monitor.log 2>&1 </dev/null &
    echo "monitor.py started"
fi
echo "==== boot_thor.sh done"
