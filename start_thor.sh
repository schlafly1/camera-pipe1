#!/usr/bin/env bash
# start_thor.sh — bring up the whole camera pipeline natively on Jetson Thor
# (no Docker; see SPARK_DEV.md -> "Native Thor runbook").
#
#   ./start_thor.sh              start (or restart) everything, then run monitor.py
#   ./start_thor.sh --no-monitor start everything and return to the shell
#   ./stop_thor.sh               stop chromadb + pipeline + query server
#
# What it does:
#   1. Starts a local `chroma run` server (chroma_data/) if not already up
#   2. Kills any old pipeline_multi.py / query_server.py
#   3. Relaunches both detached with DeepStream's native lib/plugin paths set;
#      output goes to logs/*.log
#   4. Runs monitor.py in this terminal (Ctrl+C exits monitor only —
#      the pipeline keeps running)

set -euo pipefail
cd "$(dirname "$0")"

DS_VER=9.1
export LD_LIBRARY_PATH="/opt/nvidia/deepstream/deepstream-${DS_VER}/lib:/opt/nvidia/deepstream/deepstream/lib:${LD_LIBRARY_PATH:-}"
export GST_PLUGIN_PATH="/opt/nvidia/deepstream/deepstream-${DS_VER}/lib/gst-plugins:/opt/nvidia/deepstream/deepstream/lib/gst-plugins:${GST_PLUGIN_PATH:-}"

PY=.venv/bin/python3
CHROMA=.venv/bin/chroma

# Load .env WITHOUT `source` — RTSP URLs contain `&`/`?`, which a bash
# `source` misparses as background/glob operators. Read line-by-line and
# export each raw KEY=value instead (mirrors Docker Compose's env_file
# semantics, which never re-parses the value through the shell).
if [ -f .env ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            ''|'#'*) continue ;;
            'export '*) line="${line#export }" ;;
        esac
        export "$line"
    done < .env
fi

mkdir -p logs chroma_data snapshots stats

# GPU DVFS governor smoothing — under full multi-camera + batched nvinfer
# load, the default nvhost_podgov ramp (10ms poll, k=3) jumps GPU clock/
# current to peak within one poll tick when a batch bursts in, which trips
# the board's hardware over-current protection (SOC_THERM oc3, confirmed via
# /sys/class/hwmon/*/name=soctherm_oc oc3_event_cnt climbing under load —
# throttling, not just logging: oc3_throt_en=1). Slowing the ramp (longer
# poll interval, higher EWMA smoothing) fixed it in testing: oc3_event_cnt
# stayed flat under active multi-camera load, vs. an extrapolated ~11
# events/60s at the old settings. Doesn't lower peak GPU clock/throughput,
# just how fast it gets there. Resets on reboot, so it's reapplied here.
# NOPASSWD sudo for exactly these paths is in /etc/sudoers.d/thor-power-tuning.
echo "==> Smoothing GPU DVFS ramp (over-current mitigation)..."
GPU_DEVFREQ=/sys/class/devfreq/gpu-gpc-0
# sudo -n: never prompt (this also runs unattended at boot via boot_thor.sh);
# a failure here is a warning, not a reason to leave the stack down.
echo 50 | sudo -n tee "$GPU_DEVFREQ/polling_interval" >/dev/null \
    || echo "    WARN: could not set $GPU_DEVFREQ/polling_interval" >&2
echo 10 | sudo -n tee "$GPU_DEVFREQ/nvhost_podgov/k"  >/dev/null \
    || echo "    WARN: could not set $GPU_DEVFREQ/nvhost_podgov/k" >&2
# up_freq_margin write is silently ignored by this kernel/driver (confirmed
# stuck at its default of 10 regardless of value written) — left out rather
# than implying it does something.

echo "==> Checking ChromaDB..."
# On Thor the live Chroma is the Docker container camera-pipe1-chromadb-1
# (restart=unless-stopped, chroma_data/ mounted at /data). At boot it may not
# be listening yet; starting a native `chroma run` on the same chroma_data/
# would race it for :8000 and have two Chroma versions touching the same
# sqlite files. CHROMA_NO_NATIVE=1 (set by boot_thor.sh) waits up to
# CHROMA_WAIT_S for :8000 and never starts a native server.
if [ "${CHROMA_NO_NATIVE:-0}" = "1" ]; then
    for i in $(seq 1 "${CHROMA_WAIT_S:-180}"); do
        (echo > /dev/tcp/127.0.0.1/8000) 2>/dev/null && break
        sleep 1
    done
elif ! (echo > /dev/tcp/127.0.0.1/8000) 2>/dev/null; then
    echo "    Starting chroma run --path chroma_data --port 8000"
    nohup "$CHROMA" run --path chroma_data --port 8000 \
        > logs/chromadb.log 2>&1 &
    for i in $(seq 1 15); do
        (echo > /dev/tcp/127.0.0.1/8000) 2>/dev/null && break
        sleep 1
    done
fi
if ! (echo > /dev/tcp/127.0.0.1/8000) 2>/dev/null; then
    echo "ERROR: chromadb did not come up — see logs/chromadb.log" >&2
    exit 1
fi
echo "    chromadb up on :8000"

echo "==> Stopping any old pipeline/query processes..."
# Anchored to the interpreter at the start of the command line, so a shell or
# agent whose command line merely mentions these file names is never matched.
PIPE_PAT='^[^ ]*python3?( -u)? pipeline_multi\.py'
QUERY_PAT='^[^ ]*python3?( -u)? query_server\.py'
pkill -f "$PIPE_PAT"  2>/dev/null || true
pkill -f "$QUERY_PAT" 2>/dev/null || true
for i in $(seq 1 15); do
    pgrep -f "$PIPE_PAT" >/dev/null || pgrep -f "$QUERY_PAT" >/dev/null || break
    sleep 1
done
pkill -9 -f "$PIPE_PAT"  2>/dev/null || true
pkill -9 -f "$QUERY_PAT" 2>/dev/null || true
sleep 1

echo "==> Launching pipeline_multi.py (logs/pipeline-console.log)..."
nohup "$PY" -u pipeline_multi.py > logs/pipeline-console.log 2>&1 &

echo "==> Launching query_server.py (logs/query_server.log)..."
nohup "$PY" -u query_server.py > logs/query_server.log 2>&1 &

sleep 3
ok=1
for proc in pipeline_multi query_server; do
    if pgrep -f "^[^ ]*python3?( -u)? $proc\.py" >/dev/null; then
        echo "    $proc.py running"
    else
        ok=0
        echo "ERROR: $proc.py is NOT running — last log lines:" >&2
        tail -n 20 "logs/${proc/pipeline_multi/pipeline-console}.log" 2>/dev/null >&2 || true
    fi
done
[ "$ok" = 1 ] || exit 1

echo "==> All up.  Search UI: http://localhost:8001   Logs: ./logs/"

if [ "${1:-}" = "--no-monitor" ]; then
    echo "    Run '.venv/bin/python3 monitor.py' any time to watch stats."
else
    echo "==> Starting monitor (Ctrl+C exits monitor only; pipeline keeps running)"
    exec "$PY" monitor.py
fi
