"""
monitor.py — live performance monitor for camera-pipe1.

Run on the host (Jetson Thor or DGX Spark), not inside a container:
    python3 monitor.py

Reads pipeline stats from ./stats/cam*_stats.json (written by
pipeline_multi.py, heartbeat every ~10s even when idle).
Runs tegrastats to get GPU/CPU/power/memory metrics.
Refreshes every INTERVAL seconds.

Press Ctrl+C to exit.
"""

import glob
import json
import os
import re
import subprocess
import sys
import threading
import time

STATS_GLOB   = "./stats/cam*_stats.json"
INTERVAL     = 10    # seconds between display refreshes
STALE_AFTER  = 35    # no stats write for this long => pipeline not running
FRAME_STALE  = 15    # no decoded frame for this long => camera feed down
# Must match VLM_QUEUE_MAX in pipeline_multi.py (both default to 12)
VLM_QUEUE_MAX = int(os.environ.get("VLM_QUEUE_MAX", "12"))


# ── tegrastats reader ─────────────────────────────────────────────────────────

_ts_lock   = threading.Lock()
_ts_latest = ""


def _tegrastats_thread():
    global _ts_latest
    try:
        proc = subprocess.Popen(
            ["tegrastats", "--interval", "2000"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        for line in proc.stdout:
            with _ts_lock:
                _ts_latest = line.strip()
        proc.wait()
    except FileNotFoundError:
        pass  # tegrastats not available (e.g. running off-device)
    except Exception:
        pass


def _start_tegrastats():
    t = threading.Thread(target=_tegrastats_thread, daemon=True)
    t.start()


def _get_tegrastats_line():
    with _ts_lock:
        return _ts_latest


# ── tegrastats parser ─────────────────────────────────────────────────────────

def parse_tegrastats(line):
    """Extract key metrics from a tegrastats line. Missing fields are omitted."""
    if not line:
        return {}

    out = {}

    # RAM usage: RAM 24576/65536MB
    m = re.search(r'RAM (\d+)/(\d+)MB', line)
    if m:
        out["ram_used_mb"]  = int(m.group(1))
        out["ram_total_mb"] = int(m.group(2))

    # GPU utilization — try field names used across JetPack versions
    for field in ("GPC_FREQ", "GR3D_FREQ", "GPU"):
        m = re.search(rf'{field} (\d+)%', line)
        if m:
            out["gpu_pct"] = int(m.group(1))
            break

    # CPU utilization: CPU [45%@2035,32%@2035,...]
    m = re.search(r'CPU \[([^\]]+)\]', line)
    if m:
        pcts = [int(p) for p in re.findall(r'(\d+)%', m.group(1))]
        if pcts:
            out["cpu_pct"]   = round(sum(pcts) / len(pcts))
            out["cpu_cores"] = len(pcts)

    # GPU temperature: GPU@51.2C
    m = re.search(r'GPU@([\d.]+)C', line)
    if m:
        out["gpu_temp_c"] = float(m.group(1))

    # Power: look for VIN/SYS domains first (total system draw), then largest
    # Formats seen: VIN_SYS_5V0 66000mW/80000mW  or  VDD_IN 66W/80W
    power_mw = re.findall(r'(\w+) (\d+)mW/(\d+)mW', line)
    power_w  = re.findall(r'(\w+) (\d+)W/(\d+)W',   line)

    def pick_power(matches, scale):
        for name, cur, lim in matches:
            if any(k in name for k in ("VIN", "SYS_5V", "VDD_IN")):
                return int(cur) * scale, int(lim) * scale, name
        if matches:
            # fall back to the largest reading
            best = max(matches, key=lambda x: int(x[1]))
            return int(best[1]) * scale, int(best[2]) * scale, best[0]
        return None, None, None

    cur_mw, lim_mw, pname = pick_power(power_mw, 1)
    if cur_mw is None:
        cur_mw, lim_mw, pname = pick_power(power_w, 1000)
    if cur_mw is not None:
        out["power_mw"]       = cur_mw
        out["power_limit_mw"] = lim_mw
        out["power_domain"]   = pname

    return out


# ── stats file reader ─────────────────────────────────────────────────────────

def read_cam_stats():
    results = []
    for path in sorted(glob.glob(STATS_GLOB)):
        try:
            with open(path) as fh:
                results.append(json.load(fh))
        except (OSError, json.JSONDecodeError):
            pass
    return results


# ── display ───────────────────────────────────────────────────────────────────

def _bar(value, limit, width=10):
    """ASCII fill bar, e.g. [████░░░░░░]"""
    if not limit:
        return "[" + "?" * width + "]"
    filled = round(value / limit * width)
    filled = max(0, min(width, filled))
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _fmt_ms(val):
    return f"{val:>6}ms" if val is not None else "      --"


def display(ts, cams):
    now  = time.time()
    rows = []

    rows.append("\033[2J\033[H")  # clear screen, cursor to top
    rows.append(f"camera-pipe1 monitor   {time.strftime('%Y-%m-%d %H:%M:%S')}   "
                f"refresh {INTERVAL}s   Ctrl+C to exit")
    rows.append("=" * 72)

    # ── system ────────────────────────────────────────────────────────────────
    rows.append("")
    rows.append("System")

    if ts:
        gpu_pct = ts.get("gpu_pct")
        cpu_pct = ts.get("cpu_pct")
        ram_u   = ts.get("ram_used_mb",  0)
        ram_t   = ts.get("ram_total_mb", 0)
        temp    = ts.get("gpu_temp_c")
        pw      = ts.get("power_mw")
        pw_lim  = ts.get("power_limit_mw")

        gpu_str  = f"{gpu_pct:>3}% {_bar(gpu_pct, 100)}" if gpu_pct is not None else "  --"
        cpu_str  = f"{cpu_pct:>3}%"                       if cpu_pct is not None else "  --"
        ram_str  = f"{ram_u/1024:.1f}/{ram_t/1024:.1f} GB" if ram_t else "--"
        temp_str = f"{temp:.1f}°C"                         if temp is not None else "--"

        rows.append(f"  GPU  {gpu_str}  CPU {cpu_str}  RAM {ram_str}  GPU temp {temp_str}")

        if pw is not None:
            pw_w   = pw / 1000
            lim_w  = pw_lim / 1000
            pct    = pw / pw_lim * 100
            warn   = "  *** THROTTLE RISK ***" if pct > 85 else ""
            rows.append(f"  Power  {pw_w:.1f} W / {lim_w:.1f} W  {_bar(pw, pw_lim)}  {pct:.0f}%{warn}")
    else:
        rows.append("  (tegrastats not available — run on the Jetson host)")

    # ── per-camera pipelines ──────────────────────────────────────────────────
    cams = sorted(cams, key=lambda x: x.get("camera_id", 0))
    fresh = [s for s in cams if now - s.get("updated_at", 0) <= STALE_AFTER]

    uptime = max((s.get("elapsed_s", 0) for s in fresh), default=0)
    rows.append("")
    if fresh:
        rows.append(f"Pipelines   (run uptime {uptime/3600:.1f}h — rates are per-min since start)")
    else:
        rows.append("Pipelines")
    rows.append(f"  {'cam':<10}  {'frame':>6}  {'det/m':>7}  {'qd/m':>6}  {'sv/m':>6}  "
                f"{'vlm avg':>8}  {'q':>3}  status")
    rows.append("  " + "-" * 70)

    if not cams:
        rows.append("  (no stats files — is pipeline_multi.py running inside the container?)")
    else:
        for s in cams:
            cam_id = s.get("camera_id", "?")
            ctype  = (s.get("cam_type") or "?")[:6]
            name   = f"cam{cam_id} {ctype}"
            age    = now - s.get("updated_at", 0)

            if age > STALE_AFTER:
                rows.append(f"  {name:<10}  {'--':>6}  {'--':>7}  {'--':>6}  {'--':>6}  "
                            f"{'--':>8}  {'--':>3}  NO STATS ({age:.0f}s) — pipeline down?")
                continue

            frame_at = s.get("last_frame_at")
            f_age    = (now - frame_at) if frame_at else None
            det_m  = s.get("det_per_min", 0)
            q_m    = s.get("queue_per_min", 0)
            sv_m   = s.get("saves_per_min", 0)
            dr_m   = s.get("drops_per_min", 0)
            vlm_a  = s.get("vlm_ms_avg")
            qdepth = s.get("queue_depth", 0)

            if f_age is None:
                status = "NO FRAMES YET — camera down?"
            elif f_age > FRAME_STALE:
                status = f"NO FRAMES ({f_age:.0f}s) — camera down?"
            elif dr_m > 0:
                status = "DROPPING  <-- VLM behind"
            elif qdepth >= VLM_QUEUE_MAX * 0.75:
                status = "QUEUE HIGH"
            elif q_m == 0:
                status = "quiet (frames OK, no events)"
            else:
                status = "ok"

            f_str = f"{f_age:.0f}s" if f_age is not None else "--"
            rows.append(
                f"  {name:<10}  {f_str:>6}  {det_m:>7.1f}  {q_m:>6.1f}  {sv_m:>6.1f}  "
                f"{_fmt_ms(vlm_a)}  {qdepth:>3}  {status}"
            )

    # ── event funnel ──────────────────────────────────────────────────────────
    if fresh:
        rows.append("")
        rows.append("Event funnel (totals this run — where detections stopped)")
        rows.append(f"  {'cam':<6}  {'det':>7}  {'lowconf':>7}  {'dedup':>7}  {'throttl':>7}  "
                    f"{'queued':>6}  {'drop':>5}  {'nofrm':>5}  {'stale':>5}  {'rej':>5}  "
                    f"{'err':>4}  {'saved':>6}")
        rows.append("  " + "-" * 88)
        for s in fresh:
            cam_id = s.get("camera_id", "?")
            rows.append(
                f"  cam{cam_id:<3}  {s.get('detections_total', 0):>7}  "
                f"{s.get('low_conf_total', 0):>7}  {s.get('dedup_total', 0):>7}  "
                f"{s.get('throttled_total', 0):>7}  {s.get('queued_total', 0):>6}  "
                f"{s.get('drops_total', 0):>5}  {s.get('no_frame_total', 0):>5}  "
                f"{s.get('stale_frame_total', 0):>5}  {s.get('vlm_reject_total', 0):>5}  "
                f"{s.get('errors_total', 0):>4}  {s.get('saves_total', 0):>6}"
            )

    # ── guidance ──────────────────────────────────────────────────────────────
    rows.append("")
    rows.append("Guidance")

    hints = []

    gpu_pct = ts.get("gpu_pct") if ts else None
    if gpu_pct is not None:
        if gpu_pct > 85:
            hints.append("  GPU >85% — at capacity, do not add cameras or reduce pgie interval")
        elif gpu_pct > 60:
            hints.append("  GPU 60-85% — moderate load, adding one more camera may be OK")
        else:
            hints.append(f"  GPU {gpu_pct}% — headroom available")

    pw     = ts.get("power_mw")   if ts else None
    pw_lim = ts.get("power_limit_mw") if ts else None
    if pw and pw_lim:
        pct = pw / pw_lim * 100
        if pct > 85:
            hints.append(f"  Power at {pct:.0f}% — throttling likely; try `sudo nvpmodel -m 2`")

    # VLM throughput: is the single worker keeping up with the queue rate?
    vlm_avgs = [s["vlm_ms_avg"] for s in fresh if s.get("vlm_ms_avg")]
    total_q_m = sum(s.get("queue_per_min", 0) for s in fresh)
    if vlm_avgs:
        avg_ms = sum(vlm_avgs) / len(vlm_avgs)
        capacity_m = 60000.0 / avg_ms  # events/min one serial worker can do
        if total_q_m > capacity_m:
            hints.append(f"  VLM saturated: cameras queue {total_q_m:.1f} evt/min but worker "
                         f"capacity is ~{capacity_m:.1f} evt/min at {avg_ms/1000:.0f}s/event")
            hints.append("    Options: smaller/faster VLM model, raise SAVE_INTERVAL, offload Ollama")

    total_drops = sum(s.get("drops_total", 0) for s in fresh)
    if total_drops > 0:
        hints.append(f"  {total_drops} events dropped (queue full) — VLM can't keep pace with detections")

    total_noframe = sum(s.get("no_frame_total", 0) for s in fresh)
    if total_noframe > 0:
        hints.append(f"  {total_noframe} events skipped with no usable JPEG — snapshot branch "
                     f"lagging or camera feed gaps (see logs/pipeline.log)")

    total_stale = sum(s.get("stale_frame_total", 0) for s in fresh)
    if total_stale > 0:
        hints.append(f"  {total_stale} events used a stale frame — image may not match the detection")

    total_rej   = sum(s.get("vlm_reject_total", 0) for s in fresh)
    total_saves = sum(s.get("saves_total", 0) for s in fresh)
    if total_rej > 0 and total_rej >= max(total_saves, 1) * 0.3:
        hints.append(f"  VLM rejected {total_rej} events vs {total_saves} saved — many detector "
                     f"false positives (or stale frames); consider raising DETECT_MIN_CONF")

    total_errs = sum(s.get("errors_total", 0) for s in fresh)
    if total_errs > 0:
        hints.append(f"  {total_errs} VLM worker errors — check logs/pipeline.log")

    vlm_maxes = [s["vlm_ms_max"] for s in fresh if s.get("vlm_ms_max")]
    if vlm_maxes:
        m = max(vlm_maxes)
        if m > 15000:
            hints.append(f"  VLM latency peaks {m/1000:.0f}s — GPU contention or model too large for available memory")
        elif m > 5000:
            hints.append(f"  VLM latency up to {m/1000:.0f}s — acceptable but watch for drops if traffic increases")

    ram_u  = ts.get("ram_used_mb",  0) if ts else 0
    ram_t  = ts.get("ram_total_mb", 1) if ts else 1
    if ram_t and ram_u / ram_t > 0.85:
        hints.append(f"  RAM at {ram_u/ram_t*100:.0f}% — risk of OOM; close unused processes")

    if not hints:
        hints.append("  All metrics nominal")

    rows.extend(hints)
    rows.append("")

    print("\n".join(rows), end="", flush=True)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    _start_tegrastats()
    print("Starting monitor — waiting for first tegrastats sample...", flush=True)
    time.sleep(2.5)  # let tegrastats emit at least one line

    try:
        while True:
            ts_line = _get_tegrastats_line()
            ts_data = parse_tegrastats(ts_line)
            cams    = read_cam_stats()
            display(ts_data, cams)
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
