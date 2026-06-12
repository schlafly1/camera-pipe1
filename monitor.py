"""
monitor.py — live performance monitor for camera-pipe1.

Run on the Jetson Thor host (not inside a container):
    python3 monitor.py

Reads pipeline stats from ./stats/cam*_stats.json (written by pipeline2.py).
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
STALE_AFTER  = 30    # seconds before marking a camera as inactive


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
    rows.append("")
    rows.append("Pipelines")
    rows.append(f"  {'cam':<6}  {'queued/m':>8}  {'drops/m':>8}  {'saves/m':>8}  "
                f"{'vlm avg':>8}  {'vlm max':>8}  {'q':>3}  status")
    rows.append("  " + "-" * 68)

    if not cams:
        rows.append("  (no stats files — is pipeline2.py running inside the containers?)")
    else:
        for s in sorted(cams, key=lambda x: x.get("camera_id", 0)):
            cam_id = s.get("camera_id", "?")
            age    = now - s.get("updated_at", 0)

            if age > STALE_AFTER:
                rows.append(f"  cam{cam_id:<3}   {'--':>8}  {'--':>8}  {'--':>8}  "
                            f"{'--':>8}  {'--':>8}  {'--':>3}  INACTIVE ({age:.0f}s ago)")
                continue

            q_m    = s.get("queue_per_min", 0)
            dr_m   = s.get("drops_per_min", 0)
            sv_m   = s.get("saves_per_min", 0)
            vlm_a  = s.get("vlm_ms_avg")
            vlm_x  = s.get("vlm_ms_max")
            qdepth = s.get("queue_depth", 0)

            if dr_m > 0:
                status = "DROPPING  <-- VLM behind"
            elif qdepth >= VLM_QUEUE_MAX_DISPLAY * 0.75:
                status = "QUEUE HIGH"
            elif q_m == 0:
                status = "idle"
            else:
                status = "ok"

            rows.append(
                f"  cam{cam_id:<3}   {q_m:>8.1f}  {dr_m:>8.1f}  {sv_m:>8.1f}  "
                f"{_fmt_ms(vlm_a)}  {_fmt_ms(vlm_x)}  {qdepth:>3}  {status}"
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

    total_drops = sum(s.get("drops_per_min", 0) for s in cams)
    if total_drops > 0:
        hints.append("  VLM queue drops > 0 — VLM can't keep pace with detections")
        hints.append("    Options: offload Ollama to another Jetson, raise SAVE_INTERVAL, or use a smaller model")

    vlm_maxes = [s["vlm_ms_max"] for s in cams if s.get("vlm_ms_max")]
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

VLM_QUEUE_MAX_DISPLAY = 6  # matches VLM_QUEUE_MAX in pipeline2.py


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
