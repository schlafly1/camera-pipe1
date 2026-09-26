# camera-pipe1

DeepStream 9.1 camera pipeline running natively on NVIDIA Jetson Thor
(JetPack 7.2 / L4T R39.2), with the VLM served from a separate DGX Spark box.

It detects cars, persons, and bicycles/motorcycles with RT-DETR (TrafficCamNet
Transformer Lite), sends each detection frame to a VLM for a natural-language
description, uses that description to reject detector false positives, embeds
the accepted descriptions with nomic-embed-text, and stores them in ChromaDB. A
FastAPI server (Vision Search) serves a search UI for natural-language queries.

Everything lives on the `main` branch (the old `multi-stream` branch is gone).

## Current setup (hosts)

```
6 RTSP cams ──> Thor: pipeline_multi.py (DeepStream 9.1, one batched nvstreammux
                 → nvinfer RT-DETR → nvtracker → probe)
                         │  detections queue (2 VLM worker threads)
                         v
          VLM describe:  vLLM  google/gemma-4-12B-it, thinking off   gx10-2ea8:8000
          fallback:      Ollama gemma4:12b, OLLAMA_THINK=false        spark-2251:11434
          embeddings:    Ollama nomic-embed-text                      spark-2251:11434
                         │
                         v
          Thor: ChromaDB (Docker, :8000)  <──  query_server.py (Vision Search, :8001)
```

| Host | Runs |
|------|------|
| **Thor** (`thor2`) | `pipeline_multi.py` (capture + detection + VLM workers), `query_server.py` on **:8001**, `monitor.py`, and ChromaDB on **:8000** as the Docker container `camera-pipe1-chromadb-1` (image `chromadb/chroma:latest`, `restart: unless-stopped`, `./chroma_data` mounted at `/data`; created from `cam1.yml`'s `chromadb` service). |
| **gx10** (`gx10-2ea8`, DGX Spark) | Docker `vllm-gemma4-12b` serving `google/gemma-4-12B-it` on :8000 (see SPARK_DEV.md §8). Primary VLM. |
| **Spark** (`spark-2251`) | Ollama: `gemma4:12b` (VLM fallback) and `nomic-embed-text` (all embeddings, for both the pipeline and query_server). |

Notes:
- Use `gx10-2ea8` — the short name `gx10` does **not** resolve from Thor.
- The VLM call is per event: if vLLM is unhealthy or a call fails/times out,
  that event goes to Ollama on Spark instead (no restart needed). vLLM is
  re-probed every `VLLM_HEALTHCHECK_S`.
- Embeddings always come from Spark Ollama regardless of `VLM_BACKEND`, so the
  ChromaDB vector space doesn't change when the VLM backend does.
- Spark also runs a leftover `camera-pipe1-chromadb-1` container; it is not the
  live store. Live Chroma is Thor-local (`localhost:8000`).

Search UI: http://thor2:8001 (or http://localhost:8001 on Thor)
REST: `curl "http://localhost:8001/query?text=red+car"`

## Start, stop, boot

```bash
cd ~/sd/camera-pipe1
./start_thor.sh               # (re)start pipeline + query server, then run monitor.py in this terminal
./start_thor.sh --no-monitor  # same, but return to the shell
./stop_thor.sh                # stop pipeline + query server (Chroma container and monitor.py keep running)
```

`stop_thor.sh` never stops `monitor.py`; use Ctrl+C, or for the nohup'd one
`pkill -f '^[^ ]*python3?( -u)? monitor\.py'`. `stop_thor.sh --all` also
stops a *native* `chroma run`, not the Docker container.

`start_thor.sh` loads `.env` line by line (RTSP URLs with `&`/`?` survive),
reapplies the GPU DVFS smoothing (passwordless `sudo -n` for exactly those
sysfs paths), checks Chroma on :8000, kills any old `pipeline_multi.py` /
`query_server.py` (patterns anchored to the interpreter, waits up to 15s),
and relaunches both detached. It is safe to re-run after a code or `.env`
change — that is the restart procedure. It does not touch a running
`monitor.py`. If nothing answers on :8000 it starts a native `chroma run` on
`chroma_data/` — on Thor that should not happen because the Docker container
owns that directory; `CHROMA_NO_NATIVE=1` makes it wait instead.

The TensorRT engine is rebuilt on every start (`pgie_config_rtdetr.txt` omits
`model-engine-file`), so the first detections appear ~2 minutes after a start.

**Start at boot.** A systemd *user* unit (no root; `loginctl enable-linger
roger` is enabled so the user manager starts at boot):

- Unit: `~/.config/systemd/user/camera-pipe1.service` (copy of
  `deploy/camera-pipe1.service`; install steps are in its header), enabled
  under `default.target`.
- It runs `boot_thor.sh`, which: exits if `pipeline_multi.py` is already
  running (no duplicate stack); runs `start_thor.sh --no-monitor` with
  `CHROMA_NO_NATIVE=1` (waits up to `CHROMA_WAIT_S`=180s for the Docker Chroma,
  never starts a native one); then starts `monitor.py` under nohup, rotating
  the previous `logs/monitor.log` to `logs/monitor.log.1`.
- It is `Type=oneshot` + `RemainAfterExit=yes` with no `Restart=`: it starts
  the stack at boot but **does not restart the pipeline after a crash**.
  Recover with `./start_thor.sh --no-monitor`.

```bash
systemctl --user status camera-pipe1       # enabled; "active (exited)" only after a boot run
journalctl --user -u camera-pipe1          # unit start/stop only
cat logs/boot.log                          # boot_thor.sh / start_thor.sh output
```

If the stack was started by hand with `start_thor.sh`, the unit stays
`inactive (dead)` — that's expected; `systemctl --user stop camera-pipe1`
only runs `stop_thor.sh` when the unit itself is active, so use
`./stop_thor.sh` directly.

## Logs and monitoring

| Path | What |
|------|------|
| `logs/pipeline.log` | Rotating (5 MB × 3): every detection, VLM accept and REJECT (REJECT lines end with `[vllm]`/`[ollama]`; accepts record the backend in Chroma metadata), skip, vLLM fallback/health change, error |
| `logs/pipeline-console.log` | Pipeline stdout/stderr incl. DeepStream/TensorRT output (not rotated) |
| `logs/query_server.log` | query_server output |
| `logs/monitor.log` (+ `.1`) | `monitor.py` screen output; rotated to `.1` only at boot |
| `logs/boot.log` | Boot-unit runs |
| `stats/cam*_stats.json` | Per-camera funnel counters, rewritten every 10s |

`monitor.py` (`.venv/bin/python3 monitor.py`) shows per-camera frame age,
rates, VLM latency, queue depth and the event funnel (detections → low_conf →
dedup → throttled → queued → drops/stale → reject → saved) plus tegrastats. The
stats files also carry `vlm_calls_vllm_total`, `vlm_calls_ollama_total`,
`vllm_fallback_total`, `vlm_call_ms_avg`/`vlm_call_ms_max` (pure VLM round
trip) and `vlm_backend_last`, which the monitor table doesn't display yet.

## Configuration

Set in `.env` (never committed; `env.example` is the template). "Code
default" is what applies when the variable is unset; "Live" is Thor's `.env`.

**VLM backend and models**

| Variable | Code default | Live | Notes |
|----------|--------------|------|-------|
| `VLM_BACKEND` | `ollama` | `vllm` | `vllm` = gx10 first, Ollama fallback per event |
| `VLLM_URL` | `http://gx10-2ea8:8000` | same | OpenAI-compatible endpoint |
| `VLLM_MODEL` | `google/gemma-4-12B-it` | same | Must match `curl $VLLM_URL/v1/models` |
| `VLLM_TIMEOUT_S` | `20` | unset | Per call; eval p95 was 8.6s |
| `VLLM_HEALTHCHECK_S` | `30` | unset | Min seconds between health probes; a failed call marks vLLM down immediately |
| `VLLM_MAX_TOKENS` | `300` | unset | |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` (ollama client) | `http://spark-2251:11434` | Fallback VLM + all embeddings |
| `VLM_MODEL` | `gemma4:26b` | `gemma4:12b` | Ollama model for the fallback path |
| `OLLAMA_THINK` | `false` | `false` | `false` = fast mode; `true`; `default` omits the param (gemma4 then thinks) |
| `OLLAMA_CHAT_TIMEOUT_S` / `OLLAMA_EMBED_TIMEOUT_S` | `180` / `30` | unset | |
| `VLM_NUM_CTX` | `4096` | unset | Ollama context cap (lets Spark run 2 parallel slots) |
| `VLM_WORKERS` | `2` | unset | VLM worker threads |

vLLM requests always send `chat_template_kwargs: {"enable_thinking": false}`.
The embedding model is fixed in code (`EMBED_MODEL = "nomic-embed-text"`).

**Cameras and detection**

| Variable | Code default | Live | Notes |
|----------|--------------|------|-------|
| `RTSP_URL_CAM1..N` | — | cam1–cam6 | Numbering stops at the first gap |
| `RTSP_TRANSPORT_CAMn` (or global `RTSP_TRANSPORT`) | `0` | unset | `4` forces TCP |
| `CAM_TYPE_CAMn` | `street` | cam1–2 `office`, cam3–6 `street` | Office: `SAVE_INTERVAL` throttle, dropped when the queue is full |
| `DROP_CLASSES_CAMn` | empty | cam1, cam2 = `3` | Comma list of detector class ids ignored on that camera (before any counting) |
| `MIN_CONF_PERSON_CAMn` | unset (gate stays `DETECT_MIN_CONF[2]` = 0.40) | cam6 = `0.60` | Per-camera person confidence gate; only ever raises the 0.40 gate. Blocked detections count as `lowconf` in the monitor funnel. Invalid value = gate off (warning in `pipeline-console.log`) |
| `MIN_CONF_PERSON_HOURS_CAMn` | unset (all day) | cam6 = `19:15-06:45` | Local-time `HH:MM-HH:MM` window (may wrap midnight; start inclusive, end exclusive) during which `MIN_CONF_PERSON_CAMn` applies. Invalid window = gate off. Fixed clock times: cam6's IR switch moves with sunset/DST, so widen it in winter |
| `SAVE_INTERVAL` | `30.0` | `30.0` | Office cams: min seconds between events per class |
| `STREET_SAVE_INTERVAL` | `8.0` | unset | Street cams: same, shorter |
| `MIN_TRACK_HITS` | `2` | unset | Tracker hits before a track emits its one event |
| `PGIE_CONFIG` | `pgie_config_rtdetr.txt` | unset | Detector config |
| `FRAME_W` / `FRAME_H` | `1280` / `720` | unset | Mux resolution |

Detector classes (RT-DETR NGC order): 0=Car, 1=RoadSign (filtered out in
`pgie_config_rtdetr.txt`), 2=Person, 3=Bicycle (app label `motorcycle`).
Class 3 is dropped on the office cams because it fires on empty rooms
(`notes/motorcycle_rejects_eyeball.md`). Confidence gates are hard-coded in
`pipeline_multi.py`: `DETECT_MIN_CONF = {0: 0.50, 2: 0.40, 3: 0.40}` (car,
person, motorcycle), on top of the detector's `pre-cluster-threshold=0.4`;
inference runs every 5th frame (`interval=4`). `MIN_CONF_PERSON_CAMn` (+
optional `MIN_CONF_PERSON_HOURS_CAMn`) raises the person gate per camera; cam6
uses 0.60 at night because its IR image fires ~1,750 person events a night
that the VLM rejects, with detector confidence no different from the few it
"accepts" (which, checked by eye, showed no person either). The pipeline logs
the active per-camera gates at startup (`[Detect] camN: person min conf ...`).
The VLM rejection phrases (`_vlm_says_absent` in `pipeline_multi.py`) are
covered by `LOG_DIR=/tmp/vlm_selftest .venv/bin/python3 tools/test_vlm_absent.py`;
the env parsing by `.venv/bin/python3 tools/test_person_gate.py`.

**Queue, frames, snapshots**

| Variable | Code default | Live | Notes |
|----------|--------------|------|-------|
| `VLM_QUEUE_MAX` | `12` | `12` | Office cams dropped above this |
| `STREET_QUEUE_MAX` | `VLM_QUEUE_MAX × 8` | unset | Hard cap for street cams |
| `MAX_EVENT_AGE_S` | `300.0` | unset | Skip events that waited longer in the queue |
| `VLM_STALE_FRAME_MAX_GAP_S` | `20.0` | unset | Skip the VLM call if the chosen frame is further than this from the detection |
| `SNAPSHOT_RING_FILES` | `900` | unset | Per-camera `/tmp/frame_camN_*.jpg` ring |
| `FRAME_PICK_CLOSEST` | `0` | off | `1` = pick the in-window frame closest to the detection (fixes late workers always getting "now"); not enabled yet |
| `SAVE_REJECTS` | `0` | off | `1` = keep rejected frames for labeling (below) |
| `REJECT_DIR` / `REJECT_SAVE_MAX` | `snapshots_rejected` / `2000` | unset | Oldest deleted beyond the cap |

**Other:** `ENABLE_DISPLAY` (`0`) / `HEADLESS` (`1`, legacy alias) for a live
tiled window, `LIVE_STREAM` (`0`) for an HLS stream at `/hls/stream.m3u8`,
`TILER_W`/`TILER_H` (`1280`/`720`), `LOG_DIR` (`logs`), `STATS_HEARTBEAT_S`
(`10`), `CHROMADB_HOST`/`CHROMADB_PORT` (`localhost`/`8000`; pipeline only —
`query_server.py` hard-codes `localhost:8000`). Boot/start only:
`CHROMA_NO_NATIVE` (`0`; `boot_thor.sh` sets `1`), `CHROMA_WAIT_S` (`180`).

## Rolling back to Ollama, and the self-test

Rollback is env-only: set `VLM_BACKEND=ollama` in `.env`, then
`./start_thor.sh --no-monitor`. Ollama stays in fast mode unless you also set
`OLLAMA_THINK=default` (old behavior, thinking on, much slower).

Check both backends and the fallback without touching the running pipeline
(one real call to gx10 and two to Spark, on one saved snapshot):

```bash
LOG_DIR=/tmp/vlm_selftest .venv/bin/python3 tools/vlm_backend_selftest.py [snapshots/<file>.jpg]
```

`LOG_DIR` keeps it out of `logs/pipeline.log`; it prints `SELFTEST PASS`/`FAIL`.
Health check by hand: `curl -s http://gx10-2ea8:8000/v1/models`,
`curl -s http://spark-2251:11434/api/ps`.

## Snapshots and rejected frames

- Accepted events are saved as `snapshots/cam{N}_src{S}_{label}_evt{K}.jpg`
  with a ChromaDB record (metadata includes `vlm_backend`, `vlm_model`).
  Event numbers continue after the highest `evt` already in `snapshots/` for
  each camera, so restarts no longer overwrite old snapshots or reuse Chroma
  ids.
- Rejected events are not saved by default. With `SAVE_REJECTS=1` each rejected
  frame goes to `snapshots_rejected/<time>_cam{N}_{label}_evt{K}.jpg` with a
  sidecar `.json` (camera, label, confidence, backend/model, frame staleness,
  full VLM description).

Full reset (clears search history): `./stop_thor.sh`, `docker stop
camera-pipe1-chromadb-1`, then `sudo rm -rf chroma_data/* && rm -rf
snapshots/*`, `docker start camera-pipe1-chromadb-1`.

## Evals

Offline comparisons are made with `eval_vlm_models.py` over saved snapshots;
results are the `eval_*.json` files in the repo root (e.g.
`eval_vllm_12b_n3.json`: vLLM 12B fast 5.7s mean vs Ollama 12B with thinking
326s; `eval_2x2_*`: thinking vs fast on both backends). Analysis notes:
`notes/detect_min_conf_rejects.md`, `notes/motorcycle_rejects_eyeball.md`,
`notes/vllm_cutover_applied.md`.

**There is no human ground truth.** Eval rows record only the detector label,
the model's description and a rejected flag, and the images are previously
accepted snapshots. So reject rate measures disagreement with the detector (or
an earlier VLM), not accuracy — a high reject rate can mean the detector
produced false positives. The only human check so far is the 13-image eyeball
in `notes/motorcycle_rejects_eyeball.md`.

## Other docs and legacy paths

- **SPARK_DEV.md** — native Thor install runbook, the gx10 vLLM `docker run`
  (§8), and the Spark/Docker development path (`cam_multi.yml`, `Dockerfile`,
  `start.sh`/`stop.sh`), which is Spark-only; Thor runs natively.
- **notes/vllm_cutover_applied.md** — what the VLM cutover changed;
  `notes/vllm_cutover_design.md` is the earlier design (superseded).
- **plan_multi.md** — adding a camera (`.env` + a `<option>` in `search.html`).
- Legacy per-camera pipeline (`pipeline2.py`, `cam1.yml`, `pgie_config.yml`) is
  kept for reference only; `cam1.yml` still defines the Chroma container.

## Power

Under full multi-camera load the monitor often shows power near its limit.
`start_thor.sh` slows the GPU DVFS ramp to avoid over-current throttling; if
you still see over-current warnings, cap the power mode with `sudo nvpmodel -m
2`. Watch with `tegrastats` (or `monitor.py`).
