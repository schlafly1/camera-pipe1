# Spark Development — Multi-Stream Pipeline

Develop on the **DGX Spark** with a **single DeepStream 9 container** running
all RTSP streams through one batched `nvstreammux` → `nvinfer` pipeline.
Deploy the same layout on **Jetson Thor** at the edge.

## Architecture

```
RTSP cam 1 ──> nvurisrcbin src0 ──┐
RTSP cam 2 ──> nvurisrcbin src1 ──┼──> nvstreammux (batch=N)
RTSP cam 3 ──> nvurisrcbin src2 ──┤         │
RTSP cam 4 ──> nvurisrcbin src3 ──┘         v
                                    nvinfer (TrafficCamNet)
                                            │
                              probe → VLM queue → Ollama (Spark)
                                            │
                                    ChromaDB + query_server
```

Compared to `pipeline2.py` + `cam1.yml` (one container per camera), this uses
one TRT engine, one decode batch, and far less GPU contention.

## Prerequisites (Spark)

```bash
# NVIDIA Container Toolkit + Docker
docker run --rm --gpus all nvcr.io/nvidia/deepstream:9.0-triton-multiarch \
  deepstream-app --version

# Ollama on Spark (VLM stays here, not on Thor)
OLLAMA_HOST=0.0.0.0 ollama serve   # separate terminal or systemd
ollama pull gemma4:26b
ollama pull nomic-embed-text
```

## Clone on Spark (everything you need is on GitHub)

```bash
git clone -b multi-stream https://github.com/schlafly1/camera-pipe1.git
cd camera-pipe1
```

You do **not** need to copy anything from the Windows `C:\sd\thor` folder except
your real **`.env`** (camera URLs and secrets) if you already have one on Thor.

| On GitHub (`multi-stream` branch) | Not on GitHub — copy manually if needed |
|-----------------------------------|----------------------------------------|
| All code, Docker, docs | `.env` (from Thor; never committed) |
| `env.example` template | `chroma_data/` (optional; starts empty) |
| `SPARK_DEV.md`, `readme.md` | `snapshots/`, `stats/` (created at runtime) |
| | `setup.txt`, `thor-spark-plan.txt` (local planning notes only) |

## Full setup on Spark

```bash
# 1. Clone
git clone -b multi-stream https://github.com/schlafly1/camera-pipe1.git
cd camera-pipe1

# 2. Environment — copy from Thor or start from template
cp env.example .env
nano .env    # RTSP_URL_CAM1..4, OLLAMA_HOST=http://127.0.0.1:11434

# 3. Local data dirs (gitignored; created automatically but safe to pre-create)
mkdir -p chroma_data snapshots stats

# 4. Ollama (host, not in Docker)
ollama pull gemma4:26b
ollama pull nomic-embed-text
OLLAMA_HOST=0.0.0.0 ollama serve &

# 5. Build and start containers
docker compose -f cam_multi.yml build    # first time ~3-5 min
docker compose -f cam_multi.yml up -d

# 6. Run the pipeline (one shell — handles ALL cameras together)
docker exec -it camera-pipe1-deepstream-1 bash
python3 pipeline_multi.py

# 7. Query server + search UI (another shell, or background it)
docker exec -it camera-pipe1-deepstream-1 bash
python3 query_server.py
# (or in background from host:)
# docker exec -d camera-pipe1-deepstream-1 python3 query_server.py

# 8. Monitor (on host, outside any container)
python3 monitor.py

# Search UI will be at http://localhost:8001
# REST: curl "http://localhost:8001/query?text=red+car"
```

Search UI: http://localhost:8001

## Live video feed (2×2 tile)

The legacy `pipeline2.py` never connected a display sink — it only wrote JPEGs to
disk. The multi-stream pipeline can show a **live tiled window with bounding boxes**
and still run VLM in parallel.

```bash
# On the host (once per session)
xhost +local:

# In .env
ENABLE_DISPLAY=1
DISPLAY=:0          # or :1 over SSH -Y

# If you get auth errors, also mount your cookie (add to cam_multi.yml volumes):
#   - /home/you/.Xauthority:/root/.Xauthority:ro

docker compose -f cam_multi.yml up -d --force-recreate
docker exec -it camera-pipe1-deepstream-1 bash
python3 pipeline_multi.py
```

This is not hard to add — it is the standard DeepStream path:
`nvinfer → tee → [tiler → osd → display]` plus a JPEG branch for VLM.

Headless mode (`ENABLE_DISPLAY=0`, default) skips the window and saves GPU for
more streams.

## Run (multi-stream – one process for all cameras)

The old "one container + one terminal per camera" no longer applies.

Everything runs in the single container started by `cam_multi.yml`:

```bash
# 1. Start (or restart) the compose stack
docker compose -f cam_multi.yml up -d

# 2. Run the pipeline (all cameras in one process)
docker exec -it camera-pipe1-deepstream-1 bash
python3 pipeline_multi.py

# 3. In a second terminal, run the search API/UI
docker exec -it camera-pipe1-deepstream-1 bash
python3 query_server.py

# 4. On the host (outside Docker), run the stats monitor
python3 monitor.py
```

- The pipeline runs continuously and auto-reconnects cameras.
- Query server listens on :8001 (inside the container = same as host because of `network_mode: host`).
- Use `docker compose -f cam_multi.yml stop` (not down) to preserve the TRT engine cache.

Search UI: http://localhost:8001  
API: `curl "http://localhost:8001/query?text=person"`

## Camera connection reliability (TCP + per-source snapshots)

Most IP cameras (Reolink, Dahua, etc.) are much more reliable over TCP than UDP when running inside the DeepStream container.

- The repo now defaults to forcing TCP (`RTSP_TRANSPORT_CAMn=4`).
- Additional stability properties (latency, retransmit, drop-on-latency) are applied for TCP sources.
- JPEG frames for the VLM are now captured on per-camera branches (`/tmp/frame_camN_*.jpg`) instead of a single global post-batch stream. This greatly reduces the chance of the VLM seeing the wrong scene or "no fresh frame" errors.

If a camera still shows "No data from source", try the specific ffplay command you used on the host, then set only that camera to 4 (or test variants of the URL).

See the comments at the end of your `.env` for the exact ffplay lines that worked for you.

Search UI: http://localhost:8001

## Key files

| File | Purpose |
|------|---------|
| `pipeline_multi.py` | Single batched DS9 pipeline + VLM worker |
| `streams_config.py` | Loads RTSP_URL_CAM1..N from `.env` |
| `cam_multi.yml` | One `deepstream` service (not 4) |
| `pgie_config_multi.yml` | TrafficCamNet config (batch-size overridden at runtime) |
| `pipeline2.py` | Legacy per-camera pipeline (keep for rollback) |
| `cam1.yml` | Legacy per-camera compose |

## Environment variables

| Variable | Default | Notes |
|----------|---------|-------|
| `RTSP_URL_CAM1..N` | — | One URL per camera; stop numbering at first gap |
| `RTSP_TRANSPORT_CAMn` | `0` | Set `4` for TCP (strongly recommended for most IP cams inside container; see reliability section below) |
| `FRAME_W` / `FRAME_H` | `1280` / `720` | Use substream resolution when possible |
| `ENABLE_DISPLAY` | `0` | `1` = live 2×2 tile + bounding boxes (needs X11) |
| `HEADLESS` | `1` | Legacy alias: `0` also enables display |
| `TILER_W` / `TILER_H` | `1280` / `720` | Live window size |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Spark Ollama; Thor points at Spark IP |
| `VLM_MODEL` | `gemma4:26b` | |
| `SAVE_INTERVAL` | `30.0` | Min seconds between saves per class per camera (only applies to office cams). Raise if VLM can't keep up (see monitor.py). |
| `VLM_QUEUE_MAX` | `12` | Shared queue across all cameras |
| `CAM_TYPE_CAMn` | `street` | "office" (test/high-volume, throttled + droppable when VLM busy) or "street" (real low-volume; VLM on every detection, never throttled). Set e.g. CAM_TYPE_CAM1=office, CAM_TYPE_CAM3=street. Office cams are only used for testing and can be dropped to protect street cams. |

## First run

TensorRT builds a batched engine inside the container (~5 min for batch=4).
Watch logs for `Building engine` / `deserializing trt engine`.

Use `docker compose -f cam_multi.yml stop` (not `down`) to preserve the engine.

## Adding a camera

1. Add `RTSP_URL_CAM5=...` to `.env`
2. (Optional) Add `RTSP_TRANSPORT_CAM5=4` if it needs TCP.
3. Add `<option value="5">Camera 5</option>` in `search.html`
4. Restart the pipeline inside the container (it will pick up the new stream automatically).

No changes to `cam_multi.yml` or `pipeline_multi.py`.

## Deploy to Thor

Same compose file and code. On Thor `.env`:

```env
OLLAMA_HOST=http://<spark-lan-ip>:11434
FRAME_W=1280
FRAME_H=720
HEADLESS=1
```

Run `monitor.py` on the Thor host to watch per-camera stats in `./stats/`.

## Rollback

To return to the per-camera model:

```bash
docker compose -f cam1.yml up -d
# One terminal per camera:
docker exec -it <cam1-container> python3 pipeline2.py
```