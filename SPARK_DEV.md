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

## Setup

```bash
git clone https://github.com/schlafly1/camera-pipe1.git
cd camera-pipe1
git checkout -b multi-stream   # after you push this branch

cp env.example .env
# Edit .env: RTSP_URL_CAM1..4, OLLAMA_HOST=http://127.0.0.1:11434

docker compose -f cam_multi.yml build
docker compose -f cam_multi.yml up -d
```

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

## Run

```bash
# Single pipeline for all cameras
docker exec -it camera-pipe1-deepstream-1 bash
python3 pipeline_multi.py

# Query server (same container or any shell with repo mounted)
python3 query_server.py

# Monitor (on host, outside container)
python3 monitor.py
```

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
| `RTSP_TRANSPORT_CAMn` | `0` | Set `4` for TCP-only cameras |
| `FRAME_W` / `FRAME_H` | `1280` / `720` | Use substream resolution when possible |
| `ENABLE_DISPLAY` | `0` | `1` = live 2×2 tile + bounding boxes (needs X11) |
| `HEADLESS` | `1` | Legacy alias: `0` also enables display |
| `TILER_W` / `TILER_H` | `1280` / `720` | Live window size |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Spark Ollama; Thor points at Spark IP |
| `VLM_MODEL` | `gemma4:26b` | |
| `SAVE_INTERVAL` | `5.0` | Min seconds between saves per class per camera |
| `VLM_QUEUE_MAX` | `12` | Shared queue across all cameras |

## First run

TensorRT builds a batched engine inside the container (~5 min for batch=4).
Watch logs for `Building engine` / `deserializing trt engine`.

Use `docker compose -f cam_multi.yml stop` (not `down`) to preserve the engine.

## Adding a camera

1. Add `RTSP_URL_CAM5=...` to `.env`
2. Add `<option value="5">Camera 5</option>` in `search.html`
3. Restart pipeline — `streams_config.py` picks up the new URL automatically

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