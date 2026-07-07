# camera-pipe1

DeepStream 9.0 camera pipeline for NVIDIA Jetson Thor (JetPack 7.2) and DGX Spark.

Detects vehicles, motorcycles, and persons via TrafficCamNet, sends each
detection frame to a VLM (Gemma4:26b via Ollama) for a natural-language
description, embeds the description with nomic-embed-text, and stores it in
ChromaDB. A FastAPI server serves a search UI for natural-language queries.

## Recommended: multi-stream pipeline

**One DS9 pipeline for all cameras** — develop on Spark, deploy on Thor.

See **[SPARK_DEV.md](SPARK_DEV.md)** for full up-to-date instructions (including how to run `pipeline_multi.py` and `query_server.py`).

Quick start (multi-stream):

```bash
docker compose -f cam_multi.yml up -d
docker exec -it camera-pipe1-deepstream-1 bash
python3 pipeline_multi.py     # one process handles all cameras
```

In a second shell:
```bash
docker exec -it camera-pipe1-deepstream-1 bash
python3 query_server.py
```

UI: http://localhost:8001

See SPARK_DEV.md for environment variables (especially `RTSP_TRANSPORT_CAMn=4` for reliable TCP), troubleshooting camera feeds, and the difference from the old per-camera setup.

```bash
cp env.example .env          # set RTSP_URL_CAM1..4, OLLAMA_HOST
docker compose -f cam_multi.yml build
docker compose -f cam_multi.yml up -d

docker exec -it camera-pipe1-deepstream-1 bash
python3 pipeline_multi.py    # all cameras in one process

python3 query_server.py      # search UI on :8001
python3 monitor.py           # on host — per-camera stats
```

| Multi-stream | Legacy (per-camera) |
|--------------|---------------------|
| `pipeline_multi.py` | `pipeline2.py` |
| `cam_multi.yml` | `cam1.yml` |
| `pgie_config_multi.yml` | `pgie_config.yml` |

Set `ENABLE_DISPLAY=1` in `.env` for a live 2×2 tile with bounding boxes (the legacy
pipeline never wired a display sink).

## Requirements

- NVIDIA Jetson Thor or DGX Spark with Docker + NVIDIA Container Toolkit
- DeepStream 9.0 container (`nvcr.io/nvidia/deepstream:9.0-triton-multiarch`)
- Ollama with `gemma4:26b` and `nomic-embed-text` (Spark recommended for VLM)
- One or more RTSP cameras reachable from the edge device

## Legacy setup (one container per camera)

```bash
# 1. Copy and fill in your camera URLs
cp env.example .env
# edit .env — set RTSP_URL_CAM1..4, set RTSP_TRANSPORT_CAMx=4 for TCP cameras

# 2. Allow containers to use the X11 display (once per session / reboot)
xhost +local:

# 3. Build and start containers
cd ~/sd/camera-pipe1
docker compose -f cam1.yml build   # first time only (~3-5 min)
docker compose -f cam1.yml up -d
```

## Running

Open one terminal per camera plus one for the query server, one to monitor:

```bash
# Camera 1
docker exec -it camera-pipe1-deepstream-cam1-1 bash
DISPLAY=:1 python3 pipeline2.py

# Camera 2 (repeat for cam3, cam4)
docker exec -it camera-pipe1-deepstream-cam2-1 bash
DISPLAY=:1 python3 pipeline2.py

# Camera 3
docker exec -it camera-pipe1-deepstream-cam3-1 bash
DISPLAY=:1 python3 pipeline2.py
# Camera 4
docker exec -it camera-pipe1-deepstream-cam4-1 bash
DISPLAY=:1 python3 pipeline2.py

# Query server (run in any deepstream container)
docker exec -it camera-pipe1-deepstream-cam1-1 bash
python3 query_server.py

# Monitoring app
python3 monitor.py

```

Search UI: http://localhost:8001  
REST API: `curl "http://localhost:8001/query?text=red+car"`

## Configuration

| Setting | File | Default |
|---------|------|---------|
| Inference interval (0 = every frame, 4 = every 5th) | `pgie_config.yml` → `interval` | 4 |
| Min seconds between saves per object class per camera | `pipeline_multi.py` → `SAVE_INTERVAL` (in .env) | 30.0 (raise if VLM can't keep up) |
| VLM model | `.env` → `VLM_MODEL` | `gemma4:26b` |
| Force TCP RTSP | `.env` → `RTSP_TRANSPORT_CAMx=4` | 0 (UDP) |

## Shutdown

```bash
# Preserves TRT engines (cached inside containers) — fast restart next time
docker compose -f cam1.yml stop

# After reboot: xhost +local: then docker compose -f cam1.yml up -d

# Full reset (clears ChromaDB and snapshots)
sudo rm -rf chroma_data/*
rm -rf snapshots/*
```

## Adding cameras

See `plan.md` for step-by-step instructions.

## Power

Running 4 cameras + VLM inference draws ~60-80W. If you see over-current
warnings, cap the power mode: `sudo nvpmodel -m 2`. Monitor with `tegrastats`.

## Offloading Ollama models

2. On the remote Jetson — start Ollama bound to all interfaces:
  OLLAMA_HOST=0.0.0.0 ollama serve
  And make sure models are pulled (only needs to be done once):
  ollama pull gemma4:26b
  ollama pull nomic-embed-text
  
  3. In your .env on Thor — change OLLAMA_HOST:
  OLLAMA_HOST=http://<remote-jetson-ip>:11434

  4. Restart the deepstream containers to pick up the new env:
  docker compose -f cam1.yml up -d --force-recreate

  That's it. Since the cameras are quiet, the remote Ollama latency is a
  non-issue — VLM calls are already fire-and-forget on a background thread, so a
  slow response just means a slight delay before the description lands in
  ChromaDB. DeepStream detection continues uninterrupted regardless.

