# Phase 3 Spike — DeepStream 9.1 vLLM plugin (Cosmos-Reason2-8B)

Goal: decide whether the in-pipeline `nvvllmvlm` plugin should replace the
current `/tmp`-JPEG + remote-Ollama VLM worker. This is an **isolated
experiment** — it does NOT touch the production pipeline (`pipeline_multi.py`)
or ChromaDB. Keep the Ollama path as the default until this proves out.

Status: PREPARED (not yet run). Nothing here downloads the 40 GB model or
installs vLLM until you run the steps below.

## What the plugin is (from the sample)

- Source: `~/sd/DeepStream/src/apps/reference_apps/deepstream-vllm-plugin/`
- Model: `nvidia/Cosmos-Reason2-8B` (HF-gated; needs token + license accept)
- Paradigm: **segment-based** — collects `segment.length_sec` (default 10s) of
  frames, subsamples to `selection_fps` (default 1 fps), and asks the VLM one
  prompt per segment per stream. This is the key difference from our current
  **per-detection** model (one description per tracked car/person).
- Output: prints to console (`--dry-run`) or publishes to Kafka. It does NOT
  embed or write to ChromaDB — wiring that up is only worth doing if the spike
  succeeds.
- Memory: `gpu_memory_utilization: 0.7`. On the Spark's 121 GB unified memory
  that is a large reservation (~40 GB+ for the 8B model + KV cache). Run it
  when the production pipeline is stopped, or lower the fraction.

## Prerequisites (one-time)

1. HuggingFace token: huggingface.co → Profile → Access Tokens → create.
2. Accept the license at https://huggingface.co/nvidia/Cosmos-Reason2-8B
   (NVIDIA Open Model License) with that account.

## Pre-downloading the model (do this ahead of the spike)

Sizes: ~16-20 GB on disk (8B params). The "40 GB" is the RUNTIME GPU-memory
reservation, not the download. Host has ample space (2.9 TB free).

DO NOT pip-install anything on the host. The DeepStream container already ships
the `hf` CLI (huggingface_hub 1.24.0), and `/workspace/phase3-vllm-spike` is
mounted from the host — so download FROM INSIDE the container straight into the
host-mounted cache dir. Nothing touches the host Python.

```bash
# Cache dir lives under the repo (mounted into the container as /workspace/...)
mkdir -p ~/sd/camera-pipe1/phase3-vllm-spike/hf_home

DS=$(docker compose -f ~/sd/camera-pipe1/cam_multi.yml ps -q deepstream)

# Authenticate once (interactive — paste the token; keeps it out of shell history)
docker exec -it -e HF_HOME=/workspace/phase3-vllm-spike/hf_home "$DS" hf auth login

# Download into the mounted cache (resumable; safe to run alongside the pipeline)
docker exec -it -e HF_HOME=/workspace/phase3-vllm-spike/hf_home "$DS" \
  hf download nvidia/Cosmos-Reason2-8B

# Verify it landed on the host
du -sh ~/sd/camera-pipe1/phase3-vllm-spike/hf_home/hub/models--nvidia--Cosmos-Reason2-8B
```

When the spike container runs (step 2 of "Run steps"), point it at this same
cache so nothing re-downloads:

```bash
  -v ~/sd/camera-pipe1/phase3-vllm-spike/hf_home:/root/.cache/huggingface \
  # inside the spike container: export HF_HOME=/root/.cache/huggingface
```

Notes:
- Files land root-owned on the host (the container runs as root — same as the
  existing logs/, stats/, snapshots/). `sudo chown -R $USER` if you want.
- hf_home/ is gitignored (tens of GB) — see .gitignore.
- If you ever DO need host-side Python tooling that isn't in the container, use
  an isolated env (`python3 -m venv .venv` or `pipx`/`uvx`), never system pip.

## Image note (Spark-specific)

The sample README launches the plugin in `9.1-triton-multiarch`. On the DGX
Spark that image is missing the bundled Jetson multimedia libs (same
`libnvbufsurface.so.1.0.0` problem we hit in Phase 1 — see
`../plan-cam-pipe-ds9.1.txt`). Use the **sbsa-dgx-spark** image instead:

    nvcr.io/nvidia/deepstream:9.1-triton-sbsa-dgx-spark

The plugin's `install.sh` (vLLM 0.21.0 + PyTorch 2.11.0) should be run inside
that container. If vLLM has no prebuilt aarch64/Blackwell wheel for this stack,
that is itself a finding — record it and stop; it would mean the plugin is not
yet practical on Spark.

## Run steps (when ready to spike)

```bash
# 1. Stop the production pipeline to free GPU memory
cd ~/sd/camera-pipe1 && ./stop.sh

# 2. Enter a throwaway container on the plugin source (sbsa image)
docker run -it --rm --runtime=nvidia --gpus all --network=host \
  -v ~/sd/DeepStream/src/apps/reference_apps/deepstream-vllm-plugin:/home/vllm_ds_plugin \
  -v ~/sd/camera-pipe1/phase3-vllm-spike:/spike \
  nvcr.io/nvidia/deepstream:9.1-triton-sbsa-dgx-spark

# 3. Inside the container:
cd /home/vllm_ds_plugin && chmod +x install.sh && ./install.sh
export HF_TOKEN=<your token>

# 4. Dry-run against ONE real camera (use an RTSP_URL_CAMn value from ../.env;
#    do NOT paste credentials into a committed file). Point it at a street cam
#    that actually sees traffic so the descriptions are meaningful.
python3 vllm_ds_app_kafka_publish.py "<rtsp url>" --dry-run \
  --converter-mode gpu            # Jetson/Thor/Spark need gpu or auto, not default

# Optionally use the tuned prompt in /spike/config.sample.yaml (copied from the
# sample, with user_prompt aligned to how we describe vehicles/persons today).
```

## What to evaluate (the decision)

1. **Does it run on Spark at all?** vLLM aarch64/Blackwell wheel availability,
   model load time, GPU memory headroom with the pipeline stopped.
2. **Description quality vs gemma4:12b** on OUR camera imagery — is Cosmos-Reason2
   better for search (color, make/model, direction, clothing detail)?
3. **Segment vs per-object.** Our search index is keyed on per-track events
   (`cam{n}_..._evt{k}` + snapshot + embedding). Segment output describes a 10s
   window, not a specific car. Decide:
     (a) adopt segment mode and re-architect the index around time windows, or
     (b) run the plugin in image-only mode (`video_mode: 0`) as a faster
         drop-in per-frame describer, or
     (c) stick with Ollama (plugin not worth the memory/complexity).
4. **Throughput.** Does segment batching beat the current ~40 s/event serial
   Ollama worker in events (or seconds of video) understood per minute?

## Do NOT (until the decision is made)

- Wire the plugin into `pipeline_multi.py`.
- Point it at ChromaDB.
- Make it part of `start.sh`.
