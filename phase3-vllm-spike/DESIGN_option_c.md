# Option (c) design — dual index: per-object search + per-segment "what happened"

Goal: keep the existing per-object records (for search) AND add Cosmos-Reason2
segment descriptions that summarize what happened in each ~10s window. Approved
2026-07-23.

## Measured facts this design rests on (Phase 3 spike + coexistence test)

- The vLLM plugin runs on the Spark (aarch64/Blackwell wheels OK, CUDA OK).
- Cosmos-Reason2-8B loads from the cached model in ~2 min; ~16.3 GiB weights.
- Description quality (motion-aware, per 10s/10-frame segment) clearly beats
  the per-frame gemma4:12b path.
- COEXISTENCE OK: pipeline (RT-DETR, ~35-40 GB) + 6-stream vLLM at
  gpu_memory_utilization=0.4 peaked ~62 GB of 121 GB; pipeline stayed healthy.
- THROUGHPUT: ~12 s compute per 10 s segment for ONE stream. So one GB10 is
  already slightly slower than real-time for a single continuous segment
  stream. Continuous 10 s segments on all 6 cameras is NOT feasible in real
  time; batching helps but not 6x. => segment analysis must be THROTTLED or
  EVENT-TRIGGERED, not blanket-continuous. (Exact 6-stream batch factor to be
  measured once the sidecar exists with sustained streams.)

## What stays unchanged (per-object path)

RT-DETR detect -> nvtracker -> per-object snapshot -> gemma4:12b via Ollama ->
nomic embed -> ChromaDB collection `vision_events` -> query_server / search UI.
This remains the search substrate (find a specific car/person). No change.

## What's added (segment path)

1. SEGMENT SIDECAR (new): a long-running container built from the committed
   `vllm-ds-spike:latest` image (DeepStream 9.1 sbsa + vLLM 0.21 + torch 2.11 +
   the nvvllmvlm plugin + Cosmos-Reason2-8B). It ingests camera streams in 10 s
   segments and produces a natural-language summary per segment per camera.
   - Config: gpu_memory_utilization=0.4 (fits alongside the pipeline).
   - Model cache mounted from phase3-vllm-spike/hf_home (no re-download).

2. RESULT -> CHROMADB writer (new, small): the sample app connects the plugin's
   `vlm-result` signal to a Kafka publisher. We replace that handler with one
   that: embeds the description with nomic-embed-text (same embedder as the
   object path) and writes to a NEW ChromaDB collection `vision_segments` with
   metadata {camera_id, start_s, end_s, wall_time, description, maybe a
   representative frame path}. No Kafka needed — reuse the existing signal hook.
   (Base it on vllm_ds_app_kafka_publish.py, swap the Kafka sink for a Chroma
   sink; keep --dry-run for testing.)

3. CADENCE CONTROL (the throughput constraint): do NOT analyze every 10 s
   window on every camera. Two levers, both configurable:
   a. Throttle: analyze at most one segment per camera per N seconds
      (SEGMENT_PERIOD, default e.g. 60 s) — set N so total GPU load stays in
      budget for the number of cameras enabled.
   b. Event-trigger (phase 2): only analyze a camera's segment when the
      existing RT-DETR pipeline saw activity there. The pipeline already
      writes per-camera stats/liveness; a lightweight shared signal (file or
      socket) lets the sidecar skip empty scenes. Big efficiency win for
      street cams that are idle most of the time (and at night).
   Which cameras get segment analysis is itself configurable (start with the
   street cams the user cares about).

## Query + UI

- query_server.py searches BOTH collections and merges results, tagging each as
  `object` or `segment`. A query like "white pickup at night" can match a
  segment summary even if no single object snapshot captured it.
- search.html: add a "Moments" lane (segment hits: camera + time window +
  summary) alongside the existing per-object grid. Or a unified time-sorted
  view. Minimal first cut: a separate results section.

## GPU / memory budget (measured, fits)

pipeline ~40 GB + sidecar (gpu_memory_utilization 0.4 => ~48 GB ceiling, ~22 GB
actual for model+KV) => ~62 GB peak of 121 GB. Comfortable. Keep 0.4 unless the
6-stream batch needs more KV cache.

## Build phases (incremental, each independently testable)

- C1. Segment sidecar + Chroma writer: fork the sample app, swap Kafka->Chroma,
  write to `vision_segments`. Test on ONE camera end-to-end (segment -> embed ->
  stored). Measure real batch throughput here.
- C2. Cadence throttle (SEGMENT_PERIOD + per-camera enable list) so all desired
  cameras fit the GPU budget. Run as a compose service alongside the pipeline.
- C3. query_server + UI: search and display both collections.
- C4. (optional) Event-triggering from the RT-DETR pipeline to skip idle scenes.

## Open questions / risks

- 6-stream batched throughput: measure in C1 to set SEGMENT_PERIOD correctly.
- Representative frame for a segment (for the UI thumbnail): the plugin has the
  frames; decide whether to save one JPEG per stored segment.
- Do segment embeddings and object embeddings share one collection or two? Two
  (proposed) keeps schemas clean and lets the UI treat them differently.
- Model load is ~2 min; the sidecar is a persistent service so this is paid
  once at startup, not per segment.
