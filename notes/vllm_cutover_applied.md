# vLLM cutover — applied 2026-09-25

Implements notes/vllm_cutover_design.md in pipeline_multi.py, with changes
from a Claude Code review (2026-09-25):

- `VLM_BACKEND=ollama|vllm` (code default ollama; live .env = vllm).
  `VLLM_URL` default `http://gx10-2ea8:8000` — the short name `gx10` does
  NOT resolve from Thor. `VLLM_MODEL=google/gemma-4-12B-it`,
  `VLLM_TIMEOUT_S=20`, `VLLM_HEALTHCHECK_S=30`.
- vLLM requests send `chat_template_kwargs: {"enable_thinking": false}`
  explicitly; content that looks like leaked reasoning raises -> fallback.
- Any failed vLLM call marks it unhealthy immediately (next probe in 30s),
  then falls back to Ollama for that event. Counters: `vlm_calls_vllm`,
  `vlm_calls_ollama`, `vllm_fallback` in stats/*.json; per-call VLM latency
  in `vlm_call_ms_avg`/`vlm_call_ms_max`, `vlm_backend_last`.
- Ollama path: `OLLAMA_THINK=false` by default (was: param omitted, which
  leaves gemma4 thinking ON). `default` restores the old omit behavior.
  Explicit Ollama clients with timeouts (chat 180s, embeddings 30s) — the
  module-level client had no timeout.
- Embeddings unchanged: nomic-embed-text on OLLAMA_HOST (Spark).
- Prompts and `_vlm_says_absent()` unchanged. REJECT log lines now end with
  `[vllm]` or `[ollama]`; Chroma metadata gains `vlm_backend`, `vlm_model`.

Other changes in the same restart:
- `DROP_CLASSES_CAMn` (streams_config.py): per-camera class drop, applied in
  the probe before the `detections` counter. Live: cam1, cam2 (office) drop
  class 3 (RT-DETR Bicycle / app label "motorcycle").
- `SAVE_REJECTS=1` (default off): capped copy of rejected frames + JSON
  sidecar in `snapshots_rejected/` (`REJECT_SAVE_MAX`, default 2000).
- `FRAME_PICK_CLOSEST=1` (default off, NOT enabled yet): fixes
  get_jpeg_after, where any file newer than detection+10s overrides an
  in-window pick.
- Event ids now continue past the highest evt in snapshots/ per camera, so
  restarts no longer overwrite snapshot JPEGs / collide on Chroma ids.
- VLM workers retry the Chroma connection instead of dying at boot.
- Boot auto-start: systemd user unit deploy/camera-pipe1.service ->
  boot_thor.sh (linger enabled for roger). start_thor.sh: anchored pkill
  patterns + wait, `sudo -n` for DVFS, `CHROMA_NO_NATIVE=1` mode.

Correction to the design note: live Chroma is the Thor-local Docker
container camera-pipe1-chromadb-1 (localhost:8000), not Spark.

Rollback: set `VLM_BACKEND=ollama` in .env and restart (start_thor.sh).
