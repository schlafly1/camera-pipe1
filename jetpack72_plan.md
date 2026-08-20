# JetPack 7.2 Feature Opportunities for camera-pipe1

Two JetPack 7.2 features are worth evaluating for this project:
**MIG** (hardware GPU partitioning) and **agentic AI** (proactive reasoning over accumulated data).

---

## 1. MIG — Multi-Instance GPU

### What it does

Jetson Thor's Blackwell GPU can be split into two hardware-isolated partitions.
Each gets its own dedicated compute, cache, and memory bandwidth — no interference.

| Partition        | SMs | CUDA cores | Natural fit             |
|------------------|-----|-----------|-------------------------|
| AI/graphics      | 12  | 1536      | DeepStream, video decode |
| Robotics/safety  |  8  | 1024      | LLM inference, embeddings |

### Why it matters for this project

Currently DeepStream (TrafficCamNet × 4 cameras) and Ollama (Gemma4:26b VLM +
nomic-embed-text) share the same GPU and compete for memory bandwidth. Frame
drops may be caused partly by this contention. MIG gives each workload a
guaranteed slice.

### Setup sketch (MIG is "technology preview" in JP7.2)

```bash
# 1. List what profile names the Thor actually exposes
sudo nvidia-smi mig --list-gpu-instance-profiles

# 2. Create instances (exact profile names TBD from above command)
sudo nvidia-smi mig -cgi <12sm-profile>,<8sm-profile> -C

# 3. Find the MIG device UUIDs
nvidia-smi -L   # shows  MIG-GPU-xxxx/0/0, MIG-GPU-xxxx/0/1
```

```yaml
# cam1.yml — point all deepstream services at the large partition
environment:
  CUDA_VISIBLE_DEVICES: "MIG-GPU-xxxx/0/0"   # 12-SM instance
```

```bash
# Run Ollama on the small partition
CUDA_VISIBLE_DEVICES="MIG-GPU-xxxx/0/1" ollama serve
```

### Key unknown: does Gemma4:26b fit in the 8-SM partition?

Gemma4:26b needs roughly 15–20 GB of GPU memory depending on quantization.
Thor has 96 GB of unified (CPU+GPU) memory, but MIG allocates a fixed slice to
each partition. The exact breakdown for Thor hasn't been published yet — check
`nvidia-smi mig --list-gpu-instance-profiles` once MIG is enabled.

**If Gemma4:26b doesn't fit:** use a smaller model on-device (gemma3:12b,
llava:13b) and keep the 26b on the remote Jetson. The partition boundary still
protects DeepStream from LLM memory pressure regardless.

### Recommendation

Try MIG with `gemma3:12b` in the 8-SM partition first. If description quality is
acceptable you eliminate the remote-Jetson dependency. If not, fall back to remote
Ollama but keep MIG enabled — DeepStream alone on the 12-SM instance will have
more predictable latency.

---

## 2. Agentic AI Features

### What "agentic" means in NVIDIA's framing

An agent observes → reasons → acts in a loop rather than just responding to
queries. For this project: instead of only storing VLM descriptions in ChromaDB
and answering ad-hoc searches, an agent proactively watches accumulated data,
notices patterns, and surfaces alerts without being asked.

The description of your system — ingest RTSP, detect objects, describe with VLM,
embed into ChromaDB — is exactly the Metropolis VSS (Video Search and
Summarization) blueprint pattern. The "agentic" layer sits on top of that.

### Building an agent loop on top of camera-pipe1

Add a lightweight agent loop to `query_server.py`. The Ollama client is already
available; Gemma4 is already running. The loop is ~60–80 lines:

```
Every N minutes:
  1. Query ChromaDB: get all events from the past 30 minutes
  2. Build a prompt: "Here are recent camera observations. Summarize anything
     unusual or worth flagging. Be concise."
  3. Send to Gemma4 via ollama.chat()
  4. If the response flags something notable, write to ./alerts/<timestamp>.json
     (and/or POST to a webhook, send an email, etc.)
  5. Optionally: upsert a rolling "scene summary" document back into ChromaDB
     so future queries have long-horizon context
```

This is the core of what NVIDIA means by "agentic": the system watches its own
accumulated observations and decides what matters without a human asking first.

The loop belongs in a new function `agent_loop()` in `query_server.py`, run as a
background thread (alongside the existing FastAPI app). The interval and prompt
are configurable via `.env`.

### Metropolis VSS Blueprint

NVIDIA's VSS blueprint is the closest pre-built analog to camera-pipe1:
RTSP → detection → VLM description → vector DB → NL query. It's worth reading
the blueprint source for ideas (especially its summarization chunking strategy
for long-running sessions), but migrating to it would mean replacing the current
DeepStream + pyservicemaker code. Treat it as a reference architecture, not a
migration target.

---

## Priority order

1. **MIG** — highest potential impact on the original frame-drop problem.
   Guaranteed GPU bandwidth for DeepStream regardless of what Ollama is doing.
   Test with gemma3:12b first.

2. **Agent summary loop** — add to `query_server.py` once the pipeline is stable
   and producing good descriptions. Small code change, qualitatively changes what
   the system can do.

3. **Metropolis VSS** — reference reading only.

---

## Caveats

- Thor is now confirmed running **JetPack 7.2 with DeepStream 9.1 natively**
  — there is no Docker image for it (NVIDIA doesn't publish one for Jetson;
  Spark's `Dockerfile` pins a `9.1-triton-sbsa-dgx-spark` base that's
  specific to Spark's SBSA-DGX setup and doesn't apply to a real Jetson
  device like Thor). So the plan is native install, not a container — see
  `SPARK_DEV.md` → "Native Thor runbook". **TODO on first real Thor
  session**: run `deepstream-app --version` on the host (not in a
  container) to confirm the native 9.1 install actually matches what
  `pipeline_multi.py` needs, before assuming parity with Spark's image.

- MIG is marked **technology preview** in JP7.2. Profile names and memory splits
  for Thor are not yet fully documented. Check the actual device before designing
  around specific partition sizes.
