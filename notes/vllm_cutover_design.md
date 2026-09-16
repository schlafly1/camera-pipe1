# vLLM cutover design — VLM_BACKEND switch with Ollama fallback

**Date:** 2026-09-16. Design only — nothing in this note has been applied to the
live tree or the running pipeline. Builds on eval_vllm_12b_n3.json (vLLM ~57x
faster, comparable reject rate) and notes/motorcycle_rejects_eyeball.md (this
cutover fixes latency/backlog, not the motorcycle hallucination problem).

## 1. How the live code calls Ollama today

Everything lives in pipeline_multi.py:

- Chat/description leg — vlm_worker() (pipeline_multi.py:561-686), one
  call per detection at lines 624-633:
  ```python
  resp = ollama.chat(
      model=VLM_MODEL,
      messages=[{"role": "user", "content": prompt, "images": [jpeg_b64]}],
      options={"num_ctx": VLM_NUM_CTX},
  )
  description = resp["message"]["content"].strip()
  ```
  VLM_MODEL defaults to gemma4:26b (env-overridden to gemma4:12b live),
  read once at import time (line 41). The ollama package (line 26 import
  ollama) picks up OLLAMA_HOST from the environment implicitly — no explicit
  ollama.Client(host=...) anywhere, so there's no client object to swap out,
  just this one call site.
- Embedding leg — line 651, ollama.embeddings(model=EMBED_MODEL,
  prompt=description), EMBED_MODEL = "nomic-embed-text" (line 49, not
  env-overridable today). This must stay exactly as-is — vLLM's container
  only serves the chat model, not an embedding model, and re-pointing
  embeddings anywhere else would require re-embedding the whole ChromaDB
  collection to stay comparable.
- Concurrency — VLM_WORKERS (line 60, default 2) identical vlm_worker
  threads share one queue.Queue (spawned at line 880-889). Any new global
  state (health-check cache) must be thread-safe across these.
- Reject check — _vlm_says_absent() (line 184) runs on the returned
  description string regardless of which backend produced it. No change
  needed — it's already backend-agnostic.
- Stats funnel — StatsTracker._COUNTER_KEYS (line 230-234) is the fixed
  set of counters monitor.py reads via .get(key, default) (confirmed
  additive/tolerant, e.g. monitor.py:262-264). Adding one new key here is
  safe and won't break monitor.py, though monitor.py's fixed-width table
  won't print it until someone also edits monitor.py — optional follow-up,
  not required for the cutover.

eval_vlm_models.py's _run_vllm() (its own file, not pipeline_multi.py)
already proves the exact request/response shape for gx10's vLLM server — the
patch below reuses that shape verbatim rather than inventing a new one.

## 2. Design

### Env vars (all new, all default to today's exact behavior)

| var | default | purpose |
|---|---|---|
| VLM_BACKEND | ollama | "ollama" or "vllm". Default means zero behavior change until someone opts in. |
| VLLM_URL | http://gx10-2ea8:8000 | gx10's OpenAI-compatible endpoint, matches eval_vlm_models.py's default. |
| VLLM_MODEL | google/gemma-4-12B-it | must match what's actually loaded in the vllm-gemma4-12b container. |
| VLLM_MAX_TOKENS | 300 | matches the eval's setting; vLLM requires an explicit cap (Ollama's num_ctx path doesn't need one the same way). |
| VLLM_TIMEOUT_S | 30 | generous vs. the eval's 8.6s p95 — bounds worst-case per-call hang without being trigger-happy. |
| VLLM_HEALTHCHECK_S | 30 | minimum seconds between real health-check HTTP calls; avoids paying a round trip on every single detection. |

### Request/response shape difference (why two separate helper functions)

Ollama's images field takes bare base64; vLLM's OpenAI-compatible endpoint
wants a content parts list with a data:image/jpeg;base64,... URL, and
needs max_tokens set explicitly. Response parsing also differs:
resp["message"]["content"] (Ollama) vs.
resp.json()["choices"][0]["message"]["content"] (vLLM). Thinking is off at
the vLLM server/template level already (per the gx10 container config), so no
chat_template_kwargs or _final_answer()-style thinking-stripping is needed
in production — that stripping in eval_vlm_models.py was defense-in-depth
for eval runs, not something this path needs to inherit.

### Health-check + fallback

One cached boolean (_vllm_health, module-level dict + threading.Lock),
refreshed at most once per VLLM_HEALTHCHECK_S by GET {VLLM_URL}/v1/models
(3s timeout). _vlm_describe() is the single dispatch point:

- VLM_BACKEND=ollama (default): calls _vlm_chat_ollama() directly. Never
  touches vLLM, never uses the health-check path in practice — this mode is
  exactly today's code, just refactored into a function.
- VLM_BACKEND=vllm: checks the health cache; if healthy, tries
  _vlm_chat_vllm(). On any exception (timeout, connection refused, 5xx,
  bad JSON) or a cached-unhealthy result, falls back to _vlm_chat_ollama()
  for that single call and bumps a new vllm_fallback stats counter. Ollama
  stays fully configured and warm the whole time (VLM_MODEL/OLLAMA_HOST
  unchanged), so fallback has no extra setup cost per call beyond the vLLM
  attempt's timeout.

This means a gx10 outage (it has exited 255 before, unattended) degrades
straight back to today's live behavior — slow, but working — rather than
silently dropping every detection.

### What does NOT change

- EMBED_MODEL / the ollama.embeddings() call (line 651) — always Ollama.
- Detector code, DETECT_MIN_CONF, dedup/throttle/queue logic — untouched.
- _vlm_says_absent() reject check — untouched, backend-agnostic already.
- monitor.py, query_server.py — untouched (stats stay additive).
- No live restart, no flipping VLM_BACKEND in .env as part of landing this
  patch — ships dark, defaulting to ollama.

## 3. Draft patch (sketch only — NOT applied to the working tree)

```diff
--- a/pipeline_multi.py
+++ b/pipeline_multi.py
@@
 import chromadb
 import ollama
+import requests
 from pyservicemaker import BatchMetadataOperator, Pipeline, Probe
@@
 VLM_NUM_CTX     = int(os.environ.get("VLM_NUM_CTX", "4096"))
 EMBED_MODEL     = "nomic-embed-text"
+
+# vLLM cutover (notes/vllm_cutover_design.md). Default "ollama" = today's
+# exact behavior. Embeddings (EMBED_MODEL above) always stay on Ollama/Spark
+# regardless of VLM_BACKEND -- vLLM only ever serves chat/description calls.
+VLM_BACKEND        = os.environ.get("VLM_BACKEND", "ollama")  # "ollama" | "vllm"
+VLLM_URL           = os.environ.get("VLLM_URL", "http://gx10-2ea8:8000")
+VLLM_MODEL         = os.environ.get("VLLM_MODEL", "google/gemma-4-12B-it")
+VLLM_MAX_TOKENS    = int(os.environ.get("VLLM_MAX_TOKENS", "300"))
+VLLM_TIMEOUT_S     = float(os.environ.get("VLLM_TIMEOUT_S", "30"))
+VLLM_HEALTHCHECK_S = float(os.environ.get("VLLM_HEALTHCHECK_S", "30"))
+_vllm_health = {"ok": True, "checked_at": 0.0}
+_vllm_health_lock = threading.Lock()
@@
-        "vlm_reject", "errors", "saves",
+        "vlm_reject", "vllm_fallback", "errors", "saves",
     )
@@
+def _vllm_is_healthy():
+    """Cached vLLM reachability check -- one real HTTP round trip per
+    VLLM_HEALTHCHECK_S window, shared by all VLM_WORKERS threads."""
+    now = time.time()
+    with _vllm_health_lock:
+        if now - _vllm_health["checked_at"] < VLLM_HEALTHCHECK_S:
+            return _vllm_health["ok"]
+    try:
+        r = requests.get(f"{VLLM_URL}/v1/models", timeout=3)
+        ok = r.ok and VLLM_MODEL in r.text
+    except requests.RequestException:
+        ok = False
+    with _vllm_health_lock:
+        was_ok = _vllm_health["ok"]
+        _vllm_health["ok"], _vllm_health["checked_at"] = ok, now
+    if ok != was_ok:
+        log.info(f"[VLM] vLLM health changed: {'UP' if ok else 'DOWN'} ({VLLM_URL})")
+    return ok
+
+
+def _vlm_chat_ollama(prompt, jpeg_b64):
+    resp = ollama.chat(
+        model=VLM_MODEL,
+        messages=[{"role": "user", "content": prompt, "images": [jpeg_b64]}],
+        options={"num_ctx": VLM_NUM_CTX},
+    )
+    return resp["message"]["content"].strip()
+
+
+def _vlm_chat_vllm(prompt, jpeg_b64):
+    payload = {
+        "model": VLLM_MODEL,
+        "messages": [{
+            "role": "user",
+            "content": [
+                {"type": "text", "text": prompt},
+                {"type": "image_url",
+                 "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"}},
+            ],
+        }],
+        "max_tokens": VLLM_MAX_TOKENS,
+    }
+    r = requests.post(f"{VLLM_URL}/v1/chat/completions", json=payload,
+                       timeout=VLLM_TIMEOUT_S)
+    r.raise_for_status()
+    return r.json()["choices"][0]["message"]["content"].strip()
+
+
+def _vlm_describe(prompt, jpeg_b64, stats):
+    """Dispatch to the configured backend, falling back to Ollama if vLLM
+    is unreachable or errors. VLM_BACKEND=ollama (default) never touches
+    vLLM at all."""
+    if VLM_BACKEND == "vllm" and _vllm_is_healthy():
+        try:
+            return _vlm_chat_vllm(prompt, jpeg_b64)
+        except Exception as e:
+            log.info(f"[VLM] vLLM call failed ({e}); falling back to Ollama")
+            stats.bump("vllm_fallback")
+    return _vlm_chat_ollama(prompt, jpeg_b64)
+
+
 def vlm_worker(event_queue, stats_registry):
@@ vlm_worker()
-            resp = ollama.chat(
-                model=VLM_MODEL,
-                messages=[{
-                    "role": "user",
-                    "content": prompt,
-                    "images": [jpeg_b64],
-                }],
-                options={"num_ctx": VLM_NUM_CTX},
-            )
-            description = resp["message"]["content"].strip()
+            description = _vlm_describe(prompt, jpeg_b64, stats)
```

This is intentionally a sketch (line numbers/context trimmed with @@ for
readability) — it has not been generated as a real applicable patch file
and has not been run against the working tree.

## 4. Rollout steps

1. Land the refactor with VLM_BACKEND defaulting to ollama. This is a
   no-op release — verify nothing changed (same latency/reject numbers) before
   touching the env var. Requires confirming requests imports cleanly in the
   DeepStream Python env on Thor first (see Risks — unverified).
2. Trial VLM_BACKEND=vllm in .env during a deliberately low-traffic
   window, one restart, watch for a few hours: monitor.py queue depth,
   vlm_ms_avg, vllm_fallback_total (once wired into monitor's display),
   and vlm_reject_total vs. the 88.7% live baseline
   (notes/detect_min_conf_rejects.md).
3. Compare, don't assume: expect latency/stale_skip to collapse (that's
   the whole point); do not expect reject rate to improve materially —
   notes/motorcycle_rejects_eyeball.md showed both backends hallucinate the
   same way on ambiguous motorcycle crops, so a flat-to-similar reject rate is
   the correct outcome, not a regression.
4. If stable, make vllm the new default (flip the default in code, or
   just leave VLM_BACKEND=vllm permanently in .env/start_thor.sh).
   Keep the Ollama fallback path in the code indefinitely — gx10's container
   has exited unexpectedly before (exit 255), and there's no cost to keeping
   the safety net live.

## 5. Rollback

Primary rollback is env-only, no code change: set VLM_BACKEND=ollama (or
unset it) and restart the native Thor process — ollama.chat is still fully
wired and warm, so this is instant and low-risk. Full git revert of the
patch is only needed if the abstraction itself turns out buggy (e.g. the
health-check deadlocks or the fallback path double-counts events), which the
automatic per-call fallback should mostly mask even before a revert lands.

## 6. Risks

- Motorcycle hallucination is not fixed by this cutover. Per
  notes/motorcycle_rejects_eyeball.md, both Ollama gemma4:12b and vLLM 12B
  independently hallucinated a confident-but-wrong bicycle/rider description
  on 2 of 13 sampled images. Don't present this cutover as a quality fix —
  it's a latency/backlog fix. The detector-side and prompt-side issues in that
  note are separate follow-up work.
- requests import is unverified in the live Python environment. Thor
  runs pipeline_multi.py natively (per SPARK_DEV.md's native runbook) with
  --system-site-packages; requests is ubiquitous but hasn't been confirmed
  importable in that specific venv. Check before landing, not after.
- New cross-host dependency. Adds Thor -> gx10-2ea8:8000 alongside the
  existing Thor -> spark-2251:11434 (Ollama) and Thor -> Spark ChromaDB paths.
  If gx10's network path is flakier than Spark's, that's a new failure mode —
  bounded by VLLM_TIMEOUT_S (30s) and the health-check cache, but still a
  new thing that can go wrong that didn't exist before.
- Thread-safety of the health cache is "good enough," not perfect. With
  VLM_WORKERS=2 racing on _vllm_health, a stale "healthy" reading can
  persist up to VLLM_HEALTHCHECK_S (30s) after a real outage starts — worst
  case, a couple of wasted VLLM_TIMEOUT_S-bound calls before the cache
  flips. Not correctness-critical (fallback still catches the exception per
  call regardless of the cache), just a minor latency cost during the first
  few seconds of an outage.
- monitor.py won't display vllm_fallback_total without also editing
  its fixed-width table (monitor.py:257-265) — safe to skip for the first
  trial (the counter still gets written to the stats file and is greppable),
  but worth doing before calling this "production-ready" observability.
- Env default drift risk: if VLM_BACKEND is set in .env by a future
  session without the fallback code actually being deployed yet, the env var
  simply isn't read by old code — harmless, but worth a comment in
  .env/env.example when this lands, pointing at this note.

## Caveats

- This is a design document only. No file outside notes/ has been touched;
  pipeline_multi.py is unmodified.
- The patch sketch above has not been generated as a real diff against the
  current file and may need minor adjustment (surrounding context) when
  actually applied — treat it as a specification, not a ready-to-apply patch.