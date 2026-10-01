"""Standalone check of pipeline_multi's VLM backend helpers against one saved
snapshot, without touching the running pipeline.

    LOG_DIR=/tmp/vlm_selftest .venv/bin/python3 tools/vlm_backend_selftest.py [snapshot.jpg [person|car|bicycle]]

The class defaults to the label in the file name; pass it explicitly for
snapshots saved before the 2026-09-30 class-map fix, whose names are wrong
(e.g. a car saved as cam3_src2_person_evt3469.jpg). PASS needs the expected
backends AND the vLLM reply to describe the object (not NONE/absent).

LOG_DIR keeps the import-time logger away from logs/pipeline.log. Loads .env
the same way start_thor.sh does (raw KEY=value lines). Exercises: vLLM path,
Ollama path (think per OLLAMA_THINK), and forced fallback (bogus VLLM_URL).
"""
import base64
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("LOG_DIR", "/tmp/vlm_selftest")
if os.path.exists(".env"):
    for line in open(".env"):
        line = line.rstrip("\n")
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v)
os.environ["VLM_BACKEND"] = "vllm"

import pipeline_multi as pm  # noqa: E402

img = sys.argv[1] if len(sys.argv) > 1 else sorted(
    glob.glob("snapshots/cam*_person_evt*.jpg"), key=os.path.getmtime)[-1]
if len(sys.argv) > 2:
    label = pm._norm_class_name(sys.argv[2])
else:
    label = next((c for c in pm.APP_CLASSES if f"_{c}_" in os.path.basename(img)), "person")
if label not in pm.VLM_PROMPTS:
    sys.exit(f"unknown class {label!r}; use one of {', '.join(pm.APP_CLASSES)}")
cls = label            # classes are keyed by name
prompt = pm.VLM_PROMPTS[cls]
b64 = base64.b64encode(open(img, "rb").read()).decode()
print(f"image={img} class={label} VLLM_URL={pm.VLLM_URL} OLLAMA_THINK={pm.OLLAMA_THINK}")

ok = True
absent_seen = {}
def run(name, fn):
    global ok
    t = time.time()
    try:
        out = fn()
        desc, backend = out if isinstance(out, tuple) else (out, name)
        dt = time.time() - t
        absent_seen[name] = pm._vlm_says_absent(desc, cls)
        print(f"\n[{name}] backend={backend} {dt:.1f}s absent={absent_seen[name]}\n  {desc}")
        return backend
    except Exception as e:
        ok = False
        print(f"\n[{name}] FAILED after {time.time()-t:.1f}s: {type(e).__name__}: {e}")

print("health:", pm._vllm_is_healthy())
b = run("vllm-dispatch", lambda: pm._vlm_describe(prompt, b64))
ok = ok and b == "vllm" and absent_seen.get("vllm-dispatch") is False
run("ollama-direct", lambda: pm._vlm_chat_ollama(prompt, b64))
pm.VLLM_URL = "http://127.0.0.1:9"      # nothing listens there
pm._vllm_health.update(ok=True, checked_at=0.0)
b = run("forced-fallback", lambda: pm._vlm_describe(prompt, b64))
ok = ok and b == "ollama"
print("\nSELFTEST", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
