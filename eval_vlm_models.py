"""eval_vlm_models.py — offline side-by-side comparison of VLM_MODEL candidates.

Runs each candidate model over a stratified sample of real detection crops
already captured in ./snapshots/, using the exact same prompts and
false-positive rejection check as the production worker (pipeline_multi.py's
VLM_PROMPTS / _vlm_says_absent — duplicated here so this script has no
dependency on pyservicemaker and can run outside the DeepStream container).

    python3 eval_vlm_models.py [--n-per-label N] [--models m1,m2,...]
                                [--max-images N] [--output path.json]

Two backends, picked per-model by a prefix on the --models entry:
  - Ollama (default, no prefix): needs the `ollama` package and a reachable
    Ollama server (OLLAMA_HOST env var, same as pipeline_multi.py).
  - vLLM / any OpenAI-compatible server: prefix the model id with "vllm:",
    e.g. "vllm:google/gemma-4-26B-A4B-it". Uses --vllm-url (default from
    VLLM_URL env var) and plain `requests` — no `openai` package needed.
    Get the exact model id vLLM expects from `curl <url>/v1/models`.

Example comparing the current on-pipeline Ollama model against a candidate
vLLM server on Spark:

    .venv/bin/python eval_vlm_models.py \\
        --models gemma4:12b,vllm:google/gemma-4-26B-A4B-it \\
        --vllm-url http://gx10-2ea8:8000

Not installed on the host by default (only inside the DeepStream container,
via Dockerfile) — on the host, use:

    python3 -m venv .venv && .venv/bin/pip install ollama requests
    .venv/bin/python eval_vlm_models.py ...

Quality is judged by eye: descriptions are printed grouped by image so you
can compare models side by side. Latency/reject-rate are the objective
signals (see the printed summary and the JSON report).
"""

import argparse
import base64
import json
import os
import random
import re
import time

import ollama
import requests

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
FILENAME_RE = re.compile(r"^cam(\d+)_src\d+_([a-zA-Z]+)_evt\d+\.jpg$")

# Kept in sync with pipeline_multi.py's DETECT_CLASSES / VLM_PROMPTS.
LABEL_TO_CLASS_ID = {"car": 0, "person": 2, "motorcycle": 3, "bicycle": 3}

VLM_PROMPTS = {
    0: (
        "Describe this vehicle in 2-3 sentences. Include: color, body style"
        " (sedan/SUV/truck/van/coupe), make and model if recognizable, approximate"
        " year range, any visible damage or distinctive markings, direction of travel,"
        " and license plate text if legible."
    ),
    2: (
        "Describe this person in 2-3 sentences. Include: approximate age range and"
        " gender, hair color and length, clothing (shirt/jacket color and style,"
        " pants/skirt color, footwear), any accessories (backpack, hat, bag, phone),"
        " what they are doing, and which direction they are moving."
    ),
    3: (
        "Describe this motorcycle or bicycle in 2-3 sentences. Include: type"
        " (sport/cruiser/dirt bike/bicycle/scooter), color, make if recognizable,"
        " rider's helmet color and clothing, any passenger, and direction of travel."
    ),
}

_VLM_REFUSAL = (
    "i'm sorry", "i am sorry", "i cannot", "i can't", "cannot provide",
    "unable to", "cannot find any", "don't see any", "do not see any",
    "doesn't appear to be", "does not appear to be", "no discernible",
)
_VLM_ABSENT_SUBJECT = {
    0: ("no vehicle", "no vehicles", "no car", "no cars", "not a vehicle"),
    2: ("no person", "no people", "no individual", "no humans", "no pedestrian"),
    3: ("no motorcycle", "no motorcycles", "no bicycle", "no bicycles",
        "no bike", "no bikes", "no scooter"),
}


def _vlm_says_absent(description, class_id):
    d = description.lower()
    if any(m in d for m in _VLM_REFUSAL):
        return True
    return any(s in d for s in _VLM_ABSENT_SUBJECT.get(class_id, ()))


DEFAULT_MODELS = ["gemma4:12b", "qwen3.8", "nemotron-3.5-lightning", "muse-glimmer:30b"]
# laguna-s-2.1 (96GB) deliberately excluded — too large to be a realistic
# on-pipeline VLM alongside DeepStream on this hardware.

# Rough real-time budget: street cams send every qualifying detection to the
# VLM with no throttle (streams_config.py), and pipeline_multi.py's worker
# treats an event as unrecoverable once it's waited MAX_EVENT_AGE_S (30s
# default) in the queue. p95 latency above half that is a heuristic flag,
# not a hard cutoff — a real backlog test on live traffic is the real answer.
REALTIME_WARN_S = 15.0


def sample_images(n_per_label, max_images, seed):
    by_cam_label = {}
    for name in os.listdir(SNAPSHOT_DIR):
        m = FILENAME_RE.match(name)
        if not m:
            continue
        cam_id, label = int(m.group(1)), m.group(2)
        if label not in LABEL_TO_CLASS_ID:
            continue
        by_cam_label.setdefault((cam_id, label), []).append(name)

    rng = random.Random(seed)
    sample = []
    for (cam_id, label), names in sorted(by_cam_label.items()):
        rng.shuffle(names)
        for name in names[:n_per_label]:
            sample.append({
                "path": os.path.join(SNAPSHOT_DIR, name),
                "name": name,
                "camera_id": cam_id,
                "label": label,
                "class_id": LABEL_TO_CLASS_ID[label],
            })
    rng.shuffle(sample)
    if max_images:
        sample = sample[:max_images]
    return sample


VLLM_PREFIX = "vllm:"

def _final_answer(text):
    """Drop Gemma 4 thinking preamble when the reasoning parser is off."""
    import re
    t = (text or "").strip()
    if not t:
        return t
    if "<channel|>" in t:
        t = t.split("<channel|>")[-1].strip()
    low = t.lower()
    if low.startswith("thought") or "<|channel>thought" in low[:80]:
        quoted = list(re.finditer(r"\"([^\"]{20,})\"", t))
        if quoted:
            after = t[quoted[-1].end():].strip().strip("\"")
            return after if len(after) >= 20 else quoted[-1].group(1)
        paras = [x.strip() for x in re.split(r"\n\s*\n", t) if x.strip()]
        for para in reversed(paras):
            if not para.lstrip().startswith("*") and len(para) >= 20:
                return para
    return t



def _msg_field(msg, name, default=""):
    """ollama Message supports both dict- and attribute-style access
    depending on client version; try both rather than assuming one."""
    try:
        val = msg[name]
    except (KeyError, TypeError):
        val = getattr(msg, name, default)
    return val or default


def _run_ollama(model, prompt, jpeg_b64, think=None):
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt, "images": [jpeg_b64]}],
        think=think,
    )
    msg = resp["message"]
    content = _msg_field(msg, "content").strip()
    thinking = _msg_field(msg, "thinking")
    return content, thinking


def _run_vllm(model, prompt, jpeg_b64, vllm_url, timeout=180, max_tokens=300, enable_thinking=False):
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"}},
            ],
        }],
        "max_tokens": max_tokens,
    }
    if enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    r = requests.post(f"{vllm_url}/v1/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    raw = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
    if not reasoning and raw.lower().startswith("thought"):
        reasoning = raw
    content = _final_answer(raw)
    return content, reasoning


def run_model_on_image(model, image, vllm_url, enable_thinking=False, max_tokens=300,
                        ollama_think=None):
    with open(image["path"], "rb") as f:
        jpeg_b64 = base64.b64encode(f.read()).decode()
    prompt = VLM_PROMPTS.get(image["class_id"], "Describe what you see in one sentence.")
    t0 = time.time()
    try:
        if model.startswith(VLLM_PREFIX):
            description, reasoning = _run_vllm(
                model[len(VLLM_PREFIX):], prompt, jpeg_b64, vllm_url,
                timeout=180, max_tokens=max_tokens, enable_thinking=enable_thinking,
            )
        else:
            description, reasoning = _run_ollama(model, prompt, jpeg_b64, think=ollama_think)
        latency_s = time.time() - t0
        return {
            "model": model,
            "image": image["name"],
            "camera_id": image["camera_id"],
            "label": image["label"],
            "latency_s": round(latency_s, 2),
            "description": description,
            "reasoning_chars": len(reasoning) if reasoning else 0,
            "rejected": _vlm_says_absent(description, image["class_id"]),
            "error": None,
        }
    except Exception as e:
        return {
            "model": model,
            "image": image["name"],
            "camera_id": image["camera_id"],
            "label": image["label"],
            "latency_s": None,
            "description": None,
            "rejected": None,
            "error": str(e),
        }


def summarize(rows):
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    summary = {}
    for model, model_rows in by_model.items():
        ok = [r for r in model_rows if r["error"] is None]
        errors = len(model_rows) - len(ok)
        latencies = sorted(r["latency_s"] for r in ok)
        rejects = sum(1 for r in ok if r["rejected"])
        reasoning_chars = [r.get("reasoning_chars", 0) for r in ok]
        n = len(latencies)
        p95 = latencies[int(0.95 * (n - 1))] if n else None
        mean = round(sum(latencies) / n, 2) if n else None
        summary[model] = {
            "n_ok": n,
            "n_errors": errors,
            "mean_latency_s": mean,
            "p95_latency_s": p95,
            "reject_rate": round(rejects / n, 2) if n else None,
            "mean_reasoning_chars": round(sum(reasoning_chars) / n) if n else None,
            "realtime_flag": bool(p95 and p95 > REALTIME_WARN_S),
        }
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-per-label", type=int, default=3,
                     help="Images to sample per (camera, label) pair.")
    ap.add_argument("--max-images", type=int, default=None,
                     help="Cap total images after sampling (default: no cap).")
    ap.add_argument("--models", type=str, default=None,
                     help=f"Comma-separated model list (default: {','.join(DEFAULT_MODELS)}).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=str, default="eval_vlm_results.json")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="vLLM only: set chat_template_kwargs.enable_thinking.")
    ap.add_argument("--max-tokens", type=int, default=300,
                    help="vLLM max_tokens (raise this if thinking eats the budget).")
    ap.add_argument("--ollama-think", type=str, default="default",
                    choices=["default", "true", "false", "low", "medium", "high"],
                    help="Ollama-only: think= passed to ollama.chat() for "
                         "thinking-capable models. 'default' omits the param "
                         "(whatever Ollama does when unset).")
    ap.add_argument("--vllm-url", type=str,
                     default=os.environ.get("VLLM_URL", "http://gx10-2ea8:8000"),
                     help="Base URL for vllm: -prefixed models (OpenAI-compatible server).")
    ap.add_argument("--images", type=str, default=None,
                     help="Comma-separated snapshot filenames (basenames under "
                          "./snapshots/) to run instead of sampling — for "
                          "re-testing a specific known set (e.g. prior "
                          "disagreements) under new settings.")
    args = ap.parse_args()

    models = args.models.split(",") if args.models else list(DEFAULT_MODELS)
    ollama_think = {"default": None, "true": True, "false": False}.get(
        args.ollama_think, args.ollama_think  # low/medium/high pass through as-is
    )

    if args.images:
        names = [n.strip() for n in args.images.split(",") if n.strip()]
        images = []
        for name in names:
            m = FILENAME_RE.match(name)
            if not m:
                print(f"[eval] Skipping {name!r}: doesn't match snapshot filename pattern")
                continue
            cam_id, label = int(m.group(1)), m.group(2)
            if label not in LABEL_TO_CLASS_ID:
                print(f"[eval] Skipping {name!r}: unknown label {label!r}")
                continue
            images.append({
                "path": os.path.join(SNAPSHOT_DIR, name),
                "name": name,
                "camera_id": cam_id,
                "label": label,
                "class_id": LABEL_TO_CLASS_ID[label],
            })
    else:
        images = sample_images(args.n_per_label, args.max_images, args.seed)
    if not images:
        print(f"[eval] No labeled snapshots found under {SNAPSHOT_DIR}")
        return
    print(f"[eval] {len(images)} images x {len(models)} models = "
          f"{len(images) * len(models)} calls")

    # Model outer, image inner: Ollama keeps the active model resident and
    # only pays a (multi-GB) reload cost when the model changes, so grouping
    # calls by model keeps total runtime sane and keeps per-call latency
    # measurements from being dominated by swap overhead.
    rows = []
    by_image = {image["name"]: image for image in images}
    for model in models:
        print(f"\n--- {model} ---")
        for image in images:
            row = run_model_on_image(model, image, args.vllm_url, args.enable_thinking,
                                      args.max_tokens, ollama_think=ollama_think)
            rows.append(row)
            tag = "ERROR" if row["error"] else ("REJECT" if row["rejected"] else "ok")
            desc = row["error"] or row["description"]
            print(f"[{model:24s}] {image['name']:35s} {tag:6s} "
                  f"{row['latency_s'] or 0:5.1f}s  {desc[:90]}")

    print("\n=== Side-by-side (grouped by image) ===")
    for name, image in by_image.items():
        print(f"\n-- {name} (cam{image['camera_id']}, {image['label']}) --")
        for r in [row for row in rows if row["image"] == name]:
            desc = r["error"] or r["description"]
            print(f"  {r['model']:24s} {desc}")

    summary = summarize(rows)
    print("\n=== Summary ===")
    for model, s in summary.items():
        flag = "  <-- may not keep up in real time" if s["realtime_flag"] else ""
        print(f"{model:24s} ok={s['n_ok']:3d} err={s['n_errors']:2d} "
              f"mean={s['mean_latency_s']}s p95={s['p95_latency_s']}s "
              f"reject_rate={s['reject_rate']} "
              f"mean_reasoning_chars={s['mean_reasoning_chars']}{flag}")

    with open(args.output, "w") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=2)
    print(f"\n[eval] Wrote {args.output}")


if __name__ == "__main__":
    main()
