"""
pipeline_multi.py - Single DeepStream 9 pipeline for multiple RTSP streams.

One nvstreammux batches all cameras; one nvinfer pass handles every stream.
Detections queue to a shared VLM worker (Ollama on Spark via OLLAMA_HOST).

Replaces the per-camera container model (pipeline2.py + cam1.yml).

Classes: read at startup from the nvinfer labels file (pgie_config_rtdetr.txt
labelfile-path), one line per model output slot. RT-DETR TrafficCamNet
Transformer Lite has 5 slots: 0=BG 1=bicycle 2=car 3=person 4=road_sign
(0 and 4 filtered out). Everything in the app is keyed by class NAME; the
pipeline refuses to start if the labels file, num-detected-classes and the
model's pred_logits width disagree.
"""

import datetime
import glob
import json
import logging
import logging.handlers
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import chromadb
import ollama
import requests
from pyservicemaker import BatchMetadataOperator, Pipeline, Probe

from streams_config import class_min_conf, load_streams

try:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo("America/Los_Angeles")
except Exception:
    LOCAL_TZ = datetime.timezone(datetime.timedelta(hours=-7))

# ── Config ────────────────────────────────────────────────────────────────────
CHROMADB_HOST   = os.environ.get("CHROMADB_HOST", "localhost")
CHROMADB_PORT   = int(os.environ.get("CHROMADB_PORT", "8000"))
# Chroma collection the VLM workers write events to (the search index).
# Thor .env sets CHROMA_COLLECTION=vision_events_v2 (2026-10-03 switch);
# query_server.py defaults to the same setting.
COLLECTION      = os.environ.get("CHROMA_COLLECTION", "vision_events")
VLM_MODEL       = os.environ.get("VLM_MODEL", "gemma4:26b")
# Ollama's default context (262144, Gemma 4's max) makes its automatic
# parallel-slot sizing pick num_parallel=1 regardless of OLLAMA_NUM_PARALLEL,
# since each additional slot's KV cache scales with context length. Our
# prompts are one short instruction + one image — nowhere near that — so
# request a small context explicitly to leave slot-count headroom on the
# server (confirmed via Spark's ollama process: `-c 262144 -np 1`).
VLM_NUM_CTX     = int(os.environ.get("VLM_NUM_CTX", "4096"))
EMBED_MODEL     = "nomic-embed-text"
# Embedding host. Defaults to OLLAMA_HOST (and then the ollama client's own
# default), so nothing changes when it is unset. Set EMBED_HOST to serve
# embeddings from a different Ollama than the VLM fallback (live: gx10, while
# the gemma4 fallback stays on Spark). The target MUST serve the identical
# nomic-embed-text weights (same digest), or new vectors won't be comparable
# with the ones already stored in ChromaDB.
EMBED_HOST      = (os.environ.get("EMBED_HOST") or os.environ.get("OLLAMA_HOST") or "").strip() or None

# nomic-embed-text was trained with task prefixes ("search_document: " /
# "search_query: "). Off by default so the live vision_events_v2 index
# (built without prefixes) stays comparable. Set EMBED_PREFIX_STYLE=nomic
# when pointing CHROMA_COLLECTION at a prefixed collection (e.g. v3), or
# set EMBED_DOC_PREFIX explicitly. See tools/reembed_nomic_prefixes.py.
_EMBED_PREFIX_STYLE = os.environ.get("EMBED_PREFIX_STYLE", "none").strip().lower()
if _EMBED_PREFIX_STYLE in ("nomic", "nomic-embed-text"):
    EMBED_DOC_PREFIX = os.environ.get("EMBED_DOC_PREFIX", "search_document: ")
else:
    EMBED_DOC_PREFIX = os.environ.get("EMBED_DOC_PREFIX", "")

# ── VLM backend switch (notes/vllm_cutover_design.md) ─────────────────────────
# VLM_BACKEND=ollama (default) keeps every description call on OLLAMA_HOST.
# VLM_BACKEND=vllm sends description calls to the OpenAI-compatible vLLM
# server at VLLM_URL (gx10), with a cached health check and automatic per-call
# fallback to Ollama on any error/timeout. Embeddings (EMBED_MODEL) ALWAYS stay
# on Ollama (EMBED_HOST, default OLLAMA_HOST) regardless of backend, so the
# ChromaDB vector space is unchanged. Prompts and
# _vlm_says_absent() are backend-agnostic and unchanged.
VLM_BACKEND        = os.environ.get("VLM_BACKEND", "ollama").strip().lower()
VLLM_URL           = os.environ.get("VLLM_URL", "http://gx10-2ea8:8000").rstrip("/")
VLLM_MODEL         = os.environ.get("VLLM_MODEL", "google/gemma-4-12B-it")
VLLM_MAX_TOKENS    = int(os.environ.get("VLLM_MAX_TOKENS", "300"))
VLLM_TIMEOUT_S     = float(os.environ.get("VLLM_TIMEOUT_S", "20"))  # eval p95 8.6s
VLLM_HEALTHCHECK_S = float(os.environ.get("VLLM_HEALTHCHECK_S", "30"))
# Ollama think= for thinking-capable models (gemma4). Omitting it (the old
# behavior) leaves Gemma 4 thinking ON by default: eval_vllm_12b_n3.json shows
# reasoning on every Ollama row, and eval_2x2_big_ollama_{think,nothink}.json
# shows 39.0s vs 24.4s mean. "false" (default) = fast mode; "true" = think;
# "default" = omit the parameter (pre-2026-09-25 behavior).
_OLLAMA_THINK_RAW  = os.environ.get("OLLAMA_THINK", "false").strip().lower()
OLLAMA_THINK       = {"false": False, "0": False, "no": False,
                      "true": True, "1": True, "yes": True,
                      "default": None, "": None}.get(_OLLAMA_THINK_RAW, False)
_vllm_health = {"ok": True, "checked_at": 0.0}
_vllm_health_lock = threading.Lock()
# The module-level ollama client has timeout=None, so a hung Spark call would
# block a worker forever. Explicit clients: chat host from OLLAMA_HOST (as
# before), embed host from EMBED_HOST (defaults to OLLAMA_HOST).
OLLAMA_CHAT_TIMEOUT_S  = float(os.environ.get("OLLAMA_CHAT_TIMEOUT_S", "180"))
OLLAMA_EMBED_TIMEOUT_S = float(os.environ.get("OLLAMA_EMBED_TIMEOUT_S", "30"))
_ollama_chat_client  = ollama.Client(timeout=OLLAMA_CHAT_TIMEOUT_S)
_ollama_embed_client = ollama.Client(host=EMBED_HOST, timeout=OLLAMA_EMBED_TIMEOUT_S)

# Frame selection fix (get_jpeg_after): the legacy picker lets any file newer
# than detection+WINDOW_AFTER override an in-window pick, so a worker that
# reaches an event >10s late always gets "now". FRAME_PICK_CLOSEST=1 picks the
# in-window frame closest to the detection instant and only falls back to the
# newest file when nothing is in-window. Off by default (not yet enabled live).
FRAME_PICK_CLOSEST = os.environ.get("FRAME_PICK_CLOSEST", "0") == "1"

# Optional: keep a capped copy of VLM-rejected frames for later hand labeling
# (rejected events are otherwise never written to disk). Off by default.
SAVE_REJECTS       = os.environ.get("SAVE_REJECTS", "0") == "1"
REJECT_DIR         = os.environ.get("REJECT_DIR", "snapshots_rejected")
REJECT_SAVE_MAX    = int(os.environ.get("REJECT_SAVE_MAX", "2000"))
SAVE_INTERVAL   = float(os.environ.get("SAVE_INTERVAL", "30.0"))
# Street cams still throttle per (camera, class) — shorter than office so real
# passing traffic is captured, but a persistent detection (parked car or a
# false positive on shadows/foliage) can't flood the VLM queue every frame.
STREET_SAVE_INTERVAL = float(os.environ.get("STREET_SAVE_INTERVAL", "8.0"))
VLM_QUEUE_MAX   = int(os.environ.get("VLM_QUEUE_MAX", "12"))
# Ollama on Spark now runs with OLLAMA_NUM_PARALLEL=2 (see VLM_NUM_CTX above
# for why that only became viable once num_ctx was capped) — matching worker
# count here so the pipeline actually issues concurrent requests instead of
# leaving the second server slot idle.
VLM_WORKERS     = int(os.environ.get("VLM_WORKERS", "2"))
# Hard ceiling for street cams, which otherwise queue unconditionally. Keeps a
# stalled worker from growing the queue without bound (memory leak safeguard).
STREET_QUEUE_MAX = int(os.environ.get("STREET_QUEUE_MAX", str(VLM_QUEUE_MAX * 8)))
# Max time an event may wait in the VLM queue before its frame is considered
# gone / its scene stale. Past this, the worker skips it INSTANTLY (no VLM call)
# instead of describing a stale frame the VLM would just reject. This lets a
# backlogged worker drain junk in ms and catch up to events whose frames are
# still fresh — the fix for street-cam saves=0 under VLM backlog.
# Default sized for the Thor/Spark split: capture+detect runs on Thor and never
# blocks on this, so a slow remote VLM (measured p95 60-150s per model in
# eval_vlm_results.json) just means a longer wait, not a dropped event — as
# long as street-cam volume stays low (a few detections/hour). Lower this back
# toward the old 30s default only if running the VLM locally and low latency.
MAX_EVENT_AGE_S = float(os.environ.get("MAX_EVENT_AGE_S", "300.0"))
# The per-camera JPEG ring buffer (SNAPSHOT_RING_FILES below) only covers a
# few seconds-to-tens-of-seconds of real time, far less than MAX_EVENT_AGE_S.
# Once a backlogged worker is this far past the original detection, the
# in-window frame is already rotated out and get_jpeg_after's fallback grabs
# whatever the camera sees "now" — a different moment, not a late copy of the
# same one. Sending that to the VLM almost always burns a full round trip on
# a guaranteed reject (measured: 522/522 VLM calls used a fallback frame,
# 96% rejected). Skip the VLM call outright past this gap instead.
VLM_STALE_FRAME_MAX_GAP_S = float(os.environ.get("VLM_STALE_FRAME_MAX_GAP_S", "20.0"))
# Per-camera snapshot ring buffer depth (multifilesink max-files). At 200
# this covered as little as ~8s of history for a busy camera (24fps) — far
# short of realistic VLM worker backlog. JPEGs are cheap (~50-150KB each),
# so this is a storage/nothing tradeoff, not a compute one.
SNAPSHOT_RING_FILES = int(os.environ.get("SNAPSHOT_RING_FILES", "900"))
FRAME_W         = int(os.environ.get("FRAME_W", "1280"))
FRAME_H         = int(os.environ.get("FRAME_H", "720"))
ENABLE_DISPLAY  = (
    os.environ.get("ENABLE_DISPLAY", "0") == "1"
    or os.environ.get("HEADLESS", "1") == "0"
)
LIVE_STREAM     = os.environ.get("LIVE_STREAM", "0") == "1"
JPEG_GLOB       = "/tmp/frame_*.jpg"
TILER_W         = int(os.environ.get("TILER_W", "1280"))
TILER_H         = int(os.environ.get("TILER_H", "720"))
SNAPSHOT_DIR    = "snapshots"
STATS_DIR       = "stats"
LOG_DIR         = os.environ.get("LOG_DIR", "logs")
STATS_HEARTBEAT_S = float(os.environ.get("STATS_HEARTBEAT_S", "10.0"))
RECONNECT_INTERVAL = 5
RESTART_DELAY      = 10


def _setup_logging():
    """Log to stdout AND a rotating file, so REJECT/skip/error lines survive
    the interactive terminal scrolling away (docker logs is empty — the
    pipeline runs under `docker exec`)."""
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            os.path.join(LOG_DIR, "pipeline.log"),
            maxBytes=5 * 1024 * 1024, backupCount=3,
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError as e:
        logger.warning("File logging disabled: %s", e)
    return logger


log = _setup_logging()

# ── Detector class map (by NAME, read from the labels file) ──────────────────
# Line i of the labels file names model output slot i. Before 2026-09-30 the
# app hard-coded a 4-class order (0=Car 1=RoadSign 2=Person 3=Bicycle) that
# did not match the model's 5 outputs (0=BG 1=bicycle 2=car 3=person
# 4=road_sign): cars were saved as "person" and people as "motorcycle".
# Verified offline on Thor snapshots with tools/rtdetr_offline.py.
PGIE_CONFIG     = os.environ.get("PGIE_CONFIG", "pgie_config_rtdetr.txt")
# The classes the app queues to the VLM. "bicycle" is the model's own name
# (it was displayed as "motorcycle" before the class-map fix; the prompt and
# absent-matcher still cover motorcycles/scooters).
APP_CLASSES     = ("person", "car", "bicycle")
CLASS_NAME_ALIASES = {"motorcycle": "bicycle", "roadsign": "road_sign",
                      "background": "bg"}
CLASS_MAP_VERSION = "rtdetr5-20260930"   # stored in Chroma metadata


class ClassMapError(RuntimeError):
    pass


def _norm_class_name(name):
    n = re.sub(r"[\s\-]+", "_", str(name).strip().lower())
    return CLASS_NAME_ALIASES.get(n, n)


def _read_pgie_properties(path):
    """[property] key=value pairs of an nvinfer .txt config (comments dropped)."""
    props, section = {}, None
    with open(path) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].strip()
            elif section == "property" and "=" in line:
                k, v = line.split("=", 1)
                props[k.strip()] = v.strip()
    return props


def load_class_map(pgie_config=PGIE_CONFIG):
    """Read the labels file named by the pgie config. Returns a dict:
    names (slot -> normalized name), slots (app name -> slot), filtered
    (filter-out-class-ids), labels_path, onnx_path. Raises ClassMapError if
    the labels line count != num-detected-classes, a name is duplicated, an
    app class is missing, or an app class is filtered out at the detector."""
    base = os.path.dirname(os.path.abspath(pgie_config))
    props = _read_pgie_properties(pgie_config)
    lf = props.get("labelfile-path")
    if not lf:
        raise ClassMapError(f"{pgie_config}: no labelfile-path")
    lf = lf if os.path.isabs(lf) else os.path.join(base, lf)
    with open(lf) as fh:
        raw = [ln.strip() for ln in fh if ln.strip()]
    names = [_norm_class_name(x) for x in raw]
    try:
        n_cfg = int(props.get("num-detected-classes", ""))
    except ValueError:
        raise ClassMapError(f"{pgie_config}: num-detected-classes missing/invalid")
    if len(names) != n_cfg:
        raise ClassMapError(
            f"{lf} has {len(names)} labels but {pgie_config} num-detected-classes={n_cfg}")
    if len(set(names)) != len(names):
        raise ClassMapError(f"{lf}: duplicate class names {names}")
    filtered = set()
    for tok in re.split(r"[;,\s]+", props.get("filter-out-class-ids", "")):
        if tok:
            filtered.add(int(tok))
    slots = {}
    for cname in APP_CLASSES:
        if cname not in names:
            raise ClassMapError(f"{lf} has no '{cname}' line (labels: {names})")
        slots[cname] = names.index(cname)
        if slots[cname] in filtered:
            raise ClassMapError(f"app class {cname} (slot {slots[cname]}) is in "
                                f"filter-out-class-ids")
    onnx = props.get("onnx-file")
    if onnx and not os.path.isabs(onnx):
        onnx = os.path.join(base, onnx)
    return {"names": names, "slots": slots, "filtered": filtered,
            "labels_path": lf, "onnx_path": onnx}


def model_output_width(onnx_path):
    """pred_logits width of the ONNX (tools/model_width.py: TensorRT ONNX
    parser, CPU only, run in a subprocess so this process never imports
    tensorrt). None if it can't be determined."""
    tool = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "model_width.py")
    try:
        r = subprocess.run([sys.executable, tool, onnx_path], capture_output=True,
                           text=True, timeout=300)
        if r.returncode == 0:
            return int(r.stdout.strip().splitlines()[-1])
        log.warning("[Detect] model width probe failed: %s",
                    (r.stderr or r.stdout).strip().splitlines()[-1:])
    except Exception as e:
        log.warning("[Detect] model width probe failed: %s", e)
    return None


def describe_class_map(cmap):
    parts = []
    for i, n in enumerate(cmap["names"]):
        parts.append(f"{i}={n}" + ("(filtered)" if i in cmap["filtered"] else ""))
    app = " ".join(f"{n}={s}" for n, s in cmap["slots"].items())
    return f"slots {' '.join(parts)}; app classes {app}"


try:
    # Resolve a relative PGIE_CONFIG against this file's directory so tools
    # importing pipeline_multi from another cwd still find it (nvinfer itself
    # gets PGIE_CONFIG as-is; the pipeline runs from the repo root).
    CLASS_MAP = load_class_map(PGIE_CONFIG if os.path.isabs(PGIE_CONFIG) else
                               os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            PGIE_CONFIG))
except (ClassMapError, OSError, ValueError) as _e:
    log.error(f"[Detect] REFUSING TO START: class map invalid: {_e}")
    raise SystemExit(2)
CLASS_NAMES    = CLASS_MAP["names"]          # slot -> name (all model slots)
CLASS_SLOTS    = CLASS_MAP["slots"]          # app name -> slot
DETECT_CLASSES = {slot: name for name, slot in CLASS_SLOTS.items()}   # slot -> app name


def resolve_class_tokens(tokens, key="DROP_CLASSES"):
    """DROP_CLASSES tokens (names and/or numeric slots, see
    streams_config._parse_class_list) -> frozenset of slots. Unknown tokens
    are logged and ignored."""
    out = set()
    for t in tokens:
        if isinstance(t, int):
            if 0 <= t < len(CLASS_NAMES):
                out.add(t)
                log.info(f"[Detect] {key}: numeric slot {t} = {CLASS_NAMES[t]} "
                         f"(prefer the name)")
            else:
                log.warning(f"[Detect] {key}: ignoring unknown slot {t}")
        else:
            n = _norm_class_name(t)
            if n in CLASS_NAMES:
                out.add(CLASS_NAMES.index(n))
            else:
                log.warning(f"[Detect] {key}: ignoring unknown class {t!r} "
                            f"(known: {', '.join(CLASS_NAMES)})")
    return frozenset(out)


# RT-DETR is far more precise than the old resnet18 (which hallucinated "car" on
# foliage, forcing a 0.75 gate). The detector already gates at pre-cluster-
# threshold=0.4; these are secondary per-class gates in the probe.
DETECT_MIN_CONF = {"car": 0.50, "person": 0.40, "bicycle": 0.40}

# Per-object dedup (requires nvtracker). Emit one event per unique track id so a
# single passing car = one event instead of one per inference frame.
MIN_TRACK_HITS  = int(os.environ.get("MIN_TRACK_HITS", "2"))  # frames before emit
TRACK_TTL       = 30.0        # forget a track id this long after last seen
UNTRACKED_ID    = 2 ** 63     # tracker ids at/above this are "untracked" sentinels

# Neutral wording: the prompt does not assert the object is there, and gives
# the VLM an explicit way out (NONE), which _vlm_says_absent treats as absent.
VLM_PROMPTS = {
    "car": (
        "Describe the vehicle (car, SUV, truck or van) in this image in 2-3"
        " sentences. Include: color, body style (sedan/SUV/truck/van/coupe), make"
        " and model if recognizable, approximate year range, any visible damage or"
        " distinctive markings, direction of travel, and license plate text if"
        " legible. If no vehicle is clearly visible, reply exactly NONE."
    ),
    "person": (
        "Describe the person in this image in 2-3 sentences. Include: approximate"
        " age range and gender, hair color and length, clothing (shirt/jacket color"
        " and style, pants/skirt color, footwear), any accessories (backpack, hat,"
        " bag, phone), what they are doing, and which direction they are moving."
        " If no person is clearly visible, reply exactly NONE."
    ),
    "bicycle": (
        "Describe the bicycle (or other two-wheeler such as a motorcycle or"
        " scooter) in this image in 2-3 sentences. Include: type (road/mountain/"
        "e-bike/motorcycle/scooter), color, make if recognizable, the rider's"
        " helmet and clothing if someone is riding it, any passenger, and direction"
        " of travel. If no bicycle or other two-wheeler is clearly visible, reply"
        " exactly NONE."
    ),
}


# Phrases a VLM uses when the detected object isn't actually in the frame.
# Used to reject detector false positives (e.g. TrafficCamNet firing "car" on
# dappled shadows/foliage) — the VLM is a much stronger verifier than the PGIE.
_VLM_REFUSAL = (
    "i'm sorry", "i am sorry", "i cannot", "i can't", "cannot provide",
    "unable to", "cannot find any", "don't see any", "do not see any",
    "doesn't appear to be", "does not appear to be", "no discernible",
)


# Subject nouns per class (by name) for the structural negation patterns
# below. Riders and passengers are deliberately NOT bicycle nouns: "a parked
# bicycle with no rider" still has a bicycle in it.
_VLM_SUBJECT_NOUNS = {
    "car": ("vehicles?", "cars?", "trucks?", "vans?", "suvs?", "automobiles?"),
    "person": ("people", "persons?", "humans?", "individuals?", "pedestrians?",
               "anyone", "anybody", "someone", "somebody", r"human\s+figures?"),
    "bicycle": ("motorcycles?", "motorbikes?", "bicycles?", "bikes?", "scooters?",
                "two-wheelers?"),
}
# Optional qualifier between the negation and the noun ("no clearly visible people").
_NEG_ADJ = (r"(?:(?:clearly |readily |easily |actually )?(?:visible|identifiable|"
            r"discernible|recognizable|detectable|distinguishable|actual|real|"
            r"human|living) )?")
# A trailing exclusion means the subject IS there: "no one else", "no people
# other than the cyclist", "no pedestrians, except the man", "no one but the
# guard". A comma + "but" is NOT an exclusion: "no people, but a bench".
_NEG_EXCEPT = (r"(?![\s,]+(?:else|other than|besides|except|apart from|aside from)\b)"
               r"(?!\s+but\s+(?:him|her|them|the|a|an|one)\b)")
# Where a bare "no <subject>" really means absence: sentence start, "there
# is/are no ...", "the image shows/contains no ...", "the path is empty, with
# no pedestrians". Mid-sentence "a man walks down the street with no people
# around him" is not a rejection.
_NEG_CTX = (r"(?:^|[.;:!?]\s+|\bthere (?:is|are|was|were|appears? to be|"
            r"seems? to be) |\b(?:contains?|containing|shows?|showing|depicts?|"
            r"features?|includes?|reveals?|displays?|captures?|has|have|see|"
            r"sees) |\b(?:is|are|was|were|appears?|seems?) (?:completely |entirely |"
            r"otherwise )?empty\b[^.]*?\bwith )")
# "no <subject> (is) visible/present" anywhere: "... as no identifiable
# individual is visible", "an empty room with no people present".
_NEG_SEEN = (r" (?:(?:is|are|was|were|can be|could be) (?:clearly |readily )?)?"
             r"(?:visible|present|seen|shown|detected|discernible|identifiable|"
             r"in sight|in (?:the|this) (?:image|frame|scene|picture|photo))\b")


def _build_absent_rx(cname):
    nouns = "|".join(_VLM_SUBJECT_NOUNS[cname])
    # Not a possessive: "the person's face is not visible" is a real person.
    n = r"(?:%s)\b(?!'s\b)%s" % (nouns, _NEG_EXCEPT)
    art = r"(?:any |a |an )?"   # not "the": "does not show the person clearly"
    pats = [
        # "does not contain any people", "doesn't show a person",
        # "does not appear to contain any humans"
        r"\b(?:does|do|did)(?: not|n't) (?:(?:appear|seem) to )?(?:contain|show|"
        r"include|depict|feature|have|display|capture) " + art + _NEG_ADJ + n,
        # "No people are in the frame", "there are no humans", "the image shows
        # no visible person", "no sign of any pedestrians" (not "no other people")
        _NEG_CTX + r"no (?:signs? of |traces? of )?(?:any )?" + _NEG_ADJ + n,
        r"\bno " + _NEG_ADJ + n + _NEG_SEEN,
        # "there aren't any people", "there is not a person"
        r"\bthere (?:is|are|was|were)(?: not|n't) " + art + _NEG_ADJ + n,
        # "cannot see any person", "can't see anyone", "unable to identify a
        # person", "could not find any people", "I don't see a person"
        r"\b(?:cannot|can not|can't|could not|couldn't|unable to|do not|don't|"
        r"did not|didn't) (?:see|find|identify|detect|locate|spot|make out|"
        r"discern) " + art + _NEG_ADJ + n,
        # "it is impossible to describe a person because the scene is blurry"
        # (not "impossible to determine the person's age")
        r"\b(?:impossible|not possible) to (?:describe|identify|see|find|detect|"
        r"locate) " + art + _NEG_ADJ + n,
        # Clause-initial "A person is not visible", "People are not present"
        # (not "the face of the person is not visible", not "not clearly visible")
        r"(?:^|[.;:!?]\s+)(?:the |a |an )?(?:%s) (?:is|are|was|were) not "
        r"(?:actually )?(?:visible|present|shown|in (?:the|this) (?:image|frame|"
        r"scene|picture|photo))\b" % nouns,
    ]
    if cname == "car":
        pats.append(r"\bnot a vehicle\b")
    if cname == "person":
        # "Nobody is present", "there is no one", "the image shows nobody"
        # (not "no one else").
        pats.append(_NEG_CTX + r"(?:nobody|no one|no-one)\b" + _NEG_EXCEPT)
        pats.append(r"\b(?:nobody|no one|no-one)" + _NEG_SEEN)
    return re.compile("|".join("(?:%s)" % p for p in pats))


_VLM_ABSENT_RX = {name: _build_absent_rx(name) for name in _VLM_SUBJECT_NOUNS}
# An explicit sighting overrides a hedged negation elsewhere in the reply:
# "there is no clearly visible person to describe. A person is partially
# visible in the foreground, sitting in a chair ..." (seen on cam2).
_VLM_PRESENT_RX = {
    "person": re.compile(
        r"\b(?:a|one|the) (?:person|man|woman|individual|figure|pedestrian|"
        r"child|boy|girl) (?:is|can be) (?:partially |partly |faintly |barely |"
        r"dimly |only |just |clearly )?(?:visible|seen)\b"),
}


# The prompts ask for exactly NONE when the object isn't there. Absent:
# the whole reply is NONE ("NONE", "None.", "**NONE**"); it opens with NONE
# followed by "."/"!"/a dash/colon ("NONE. The driveway is empty.", "NONE -
# no car here"); or it ends with a separate all-caps NONE sentence ("The
# path is empty. NONE"). Not absent: "None, but a white sedan is partly
# visible", "None of the windows are broken", field values like "License
# plate: none." / "; none." / "Any passenger? None." (not all caps).
_VLM_NONE_START_RX = re.compile(r"^\W*none(?:\W*$|\s*[.!\-\u2013\u2014:])", re.I)
_VLM_NONE_END_RX = re.compile(r"(?:^|[.!?]\s+)\W*NONE\W*$")


def _vlm_says_absent(description, cls):
    """True if the VLM's reply indicates the detected object isn't present.
    cls is the class NAME ("person", "car", "bicycle"); a numeric detector
    slot is accepted too and mapped through DETECT_CLASSES.

    Deliberately narrow: matches explicit refusals and subject-specific
    negations ("no vehicles", "does not contain any people", "nobody",
    "cannot see any person", "the person is not visible") but NOT incidental
    negations that appear in valid descriptions ("no visible damage", "no
    passenger", "no backpack", "the person's face is not visible") or
    exclusions that imply the subject IS there ("no other people besides the
    man", "no one else"). Offline test: tools/test_vlm_absent.py.
    """
    if isinstance(cls, int):
        cls = DETECT_CLASSES.get(cls, cls)
    raw = re.sub(r"\s+", " ", description).strip()
    if not raw or _VLM_NONE_START_RX.search(raw) or _VLM_NONE_END_RX.search(raw):
        return True
    d = raw.lower().replace("\u2019", "'")
    if any(m in d for m in _VLM_REFUSAL):
        return True
    rx = _VLM_ABSENT_RX.get(cls)
    if not (rx and rx.search(d)):
        return False
    pos = _VLM_PRESENT_RX.get(cls)
    return not (pos and pos.search(d))


def camera_id_for_source(source_id, streams):
    """Map nvstreammux source_id (0-based) to configured camera_id."""
    for stream in streams:
        if stream["source_index"] == source_id:
            return stream["camera_id"]
    return int(source_id) + 1


# ── Performance stats (one file per camera for monitor.py) ────────────────────
class StatsTracker:
    """Per-camera event-funnel counters.

    The funnel, in order (each stage counts events that STOPPED there):
      detections  objects of an interesting class seen by the PGIE
      low_conf    rejected by DETECT_MIN_CONF (or MIN_CONF_<CLASS>_CAMn)
      dedup       suppressed by tracker-id dedup (already emitted / probation)
      throttled   suppressed by the (camera, class) save-interval throttle
      queued      handed to the VLM worker
      drops       dropped because the VLM queue was full
      stale_skip  skipped un-processed: waited > MAX_EVENT_AGE_S in the queue,
                  so its frame is gone (worker backlog) — no VLM call made
      no_frame    worker found no usable JPEG for the camera
      stale_frame_skip frame is >VLM_STALE_FRAME_MAX_GAP_S from the detection
                  instant (fallback shows "now", not the moment detected) —
                  skipped, no VLM call made
      stale_frame worker used a stale JPEG anyway, within gap tolerance
                  (counted, not a stop — save may follow)
      vlm_reject  VLM said the object isn't in the frame (detector false positive)
      errors      worker exception (Ollama/ChromaDB/etc.)
      saves       embedded + stored in ChromaDB with a snapshot
    """

    _LATENCY_WINDOW = 20
    _COUNTER_KEYS = (
        "detections", "low_conf", "dedup", "throttled", "queued", "drops",
        "stale_skip", "no_frame", "stale_frame_skip", "stale_frame",
        "vlm_reject", "errors", "saves",
        "vlm_calls_vllm", "vlm_calls_ollama", "vllm_fallback", "reject_frames_saved",
    )

    def __init__(self, camera_id, path, cam_type="street"):
        self._path = path
        self._camera_id = camera_id
        self._cam_type = cam_type
        self._lock = threading.Lock()
        self._start = time.time()
        self._counts = {k: 0 for k in self._COUNTER_KEYS}
        self._last_frame = None   # wall time of last decoded frame (liveness)
        self._frames = 0
        self._latencies = []
        self._vlm_call_lat = []      # pure VLM round-trip seconds (last N)
        self._vlm_backend_last = None

    def record_frame(self, now):
        # No lock: single float/int store per frame, torn reads are harmless.
        self._last_frame = now
        self._frames += 1

    def bump(self, key):
        with self._lock:
            self._counts[key] += 1

    def record_save(self, latency_s):
        with self._lock:
            self._counts["saves"] += 1
            self._latencies.append(latency_s)
            if len(self._latencies) > self._LATENCY_WINDOW:
                self._latencies.pop(0)

    def record_vlm_call(self, backend, latency_s):
        with self._lock:
            self._vlm_call_lat.append(latency_s)
            if len(self._vlm_call_lat) > self._LATENCY_WINDOW:
                self._vlm_call_lat.pop(0)
            self._vlm_backend_last = backend

    def write(self, queue_depth=0):
        now = time.time()
        elapsed = max(now - self._start, 1.0)
        with self._lock:
            counts = dict(self._counts)
            lats = self._latencies[:]
            call_lats = self._vlm_call_lat[:]
            backend_last = self._vlm_backend_last
        data = {
            "camera_id":       self._camera_id,
            "cam_type":        self._cam_type,
            "run_started_at":  round(self._start, 3),
            "updated_at":      round(now, 3),
            "elapsed_s":       round(elapsed, 1),
            "frames_total":    self._frames,
            "last_frame_at":   round(self._last_frame, 3) if self._last_frame else None,
            "det_per_min":     round(counts["detections"] / elapsed * 60, 1),
            "queue_per_min":   round(counts["queued"] / elapsed * 60, 1),
            "drops_per_min":   round(counts["drops"] / elapsed * 60, 1),
            "saves_per_min":   round(counts["saves"] / elapsed * 60, 1),
            "vlm_ms_avg":      round(sum(lats) / len(lats) * 1000) if lats else None,
            "vlm_ms_max":      round(max(lats) * 1000) if lats else None,
            "vlm_ms_last":     round(lats[-1] * 1000) if lats else None,
            "queue_depth":     queue_depth,
            "vlm_call_ms_avg": round(sum(call_lats) / len(call_lats) * 1000) if call_lats else None,
            "vlm_call_ms_max": round(max(call_lats) * 1000) if call_lats else None,
            "vlm_backend_last": backend_last,
        }
        for k in self._COUNTER_KEYS:
            data[f"{k}_total"] = counts[k]
        try:
            with open(self._path, "w") as fh:
                json.dump(data, fh)
        except OSError:
            pass


class StatsRegistry:
    def __init__(self, streams):
        self._trackers = {}
        os.makedirs(STATS_DIR, exist_ok=True)
        # Remove stats files from previous runs so monitor.py never shows a
        # dead run's numbers as current (observed: 4-day-old files displayed
        # as live). Every configured camera gets a fresh file immediately.
        for old in glob.glob(os.path.join(STATS_DIR, "cam*_stats.json")):
            try:
                os.unlink(old)
            except OSError:
                pass
        for stream in streams:
            cam_id = stream["camera_id"]
            path = os.path.join(STATS_DIR, f"cam{cam_id}_stats.json")
            self._trackers[cam_id] = StatsTracker(
                cam_id, path, cam_type=stream.get("cam_type", "street")
            )

    def for_camera(self, camera_id):
        return self._trackers[camera_id]

    def write_all(self, queue_depth=0):
        for tracker in self._trackers.values():
            tracker.write(queue_depth=queue_depth)


def stats_heartbeat(stats_registry, event_queue):
    """Rewrite every stats file periodically, even with zero events, so
    monitor.py can tell 'pipeline down' (stale file) from 'camera quiet'
    (fresh file, old last_frame_at) from 'no frames' (fresh file, no frames)."""
    while True:
        time.sleep(STATS_HEARTBEAT_S)
        stats_registry.write_all(queue_depth=event_queue.qsize())


# ── Probe: queue detections from all streams in the batch ─────────────────────
class ObjectDetector(BatchMetadataOperator):
    def __init__(self, event_queue, stats_registry, streams):
        super().__init__()
        self._q = event_queue
        self._stats = stats_registry
        self._streams = streams
        self._last_save = {}
        self._event_ids = self._seed_event_ids()
        if self._event_ids:
            log.info(f"[Detect] event ids continue after snapshots/ max: {self._event_ids}")
        self._track_seen = {}   # (camera_id, track_id) -> {count, emitted, last}
        # Per-camera class drops (DROP_CLASSES_CAMn in .env, see
        # streams_config.py) — e.g. bicycle on the office cams, where it
        # fires on empty rooms (notes/motorcycle_rejects_eyeball.md).
        # Values are class NAMES (numeric slots still accepted), resolved
        # against the labels file here and logged. Dropped classes are
        # treated as if not in DETECT_CLASSES for that camera: not counted,
        # not deduped, not queued.
        self._drop_classes = {
            s["camera_id"]: resolve_class_tokens(
                s.get("drop_classes", ()), f"DROP_CLASSES_CAM{s['camera_id']}")
            for s in streams
        }
        for cam, cls_set in sorted(self._drop_classes.items()):
            if cls_set:
                names = ", ".join(f"{CLASS_NAMES[c]} (slot {c})" for c in sorted(cls_set))
                log.info(f"[Detect] cam{cam}: dropping classes {names}")
        # Per-camera, per-class confidence gates (MIN_CONF_<CLASS>_CAMn,
        # optionally limited to MIN_CONF_<CLASS>_HOURS_CAMn local time) — e.g.
        # cam6's night-IR false positives. Only ever raise DETECT_MIN_CONF;
        # unset = unchanged. Keyed by class NAME.
        self._class_gates = {}
        for s in streams:
            g = dict(s.get("min_conf_gates") or {})
            if not g and s.get("min_conf_person") is not None:     # old dict shape
                g["person"] = (s["min_conf_person"], s.get("min_conf_person_hours"))
            if g:
                self._class_gates[s["camera_id"]] = g
        for cam, g in sorted(self._class_gates.items()):
            for cname, (conf, hours) in sorted(g.items()):
                when = ("all day" if hours is None else
                        "%02d:%02d-%02d:%02d local" % (hours[0] // 60, hours[0] % 60,
                                                       hours[1] // 60, hours[1] % 60))
                log.info(f"[Detect] cam{cam}: {cname} min conf {conf:.2f} ({when})")
        if (any(h is not None for g in self._class_gates.values() for _, h in g.values())
                and getattr(LOCAL_TZ, "key", None) is None):
            log.warning("[Detect] zoneinfo unavailable: MIN_CONF_<CLASS>_HOURS "
                        "windows use fixed UTC-7 (an hour off in winter)")

    @staticmethod
    def _seed_event_ids():
        """Start each camera's event counter past the highest evt already in
        snapshots/, so a restart can't reuse cam{N}_src{S}_{label}_evt{K}
        (which overwrote old snapshot JPEGs while ChromaDB kept the old
        description for the duplicate id)."""
        import re
        # Also matches the post-2026-09-30 names with a ms timestamp and an
        # optional collision suffix: cam3_src2_car_evt812_1790000000123[_1].jpg
        pat = re.compile(r"^cam(\d+)_src\d+_[a-zA-Z_]+?_evt(\d+)(?:_\d+)*\.jpg$")
        seeds = {}
        try:
            for name in os.listdir(SNAPSHOT_DIR):
                m = pat.match(name)
                if m:
                    cam, evt = int(m.group(1)), int(m.group(2))
                    if evt > seeds.get(cam, 0):
                        seeds[cam] = evt
        except OSError:
            pass
        return seeds

    def _next_event_id(self, camera_id):
        n = self._event_ids.get(camera_id, 0) + 1
        self._event_ids[camera_id] = n
        return n

    def handle_metadata(self, batch_meta):
        now = time.time()
        # Forget track ids we haven't seen recently so the dict can't grow
        # unbounded and old ids can't suppress a re-appearing object forever.
        if self._track_seen:
            stale = [k for k, v in self._track_seen.items()
                     if now - v["last"] > TRACK_TTL]
            for k in stale:
                del self._track_seen[k]
        minute_of_day = None
        if self._class_gates:
            lt = datetime.datetime.now(tz=LOCAL_TZ)
            minute_of_day = lt.hour * 60 + lt.minute
        for frame_meta in batch_meta.frame_items:
            source_id = frame_meta.source_id
            camera_id = camera_id_for_source(source_id, self._streams)
            stats = self._stats.for_camera(camera_id)
            stats.record_frame(now)
            pts_ns = frame_meta.buffer_pts

            # Determine if this is an office cam (throttled, droppable)
            # or street cam (always process, protect from drops)
            is_office = False
            for s in self._streams:
                if s["camera_id"] == camera_id:
                    is_office = s.get("is_office", False)
                    break

            drop_classes = self._drop_classes.get(camera_id, ())
            gates = self._class_gates.get(camera_id) or {}
            for obj_meta in frame_meta.object_items:
                cls = obj_meta.class_id
                label = DETECT_CLASSES.get(cls)
                if label is None or cls in drop_classes:
                    continue
                stats.bump("detections")
                min_conf = class_min_conf(DETECT_MIN_CONF[label], gates.get(label),
                                          minute_of_day)
                if obj_meta.confidence < min_conf:
                    stats.bump("low_conf")
                    continue

                # Per-object dedup via tracker id: emit once per unique track.
                # The tracker's probationAge already discards single-frame
                # flicker; MIN_TRACK_HITS is a second guard. Untracked objects
                # fall through to the class-level throttle below.
                tid = int(getattr(obj_meta, "object_id", -1))
                tracked = 0 <= tid < UNTRACKED_ID
                te = None
                if tracked:
                    tkey = (camera_id, tid)
                    te = self._track_seen.get(tkey)
                    if te is None:
                        te = {"count": 0, "emitted": False, "last": now}
                        self._track_seen[tkey] = te
                    te["count"] += 1
                    te["last"] = now
                    if te["emitted"] or te["count"] < MIN_TRACK_HITS:
                        stats.bump("dedup")
                        continue

                key = (camera_id, cls)
                # Throttle per (camera, class) as a backstop against tracker-id
                # churn / untracked frames. Office cams use a long interval;
                # street cams a shorter one so real passing traffic is captured.
                throttle = SAVE_INTERVAL if is_office else STREET_SAVE_INTERVAL
                if now - self._last_save.get(key, 0) < throttle:
                    stats.bump("throttled")
                    continue
                self._last_save[key] = now
                if te is not None:
                    te["emitted"] = True
                event_id = self._next_event_id(camera_id)
                log.info(
                    f"[Detect] cam{camera_id} {label} "
                    f"conf={obj_meta.confidence:.2f} evt={event_id}"
                )
                if self._q.qsize() >= VLM_QUEUE_MAX:
                    if is_office:
                        log.info(
                            f"[Detect] VLM queue full ({VLM_QUEUE_MAX}), "
                            f"dropping office cam{camera_id} evt={event_id}"
                        )
                        stats.bump("drops")
                        break
                    elif self._q.qsize() >= STREET_QUEUE_MAX:
                        # Street cams get priority, but still bail out if the
                        # worker has stalled — otherwise the queue grows without
                        # bound and leaks memory (observed at 120k+ items).
                        log.info(
                            f"[Detect] VLM queue hard cap ({STREET_QUEUE_MAX}), "
                            f"dropping street cam{camera_id} evt={event_id} "
                            f"(worker stalled?)"
                        )
                        stats.bump("drops")
                        break
                    else:
                        # Street cam: queue anyway (events are rare; protect them)
                        log.info(
                            f"[Detect] VLM queue full but queuing street cam{camera_id} "
                            f"evt={event_id} (protecting real traffic)"
                        )
                now_dt = datetime.datetime.now(tz=LOCAL_TZ)
                self._q.put({
                    "class_id":    cls,
                    "label":       label,
                    "confidence":  round(float(obj_meta.confidence), 3),
                    "pts_ns":      int(pts_ns),
                    "source_id":   int(source_id),
                    "camera_id":   camera_id,
                    "wall_time":   now_dt.isoformat(),
                    "wall_time_s": round(now_dt.timestamp(), 3),
                    "event_id":    event_id,
                    "queued_at":   now,
                })
                stats.bump("queued")
                break


def get_jpeg_after(after_time, camera_id=None, timeout=5.0):
    """Return (jpeg_bytes, is_stale, gap_s) for the given camera;
    (None, False, None) if none.

    gap_s is how far the chosen file's mtime sits from after_time (the
    detection instant) — 0 for an in-window match, and for a stale fallback
    it's the caller's signal for whether the frame is close enough that the
    detected object might still plausibly be in it, vs. so old/late that the
    scene has certainly moved on (see VLM_STALE_FRAME_MAX_GAP_S).

    Uses a time window (before and after detection time) to account for
    the snapshot being written slightly before/after the probe fires.
    Prefers per-camera files. Never falls back to the global stream when
    a camera_id is given — this prevents cam3/4 from getting images from
    cam1/2.

    If no file in the "fresh" window for the camera, we fall back to the
    most recent file that exists for that camera (stale frame is better
    than wrong camera or nothing) — but note "stale" here usually means
    *late* (mtime near "now", far past after_time) because a backlogged
    worker outruns the snapshot ring buffer, not that the file is
    chronologically old.

    The chosen file is copied to a temp location immediately to avoid
    race with multifilesink rotation.
    """
    deadline = time.time() + timeout
    WINDOW_BEFORE = 30.0   # allow snapshot written up to 30s before detection
    WINDOW_AFTER = 10.0

    if camera_id is not None:
        per_cam_pat = f"/tmp/frame_cam{camera_id}_*.jpg"
        patterns = [per_cam_pat]
    else:
        patterns = [JPEG_GLOB]

    if FRAME_PICK_CLOSEST:
        return _get_jpeg_closest(after_time, camera_id, patterns, deadline,
                                 WINDOW_BEFORE, WINDOW_AFTER)

    best_path = None
    best_mtime = -1
    best_is_stale = False

    while time.time() < deadline:
        for pat in patterns:
            try:
                files = glob.glob(pat)
            except Exception:
                files = []

            for f in files:
                try:
                    m = os.path.getmtime(f)
                except OSError:
                    continue

                # Prefer files within the window around after_time
                if (after_time - WINDOW_BEFORE) <= m <= (after_time + WINDOW_AFTER):
                    if m > best_mtime:
                        best_mtime = m
                        best_path = f
                        best_is_stale = False
                elif camera_id is not None and m > best_mtime:
                    # Track most recent as potential stale fallback for this cam
                    best_mtime = m
                    best_path = f
                    best_is_stale = True

        if best_path:
            break
        time.sleep(0.1)

    if not best_path:
        return None, False, None

    return _read_jpeg_copy(best_path, best_mtime, best_is_stale, after_time, camera_id)


def _get_jpeg_closest(after_time, camera_id, patterns, deadline,
                      window_before, window_after):
    """FRAME_PICK_CLOSEST=1 picker: in-window file closest to after_time;
    newest file for the camera only if nothing is in-window."""
    while True:
        in_path, in_mtime, in_gap = None, None, None
        new_path, new_mtime = None, -1
        for pat in patterns:
            try:
                files = glob.glob(pat)
            except Exception:
                files = []
            for f in files:
                try:
                    m = os.path.getmtime(f)
                except OSError:
                    continue
                if (after_time - window_before) <= m <= (after_time + window_after):
                    g = abs(m - after_time)
                    if in_gap is None or g < in_gap:
                        in_path, in_mtime, in_gap = f, m, g
                elif camera_id is not None and m > new_mtime:
                    new_path, new_mtime = f, m
        if in_path:
            return _read_jpeg_copy(in_path, in_mtime, False, after_time, camera_id)
        if new_path:
            return _read_jpeg_copy(new_path, new_mtime, True, after_time, camera_id)
        if time.time() >= deadline:
            return None, False, None
        time.sleep(0.1)


def _read_jpeg_copy(best_path, best_mtime, best_is_stale, after_time, camera_id):
    gap_s = abs(best_mtime - after_time)

    # Immediately copy to a safe temp file so multifilesink can't delete it
    # while we (or the caller) are reading / using the bytes.
    try:
        fd, safe_path = tempfile.mkstemp(suffix=".jpg", prefix=f"vlm_cam{camera_id or '0'}_")
        os.close(fd)
        shutil.copy2(best_path, safe_path)
        with open(safe_path, "rb") as fh:
            data = fh.read()
        os.unlink(safe_path)  # clean temp
        if len(data) < 1000:
            return None, False, None
        if best_is_stale and camera_id is not None:
            log.info(
                f"[VLM] Stale frame for cam{camera_id}: chosen file is "
                f"{gap_s:.1f}s from the detection instant"
            )
        return data, best_is_stale, gap_s
    except Exception:
        # Fallback: try direct read (may race)
        try:
            with open(best_path, "rb") as fh:
                data = fh.read()
            if len(data) > 1000:
                return data, best_is_stale, gap_s
        except OSError:
            pass
    return None, False, None


def _vllm_is_healthy():
    """Cached vLLM reachability check: at most one real HTTP round trip per
    VLLM_HEALTHCHECK_S, shared by all VLM_WORKERS threads."""
    now = time.time()
    with _vllm_health_lock:
        if now - _vllm_health["checked_at"] < VLLM_HEALTHCHECK_S:
            return _vllm_health["ok"]
        # Claim this window so concurrent workers don't all probe at once.
        _vllm_health["checked_at"] = now
    try:
        r = requests.get(f"{VLLM_URL}/v1/models", timeout=3)
        ok = r.ok and VLLM_MODEL in r.text
    except requests.RequestException:
        ok = False
    with _vllm_health_lock:
        was_ok = _vllm_health["ok"]
        _vllm_health["ok"] = ok
    if ok != was_ok:
        log.info(f"[VLM] vLLM health changed: {'UP' if ok else 'DOWN'} ({VLLM_URL})")
    return ok


def _vllm_mark_unhealthy():
    with _vllm_health_lock:
        _vllm_health["ok"] = False
        _vllm_health["checked_at"] = time.time()


def _vlm_chat_ollama(prompt, jpeg_b64):
    kwargs = {}
    if OLLAMA_THINK is not None:
        kwargs["think"] = OLLAMA_THINK
    resp = _ollama_chat_client.chat(
        model=VLM_MODEL,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [jpeg_b64],
        }],
        options={"num_ctx": VLM_NUM_CTX},
        **kwargs,
    )
    return resp["message"]["content"].strip()


def _vlm_chat_vllm(prompt, jpeg_b64):
    payload = {
        "model": VLLM_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"}},
            ],
        }],
        "max_tokens": VLLM_MAX_TOKENS,
        # Thinking is already off in the gx10 container's template; set it
        # explicitly too so a server-side default change can't turn it on.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = requests.post(f"{VLLM_URL}/v1/chat/completions", json=payload,
                      timeout=VLLM_TIMEOUT_S)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    if not content or not content.strip():
        raise ValueError("vLLM returned empty content")
    head = content.lstrip()[:20].lower()
    if head.startswith("thought") or head.startswith("<|channel") or head.startswith("<think"):
        # Reasoning leaked into content (no --reasoning-parser on gx10);
        # _vlm_says_absent would scan it and false-reject. Fall back instead.
        raise ValueError("vLLM content looks like leaked reasoning")
    return content.strip()


def _vlm_describe(prompt, jpeg_b64, stats=None):
    """Return (description, backend_used). VLM_BACKEND=ollama never touches
    vLLM. VLM_BACKEND=vllm falls back to Ollama for this call if vLLM is
    marked unhealthy or the call raises (timeout, refused, 5xx, bad JSON)."""
    if VLM_BACKEND == "vllm":
        if _vllm_is_healthy():
            try:
                desc = _vlm_chat_vllm(prompt, jpeg_b64)
                if stats is not None:
                    stats.bump("vlm_calls_vllm")
                return desc, "vllm"
            except Exception as e:
                _vllm_mark_unhealthy()
                log.info(f"[VLM] vLLM call failed ({type(e).__name__}: {e}); falling back to Ollama")
                if stats is not None:
                    stats.bump("vllm_fallback")
        elif stats is not None:
            stats.bump("vllm_fallback")
    desc = _vlm_chat_ollama(prompt, jpeg_b64)
    if stats is not None:
        stats.bump("vlm_calls_ollama")
    return desc, "ollama"


def _save_reject_frame(jpeg_bytes, det, description, backend,
                       jpeg_is_stale=None, jpeg_gap_s=None):
    """SAVE_REJECTS=1: keep the rejected frame + a JSON sidecar for later
    labeling, capped at REJECT_SAVE_MAX frames (oldest deleted first).
    Never raises — this must not affect the worker."""
    try:
        os.makedirs(REJECT_DIR, exist_ok=True)
        ts = datetime.datetime.now(tz=LOCAL_TZ).strftime("%Y%m%d-%H%M%S")
        base = (f"{ts}_cam{det['camera_id']}_{det['label']}_evt{det['event_id']}")
        with open(os.path.join(REJECT_DIR, base + ".jpg"), "wb") as fh:
            fh.write(jpeg_bytes)
        with open(os.path.join(REJECT_DIR, base + ".json"), "w") as fh:
            json.dump({
                "camera_id": det["camera_id"], "label": det["label"],
                "class_id": det["class_id"], "confidence": det["confidence"],
                "event_id": det["event_id"], "wall_time": det["wall_time"],
                "backend": backend,
                "model": VLLM_MODEL if backend == "vllm" else VLM_MODEL,
                "jpeg_is_stale": jpeg_is_stale,
                "jpeg_gap_s": None if jpeg_gap_s is None else round(jpeg_gap_s, 2),
                "description": description,
            }, fh)
        jpgs = sorted(f for f in os.listdir(REJECT_DIR) if f.endswith(".jpg"))
        for old in jpgs[:max(0, len(jpgs) - REJECT_SAVE_MAX)]:
            for ext in (".jpg", ".json"):
                try:
                    os.unlink(os.path.join(REJECT_DIR, old[:-4] + ext))
                except OSError:
                    pass
        return True
    except Exception as e:
        log.info(f"[VLM] could not save reject frame: {e}")
        return False


def _claim_event_name(base, jpeg_bytes, collection=None, directory=SNAPSHOT_DIR):
    """Write directory/<name>.jpg for the first name in base, base_1, base_2,
    ... that is free both on disk (O_EXCL create, so an existing snapshot is
    never overwritten) and as a Chroma id. Returns the name used."""
    for i in range(1000):
        name = base if i == 0 else f"{base}_{i}"
        if collection is not None:
            try:
                if collection.get(ids=[name], include=[])["ids"]:
                    continue
            except Exception:
                pass   # can't check; the on-disk O_EXCL still guarantees no overwrite
        try:
            with open(os.path.join(directory, name + ".jpg"), "xb") as fh:
                fh.write(jpeg_bytes)
            return name
        except FileExistsError:
            continue
    raise RuntimeError(f"no free snapshot name for {base}")


def vlm_worker(event_queue, stats_registry):
    import base64

    # Retry: at boot Chroma may not be accepting connections yet, and an
    # exception here would silently kill this worker thread.
    while True:
        try:
            client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
            collection = client.get_or_create_collection(COLLECTION)
            break
        except Exception as e:
            log.info(f"[VLM Worker] ChromaDB not ready ({e}); retrying in 5s")
            time.sleep(5)
    if VLM_BACKEND == "vllm":
        log.info(
            f"[VLM Worker] Ready (collection={COLLECTION} backend=vllm model={VLLM_MODEL} url={VLLM_URL}; "
            f"fallback=ollama model={VLM_MODEL} think={OLLAMA_THINK} "
            f"host={os.environ.get('OLLAMA_HOST') or 'ollama default'}; "
            f"embed={EMBED_MODEL} host={EMBED_HOST or 'ollama default'} prefix_style={_EMBED_PREFIX_STYLE!r})"
        )
    else:
        log.info(
            f"[VLM Worker] Ready (collection={COLLECTION} model={VLM_MODEL} think={OLLAMA_THINK} "
            f"host={os.environ.get('OLLAMA_HOST') or 'ollama default'}; "
            f"embed={EMBED_MODEL} host={EMBED_HOST or 'ollama default'} prefix_style={_EMBED_PREFIX_STYLE!r})"
        )

    while True:
        det = event_queue.get()
        if det is None:
            break
        t_start = time.time()
        camera_id = det["camera_id"]
        stats = stats_registry.for_camera(camera_id)

        # Backlog guard: if this event sat in the queue longer than the frame
        # is retained, the detection-moment JPEG is gone and the scene has
        # moved on. Skip instantly (no glob wait, no VLM) so the worker can
        # reach events whose frames are still fresh. Without this, a slow VLM
        # feeds every backlogged event a stale frame -> guaranteed reject.
        age = t_start - det["queued_at"]
        if age > MAX_EVENT_AGE_S:
            log.info(
                f"[VLM] cam{camera_id} evt={det['event_id']} SKIP stale "
                f"(waited {age:.0f}s > {MAX_EVENT_AGE_S:.0f}s in queue)"
            )
            stats.bump("stale_skip")
            stats.write(queue_depth=event_queue.qsize())
            continue

        try:
            jpeg_bytes, jpeg_is_stale, jpeg_gap_s = get_jpeg_after(
                det["queued_at"], camera_id=camera_id
            )
            if not jpeg_bytes:
                log.info(
                    f"[VLM] No fresh frame within timeout, "
                    f"skipping cam{camera_id} evt={det['event_id']}"
                )
                stats.bump("no_frame")
                stats.write(queue_depth=event_queue.qsize())
                continue
            if jpeg_is_stale and jpeg_gap_s > VLM_STALE_FRAME_MAX_GAP_S:
                # The ring buffer has already rotated past the detection
                # instant; the fallback frame shows "now", a different scene,
                # not a late copy of the same one. Calling the VLM on it is a
                # guaranteed-reject round trip that only deepens the backlog
                # for events that could still make their window.
                log.info(
                    f"[VLM] cam{camera_id} evt={det['event_id']} SKIP "
                    f"stale frame (gap {jpeg_gap_s:.0f}s > "
                    f"{VLM_STALE_FRAME_MAX_GAP_S:.0f}s, no VLM call)"
                )
                stats.bump("stale_frame_skip")
                stats.write(queue_depth=event_queue.qsize())
                continue
            if jpeg_is_stale:
                stats.bump("stale_frame")
            jpeg_b64 = base64.b64encode(jpeg_bytes).decode()

            prompt = VLM_PROMPTS.get(
                det["label"], "Describe what you see in one sentence."
            )
            t_vlm = time.time()
            description, backend = _vlm_describe(prompt, jpeg_b64, stats)
            stats.record_vlm_call(backend, time.time() - t_vlm)

            # Second-stage verification: if the VLM says the object isn't
            # there, it's a detector false positive — drop it (don't embed
            # or persist an empty-scene "car").
            if _vlm_says_absent(description, det["label"]):
                log.info(
                    f"[VLM] cam{camera_id} evt={det['event_id']} REJECT "
                    f"{det['label']} (VLM sees none): {description[:70]} [{backend}]"
                )
                stats.bump("vlm_reject")
                if SAVE_REJECTS and _save_reject_frame(
                        jpeg_bytes, det, description, backend,
                        jpeg_is_stale, jpeg_gap_s):
                    stats.bump("reject_frames_saved")
                continue

            log.info(
                f"[VLM] cam{camera_id} evt={det['event_id']} "
                f"{det['label']}: {description}"
            )

            embed_prompt = (EMBED_DOC_PREFIX + description) if EMBED_DOC_PREFIX else description
            embed_resp = _ollama_embed_client.embeddings(model=EMBED_MODEL, prompt=embed_prompt)
            embedding = embed_resp["embedding"]

            # Unique id + never-overwrite snapshot: the ms timestamp makes the
            # id unique across restarts, and the file is created exclusively
            # (suffix _1, _2, ... on a clash), so a reused event number can
            # never again replace an old picture while Chroma keeps its text.
            doc_id = _claim_event_name(
                f"cam{camera_id}_src{det['source_id']}_{det['label']}_"
                f"evt{det['event_id']}_{int(round(det['wall_time_s'] * 1000))}",
                jpeg_bytes, collection)
            snap_name = f"{doc_id}.jpg"

            try:
                _chroma_add(collection, embedding, description, det, camera_id,
                            snap_name, backend, doc_id)
            except Exception:
                # Don't leave an orphan snapshot with no Chroma record.
                try:
                    os.unlink(os.path.join(SNAPSHOT_DIR, snap_name))
                except OSError:
                    pass
                raise
            log.info(f"[ChromaDB] Saved {doc_id} @ {det['wall_time']}")
            stats.record_save(time.time() - t_start)
        except Exception as e:
            log.info(f"[VLM Worker] Error cam{camera_id} evt={det['event_id']}: {e}")
            stats.bump("errors")
        finally:
            stats.write(queue_depth=event_queue.qsize())


def _chroma_add(collection, embedding, description, det, camera_id, snap_name,
                backend, doc_id):
    collection.add(
        embeddings=[embedding],
        documents=[description],
        metadatas=[{
            "timestamp_s":  round(float(det["pts_ns"] / 1e9), 3),
            "timestamp_ns": det["pts_ns"],
            "wall_time":    det["wall_time"],
            "wall_time_s":  det["wall_time_s"],
            "camera_id":    camera_id,
            "source_id":    det["source_id"],
            "class_id":     det["class_id"],
            "label":        det["label"],
            "confidence":   det["confidence"],
            "image_path":   f"/snapshots/{snap_name}",
            "vlm_backend":  backend,
            "vlm_model":    VLLM_MODEL if backend == "vllm" else VLM_MODEL,
            "class_map":    CLASS_MAP_VERSION,
        }],
        ids=[doc_id],
    )


def _mask_url(url):
    """'rtsp://user:pw@10.0.0.5:554/x?password=..' -> 'rtsp://10.0.0.5:554/…'."""
    try:
        from urllib.parse import urlsplit
        u = urlsplit(url)
        host = u.hostname or "?"
        port = f":{u.port}" if u.port else ""
        return f"{u.scheme}://{host}{port}/…"
    except Exception:
        return "<url hidden>"


def _tiler_layout(n):
    import math
    rows = int(math.sqrt(n))
    cols = int(math.ceil(n / max(1, rows)))
    return rows, cols


def _add_jpeg_branch(pipeline):
    """Frames for VLM worker — shared across display and headless modes."""
    pipeline.add("nvjpegenc", "encoder", {"quality": 85})
    pipeline.add("multifilesink", "filesink", {
        "location":  "/tmp/frame_%05d.jpg",
        "max-files": 200,
        "async":     0,
        "sync":      0,
    })
    pipeline.link("encoder", "filesink")




def build_pipeline(detector, streams):
    n = len(streams)
    pipeline = Pipeline("multi-cam-vlm-pipeline")

    pipeline.add("nvstreammux", "mux", {
        "batch-size":           n,
        "width":                FRAME_W,
        "height":               FRAME_H,
        "batched-push-timeout": 40000,
        "live-source":          1,
        "compute-hw":           1,
    })

    for i, stream in enumerate(streams):
        name = f"src{i}"
        cam = stream["camera_id"]
        props = {
            "uri":                     stream["url"],
            "select-rtp-protocol":     stream["rtsp_transport"],
            "rtsp-reconnect-interval": RECONNECT_INTERVAL,
        }
        if stream.get("rtsp_transport") == 4:
            # TCP is often more reliable inside containers / for Reolink/Dahua
            props.update({
                "latency": 2000,
                "drop-on-latency": 1,
            })
        pipeline.add("nvurisrcbin", name, props)

        # Per-source tee so we can feed a clean JPEG branch per camera for VLM.
        # One leg goes to the mux (for batched inference), the other produces
        # /tmp/frame_camN_*.jpg that get_jpeg_after(camera_id=...) can use reliably.
        tee_name = f"srctee{i}"
        q_mux = f"qsrc{i}_mux"
        q_snap = f"qsrc{i}_snap"
        pipeline.add("tee", tee_name)
        pipeline.add("queue", q_mux, {"max-size-buffers": 6, "leaky": 2})
        pipeline.add("queue", q_snap, {"max-size-buffers": 3, "leaky": 2})

        pipeline.link(name, tee_name)
        pipeline.link(tee_name, q_mux)
        pipeline.link(tee_name, q_snap)
        pipeline.link((q_mux, "mux"), ("", "sink_%u"))

        # Dedicated low-overhead JPEG snapshot for this camera
        enc = f"snapenc{cam}"
        fsink = f"fsnap{cam}"
        pipeline.add("nvjpegenc", enc, {"quality": 82})
        pipeline.add("multifilesink", fsink, {
            "location":  f"/tmp/frame_cam{cam}_%05d.jpg",
            "max-files": SNAPSHOT_RING_FILES,
            "async":     0,
            "sync":      0,
        })
        pipeline.link(q_snap, enc)
        pipeline.link(enc, fsink)

    pipeline.add("nvinfer", "infer", {
        "config-file-path": os.environ.get("PGIE_CONFIG", "pgie_config_rtdetr.txt"),
        "batch-size":       n,
    })

    # Multi-object tracker: assigns a persistent object_id per target so the
    # probe can emit one event per unique car/person instead of one per frame.
    # NvDCF_perf is self-contained (no ReID model); probationAge filters flicker.
    _DS = "/opt/nvidia/deepstream/deepstream"
    pipeline.add("nvtracker", "tracker", {
        "ll-lib-file":    f"{_DS}/lib/libnvds_nvmultiobjecttracker.so",
        "ll-config-file": f"{_DS}/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml",
        "tracker-width":  640,
        "tracker-height": 384,
        "gpu-id":         0,
    })

    if ENABLE_DISPLAY or LIVE_STREAM:
        # tee after infer for (optional) display + live web stream
        # (VLM JPEGs are now provided by the per-camera snapshot branches created earlier)
        pipeline.add("tee", "tee")
        pipeline.add("queue", "q_display", {"max-size-buffers": 2, "leaky": 2})

        pipeline.link("mux", "infer", "tracker", "tee")

        # Build tiled + osd path (used for both display and live HLS stream)
        rows, cols = _tiler_layout(n)
        pipeline.add("nvmultistreamtiler", "tiler", {
            "rows":    rows,
            "columns": cols,
            "width":   TILER_W,
            "height":  TILER_H,
            "compute-hw": 1,
        })
        pipeline.add("nvosdbin", "osd")

        # Link the display branch *downstream first*, then attach the tee.
        # This helps caps negotiation across the tee.
        # Use a converter on the display branch to make caps negotiation
        # happy when coming from the post-infer tee (common in DeepStream).
        pipeline.add("nvvideoconvert", "dispconv")
        pipeline.link("q_display", "dispconv")
        pipeline.link("dispconv", "tiler")
        pipeline.link("tiler", "osd")
        pipeline.link("tee", "q_display")

        # After OSD, branch for display sink and/or HLS stream.
        # We always use a converter before the hardware encoder for robust caps negotiation.
        if ENABLE_DISPLAY and LIVE_STREAM:
            pipeline.add("tee", "osdtee")
            pipeline.add("queue", "q_osd_disp", {"max-size-buffers": 2, "leaky": 2})
            pipeline.add("queue", "q_osd_hls", {"max-size-buffers": 2, "leaky": 2})
            pipeline.link("osd", "osdtee")
            pipeline.link("osdtee", "q_osd_disp")
            pipeline.link("osdtee", "q_osd_hls")
            disp_sink = "q_osd_disp"
            stream_in = "q_osd_hls"
        else:
            disp_sink = "osd"
            stream_in = "osd"

        if ENABLE_DISPLAY:
            import platform
            sink = "nv3dsink" if platform.processor() == "aarch64" else "nveglglessink"
            pipeline.add(sink, "display", {"sync": 0, "qos": 0})
            pipeline.link(disp_sink, "display")
            log.info(f"[Main] Live display ON — {n} streams in {rows}x{cols} tile")

        if LIVE_STREAM:
            os.makedirs("/tmp/hls", exist_ok=True)
            # Always go through a queue + converter before the encoder for stability
            # and correct caps negotiation from the OSD/tiler output.
            pipeline.add("queue", "q_hls", {"max-size-buffers": 4, "leaky": 2})
            pipeline.add("nvvideoconvert", "streamconv")
            pipeline.add("nvv4l2h264enc", "streamenc", {"preset-level": 1, "insert-sps-pps": 1})
            pipeline.add("hlssink2", "hls", {
                "location": "/tmp/hls/segment%05d.ts",
                "playlist-location": "/tmp/hls/stream.m3u8",
                "max-files": 8,
                "target-duration": 2,
                "playlist-type": 1,
            })
            # Link the appropriate upstream into our hls queue (then conv -> enc)
            pipeline.link(stream_in, "q_hls")
            pipeline.link("q_hls", "streamconv")
            pipeline.link("streamconv", "streamenc")
            pipeline.link("streamenc", "hls")
            log.info(f"[Main] Live HLS stream enabled at /hls/stream.m3u8 ({n} streams tiled)")

    else:
        _add_jpeg_branch(pipeline)
        pipeline.link("mux", "infer", "tracker", "encoder")
        log.info("[Main] Headless mode (set ENABLE_DISPLAY=1 or LIVE_STREAM=1 for live video)")

    # Probe on the tracker (not infer) so obj_meta.object_id is populated.
    pipeline.attach("tracker", Probe("detector", detector))
    return pipeline


def main():
    streams = load_streams()
    n = len(streams)
    log.info(f"[Main] Starting multi-stream pipeline: {n} camera(s)")
    for s in streams:
        # Host only: RTSP URLs carry credentials (userinfo and/or query string).
        log.info(f"  cam{s['camera_id']}: {_mask_url(s['url'])} ({s.get('cam_type', 'street')})")

    # Refuse to start on a class-map mismatch (labels vs config was checked at
    # import; here: labels vs the model's actual output width).
    width = model_output_width(CLASS_MAP["onnx_path"]) if CLASS_MAP["onnx_path"] else None
    if width is None:
        log.warning("[Detect] could not read the model's pred_logits width; "
                    "class map checked against the config only")
    elif width != len(CLASS_NAMES):
        log.error(f"[Detect] REFUSING TO START: model outputs {width} class slots but "
                  f"{CLASS_MAP['labels_path']} has {len(CLASS_NAMES)} lines")
        raise SystemExit(2)
    log.info(f"[Detect] class map ({os.path.basename(CLASS_MAP['labels_path'])}, "
             f"{len(CLASS_NAMES)} lines = num-detected-classes"
             f"{'' if width is None else ' = model width'}): {describe_class_map(CLASS_MAP)}")
    log.info("[Detect] min conf: " + ", ".join(f"{k}={v:.2f}" for k, v in DETECT_MIN_CONF.items()))

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    stats_registry = StatsRegistry(streams)
    event_queue = queue.Queue()
    detector = ObjectDetector(event_queue, stats_registry, streams)
    stats_registry.write_all()  # fresh files immediately, so monitor sees all cams

    workers = [
        threading.Thread(
            target=vlm_worker,
            args=(event_queue, stats_registry),
            daemon=True,
        )
        for _ in range(VLM_WORKERS)
    ]
    for worker in workers:
        worker.start()

    heartbeat = threading.Thread(
        target=stats_heartbeat,
        args=(stats_registry, event_queue),
        daemon=True,
    )
    heartbeat.start()

    try:
        while True:
            log.info("[Main] Building pipeline...")
            pipeline = build_pipeline(detector, streams)
            try:
                pipeline.start().wait()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log.info(f"[Main] Pipeline error: {e}")
            log.info(f"[Main] Pipeline stopped, restarting in {RESTART_DELAY}s...")
            time.sleep(RESTART_DELAY)
    except KeyboardInterrupt:
        log.info("Stopping...")
    finally:
        for _ in workers:
            event_queue.put(None)
        for worker in workers:
            worker.join()


if __name__ == "__main__":
    main()