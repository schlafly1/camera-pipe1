#!/usr/bin/env python3
"""Rebuild the search index into a NEW Chroma collection (default
vision_events_v2) with correct class labels. The source collection
(vision_events) is only read, never modified; snapshots are only read.

    cd ~/sd/camera-pipe1
    setsid nohup .venv/bin/python3 -u tools/rebuild_index.py > logs/rebuild_index.out 2>&1 < /dev/null &

Why: from the RT-DETR switch (2026-08-21) until the 2026-09-30 class-map
fix, cars were saved as "person" and people as "motorcycle", with prompts
for the wrong class. Before commit 9302022 event numbers also restarted at
each restart, so some snapshots were overwritten while Chroma kept the old
text.

Per source entry, by era:
  pre-RT-DETR (before 2026-08-21, the June resnet18 data, labels correct):
      copied as-is (same id, document and embedding; label "motorcycle" is
      normalized to "bicycle", orig_label kept) unless its text is a
      refusal/absent by the current _vlm_says_absent, or its image is
      missing/overwritten.
  RT-DETR era (2026-08-21 .. fix): skipped if the image is missing or its
      mtime is more than 1 h after the event (overwritten). Otherwise the
      fixed detector re-runs offline on the saved image (tools/rtdetr_offline,
      a copy of the TensorRT engine, batch 1) with the live per-class gates,
      per-camera drops and time-of-day gates; the class is the most confident
      detected app class (near-full-frame boxes, the typical office "car"
      false positive, are tried after smaller ones; at most 2 classes tried),
      or the original detection's true class if nothing is re-detected. The
      image is re-described by vLLM (VLLM_URL) with that class's prompt; a
      NONE/absent reply drops the entry (or tries the next class). Accepted
      text is embedded with nomic-embed-text at EMBED_HOST and upserted with
      corrected metadata (label, class_id slot, original timestamps, camera,
      image_path, source_doc_id, rebuilt_at, ...). New id:
      cam{N}_src{S}_{label}_evt{K}_{ms}.
  live after the fix (metadata class_map=rtdetr5-20260930): copied as-is.

Rate limit: one VLM request at a time, never while the live pipeline's VLM
queue (stats/cam*_stats.json queue_depth) is non-empty or vLLM has waiting
requests (or 2+ running). vLLM only (no Ollama fallback): on errors it backs
off and retries.

Resumable: every finished source id is appended to
logs/rebuild_index_state.jsonl (ok/copied/skip/dropped are final; errors
are retried next run). Progress: logs/rebuild_index.log and
logs/rebuild_index_status.json (counts, rate, ETA). After the main pass it
re-lists the source and handles entries that arrived meanwhile; rerun any
time to sync new live entries. SIGTERM stops after the current entry.
"""
import argparse
import base64
import collections
import datetime
import json
import logging
import os
import re
import signal
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
# pipeline_multi's import-time logger must not write into logs/pipeline.log.
os.environ["LOG_DIR"] = os.environ.get("REBUILD_PM_LOG_DIR", "/tmp/rebuild_index_pm")
if os.path.exists(".env"):                  # same raw KEY=value parsing as start_thor.sh
    for line in open(".env"):
        line = line.rstrip("\r\n")
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v)

import chromadb  # noqa: E402
import requests  # noqa: E402
from PIL import Image  # noqa: E402

import pipeline_multi as pm  # noqa: E402
import streams_config as sc  # noqa: E402

TZ = pm.LOCAL_TZ
RTDETR_SINCE = datetime.datetime(2026, 8, 21, tzinfo=TZ).timestamp()
OVERWRITE_GAP_S = 3600.0
REBUILD_TAG = "v2-20260930"
LOG_PATH = "logs/rebuild_index.log"
STATE_PATH = "logs/rebuild_index_state.jsonl"
STATUS_PATH = "logs/rebuild_index_status.json"
FINAL = {"ok", "copied", "skip", "dropped"}
EVT_RX = re.compile(r"_evt(\d+)")
_stop = False


def _on_term(signum, frame):
    global _stop
    _stop = True


def setup_log():
    lg = logging.getLogger("rebuild")
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        lg.addHandler(h)
    return lg


log = None


def get_all(col, include, where=None, page=2000):
    out = {"ids": [], "metadatas": [], "documents": [], "embeddings": []}
    off = 0
    while True:
        kw = {"include": include, "limit": page, "offset": off}
        if where:
            kw["where"] = where
        r = col.get(**kw)
        if not r["ids"]:
            break
        out["ids"] += r["ids"]
        for k in ("metadatas", "documents", "embeddings"):
            v = r.get(k)
            out[k] += list(v) if v is not None else [None] * len(r["ids"])
        off += len(r["ids"])
        if len(r["ids"]) < page:
            break
    return out


def load_state():
    st = {}
    if os.path.exists(STATE_PATH):
        for line in open(STATE_PATH):
            try:
                d = json.loads(line)
                st[d["id"]] = d
            except (ValueError, KeyError):
                continue
    return st


class State:
    def __init__(self):
        self.done = load_state()
        self.fh = open(STATE_PATH, "a")

    def record(self, src_id, outcome, reason="", **kw):
        d = {"id": src_id, "outcome": outcome, "reason": reason,
             "t": datetime.datetime.now(TZ).isoformat(timespec="seconds"), **kw}
        self.done[src_id] = d
        self.fh.write(json.dumps(d) + "\n")
        self.fh.flush()

    def final(self, src_id):
        d = self.done.get(src_id)
        return d is not None and d["outcome"] in FINAL


def live_busy(metrics_url):
    """(busy, why). Live VLM queue from the pipeline's stats files (fresh ones
    only) + vLLM's own queue."""
    q = 0
    now = time.time()
    for p in os.listdir("stats"):
        if p.startswith("cam") and p.endswith("_stats.json"):
            try:
                with open(os.path.join("stats", p)) as fh:
                    s = json.load(fh)
                if now - float(s.get("updated_at") or 0) < 60:
                    q = max(q, int(s.get("queue_depth") or 0))
            except (OSError, ValueError):
                continue
    if q > 0:
        return True, f"live queue {q}"
    try:
        txt = requests.get(metrics_url, timeout=3).text
    except requests.RequestException:
        return True, "vLLM metrics unreachable"
    run = wait = 0.0
    for line in txt.splitlines():
        if line.startswith("vllm:num_requests_running{"):
            run += float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:num_requests_waiting{"):
            wait += float(line.rsplit(" ", 1)[1])
    if wait > 0 or run >= 2:
        return True, f"vLLM running={run:.0f} waiting={wait:.0f}"
    return False, ""


class Rebuilder:
    def __init__(self, args):
        self.a = args
        self.client = chromadb.HttpClient(host=pm.CHROMADB_HOST, port=pm.CHROMADB_PORT)
        self.src = self.client.get_collection(args.source)
        # Same collection settings (e.g. distance space) as the source.
        self.dst = None if args.dry_run else self.client.get_or_create_collection(
            args.dest, metadata=self.src.metadata or None)
        self.state = State()
        self.counts = collections.Counter()
        self.recent = collections.deque(maxlen=200)   # finish times of VLM-processed entries
        self.t0 = time.time()
        self.paused_s = 0.0
        self.metrics_url = pm.VLLM_URL + "/metrics"
        streams = sc.load_streams()
        self.drop = {s["camera_id"]: pm.resolve_class_tokens(
            s.get("drop_classes", ()), f"DROP_CLASSES_CAM{s['camera_id']}") for s in streams}
        self.gates = {s["camera_id"]: dict(s.get("min_conf_gates") or {}) for s in streams}
        self.office = {s["camera_id"]: s.get("is_office", False) for s in streams}
        self.det = None
        self.last_call = 0.0
        self.remaining = 0

    # ── helpers ──────────────────────────────────────────────────────────
    def detector(self):
        if self.det is None:
            from rtdetr_offline import RTDETR
            self.det = RTDETR(self.a.engine, threshold=0.4)
            log.info(f"[init] offline detector ready ({self.det.num_slots} slots)")
        return self.det

    def wait_turn(self):
        t = time.time()
        last_why, last_log = None, 0.0
        while not _stop:
            busy, why = live_busy(self.metrics_url)
            if not busy:
                break
            if why != last_why or time.time() - last_log > 60:
                log.info(f"[pause] {why}")
                last_why, last_log = why, time.time()
            time.sleep(2)
        gap = self.a.min_gap - (time.time() - self.last_call)
        if gap > 0:
            time.sleep(gap)
        self.paused_s += time.time() - t

    def vlm(self, prompt, b64):
        delay = 10
        while not _stop:
            self.wait_turn()
            if _stop:
                break
            self.last_call = time.time()
            try:
                t = time.time()
                desc = pm._vlm_chat_vllm(prompt, b64)
                return desc, time.time() - t
            except Exception as e:
                log.info(f"[vlm] error {type(e).__name__}: {str(e)[:120]}; retry in {delay}s")
                time.sleep(delay)
                delay = min(delay * 2, 300)
        raise KeyboardInterrupt

    def embed(self, text):
        # Honor pipeline EMBED_DOC_PREFIX when EMBED_PREFIX_STYLE=nomic.
        prompt = (pm.EMBED_DOC_PREFIX + text) if getattr(pm, "EMBED_DOC_PREFIX", "") else text
        delay = 5
        while True:
            try:
                return pm._ollama_embed_client.embeddings(model=pm.EMBED_MODEL, prompt=prompt)["embedding"]
            except Exception as e:
                if _stop:
                    raise KeyboardInterrupt
                log.info(f"[embed] error {type(e).__name__}: {str(e)[:120]}; retry in {delay}s")
                time.sleep(delay)
                delay = min(delay * 2, 300)

    @staticmethod
    def image_check(meta):
        """(path, None) if usable, else (path, reason)."""
        name = os.path.basename(meta.get("image_path") or "")
        path = os.path.join(pm.SNAPSHOT_DIR, name)
        if not name or not os.path.exists(path):
            return path, "missing_image"
        gap = os.path.getmtime(path) - float(meta.get("wall_time_s") or 0)
        if gap > OVERWRITE_GAP_S:
            return path, "overwritten"
        return path, None

    def write_status(self, phase):
        el = time.time() - self.t0
        rate_h = None
        if len(self.recent) >= 5:
            span = self.recent[-1] - self.recent[0]
            if span > 0:
                rate_h = (len(self.recent) - 1) / span * 3600
        eta = None
        if rate_h and self.remaining:
            eta = (datetime.datetime.now(TZ) + datetime.timedelta(
                hours=self.remaining / rate_h)).isoformat(timespec="minutes")
        d = {"phase": phase, "updated": datetime.datetime.now(TZ).isoformat(timespec="seconds"),
             "elapsed_h": round(el / 3600, 2), "paused_h": round(self.paused_s / 3600, 2),
             "rtdetr_remaining": self.remaining,
             "rate_per_h": None if rate_h is None else round(rate_h, 1),
             "eta": eta, "counts": dict(sorted(self.counts.items())),
             "dest": self.a.dest, "dest_count": self.dst.count() if self.dst else None}
        tmp = STATUS_PATH + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(d, fh, indent=1)
        os.replace(tmp, STATUS_PATH)

    # ── copies ───────────────────────────────────────────────────────────
    def copy_entries(self, ids, era):
        todo = [i for i in ids if not self.state.final(i)]
        for k in range(0, len(todo), 200):
            if _stop:
                return
            chunk = todo[k:k + 200]
            r = self.src.get(ids=chunk, include=["metadatas", "documents", "embeddings"])
            add_ids, add_emb, add_doc, add_meta = [], [], [], []
            for i, m, doc, emb in zip(r["ids"], r["metadatas"], r["documents"], r["embeddings"]):
                lbl = pm._norm_class_name(m.get("label") or "")
                if era == "june":
                    path, bad = self.image_check(m)
                    if bad:
                        self.state.record(i, "skip", bad, era=era)
                        self.counts[f"{era}_skip_{bad}"] += 1
                        continue
                    if pm._vlm_says_absent(doc or "", lbl):
                        self.state.record(i, "skip", "absent_text", era=era, doc=(doc or "")[:120])
                        self.counts[f"{era}_skip_absent_text"] += 1
                        continue
                meta = dict(m)
                meta.update({"label": lbl or m.get("label") or "?",
                             "orig_label": m.get("label") or "",
                             "source_doc_id": i, "source_collection": self.a.source,
                             "rebuilt_at": datetime.datetime.now(TZ).isoformat(timespec="seconds"),
                             "rebuild": REBUILD_TAG, "era": era, "class_source": "copied"})
                add_ids.append(i)
                add_emb.append(list(emb))
                add_doc.append(doc)
                add_meta.append(meta)
            if add_ids and self.dst is not None:
                self.dst.upsert(ids=add_ids, embeddings=add_emb, documents=add_doc, metadatas=add_meta)
            for i in add_ids:
                self.state.record(i, "copied", era=era)
            self.counts[f"{era}_copied"] += len(add_ids)
            self.write_status(f"copy {era}")
        log.info(f"[copy] {era}: " + ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items())
                                               if k.startswith(era)))

    # ── RT-DETR era ──────────────────────────────────────────────────────
    def candidates(self, img, cam, wall_s):
        det = self.detector()
        dets = det.detect(img)
        W, H = img.size
        minute = None
        lt = datetime.datetime.fromtimestamp(wall_s, TZ)
        minute = lt.hour * 60 + lt.minute
        best = {}
        for d in dets:
            name = pm.DETECT_CLASSES.get(d["slot"])
            if name is None or d["slot"] in self.drop.get(cam, ()):
                continue
            thr = sc.class_min_conf(pm.DETECT_MIN_CONF[name],
                                    self.gates.get(cam, {}).get(name), minute)
            if d["conf"] < thr:
                continue
            x1, y1, x2, y2 = d["box"]
            frac = max(0, min(x2, W) - max(x1, 0)) * max(0, min(y2, H) - max(y1, 0)) / float(W * H)
            if name not in best or d["conf"] > best[name][0]:
                best[name] = (d["conf"], frac)
        # Most confident first, but near-full-frame boxes (>= 60% of the
        # image; office "car" false positives) after the others.
        order = sorted(best.items(), key=lambda kv: (kv[1][1] >= 0.6, -kv[1][0]))
        return order, dets

    def process(self, i, m):
        path, bad = self.image_check(m)
        if bad:
            self.state.record(i, "skip", bad, era="rtdetr")
            self.counts[f"rtdetr_skip_{bad}"] += 1
            return
        cam = int(m.get("camera_id") or 0)
        wall_s = float(m.get("wall_time_s") or 0)
        try:
            img = Image.open(path)
            img.load()
        except Exception as e:
            self.state.record(i, "skip", "bad_image", era="rtdetr", err=str(e)[:80])
            self.counts["rtdetr_skip_bad_image"] += 1
            return
        order, dets = self.candidates(img, cam, wall_s)
        orig = pm.DETECT_CLASSES.get(int(m.get("class_id", -1)))
        redet = ",".join(f"{n}:{c:.2f}" + ("F" if fr >= 0.6 else "") for n, (c, fr) in order)
        if order:
            tries = [(n, c, "redetect") for n, (c, fr) in order][:2]
        elif orig and pm.CLASS_SLOTS[orig] not in self.drop.get(cam, ()):
            tries = [(orig, float(m.get("confidence") or 0), "original")]
        else:
            self.state.record(i, "dropped", "no_class", era="rtdetr", orig=orig or "")
            self.counts["rtdetr_drop_no_class"] += 1
            return
        with open(path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        absent = []
        for name, conf, how in tries:
            desc, dt = self.vlm(pm.VLM_PROMPTS[name], b64)
            self.counts["vlm_calls"] += 1
            if pm._vlm_says_absent(desc, name):
                absent.append(f"{name}: {desc[:80]}")
                log.info(f"[absent] {i} as {name} ({how} {conf:.2f}) {dt:.1f}s: {desc[:90]}")
                continue
            emb = self.embed(desc)
            evt = EVT_RX.search(i)
            new_id = (f"cam{cam}_src{int(m.get('source_id') or 0)}_{name}_"
                      f"evt{evt.group(1) if evt else 0}_{int(round(wall_s * 1000))}")
            meta = {
                "wall_time": m.get("wall_time") or "", "wall_time_s": wall_s,
                "timestamp_s": float(m.get("timestamp_s") or 0),
                "timestamp_ns": int(m.get("timestamp_ns") or 0),
                "camera_id": cam, "source_id": int(m.get("source_id") or 0),
                "class_id": pm.CLASS_SLOTS[name], "label": name,
                "confidence": round(float(conf), 3),
                "orig_confidence": float(m.get("confidence") or 0),
                "orig_label": m.get("label") or "", "orig_class_id": int(m.get("class_id", -1)),
                "image_path": m.get("image_path") or "",
                "source_doc_id": i, "source_collection": self.a.source,
                "rebuilt_at": datetime.datetime.now(TZ).isoformat(timespec="seconds"),
                "rebuild": REBUILD_TAG, "era": "rtdetr", "class_map": pm.CLASS_MAP_VERSION,
                "class_source": how, "redetect": redet,
                "vlm_backend": "vllm", "vlm_model": pm.VLLM_MODEL,
            }
            if self.dst is not None:
                self.dst.upsert(ids=[new_id], embeddings=[emb], documents=[desc], metadatas=[meta])
            self.state.record(i, "ok", label=name, new_id=new_id, how=how, redetect=redet,
                              orig=orig or "")
            self.counts[f"rtdetr_ok_{name}"] += 1
            if orig and name != orig:
                self.counts["rtdetr_ok_class_changed_vs_orig"] += 1
            log.info(f"[ok] {i} -> {new_id} ({how} {conf:.2f}; redetect {redet or '-'}) "
                     f"{dt:.1f}s: {desc[:90]}")
            return
        self.state.record(i, "dropped", "vlm_absent", era="rtdetr", redetect=redet,
                          absent=absent)
        self.counts["rtdetr_drop_vlm_absent"] += 1

    # ── main ─────────────────────────────────────────────────────────────
    def list_source(self):
        allm = get_all(self.src, ["metadatas"])
        june, live, rt = [], [], []
        for i, m in zip(allm["ids"], allm["metadatas"]):
            m = m or {}
            if m.get("class_map") == pm.CLASS_MAP_VERSION:
                live.append(i)
            elif float(m.get("wall_time_s") or 0) < RTDETR_SINCE:
                june.append(i)
            else:
                rt.append((float(m.get("wall_time_s") or 0), i, m))
        rt.sort(key=lambda x: -x[0])                 # newest first
        return june, live, rt

    def run(self):
        log.info(f"[init] {self.a.source} -> {self.a.dest}{' (DRY RUN)' if self.a.dry_run else ''}; "
                 f"vLLM {pm.VLLM_URL} {pm.VLLM_MODEL}; embed {pm.EMBED_MODEL} @ {pm.EMBED_HOST}; "
                 f"state {len(self.state.done)} ids already recorded")
        log.info(f"[init] class map: {pm.describe_class_map(pm.CLASS_MAP)}; drops "
                 + str({c: sorted(pm.CLASS_NAMES[x] for x in v) for c, v in self.drop.items() if v})
                 + f"; gates {self.gates}")
        passes = 0
        while not _stop:
            passes += 1
            june, live, rt = self.list_source()
            todo = [x for x in rt if not self.state.final(x[1])]
            self.remaining = len(todo)
            log.info(f"[pass {passes}] source: june={len(june)} rtdetr={len(rt)} live_fixed={len(live)}; "
                     f"rtdetr to do {len(todo)}")
            if not self.a.no_copy:
                self.copy_entries(june, "june")
                self.copy_entries(live, "live_fixed")
            n = 0
            for wall_s, i, m in todo:
                if _stop or (self.a.limit and n >= self.a.limit):
                    break
                self.process(i, m)
                n += 1
                self.remaining -= 1
                if self.state.done.get(i, {}).get("outcome") in ("ok", "dropped"):
                    self.recent.append(time.time())
                if n % 10 == 0:
                    self.write_status("rtdetr")
                if n % 200 == 0:
                    log.info("[progress] " + ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items()))
                             + f"; remaining {self.remaining}")
            self.write_status("rtdetr")
            if self.a.limit or not todo or self.a.once:
                break
        self.write_status("done" if not _stop else "stopped")
        log.info("[done] " + ", ".join(f"{k}={v}" for k, v in sorted(self.counts.items())))


def main():
    global log
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default="vision_events")
    ap.add_argument("--dest", default="vision_events_v2")
    ap.add_argument("--engine", help="TensorRT engine (default: copy of the newest nvinfer engine)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N RT-DETR entries")
    ap.add_argument("--min-gap", type=float, default=float(os.environ.get("REBUILD_MIN_GAP_S", "0.5")),
                    help="min seconds between VLM requests")
    ap.add_argument("--no-copy", action="store_true", help="skip the June/live copy phase")
    ap.add_argument("--once", action="store_true", help="single pass (no re-list at the end)")
    ap.add_argument("--dry-run", action="store_true", help="no writes to Chroma (state still recorded)")
    a = ap.parse_args()
    if a.dest == a.source:
        sys.exit("refusing: --dest must differ from --source")
    os.makedirs("logs", exist_ok=True)
    global STATE_PATH, STATUS_PATH
    if a.dry_run:
        STATE_PATH = "/tmp/rebuild_index_dryrun_state.jsonl"
        STATUS_PATH = "/tmp/rebuild_index_dryrun_status.json"
    log = setup_log()
    signal.signal(signal.SIGTERM, _on_term)
    try:
        Rebuilder(a).run()
    except KeyboardInterrupt:
        log.info("[stop] interrupted")


if __name__ == "__main__":
    main()
