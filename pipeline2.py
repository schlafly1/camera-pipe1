"""
pipeline2.py - Vision-enhanced pipeline.

Saves frames to disk via nvjpegenc + multifilesink.  Detections trigger
the VLM worker to read the latest JPEG, send to Gemma4:26b, embed with
nomic-embed-text, and save to ChromaDB.

Classes detected by TrafficCamNet:
  0=Car  1=TwoWheeler  2=Person  3=RoadSign (skipped)
"""

import datetime
import glob
import json
import os
import queue
import threading
import time

import chromadb
import ollama
from pyservicemaker import (
    BatchMetadataOperator,
    Pipeline,
    Probe,
)

try:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo("America/Los_Angeles")
except Exception:
    LOCAL_TZ = datetime.timezone(datetime.timedelta(hours=-7))  # PDT fallback

# ── Config ────────────────────────────────────────────────────────────────────
RTSP_URL      = os.environ.get("RTSP_URL", "")
CHROMADB_HOST = "localhost"
CHROMADB_PORT = 8000
COLLECTION    = "vision_events"
VLM_MODEL     = os.environ.get("VLM_MODEL", "gemma4:26b")
EMBED_MODEL   = "nomic-embed-text"
SAVE_INTERVAL = 5.0
VLM_QUEUE_MAX = 6       # drop detections when VLM is this far behind
FRAME_W       = 1920
FRAME_H       = 1080

CAMERA_ID     = int(os.environ.get("CAMERA_ID", "1"))
JPEG_PATH     = f"/tmp/cam{CAMERA_ID}_%05d.jpg"
JPEG_GLOB     = f"/tmp/cam{CAMERA_ID}_*.jpg"
SNAPSHOT_DIR  = "/workspace/snapshots"
STATS_DIR     = "/workspace/stats"
# 0=UDP (default), 4=TCP — set RTSP_TRANSPORT=4 for cameras that require TCP
RTSP_TRANSPORT = int(os.environ.get("RTSP_TRANSPORT", "0"))

DETECT_CLASSES   = {0: "car", 1: "motorcycle", 2: "person"}
DETECT_MIN_CONF  = {0: 0.50, 1: 0.50, 2: 0.30}  # raise car/moto threshold

VLM_PROMPTS = {
    # ── Car (class 0) ──────────────────────────────────────────────────────────
    0: (
        "Describe this vehicle in 2–3 sentences. Include: color, body style"
        " (sedan/SUV/truck/van/coupe), make and model if recognizable, approximate"
        " year range, any visible damage or distinctive markings, direction of travel,"
        " and license plate text if legible."
    ),
    # Alternatives — uncomment to try:
    # 0: "Describe this vehicle in one sentence: color, type (car/truck/van/SUV), and make or model if recognizable.",
    # 0: "What vehicle is this? Give color, make/model if known, and one notable feature.",
    #   "... Include the license plate number and state if it is legible; write 'plate not visible' if not." 

    # ── Motorcycle / bicycle (class 1) ─────────────────────────────────────────
    1: (
        "Describe this motorcycle or bicycle in 2–3 sentences. Include: type"
        " (sport/cruiser/dirt bike/bicycle/scooter), color, make if recognizable,"
        " rider's helmet color and clothing, any passenger, and direction of travel."
    ),
    # Alternatives:
    # 1: "Describe this motorcycle or bicycle in one sentence: color, type, and rider if visible.",
    # 1: "What kind of two-wheeled vehicle is this, and what does the rider look like?",

    # ── Person (class 2) ───────────────────────────────────────────────────────
    2: (
        "Describe this person in 2–3 sentences. Include: approximate age range and"
        " gender, hair color and length, clothing (shirt/jacket color and style,"
        " pants/skirt color, footwear), any accessories (backpack, hat, bag, phone),"
        " what they are doing, and which direction they are moving."
    ),
    # Alternatives:
    # 2: "Describe this person in one sentence: appearance, clothing color, and what they are doing.",
    # 2: "Describe the person's appearance and behavior. Focus on clothing colors and any items they are carrying.",
}


# ── Performance stats: written to STATS_DIR for monitor.py ───────────────────
class StatsTracker:
    _LATENCY_WINDOW = 20  # rolling window size for VLM latency

    def __init__(self, camera_id, path):
        self._path       = path
        self._camera_id  = camera_id
        self._lock       = threading.Lock()
        self._start      = time.time()
        self._queued     = 0   # events passed to VLM queue
        self._drops      = 0   # events dropped (queue full)
        self._saves      = 0   # events saved to ChromaDB
        self._latencies  = []  # recent VLM latencies in seconds

    def record_queued(self):
        with self._lock:
            self._queued += 1

    def record_drop(self):
        with self._lock:
            self._drops += 1
        self.write(queue_depth=VLM_QUEUE_MAX)  # write immediately on drop

    def record_save(self, latency_s):
        with self._lock:
            self._saves += 1
            self._latencies.append(latency_s)
            if len(self._latencies) > self._LATENCY_WINDOW:
                self._latencies.pop(0)

    def write(self, queue_depth=0):
        now = time.time()
        elapsed = max(now - self._start, 1.0)
        with self._lock:
            lats = self._latencies[:]
            data = {
                "camera_id":    self._camera_id,
                "updated_at":   round(now, 3),
                "elapsed_s":    round(elapsed, 1),
                "queued_total": self._queued,
                "drops_total":  self._drops,
                "saves_total":  self._saves,
                "queue_per_min":  round(self._queued / elapsed * 60, 1),
                "drops_per_min":  round(self._drops  / elapsed * 60, 1),
                "saves_per_min":  round(self._saves   / elapsed * 60, 1),
                "vlm_ms_avg":   round(sum(lats) / len(lats) * 1000) if lats else None,
                "vlm_ms_max":   round(max(lats) * 1000) if lats else None,
                "vlm_ms_last":  round(lats[-1] * 1000) if lats else None,
                "queue_depth":  queue_depth,
            }
        try:
            with open(self._path, "w") as fh:
                json.dump(data, fh)
        except OSError:
            pass


# ── Probe: fires on nvinfer output, queues detections ────────────────────────
class ObjectDetector(BatchMetadataOperator):
    def __init__(self, event_queue, stats):
        super().__init__()
        self._q         = event_queue
        self._stats     = stats
        self._last_save = {}
        self._event_id  = 0

    def handle_metadata(self, batch_meta):
        now = time.time()
        for frame_meta in batch_meta.frame_items:
            source_id = frame_meta.source_id
            pts_ns    = frame_meta.buffer_pts

            for obj_meta in frame_meta.object_items:
                cls = obj_meta.class_id
                if cls not in DETECT_CLASSES:
                    continue
                if obj_meta.confidence < DETECT_MIN_CONF[cls]:
                    continue
                key = (source_id, cls)
                if now - self._last_save.get(key, 0) < SAVE_INTERVAL:
                    continue
                self._last_save[key] = now
                self._event_id += 1
                label = DETECT_CLASSES[cls]
                print(f"[Detect] {label} conf={obj_meta.confidence:.2f} evt={self._event_id}")
                if self._q.qsize() >= VLM_QUEUE_MAX:
                    print(f"[Detect] VLM queue full ({VLM_QUEUE_MAX}), dropping evt={self._event_id}")
                    self._stats.record_drop()
                    break
                now_dt = datetime.datetime.now(tz=LOCAL_TZ)
                self._q.put({
                    "class_id":   cls,
                    "label":      label,
                    "confidence": round(float(obj_meta.confidence), 3),
                    "pts_ns":     int(pts_ns),
                    "source_id":  int(source_id),
                    "wall_time":  now_dt.isoformat(),
                    "wall_time_s": round(now_dt.timestamp(), 3),
                    "event_id":   self._event_id,
                    "queued_at":  now,
                })
                self._stats.record_queued()
                break


# ── Read the JPEG written by multifilesink after `after_time` ─────────────────
def get_jpeg_after(after_time, timeout=2.0):
    """Return bytes of the first JPEG whose mtime is newer than after_time."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        files = glob.glob(JPEG_GLOB)
        fresh = [f for f in files if os.path.getmtime(f) > after_time]
        if fresh:
            path = max(fresh, key=os.path.getmtime)
            try:
                with open(path, "rb") as fh:
                    return fh.read()
            except OSError:
                pass
        time.sleep(0.05)
    return None


# ── Worker: latest frame → Gemma4 description → embed → ChromaDB ─────────────
def vlm_worker(event_queue, stats):
    import base64
    client     = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
    collection = client.get_or_create_collection(COLLECTION)
    print("[VLM Worker] Ready")

    while True:
        det = event_queue.get()
        if det is None:
            break
        t_start = time.time()
        try:
            jpeg_bytes = get_jpeg_after(det["queued_at"])
            if not jpeg_bytes:
                print(f"[VLM] No fresh frame within 2s, skipping evt={det['event_id']}")
                stats.write(queue_depth=event_queue.qsize())
                continue
            jpeg_b64 = base64.b64encode(jpeg_bytes).decode()

            prompt = VLM_PROMPTS.get(det["class_id"], "Describe what you see in one sentence.")
            resp = ollama.chat(
                model    = VLM_MODEL,
                messages = [{"role": "user", "content": prompt, "images": [jpeg_b64]}],
            )
            description = resp["message"]["content"].strip()
            print(f"[VLM] evt={det['event_id']} {det['label']}: {description}")

            embed_resp = ollama.embeddings(model=EMBED_MODEL, prompt=description)
            embedding  = embed_resp["embedding"]

            doc_id = f"cam{CAMERA_ID}_src{det['source_id']}_{det['label']}_evt{det['event_id']}"

            snap_name = f"{doc_id}.jpg"
            snap_path = os.path.join(SNAPSHOT_DIR, snap_name)
            with open(snap_path, "wb") as f:
                f.write(jpeg_bytes)

            collection.add(
                embeddings = [embedding],
                documents  = [description],
                metadatas  = [{
                    "timestamp_s":  round(float(det["pts_ns"] / 1e9), 3),
                    "timestamp_ns": det["pts_ns"],
                    "wall_time":    det["wall_time"],
                    "wall_time_s":  det["wall_time_s"],
                    "camera_id":    CAMERA_ID,
                    "source_id":    det["source_id"],
                    "class_id":     det["class_id"],
                    "label":        det["label"],
                    "confidence":   det["confidence"],
                    "image_path":   f"/snapshots/{snap_name}",
                }],
                ids = [doc_id],
            )
            print(f"[ChromaDB] Saved {doc_id} @ {det['wall_time']}")
            stats.record_save(time.time() - t_start)
        except Exception as e:
            print(f"[VLM Worker] Error evt={det['event_id']}: {e}")
        finally:
            stats.write(queue_depth=event_queue.qsize())


RECONNECT_INTERVAL = 5   # seconds nvurisrcbin waits before reconnecting
RESTART_DELAY      = 10  # seconds to wait before rebuilding the pipeline


# ── Pipeline ──────────────────────────────────────────────────────────────────
def build_pipeline(detector):
    return (
        Pipeline("cam1-vlm-pipeline")
        .add("nvurisrcbin", "src", {
            "uri":                    RTSP_URL,
            "select-rtp-protocol":    RTSP_TRANSPORT,
            "rtsp-reconnect-interval": RECONNECT_INTERVAL,
        })
        .add("nvstreammux", "mux", {
            "batch-size":           1,
            "width":                FRAME_W,
            "height":               FRAME_H,
            "batched-push-timeout": 40000,
            "live-source":          1,
        })
        .add("nvinfer", "infer", {"config-file-path": "pgie_config.yml"})
        .add("nvjpegenc", "encoder", {"quality": 85})
        .add("multifilesink", "sink", {
            "location":  JPEG_PATH,
            "max-files": 2,
            "async":     0,
            "sync":      0,
        })
        .link(("src", "mux"), ("", "sink_%u"))
        .link("mux", "infer", "encoder", "sink")
        .attach("infer", Probe("detector", detector))
    )


def main():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)

    stats_path = os.path.join(STATS_DIR, f"cam{CAMERA_ID}_stats.json")
    stats      = StatsTracker(CAMERA_ID, stats_path)

    event_queue = queue.Queue()
    detector    = ObjectDetector(event_queue, stats)

    worker = threading.Thread(
        target=vlm_worker, args=(event_queue, stats), daemon=True
    )
    worker.start()

    try:
        while True:
            print(f"[Main] Starting pipeline for cam{CAMERA_ID}...")
            pipeline = build_pipeline(detector)
            try:
                pipeline.start().wait()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"[Main] Pipeline error: {e}")
            print(f"[Main] Pipeline stopped, restarting in {RESTART_DELAY}s...")
            time.sleep(RESTART_DELAY)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        event_queue.put(None)
        worker.join()


if __name__ == "__main__":
    main()
