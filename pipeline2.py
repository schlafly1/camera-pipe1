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


# ── Probe: fires on nvinfer output, queues detections ────────────────────────
class ObjectDetector(BatchMetadataOperator):
    def __init__(self, event_queue):
        super().__init__()
        self._q         = event_queue
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
                    break
                self._q.put({
                    "class_id":   cls,
                    "label":      label,
                    "confidence": round(float(obj_meta.confidence), 3),
                    "pts_ns":     int(pts_ns),
                    "source_id":  int(source_id),
                    "wall_time":  datetime.datetime.now(tz=LOCAL_TZ).isoformat(),
                    "wall_time_s": round(datetime.datetime.now(tz=LOCAL_TZ).timestamp(), 3),
                    "event_id":   self._event_id,
                })
                break


# ── Read the latest JPEG written by multifilesink ────────────────────────────
def get_latest_jpeg():
    files = glob.glob(JPEG_GLOB)
    if not files:
        return None
    latest = max(files, key=os.path.getmtime)
    try:
        with open(latest, "rb") as f:
            return f.read()
    except OSError:
        return None


# ── Worker: latest frame → Gemma4 description → embed → ChromaDB ─────────────
def vlm_worker(event_queue):
    import base64
    client     = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
    collection = client.get_or_create_collection(COLLECTION)
    print("[VLM Worker] Ready")

    while True:
        det = event_queue.get()
        if det is None:
            break
        try:
            time.sleep(0.1)  # let multifilesink finish the current write
            jpeg_bytes = get_latest_jpeg()
            if not jpeg_bytes:
                print(f"[VLM] No frame file yet, skipping evt={det['event_id']}")
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
        except Exception as e:
            print(f"[VLM Worker] Error evt={det['event_id']}: {e}")


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

    event_queue = queue.Queue()
    detector    = ObjectDetector(event_queue)

    worker = threading.Thread(
        target=vlm_worker, args=(event_queue,), daemon=True
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
