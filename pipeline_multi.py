"""
pipeline_multi.py - Single DeepStream 9 pipeline for multiple RTSP streams.

One nvstreammux batches all cameras; one nvinfer pass handles every stream.
Detections queue to a shared VLM worker (Ollama on Spark via OLLAMA_HOST).

Replaces the per-camera container model (pipeline2.py + cam1.yml).

Classes detected by TrafficCamNet:
  0=Car  1=TwoWheeler  2=Person  3=RoadSign (skipped)
"""

import datetime
import glob
import json
import os
import queue
import shutil
import tempfile
import threading
import time

import chromadb
import ollama
from pyservicemaker import BatchMetadataOperator, Pipeline, Probe

from streams_config import load_streams

try:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo("America/Los_Angeles")
except Exception:
    LOCAL_TZ = datetime.timezone(datetime.timedelta(hours=-7))

# ── Config ────────────────────────────────────────────────────────────────────
CHROMADB_HOST   = os.environ.get("CHROMADB_HOST", "localhost")
CHROMADB_PORT   = int(os.environ.get("CHROMADB_PORT", "8000"))
COLLECTION      = "vision_events"
VLM_MODEL       = os.environ.get("VLM_MODEL", "gemma4:26b")
EMBED_MODEL     = "nomic-embed-text"
SAVE_INTERVAL   = float(os.environ.get("SAVE_INTERVAL", "30.0"))
# Street cams still throttle per (camera, class) — shorter than office so real
# passing traffic is captured, but a persistent detection (parked car or a
# false positive on shadows/foliage) can't flood the VLM queue every frame.
STREET_SAVE_INTERVAL = float(os.environ.get("STREET_SAVE_INTERVAL", "8.0"))
VLM_QUEUE_MAX   = int(os.environ.get("VLM_QUEUE_MAX", "12"))
# Hard ceiling for street cams, which otherwise queue unconditionally. Keeps a
# stalled worker from growing the queue without bound (memory leak safeguard).
STREET_QUEUE_MAX = int(os.environ.get("STREET_QUEUE_MAX", str(VLM_QUEUE_MAX * 8)))
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
SNAPSHOT_DIR    = "/workspace/snapshots"
STATS_DIR       = "/workspace/stats"
RECONNECT_INTERVAL = 5
RESTART_DELAY      = 10

DETECT_CLASSES  = {0: "car", 1: "motorcycle", 2: "person"}
# Cars raised 0.50 -> 0.75: TrafficCamNet hallucinates "car" on shadows/foliage
# in residential scenes at 0.54-0.70; real cars score ~0.9+. Streets see ~1 car/hr,
# so precision matters far more than recall here.
DETECT_MIN_CONF = {0: 0.75, 1: 0.50, 2: 0.30}

# Per-object dedup (requires nvtracker). Emit one event per unique track id so a
# single passing car = one event instead of one per inference frame.
MIN_TRACK_HITS  = int(os.environ.get("MIN_TRACK_HITS", "2"))  # frames before emit
TRACK_TTL       = 30.0        # forget a track id this long after last seen
UNTRACKED_ID    = 2 ** 63     # tracker ids at/above this are "untracked" sentinels

VLM_PROMPTS = {
    0: (
        "Describe this vehicle in 2-3 sentences. Include: color, body style"
        " (sedan/SUV/truck/van/coupe), make and model if recognizable, approximate"
        " year range, any visible damage or distinctive markings, direction of travel,"
        " and license plate text if legible."
    ),
    1: (
        "Describe this motorcycle or bicycle in 2-3 sentences. Include: type"
        " (sport/cruiser/dirt bike/bicycle/scooter), color, make if recognizable,"
        " rider's helmet color and clothing, any passenger, and direction of travel."
    ),
    2: (
        "Describe this person in 2-3 sentences. Include: approximate age range and"
        " gender, hair color and length, clothing (shirt/jacket color and style,"
        " pants/skirt color, footwear), any accessories (backpack, hat, bag, phone),"
        " what they are doing, and which direction they are moving."
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
_VLM_ABSENT_SUBJECT = {
    0: ("no vehicle", "no vehicles", "no car", "no cars", "not a vehicle"),
    1: ("no motorcycle", "no motorcycles", "no bicycle", "no bicycles",
        "no bike", "no bikes", "no scooter"),
    2: ("no person", "no people", "no individual", "no humans", "no pedestrian"),
}


def _vlm_says_absent(description, class_id):
    """True if the VLM's reply indicates the detected object isn't present.

    Deliberately narrow: matches explicit refusals and subject-specific
    negations ("no vehicles") but NOT incidental negations that appear in
    valid descriptions ("no visible damage", "no passenger", "no backpack").
    """
    d = description.lower()
    if any(m in d for m in _VLM_REFUSAL):
        return True
    return any(s in d for s in _VLM_ABSENT_SUBJECT.get(class_id, ()))


def camera_id_for_source(source_id, streams):
    """Map nvstreammux source_id (0-based) to configured camera_id."""
    for stream in streams:
        if stream["source_index"] == source_id:
            return stream["camera_id"]
    return int(source_id) + 1


# ── Performance stats (one file per camera for monitor.py) ────────────────────
class StatsTracker:
    _LATENCY_WINDOW = 20

    def __init__(self, camera_id, path):
        self._path = path
        self._camera_id = camera_id
        self._lock = threading.Lock()
        self._start = time.time()
        self._queued = 0
        self._drops = 0
        self._saves = 0
        self._latencies = []

    def record_queued(self):
        with self._lock:
            self._queued += 1

    def record_drop(self):
        with self._lock:
            self._drops += 1
        self.write(queue_depth=VLM_QUEUE_MAX)

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
                "camera_id":     self._camera_id,
                "updated_at":    round(now, 3),
                "elapsed_s":     round(elapsed, 1),
                "queued_total":  self._queued,
                "drops_total":   self._drops,
                "saves_total":   self._saves,
                "queue_per_min": round(self._queued / elapsed * 60, 1),
                "drops_per_min": round(self._drops / elapsed * 60, 1),
                "saves_per_min": round(self._saves / elapsed * 60, 1),
                "vlm_ms_avg":    round(sum(lats) / len(lats) * 1000) if lats else None,
                "vlm_ms_max":    round(max(lats) * 1000) if lats else None,
                "vlm_ms_last":   round(lats[-1] * 1000) if lats else None,
                "queue_depth":   queue_depth,
            }
        try:
            with open(self._path, "w") as fh:
                json.dump(data, fh)
        except OSError:
            pass


class StatsRegistry:
    def __init__(self, streams):
        self._trackers = {}
        os.makedirs(STATS_DIR, exist_ok=True)
        for stream in streams:
            cam_id = stream["camera_id"]
            path = os.path.join(STATS_DIR, f"cam{cam_id}_stats.json")
            self._trackers[cam_id] = StatsTracker(cam_id, path)

    def for_camera(self, camera_id):
        return self._trackers[camera_id]


# ── Probe: queue detections from all streams in the batch ─────────────────────
class ObjectDetector(BatchMetadataOperator):
    def __init__(self, event_queue, stats_registry, streams):
        super().__init__()
        self._q = event_queue
        self._stats = stats_registry
        self._streams = streams
        self._last_save = {}
        self._event_ids = {}
        self._track_seen = {}   # (camera_id, track_id) -> {count, emitted, last}

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
        for frame_meta in batch_meta.frame_items:
            source_id = frame_meta.source_id
            camera_id = camera_id_for_source(source_id, self._streams)
            stats = self._stats.for_camera(camera_id)
            pts_ns = frame_meta.buffer_pts

            # Determine if this is an office cam (throttled, droppable)
            # or street cam (always process, protect from drops)
            is_office = False
            for s in self._streams:
                if s["camera_id"] == camera_id:
                    is_office = s.get("is_office", False)
                    break

            for obj_meta in frame_meta.object_items:
                cls = obj_meta.class_id
                if cls not in DETECT_CLASSES:
                    continue
                if obj_meta.confidence < DETECT_MIN_CONF[cls]:
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
                        continue

                key = (camera_id, cls)
                # Throttle per (camera, class) as a backstop against tracker-id
                # churn / untracked frames. Office cams use a long interval;
                # street cams a shorter one so real passing traffic is captured.
                throttle = SAVE_INTERVAL if is_office else STREET_SAVE_INTERVAL
                if now - self._last_save.get(key, 0) < throttle:
                    continue
                self._last_save[key] = now
                if te is not None:
                    te["emitted"] = True
                event_id = self._next_event_id(camera_id)
                label = DETECT_CLASSES[cls]
                print(
                    f"[Detect] cam{camera_id} {label} "
                    f"conf={obj_meta.confidence:.2f} evt={event_id}"
                )
                if self._q.qsize() >= VLM_QUEUE_MAX:
                    if is_office:
                        print(
                            f"[Detect] VLM queue full ({VLM_QUEUE_MAX}), "
                            f"dropping office cam{camera_id} evt={event_id}"
                        )
                        stats.record_drop()
                        break
                    elif self._q.qsize() >= STREET_QUEUE_MAX:
                        # Street cams get priority, but still bail out if the
                        # worker has stalled — otherwise the queue grows without
                        # bound and leaks memory (observed at 120k+ items).
                        print(
                            f"[Detect] VLM queue hard cap ({STREET_QUEUE_MAX}), "
                            f"dropping street cam{camera_id} evt={event_id} "
                            f"(worker stalled?)"
                        )
                        stats.record_drop()
                        break
                    else:
                        # Street cam: queue anyway (events are rare; protect them)
                        print(
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
                stats.record_queued()
                break


def get_jpeg_after(after_time, camera_id=None, timeout=5.0):
    """Return bytes of a reasonably fresh JPEG for the given camera.

    Uses a time window (before and after detection time) to account for
    the snapshot being written slightly before/after the probe fires.
    Prefers per-camera files. Never falls back to the global stream when
    a camera_id is given — this prevents cam3/4 from getting images from
    cam1/2.

    If no file in the "fresh" window for the camera, we fall back to the
    most recent file that exists for that camera (stale frame is better
    than wrong camera or nothing).

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
        return None

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
            return None
        if best_is_stale and camera_id is not None:
            age = time.time() - best_mtime
            print(f"[VLM] Using stale frame for cam{camera_id} (age ~{age:.1f}s)")
        return data
    except Exception:
        # Fallback: try direct read (may race)
        try:
            with open(best_path, "rb") as fh:
                data = fh.read()
            if len(data) > 1000:
                return data
        except OSError:
            pass
    return None


def vlm_worker(event_queue, stats_registry):
    import base64

    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
    collection = client.get_or_create_collection(COLLECTION)
    print(f"[VLM Worker] Ready (model={VLM_MODEL})")

    while True:
        det = event_queue.get()
        if det is None:
            break
        t_start = time.time()
        camera_id = det["camera_id"]
        stats = stats_registry.for_camera(camera_id)
        try:
            jpeg_bytes = get_jpeg_after(det["queued_at"], camera_id=camera_id)
            if not jpeg_bytes:
                print(
                    f"[VLM] No fresh frame within timeout, "
                    f"skipping cam{camera_id} evt={det['event_id']}"
                )
                stats.write(queue_depth=event_queue.qsize())
                continue
            jpeg_b64 = base64.b64encode(jpeg_bytes).decode()

            prompt = VLM_PROMPTS.get(
                det["class_id"], "Describe what you see in one sentence."
            )
            resp = ollama.chat(
                model=VLM_MODEL,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [jpeg_b64],
                }],
            )
            description = resp["message"]["content"].strip()

            # Second-stage verification: if the VLM says the object isn't
            # there, it's a detector false positive — drop it (don't embed
            # or persist an empty-scene "car").
            if _vlm_says_absent(description, det["class_id"]):
                print(
                    f"[VLM] cam{camera_id} evt={det['event_id']} REJECT "
                    f"{det['label']} (VLM sees none): {description[:70]}"
                )
                continue

            print(
                f"[VLM] cam{camera_id} evt={det['event_id']} "
                f"{det['label']}: {description}"
            )

            embed_resp = ollama.embeddings(model=EMBED_MODEL, prompt=description)
            embedding = embed_resp["embedding"]

            doc_id = (
                f"cam{camera_id}_src{det['source_id']}_"
                f"{det['label']}_evt{det['event_id']}"
            )
            snap_name = f"{doc_id}.jpg"
            snap_path = os.path.join(SNAPSHOT_DIR, snap_name)
            with open(snap_path, "wb") as f:
                f.write(jpeg_bytes)

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
                }],
                ids=[doc_id],
            )
            print(f"[ChromaDB] Saved {doc_id} @ {det['wall_time']}")
            stats.record_save(time.time() - t_start)
        except Exception as e:
            print(f"[VLM Worker] Error cam{camera_id} evt={det['event_id']}: {e}")
        finally:
            stats.write(queue_depth=event_queue.qsize())


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
            "max-files": 200,
            "async":     0,
            "sync":      0,
        })
        pipeline.link(q_snap, enc)
        pipeline.link(enc, fsink)

    pipeline.add("nvinfer", "infer", {
        "config-file-path": "pgie_config_multi.yml",
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
            print(f"[Main] Live display ON — {n} streams in {rows}x{cols} tile")

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
            print(f"[Main] Live HLS stream enabled at /hls/stream.m3u8 ({n} streams tiled)")

    else:
        _add_jpeg_branch(pipeline)
        pipeline.link("mux", "infer", "tracker", "encoder")
        print("[Main] Headless mode (set ENABLE_DISPLAY=1 or LIVE_STREAM=1 for live video)")

    # Probe on the tracker (not infer) so obj_meta.object_id is populated.
    pipeline.attach("tracker", Probe("detector", detector))
    return pipeline


def main():
    streams = load_streams()
    n = len(streams)
    print(f"[Main] Starting multi-stream pipeline: {n} camera(s)")
    for s in streams:
        print(f"  cam{s['camera_id']}: {s['url']}")

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    stats_registry = StatsRegistry(streams)
    event_queue = queue.Queue()
    detector = ObjectDetector(event_queue, stats_registry, streams)

    worker = threading.Thread(
        target=vlm_worker,
        args=(event_queue, stats_registry),
        daemon=True,
    )
    worker.start()

    try:
        while True:
            print("[Main] Building pipeline...")
            pipeline = build_pipeline(detector, streams)
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