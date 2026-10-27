#!/usr/bin/env python3
"""Offline unit tests for Phase 1:
1. Main-stream URL derivation (streams_config.derive_main_urls)
2. Street cooldown fix (on_vlm_reject releases camera throttle immediately,
   suppresses repeat static low-conf bbox for STREET_REJECT_BBOX_TTL_S, and
   allows distinct moving/high-conf vehicles to bypass an unconfirmed pending FP)
3. MainStreamRingBuffer + _save_vehicle_event for cam3-5 (with main stream)
   and cam6 (substream only), writing crops and logs/vehicle_events.jsonl.
"""
import json
import os
import queue
import shutil
import sys
import tempfile
import time

os.environ.setdefault("LOG_DIR", "/tmp/vlm_selftest")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import streams_config
import pipeline_multi as pm


class _DummyRect:
    def __init__(self, left, top, width, height):
        self.left = left
        self.top = top
        self.width = width
        self.height = height


class _DummyObj:
    def __init__(self, class_id, confidence, object_id, rect):
        self.class_id = class_id
        self.confidence = confidence
        self.object_id = object_id
        self.rect_params = rect


class _DummyFrame:
    def __init__(self, source_id, pts, objects, w=640, h=480):
        self.source_id = source_id
        self.buffer_pts = pts
        self.object_items = objects
        self.pipeline_width = w
        self.pipeline_height = h


class _DummyBatch:
    def __init__(self, frames):
        self.frame_items = frames


def test_derive_main_urls():
    # Reolink (_sub -> h265Preview_01_main, h264Preview_01_main)
    u3 = "rtsp://user:secret@192.168.1.50:554/h264Preview_01_sub"
    m3 = streams_config.derive_main_urls(u3)
    assert len(m3) == 2, f"expected 2 candidates for Reolink, got {m3}"
    assert m3[0].endswith("/h265Preview_01_main")
    assert m3[1].endswith("/h264Preview_01_main")

    # Amcrest/Lorex (subtype=1 -> subtype=0)
    u4 = "rtsp://admin:secret@192.168.1.51:554/cam/realmonitor?channel=1&subtype=1"
    m4 = streams_config.derive_main_urls(u4)
    assert m4 == ("rtsp://admin:secret@192.168.1.51:554/cam/realmonitor?channel=1&subtype=0",)

    # Cam6 (single stream -> empty)
    u6 = "rtsp://admin:secret@192.168.1.52:554/live/ch0"
    assert streams_config.derive_main_urls(u6) == ()

    # Explicit override and explicit disable
    assert streams_config.derive_main_urls(u4, "none") == ()
    assert streams_config.derive_main_urls(u6, "rtsp://x/main") == ("rtsp://x/main",)
    print("  [PASS] test_derive_main_urls")


def test_street_cooldown_and_static_fp_suppression():
    streams = [
        {"camera_id": 4, "source_index": 0, "url": "rtsp://127.0.0.1/sub",
         "cam_type": "street", "is_office": False, "main_urls": ()},
    ]
    q = queue.Queue()
    stats = pm.StatsRegistry(streams)
    det_op = pm.ObjectDetector(q, stats, streams)
    car_slot = pm.CLASS_SLOTS["car"]

    # Step 1: Bush false positive on cam4 (track_id=10, conf=0.55, static box at [10,20,60,50])
    bush_rect = _DummyRect(10, 20, 60, 50)
    for step in range(pm.MIN_TRACK_HITS):
        b = _DummyBatch([_DummyFrame(0, 1000 + step, [_DummyObj(car_slot, 0.55, 10, bush_rect)])])
        det_op.handle_metadata(b)

    assert q.qsize() == 1, f"expected bush event queued, got {q.qsize()}"
    ev_bush = q.get_nowait()
    assert ev_bush["track_id"] == 10
    assert ev_bush["bbox_norm"] is not None

    # Step 2: VLM rejects the bush false positive -> cooldown must be released!
    det_op.on_vlm_reject(ev_bush)

    # Step 3: 0.5s later, a flicker track (track_id=11, conf=0.56) at the SAME static bush bbox
    # should be suppressed by _is_static_rejected_fp!
    for step in range(pm.MIN_TRACK_HITS):
        b = _DummyBatch([_DummyFrame(0, 2000 + step, [_DummyObj(car_slot, 0.56, 11, bush_rect)])])
        det_op.handle_metadata(b)
    assert q.qsize() == 0, "expected repeat static bush FP at rejected bbox to be suppressed"
    assert stats.for_camera(4)._counts["static_fp_throttled"] >= 1

    # Step 4: Immediately (still <8s since bush!), a real car (track_id=12, conf=0.81) drives through
    car_rect1 = _DummyRect(200, 140, 160, 110)
    car_rect2 = _DummyRect(240, 145, 165, 112)
    det_op.handle_metadata(_DummyBatch([_DummyFrame(0, 3000, [_DummyObj(car_slot, 0.78, 12, car_rect1)])]))
    det_op.handle_metadata(_DummyBatch([_DummyFrame(0, 3100, [_DummyObj(car_slot, 0.84, 12, car_rect2)])]))
    assert q.qsize() == 1, "real car after VLM-rejected bush must NOT be blocked by street cooldown!"
    ev_car = q.get_nowait()
    assert ev_car["track_id"] == 12

    # Confirm the real car -> now street cooldown IS locked in for STREET_SAVE_INTERVAL
    det_op.on_vlm_confirm(ev_car)
    print("  [PASS] test_street_cooldown_and_static_fp_suppression")


def test_ring_buffer_and_vehicle_event_saving():
    tmp = tempfile.mkdtemp(prefix="p1_veh_test_")
    old_dir = pm.VEHICLE_SNAPSHOT_DIR
    old_log = pm.VEHICLE_EVENTS_LOG
    try:
        pm.VEHICLE_SNAPSHOT_DIR = os.path.join(tmp, "vehicles")
        pm.VEHICLE_EVENTS_LOG = os.path.join(tmp, "vehicle_events.jsonl")

        streams = [
            {"camera_id": 4, "source_index": 0, "url": "rtsp://127.0.0.1/c4_sub",
             "cam_type": "street", "is_office": False, "main_urls": ("rtsp://127.0.0.1/c4_main",)},
            {"camera_id": 6, "source_index": 1, "url": "rtsp://127.0.0.1/c6_sub",
             "cam_type": "street", "is_office": False, "main_urls": ()},
        ]
        stats = pm.StatsRegistry(streams)
        ring = pm.MainStreamRingBuffer(streams, stats_registry=stats)

        t0 = time.time()
        # Populate main-stream ring buffer for cam4 (2592x1944 synthetic frame)
        main_img = np.full((1944, 2592, 3), (40, 90, 180), dtype=np.uint8)
        cv2.rectangle(main_img, (800, 600), (1600, 1200), (220, 220, 220), -1)
        ring.put_frame(4, t0 - 0.2, main_img)
        ring.put_frame(4, t0 + 0.2, main_img)

        # Substream JPEG (640x480)
        sub_img = np.full((480, 640, 3), (60, 120, 40), dtype=np.uint8)
        ok, enc = cv2.imencode(".jpg", sub_img)
        assert ok
        jpeg_bytes = enc.tobytes()

        # Test cam4 (has main stream)
        det4 = {
            "class_id": pm.CLASS_SLOTS["car"],
            "label": "car",
            "confidence": 0.86,
            "pts_ns": int(t0 * 1e9),
            "source_id": 0,
            "camera_id": 4,
            "is_office": False,
            "track_id": 42,
            "bbox_norm": [0.30, 0.30, 0.62, 0.62],
            "track_entry": {
                "samples": [
                    (t0 - 0.2, 0.78, [0.28, 0.30, 0.58, 0.60]),
                    (t0 + 0.15, 0.91, [0.31, 0.31, 0.62, 0.62]),
                ]
            },
            "pinned_main": [],
            "wall_time": "2026-10-26T17:30:00-07:00",
            "wall_time_s": round(t0, 3),
            "event_id": 101,
            "queued_at": t0,
        }
        doc4 = "cam4_src0_car_evt101_test"
        meta4 = pm._save_vehicle_event(
            det4, doc4, f"{doc4}.jpg", jpeg_bytes,
            "A white SUV traveling east on the street.", "vllm",
            main_ring=ring, stats=stats.for_camera(4),
        )
        assert "main_crop_path" in meta4 and "sub_crop_path" in meta4
        assert os.path.isfile(os.path.join(pm.VEHICLE_SNAPSHOT_DIR, f"{doc4}_main.jpg"))
        assert os.path.isfile(os.path.join(pm.VEHICLE_SNAPSHOT_DIR, f"{doc4}_main_crop.jpg"))
        assert os.path.isfile(os.path.join(pm.VEHICLE_SNAPSHOT_DIR, f"{doc4}_sub_crop.jpg"))

        # Test cam6 (substream only, no main stream)
        det6 = dict(det4, camera_id=6, source_id=1, event_id=202, track_id=77)
        doc6 = "cam6_src1_car_evt202_test"
        meta6 = pm._save_vehicle_event(
            det6, doc6, f"{doc6}.jpg", jpeg_bytes,
            "A dark sedan driving down the road.", "vllm",
            main_ring=ring, stats=stats.for_camera(6),
        )
        assert "sub_crop_path" in meta6 and "main_crop_path" not in meta6
        assert os.path.isfile(os.path.join(pm.VEHICLE_SNAPSHOT_DIR, f"{doc6}_sub_crop.jpg"))

        with open(pm.VEHICLE_EVENTS_LOG) as fh:
            lines = [json.loads(line) for line in fh if line.strip()]
        assert len(lines) == 2
        assert lines[0]["cam"] == 4 and lines[0]["main_crop_shape"][0] > 500
        assert lines[1]["cam"] == 6 and lines[1]["main_crop_path"] is None and lines[1]["sub_crop_shape"][0] > 100
        print("  [PASS] test_ring_buffer_and_vehicle_event_saving")
    finally:
        pm.VEHICLE_SNAPSHOT_DIR = old_dir
        pm.VEHICLE_EVENTS_LOG = old_log
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_derive_main_urls()
    test_street_cooldown_and_static_fp_suppression()
    test_ring_buffer_and_vehicle_event_saving()
    print("All Phase 1 vehicle tests passed.")
