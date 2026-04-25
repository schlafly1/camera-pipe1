"""
Multi-camera DeepStream pipeline (up to 6 cameras).
All cameras share one ChromaDB collection; each event is tagged with source_id.

To add cameras: append RTSP URLs to RTSP_URLS below.
"""

import datetime
import queue
import threading
import time

import chromadb
import ollama
from pyservicemaker import BatchMetadataOperator, Pipeline, Probe

RTSP_URLS = [
    "rtsp://192.168.8.130:554/11",  # camera 0
    # "rtsp://192.168.8.131:554/11",  # camera 1
    # "rtsp://192.168.8.132:554/11",  # camera 2
    # "rtsp://192.168.8.133:554/11",  # camera 3
    # "rtsp://192.168.8.134:554/11",  # camera 4
    # "rtsp://192.168.8.135:554/11",  # camera 5
]

CHROMADB_HOST = "localhost"
CHROMADB_PORT = 8000
COLLECTION_NAME = "vision_events"
OLLAMA_MODEL = "gemma4:26b"
PERSON_CLASS_ID = 2
SAVE_INTERVAL_S = 5.0
STREAM_WIDTH = 1920
STREAM_HEIGHT = 1080

_MODEL_DIR = "/opt/nvidia/deepstream/deepstream/samples/models/Primary_Detector"
_MODEL_ONNX = f"{_MODEL_DIR}/resnet18_trafficcamnet_pruned.onnx"


class PersonDetector(BatchMetadataOperator):
    def __init__(self, event_queue):
        super().__init__()
        self._q = event_queue
        self._last_save = {}
        self._event_id = 0

    def handle_metadata(self, batch_meta):
        now = time.time()
        for frame_meta in batch_meta.frame_items:
            source_id = frame_meta.source_id
            pts_ns = frame_meta.buffer_pts
            pts_s = pts_ns / 1e9
            frame_num = frame_meta.frame_number

            for obj_meta in frame_meta.object_items:
                if obj_meta.class_id != PERSON_CLASS_ID:
                    continue

                conf = obj_meta.confidence
                r = obj_meta.rect_params
                print(
                    f"[PERSON] cam={source_id} frame={frame_num} "
                    f"t={pts_s:.2f}s conf={conf:.2f}"
                )

                if now - self._last_save.get(source_id, 0) < SAVE_INTERVAL_S:
                    break

                self._last_save[source_id] = now
                self._event_id += 1
                wall_time = datetime.datetime.now().isoformat()
                text = (
                    f"Camera {source_id}: Person detected at {pts_s:.2f} seconds "
                    f"with confidence {conf:.2f}, "
                    f"bounding box x={r.left:.0f} y={r.top:.0f} "
                    f"width={r.width:.0f} height={r.height:.0f}"
                )
                self._q.put({
                    "id": f"cam{source_id}_evt{self._event_id}",
                    "text": text,
                    "metadata": {
                        "timestamp_s": float(pts_s),
                        "timestamp_ns": int(pts_ns),
                        "wall_time": wall_time,
                        "frame_number": int(frame_num),
                        "source_id": int(source_id),
                        "confidence": float(conf),
                    },
                })
                break


def embedding_worker(event_queue):
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    print(f"[Writer] Connected to ChromaDB, collection='{COLLECTION_NAME}'")

    while True:
        event = event_queue.get()
        if event is None:
            break
        try:
            resp = ollama.embeddings(model=OLLAMA_MODEL, prompt=event["text"])
            embedding = resp["embedding"]
            collection.add(
                embeddings=[embedding],
                documents=[event["text"]],
                metadatas=[event["metadata"]],
                ids=[event["id"]],
            )
            print(f"[Writer] Saved {event['id']} @ {event['metadata']['wall_time']}")
        except Exception as e:
            print(f"[Writer] Error: {e}")


def main():
    rtsp_urls = RTSP_URLS[:6]
    num_streams = len(rtsp_urls)
    if num_streams == 0:
        print("No RTSP URLs configured.")
        return

    print(f"Starting pipeline with {num_streams} camera(s)...")

    event_queue = queue.Queue()
    worker = threading.Thread(target=embedding_worker, args=(event_queue,), daemon=True)
    worker.start()

    pipeline = Pipeline("multi-cam-pipeline")

    for i, url in enumerate(rtsp_urls):
        pipeline.add("nvurisrcbin", f"src{i}", {"uri": url})

    pipeline.add("nvstreammux", "mux", {
        "batch-size": num_streams,
        "width": STREAM_WIDTH,
        "height": STREAM_HEIGHT,
        "batched-push-timeout": 40000,
        "live-source": 1,
    })

    engine_file = f"{_MODEL_ONNX}_b{num_streams}_gpu0_fp16.engine"
    pipeline.add("nvinfer", "infer", {
        "config-file-path": "pgie_config.yml",
        "batch-size": num_streams,
        "model-engine-file": engine_file,
    })

    pipeline.add("fakesink", "sink", {"sync": 0, "async": 0})

    for i in range(num_streams):
        pipeline.link((f"src{i}", "mux"), ("", "sink_%u"))

    pipeline.link("mux", "infer", "sink")
    pipeline.attach("infer", Probe("person-detector", PersonDetector(event_queue)))

    try:
        pipeline.start().wait()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        event_queue.put(None)
        worker.join()


if __name__ == "__main__":
    main()
