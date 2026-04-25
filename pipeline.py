"""
Single-camera DeepStream pipeline.
Detects persons via TrafficCamNet and saves detection embeddings to ChromaDB.
Uses Ollama (gemma4:26b) to generate embeddings from text descriptions.

Classes: 0=Car  1=TwoWheeler  2=Person  3=RoadSign
"""

import datetime
import queue
import threading
import time

import chromadb
import ollama
from pyservicemaker import BatchMetadataOperator, Pipeline, Probe

RTSP_URL = "rtsp://192.168.9.130:554/11"
CHROMADB_HOST = "localhost"
CHROMADB_PORT = 8000
COLLECTION_NAME = "vision_events"
OLLAMA_MODEL = "nomic-embed-text"
PERSON_CLASS_ID = 2
SAVE_INTERVAL_S = 5.0  # min seconds between ChromaDB saves per source


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
                    f"[PERSON] frame={frame_num} t={pts_s:.2f}s "
                    f"conf={conf:.2f} "
                    f"bbox=({r.left:.0f},{r.top:.0f} {r.width:.0f}x{r.height:.0f})"
                )

                if now - self._last_save.get(source_id, 0) < SAVE_INTERVAL_S:
                    break

                self._last_save[source_id] = now
                self._event_id += 1
                wall_time = datetime.datetime.now().isoformat()
                text = (
                    f"Person detected at {pts_s:.2f} seconds "
                    f"with confidence {conf:.2f}, "
                    f"bounding box x={r.left:.0f} y={r.top:.0f} "
                    f"width={r.width:.0f} height={r.height:.0f}"
                )
                self._q.put({
                    "id": f"src{source_id}_evt{self._event_id}",
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
                break  # one save per frame per source


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
    event_queue = queue.Queue()
    worker = threading.Thread(target=embedding_worker, args=(event_queue,), daemon=True)
    worker.start()

    pipeline = (
        Pipeline("cam1-pipeline")
        .add("nvurisrcbin", "src", {"uri": RTSP_URL})
        .add("nvstreammux", "mux", {
            "batch-size": 1,
            "width": 1920,
            "height": 1080,
            "batched-push-timeout": 40000,
            "live-source": 1,
        })
        .add("nvinfer", "infer", {"config-file-path": "pgie_config.yml"})
        .add("fakesink", "sink", {"sync": 0, "async": 0})
        .link(("src", "mux"), ("", "sink_%u"))
        .link("mux", "infer", "sink")
        .attach("infer", Probe("person-detector", PersonDetector(event_queue)))
    )

    try:
        pipeline.start().wait()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        event_queue.put(None)
        worker.join()


if __name__ == "__main__":
    main()
