"""
segment_ingest.py — Option (c) segment sidecar (C1).

Runs the DeepStream 9.1 nvvllmvlm plugin (Cosmos-Reason2-8B) over the STREET
cameras, and writes each ~10s segment summary to ChromaDB (collection
`vision_segments`), embedded with the same nomic-embed-text model the per-object
path uses. This is the "what happened in this window" index that complements the
per-object `vision_events` search index — the two are kept separate on purpose.

Runs INSIDE the vllm-ds-spike image (has vLLM + torch + the plugin). Reuses the
sample app's pipeline machinery and only swaps the Kafka sink for a Chroma sink.

  cd /workspace/segments && python3 segment_ingest.py [--converter-mode gpu]

Street cameras + URLs come from .env via streams_config (CAM_TYPE_CAMn=street).
"""

import datetime
import os
import sys
import time

import chromadb
import ollama

# The plugin + sample app live in the mounted plugin dir; put them on the path.
PLUGIN_DIR = os.environ.get("VLLM_PLUGIN_DIR", "/home/vllm_ds_plugin")
sys.path.insert(0, PLUGIN_DIR)
sys.path.insert(0, "/workspace")  # streams_config.py

import gstnvvllmvlm  # noqa: E402,F401  (registers the nvvllmvlm GStreamer element)
from vllm_ds_app_kafka_publish import VLMKafkaApp  # noqa: E402
from streams_config import load_streams  # noqa: E402

try:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo("America/Los_Angeles")
except Exception:
    LOCAL_TZ = datetime.timezone(datetime.timedelta(hours=-7))

CHROMADB_HOST = os.environ.get("CHROMADB_HOST", "localhost")
CHROMADB_PORT = int(os.environ.get("CHROMADB_PORT", "8000"))
COLLECTION    = os.environ.get("SEGMENT_COLLECTION", "vision_segments")
EMBED_MODEL   = os.environ.get("EMBED_MODEL", "nomic-embed-text")


class VLMChromaSink:
    """Drop-in replacement for the sample app's Kafka publisher: embeds each
    segment description and writes it to ChromaDB. Must expose on_vlm_result
    (the vlm-result signal handler) and close()."""

    def __init__(self, stream_to_camera):
        self.stream_to_camera = stream_to_camera
        self.client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
        self.collection = self.client.get_or_create_collection(COLLECTION)
        self.saved = 0
        self.errors = 0
        print(f"[Segment] Chroma sink ready (collection={COLLECTION}, "
              f"{CHROMADB_HOST}:{CHROMADB_PORT})")

    def on_vlm_result(self, element, stream_id, start_time, end_time, result_text):
        cam = self.stream_to_camera.get(int(stream_id), int(stream_id) + 1)
        desc = (result_text or "").strip()
        if not desc:
            return
        try:
            emb = ollama.embeddings(model=EMBED_MODEL, prompt=desc)["embedding"]
            now = datetime.datetime.now(tz=LOCAL_TZ)
            doc_id = f"seg_cam{cam}_{int(now.timestamp() * 1000)}"
            self.collection.add(
                embeddings=[emb],
                documents=[desc],
                metadatas=[{
                    "camera_id":   cam,
                    "kind":        "segment",
                    "start_s":     round(float(start_time), 2),
                    "end_s":       round(float(end_time), 2),
                    "duration_s":  round(float(end_time) - float(start_time), 2),
                    "wall_time":   now.isoformat(),
                    "wall_time_s": round(now.timestamp(), 3),
                }],
                ids=[doc_id],
            )
            self.saved += 1
            print(f"[Segment] cam{cam} {start_time:.1f}-{end_time:.1f}s "
                  f"SAVED: {desc[:90]}")
        except Exception as e:
            self.errors += 1
            print(f"[Segment] cam{cam} ERROR: {e}")

    def close(self):
        print(f"[Segment] done — saved={self.saved} errors={self.errors}")


def main():
    converter_mode = "gpu"
    if "--converter-mode" in sys.argv:
        converter_mode = sys.argv[sys.argv.index("--converter-mode") + 1]

    streams = [s for s in load_streams() if not s.get("is_office", False)]
    if not streams:
        print("[Segment] No street cameras configured (CAM_TYPE_CAMn=street). Exiting.")
        return
    uris = [s["url"] for s in streams]
    stream_to_camera = {i: s["camera_id"] for i, s in enumerate(streams)}
    print(f"[Segment] Street cameras: "
          + ", ".join(f"stream{i}->cam{s['camera_id']}" for i, s in enumerate(streams)))

    # Reuse the sample app's pipeline; swap its sink for the Chroma sink.
    app = VLMKafkaApp(
        input_uris=uris,
        kafka_config={},
        topic="unused",
        dry_run=True,           # keeps the Kafka publisher inert; we replace it
        converter_mode=converter_mode,
    )
    app.kafka_publisher = VLMChromaSink(stream_to_camera)
    app.run()


if __name__ == "__main__":
    main()
