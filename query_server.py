"""
FastAPI server to query ChromaDB vision events by natural language.

Serves the search UI at / and static snapshots at /snapshots/*.

Usage:
    python3 query_server.py

Endpoints:
    GET /               search UI (search.html)
    GET /query          JSON search results
    GET /snapshots/...  snapshot images
"""

import datetime
import os

import chromadb
import ollama
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo("America/Los_Angeles")
except Exception:
    LOCAL_TZ = datetime.timezone(datetime.timedelta(hours=-7))  # PDT fallback

CHROMADB_HOST  = "localhost"
CHROMADB_PORT  = 8000
COLLECTION_NAME = "vision_events"       # per-object detections (search substrate)
SEGMENT_COLLECTION = "vision_segments"  # per-10/30s "what happened" summaries (option c)
OLLAMA_MODEL   = "nomic-embed-text"
SNAPSHOT_DIR   = "snapshots"
SEARCH_HTML    = "search.html"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs("/tmp/hls", exist_ok=True)

app = FastAPI(title="Vision Query API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"],
)
app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")
app.mount("/hls", StaticFiles(directory="/tmp/hls", html=True), name="hls")

chroma_client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)


def parse_local_dt(s: str):
    """Parse a datetime-local string (no tz) as PDT → Unix timestamp."""
    if not s:
        return None
    try:
        dt = datetime.datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=LOCAL_TZ)
        return dt.timestamp()
    except ValueError:
        return None


def build_where(start_time: str, end_time: str, label: str, camera_id: str = ""):
    conditions = []
    start_ts = parse_local_dt(start_time)
    end_ts   = parse_local_dt(end_time)
    if start_ts is not None:
        conditions.append({"wall_time_s": {"$gte": start_ts}})
    if end_ts is not None:
        conditions.append({"wall_time_s": {"$lte": end_ts}})
    if label:
        conditions.append({"label": {"$eq": label}})
    if camera_id:
        conditions.append({"camera_id": {"$eq": int(camera_id)}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def fmt(doc_id, doc, meta, distance=None):
    return {
        "id":          doc_id,
        "kind":        "object",
        "wall_time":   meta.get("wall_time"),
        "camera_id":   meta.get("camera_id"),
        "label":       meta.get("label"),
        "confidence":  round(float(meta.get("confidence") or 0), 3),
        "document":    doc,
        "distance":    round(float(distance), 3) if distance is not None else None,
        "image_url":   meta.get("image_path"),
        "wall_time_s": round(float(meta.get("wall_time_s") or 0), 3),
    }


def fmt_segment(doc_id, doc, meta, distance=None):
    """Format a vision_segments hit. Segments have a time window instead of a
    snapshot/label/confidence."""
    return {
        "id":          doc_id,
        "kind":        "segment",
        "wall_time":   meta.get("wall_time"),
        "camera_id":   meta.get("camera_id"),
        "label":       "segment",
        "confidence":  None,
        "document":    doc,
        "distance":    round(float(distance), 3) if distance is not None else None,
        "image_url":   None,
        "wall_time_s": round(float(meta.get("wall_time_s") or 0), 3),
        "start_s":     meta.get("start_s"),
        "end_s":       meta.get("end_s"),
        "duration_s":  meta.get("duration_s"),
    }


def build_where_segment(start_time: str, end_time: str, camera_id: str = ""):
    """Where-filter for segments: time + camera only (segments have no label)."""
    conditions = []
    start_ts = parse_local_dt(start_time)
    end_ts   = parse_local_dt(end_time)
    if start_ts is not None:
        conditions.append({"wall_time_s": {"$gte": start_ts}})
    if end_ts is not None:
        conditions.append({"wall_time_s": {"$lte": end_ts}})
    if camera_id:
        conditions.append({"camera_id": {"$eq": int(camera_id)}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def _search_collection(collection, text, search_type, where, n, formatter, embedding=None):
    """Run exact / semantic / browse search over one collection; return formatted
    rows. `embedding` is precomputed for semantic search so we embed once."""
    out = []
    t = text.strip()
    if t and search_type == "exact":
        kwargs = {"where_document": {"$contains": t}, "limit": n,
                  "include": ["documents", "metadatas"]}
        if where:
            kwargs["where"] = where
        res = collection.get(**kwargs)
        for i, doc_id in enumerate(res["ids"]):
            out.append(formatter(doc_id, res["documents"][i], res["metadatas"][i]))
    elif t:
        kwargs = {"query_embeddings": [embedding], "n_results": n}
        if where:
            kwargs["where"] = where
        res = collection.query(**kwargs)
        for i, doc_id in enumerate(res["ids"][0]):
            out.append(formatter(doc_id, res["documents"][0][i],
                                 res["metadatas"][0][i], res["distances"][0][i]))
    else:
        kwargs = {"limit": n, "include": ["documents", "metadatas"]}
        if where:
            kwargs["where"] = where
        res = collection.get(**kwargs)
        for i, doc_id in enumerate(res["ids"]):
            out.append(formatter(doc_id, res["documents"][i], res["metadatas"][i]))
    return out


@app.get("/count")
def count(
    start_time: str = "",
    end_time: str = "",
    label: str = "",
    camera_id: str = "",
):
    """Count events matching filters, broken down by label and camera."""
    where = build_where(start_time, end_time, label, camera_id)

    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ChromaDB error: {e}")

    kwargs = {"include": ["metadatas"]}
    if where:
        kwargs["where"] = where
    try:
        results = collection.get(**kwargs)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ChromaDB error: {e}")

    by_label  = {}
    by_camera = {}
    for meta in results["metadatas"]:
        lbl = meta.get("label") or "unknown"
        cam = str(meta.get("camera_id") or "?")
        by_label[lbl]   = by_label.get(lbl, 0) + 1
        by_camera[cam]  = by_camera.get(cam, 0) + 1

    return {
        "total":     len(results["ids"]),
        "by_label":  dict(sorted(by_label.items())),
        "by_camera": dict(sorted(by_camera.items())),
    }


@app.get("/")
def serve_search():
    return FileResponse(SEARCH_HTML)


@app.get("/query")
def query(
    text: str = "",
    n: int = 20,
    start_time: str = "",
    end_time: str = "",
    sort_by: str = "relevance",
    label: str = "",
    camera_id: str = "",
    search_type: str = "semantic",
    sources: str = "both",   # "objects", "segments", or "both"
):
    t = text.strip()

    # A label filter implies object search (segments have no label).
    want_objects  = sources in ("both", "objects")
    want_segments = sources in ("both", "segments") and not label

    # Embed once (semantic) and reuse for both collections.
    embedding = None
    if t and search_type != "exact":
        try:
            embedding = ollama.embeddings(model=OLLAMA_MODEL, prompt=t)["embedding"]
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Ollama error: {e}")

    output = []

    if want_objects:
        try:
            col = chroma_client.get_collection(COLLECTION_NAME)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"ChromaDB error: {e}")
        where = build_where(start_time, end_time, label, camera_id)
        try:
            output += _search_collection(col, text, search_type, where, n, fmt, embedding)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"ChromaDB query error: {e}")

    if want_segments:
        # The segments collection may not exist yet (sidecar never run) — skip quietly.
        try:
            seg_col = chroma_client.get_collection(SEGMENT_COLLECTION)
        except Exception:
            seg_col = None
        if seg_col is not None:
            where_seg = build_where_segment(start_time, end_time, camera_id)
            try:
                output += _search_collection(seg_col, text, search_type, where_seg,
                                             n, fmt_segment, embedding)
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"ChromaDB segment query error: {e}")

    # Merge/sort across both collections. Distances are comparable (same embedder).
    if sort_by == "time_desc":
        output.sort(key=lambda x: x["wall_time_s"] or 0, reverse=True)
    elif sort_by == "time_asc":
        output.sort(key=lambda x: x["wall_time_s"] or 0)
    elif t and search_type != "exact":
        # relevance: nearest first; rows without a distance (browse) go last
        output.sort(key=lambda x: x["distance"] if x["distance"] is not None else 1e9)

    output = output[:n]   # respect max results across the merged set
    return {"query": text, "count": len(output), "results": output}


LIVE_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Live Feeds</title>
<style>
body { background:#0f1117; color:#e2e4ef; font-family:system-ui,sans-serif; padding:20px; margin:0; }
h1 { margin-bottom:12px; }
#player-container { max-width:1280px; margin:0 auto; }
video { width:100%; height:auto; background:#000; border-radius:8px; }
.info { color:#8890a8; font-size:0.85rem; margin:8px 0 16px; }
</style>
</head>
<body>
<h1>Live Camera Feeds (tiled + detections)</h1>
<div id="player-container">
  <video id="video" controls autoplay muted playsinline></video>
</div>
<p class="info">HLS stream from the DeepStream pipeline (2x2 tiled view with bounding boxes when LIVE_STREAM=1).</p>
<script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
<script>
const video = document.getElementById('video');
if (Hls.isSupported()) {
  const hls = new Hls({ lowLatencyMode: true });
  hls.loadSource('/hls/stream.m3u8');
  hls.attachMedia(video);
  hls.on(Hls.Events.MANIFEST_PARSED, () => video.play().catch(()=>{}));
} else if (video.canPlayType('application/vnd.apple.mpegurl')) {
  video.src = '/hls/stream.m3u8';
  video.addEventListener('loadedmetadata', () => video.play().catch(()=>{}));
} else {
  document.getElementById('player-container').innerHTML = '<p>Your browser does not support HLS playback.</p>';
}
</script>
</body>
</html>
"""

@app.get("/live")
def live():
    """Simple live video player page."""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=LIVE_HTML)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
