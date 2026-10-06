"""
FastAPI server to query ChromaDB vision events by natural language.

Serves the search UI at / and static snapshots at /snapshots/*.

Usage:
    python3 query_server.py                       # :8001, collection $QUERY_COLLECTION,
                                                  # else $CHROMA_COLLECTION, else vision_events
    QUERY_PORT=8002 QUERY_COLLECTION=vision_events_v2 python3 query_server.py

Any page/endpoint also takes ?collection=<name> (allow-listed below), e.g.
http://thor2:8001/?collection=vision_events_v3 shows the nomic-prefix index
side by side without changing the default view.

Endpoints:
    GET /               search UI (search.html)
    GET /query          JSON search results
    GET /collections    allow-listed collections + counts (page switcher)
    GET /snapshots/...  snapshot images
"""

import datetime
import os
import time

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
# Default per-object collection (search substrate). ?collection= may pick
# another one from ALLOWED_COLLECTIONS (side-by-side review of a rebuild).
# Follows CHROMA_COLLECTION (the collection pipeline_multi.py writes to)
# unless QUERY_COLLECTION overrides it.
COLLECTION_NAME = (os.environ.get("QUERY_COLLECTION")
                   or os.environ.get("CHROMA_COLLECTION", "vision_events"))
ALLOWED_COLLECTIONS = {COLLECTION_NAME, "vision_events", "vision_events_v2",
                       "vision_events_v3"} | {
    c.strip() for c in os.environ.get("QUERY_COLLECTIONS_EXTRA", "").split(",") if c.strip()}
QUERY_PORT     = int(os.environ.get("QUERY_PORT", "8001"))
SEGMENT_COLLECTION = "vision_segments"  # per-10/30s "what happened" summaries (option c)
OLLAMA_MODEL   = "nomic-embed-text"
# Embedding host: EMBED_HOST, else OLLAMA_HOST, else the ollama client default
# (same resolution as pipeline_multi.py, so queries and stored events always
# use the same embedder).
EMBED_HOST     = (os.environ.get("EMBED_HOST") or os.environ.get("OLLAMA_HOST") or "").strip() or None
# nomic-embed-text task prefixes. Live default stays "none" (v2 was built
# without them). vision_events_v3 (and any name in QUERY_PREFIXED_COLLECTIONS)
# always get search_query: on semantic search. When EMBED_PREFIX_STYLE=nomic,
# the default CHROMA/QUERY collection also uses prefixes (for the later cutover).
_EMBED_PREFIX_STYLE = os.environ.get("EMBED_PREFIX_STYLE", "none").strip().lower()
if _EMBED_PREFIX_STYLE in ("nomic", "nomic-embed-text"):
    EMBED_QUERY_PREFIX = os.environ.get("EMBED_QUERY_PREFIX", "search_query: ")
    EMBED_DOC_PREFIX = os.environ.get("EMBED_DOC_PREFIX", "search_document: ")
else:
    EMBED_QUERY_PREFIX = os.environ.get("EMBED_QUERY_PREFIX", "")
    EMBED_DOC_PREFIX = os.environ.get("EMBED_DOC_PREFIX", "")
PREFIXED_COLLECTIONS = {
    c.strip() for c in os.environ.get(
        "QUERY_PREFIXED_COLLECTIONS", "vision_events_v3").split(",") if c.strip()}
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
embed_client  = ollama.Client(host=EMBED_HOST)
print(f"[query_server] embeddings: {OLLAMA_MODEL} @ {EMBED_HOST or 'ollama default'} "
      f"style={_EMBED_PREFIX_STYLE!r} prefixed={sorted(PREFIXED_COLLECTIONS)}", flush=True)


def _collection_uses_nomic_prefixes(coll_name: str) -> bool:
    """True when stored vectors in this object collection used document prefixes."""
    if coll_name in PREFIXED_COLLECTIONS:
        return True
    if _EMBED_PREFIX_STYLE in ("nomic", "nomic-embed-text") and coll_name == COLLECTION_NAME:
        return True
    return False


def _embed_query(text: str, coll_name: str):
    """Embed a search query; prepend search_query: when the target collection is prefixed."""
    prefix = ""
    if _collection_uses_nomic_prefixes(coll_name):
        prefix = EMBED_QUERY_PREFIX or "search_query: "
    prompt = (prefix + text) if prefix else text
    return embed_client.embeddings(model=OLLAMA_MODEL, prompt=prompt)["embedding"]


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


def _objects_collection(name: str = ""):
    """The object collection to search: ?collection= if allow-listed, else
    the default. Raises 400 for an unknown name, 503 if Chroma fails."""
    name = (name or COLLECTION_NAME).strip()
    if name not in ALLOWED_COLLECTIONS:
        raise HTTPException(status_code=400,
                            detail=f"unknown collection {name!r}; allowed: {sorted(ALLOWED_COLLECTIONS)}")
    try:
        return chroma_client.get_collection(name)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ChromaDB error ({name}): {e}")


def _and(*conds):
    conds = [c for c in conds if c]
    if not conds:
        return None
    flat = []
    for c in conds:
        flat.extend(c["$and"] if "$and" in c else [c])
    return flat[0] if len(flat) == 1 else {"$and": flat}


def _get_all(collection, where=None, include=("metadatas",), page=5000):
    """collection.get() in pages: an unpaged get over ~30k+ rows fails in
    Chroma with "too many SQL variables"."""
    ids, metas, docs, off = [], [], [], 0
    while True:
        kw = {"include": list(include), "limit": page, "offset": off}
        if where:
            kw["where"] = where
        res = collection.get(**kw)
        if not res["ids"]:
            break
        ids += res["ids"]
        metas += res.get("metadatas") or [None] * len(res["ids"])
        docs += res.get("documents") or [None] * len(res["ids"])
        off += len(res["ids"])
        if len(res["ids"]) < page:
            break
    return {"ids": ids, "metadatas": metas, "documents": docs}


# Browse window growth: 1h, 4h, ... 4**6 h (~170 days), then one paged scan.
_BROWSE_STEPS = 7


def _browse(collection, where, n, formatter, oldest_first=False, end_ts=None, start_ts=None):
    """Empty-text browse sorted by wall_time_s ON THE SERVER (Chroma's get()
    has no ORDER BY and returns rows in storage order, i.e. oldest first).
    Newest-first: fetch only metadata over a growing time window ending at
    end_ts/now until n rows are found, then documents for just the top n.
    Oldest-first: same, with a window growing forward from start_ts (or the
    whole collection if no start)."""
    def rows_in(window_where):
        res = _get_all(collection, _and(where, window_where))
        return list(zip(res["ids"], res["metadatas"]))

    rows = None
    if not oldest_first:
        anchor = end_ts if end_ts is not None else time.time() + 3600
        span = 3600.0
        for _ in range(_BROWSE_STEPS):
            lo = anchor - span
            if start_ts is not None and lo <= start_ts:
                break
            rows = rows_in({"wall_time_s": {"$gte": lo}} if end_ts is None else
                           {"$and": [{"wall_time_s": {"$gte": lo}},
                                     {"wall_time_s": {"$lte": anchor}}]})
            if len(rows) >= n:
                break
            rows = None
            span *= 4
    elif start_ts is not None:
        span = 3600.0
        for _ in range(_BROWSE_STEPS):
            hi = start_ts + span
            if end_ts is not None and hi >= end_ts:
                break
            rows = rows_in({"$and": [{"wall_time_s": {"$gte": start_ts}},
                                     {"wall_time_s": {"$lte": hi}}]})
            if len(rows) >= n:
                break
            rows = None
            span *= 4
    if rows is None:                               # small/filtered set: take it all
        res = _get_all(collection, where)
        rows = list(zip(res["ids"], res["metadatas"]))
    rows.sort(key=lambda r: float(r[1].get("wall_time_s") or 0), reverse=not oldest_first)
    top = [r[0] for r in rows[:n]]
    if not top:
        return []
    res = collection.get(ids=top, include=["documents", "metadatas"])
    by_id = {i: (d, m) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])}
    return [formatter(i, *by_id[i]) for i in top if i in by_id]


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


def _search_collection(collection, text, search_type, where, n, formatter, embedding=None,
                       oldest_first=False, start_ts=None, end_ts=None):
    """Run exact / semantic / browse search over one collection; return formatted
    rows. `embedding` is precomputed for semantic search so we embed once.
    Browse (empty text) is sorted by time on the server, newest first unless
    oldest_first."""
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
        out = _browse(collection, where, n, formatter, oldest_first=oldest_first,
                      start_ts=start_ts, end_ts=end_ts)
    return out


@app.get("/count")
def count(
    start_time: str = "",
    end_time: str = "",
    label: str = "",
    camera_id: str = "",
    collection: str = "",
):
    """Count events matching filters, broken down by label and camera."""
    where = build_where(start_time, end_time, label, camera_id)
    coll_name = collection
    collection = _objects_collection(coll_name)

    try:
        results = _get_all(collection, where)
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
        "collection": collection.name,
        "total":     len(results["ids"]),
        "by_label":  dict(sorted(by_label.items())),
        "by_camera": dict(sorted(by_camera.items())),
    }


@app.get("/")
def serve_search():
    # no-cache: browsers otherwise heuristically cache the page and keep
    # running stale JS after search.html changes.
    return FileResponse(SEARCH_HTML, headers={"Cache-Control": "no-cache"})


@app.get("/collections")
def collections():
    """Allow-listed object collections (for the page's switcher) with counts;
    count is None if the collection doesn't exist / Chroma is unreachable."""
    out = []
    for name in sorted(ALLOWED_COLLECTIONS):
        try:
            cnt = chroma_client.get_collection(name).count()
        except Exception:
            cnt = None
        out.append({"name": name, "count": cnt})
    return {"default": COLLECTION_NAME, "collections": out}


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
    collection: str = "",    # object collection (default COLLECTION_NAME)
):
    t = text.strip()
    oldest_first = sort_by == "time_asc"
    start_ts, end_ts = parse_local_dt(start_time), parse_local_dt(end_time)

    # A label filter implies object search (segments have no label).
    want_objects  = sources in ("both", "objects")
    want_segments = sources in ("both", "segments") and not label

    # Resolve object collection early so the query embedding can use the
    # matching nomic prefix (v3) or none (v2 / legacy).
    coll_name = (collection or COLLECTION_NAME).strip()
    embedding = None
    if t and search_type != "exact":
        try:
            embedding = _embed_query(t, coll_name)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Ollama error: {e}")

    output = []

    col = _objects_collection(collection) if want_objects else None
    if want_objects:
        where = build_where(start_time, end_time, label, camera_id)
        try:
            output += _search_collection(col, text, search_type, where, n, fmt, embedding,
                                         oldest_first, start_ts, end_ts)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"ChromaDB query error: {e}")

    if want_segments:
        # vision_segments (when present) was built without nomic prefixes. Do
        # not mix it into a prefixed object-collection search — the query
        # vector would be in the wrong space. Skip quietly; a matching
        # vision_segments_v3 can be added later.
        if _collection_uses_nomic_prefixes(coll_name):
            seg_col = None
        else:
            try:
                seg_col = chroma_client.get_collection(SEGMENT_COLLECTION)
            except Exception:
                seg_col = None
        if seg_col is not None:
            where_seg = build_where_segment(start_time, end_time, camera_id)
            try:
                output += _search_collection(seg_col, text, search_type, where_seg,
                                             n, fmt_segment, embedding,
                                             oldest_first, start_ts, end_ts)
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"ChromaDB segment query error: {e}")

    # Merge/sort across collections. Distances are comparable when both used
    # the same embed prompt style (unprefixed objects+segments).
    if sort_by == "time_desc":
        output.sort(key=lambda x: x["wall_time_s"] or 0, reverse=True)
    elif sort_by == "time_asc":
        output.sort(key=lambda x: x["wall_time_s"] or 0)
    elif t and search_type != "exact":
        # relevance: nearest first; rows without a distance (browse) go last
        output.sort(key=lambda x: x["distance"] if x["distance"] is not None else 1e9)
    elif not t:
        # browse: newest first (each collection is already server-sorted;
        # this merges objects + segments in time order)
        output.sort(key=lambda x: x["wall_time_s"] or 0, reverse=True)

    output = output[:n]   # respect max results across the merged set
    return {"query": text, "collection": col.name if col is not None else (collection or COLLECTION_NAME), "count": len(output), "results": output}


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
    print(f"[query_server] default collection {COLLECTION_NAME} on :{QUERY_PORT}; "
          f"?collection= allows {sorted(ALLOWED_COLLECTIONS)}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=QUERY_PORT)
