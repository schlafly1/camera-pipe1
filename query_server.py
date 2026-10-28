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
from dotenv import load_dotenv
load_dotenv()
import time

import chromadb
import ollama
import json
from typing import Any, Dict, List, Optional
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import vehicle_reid_worker as vrw
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
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "PUT", "OPTIONS"], allow_headers=["*"],
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
        "id":            doc_id,
        "kind":          "object",
        "wall_time":     meta.get("wall_time"),
        "camera_id":     meta.get("camera_id"),
        "label":         meta.get("label"),
        "confidence":    round(float(meta.get("confidence") or 0), 3),
        "document":      doc,
        "distance":      round(float(distance), 3) if distance is not None else None,
        "image_url":     meta.get("image_path"),
        "wall_time_s":   round(float(meta.get("wall_time_s") or 0), 3),
        "vehicle_id":    meta.get("vehicle_id") or None,
        "vehicle_label": meta.get("vehicle_label") or None,
        "pass_id":       meta.get("pass_id"),
        "is_unfamiliar": meta.get("is_unfamiliar"),
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




# ── Phase 3: Vehicles & Re-ID REST API ────────────────────────────────────────
VEHICLES_DB_PATH = os.environ.get("VEHICLES_DB_PATH") or os.environ.get("VEHICLE_DB_PATH") or os.path.join("data", "vehicles.db")


def _img_url(path_str: Optional[str]) -> Optional[str]:
    if not path_str:
        return None
    s = str(path_str).strip()
    if not s:
        return None
    return s if s.startswith("/") else "/" + s


def _get_vdb_conn():
    return vrw.init_db(VEHICLES_DB_PATH)


def _get_vdb_worker():
    return vrw.VehicleReidWorker(db_path=VEHICLES_DB_PATH, update_chroma=True)


def _format_vehicle_row(v_row, recent_passes=None):
    d = dict(v_row)
    cams = []
    try:
        cams = json.loads(d.get("cameras_seen") or "[]")
    except Exception:
        cams = []
    return {
        "vehicle_id": d.get("vehicle_id"),
        "cls": d.get("cls") or "car",
        "label": d.get("label") or d.get("vehicle_id"),
        "custom_label": int(d.get("custom_label") or 0),
        "notes": d.get("notes") or "",
        "color": d.get("color") or "unknown",
        "body_type": d.get("body_type") or "unknown",
        "roof_trim": d.get("roof_trim") or "none",
        "make_model": d.get("make_model") or "unknown",
        "plate_partial": d.get("plate_partial") or "",
        "first_seen_ts": float(d.get("first_seen_ts") or 0.0),
        "first_seen_iso": d.get("first_seen_iso"),
        "last_seen_ts": float(d.get("last_seen_ts") or 0.0),
        "last_seen_iso": d.get("last_seen_iso"),
        "sighting_count": int(d.get("sighting_count") or 0),
        "pass_count": int(d.get("pass_count") or 0),
        "cameras_seen": cams,
        "is_familiar": int(d.get("is_familiar") or 0),
        "rep_crop_path": d.get("rep_crop_path"),
        "rep_crop_url": _img_url(d.get("rep_crop_path")),
        "merged_into": d.get("merged_into"),
        "recent_passes": recent_passes if recent_passes is not None else [],
    }


def _format_pass_row(p_row):
    d = dict(p_row)
    cams = []
    try:
        cams = json.loads(d.get("cameras") or "[]")
    except Exception:
        cams = []
    start_ts = float(d.get("start_ts") or 0.0)
    end_ts = float(d.get("end_ts") or start_ts)
    return {
        "pass_id": int(d["pass_id"]),
        "vehicle_id": d.get("vehicle_id"),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "duration_s": round(max(0.0, end_ts - start_ts), 1),
        "start_iso": d.get("start_iso"),
        "direction": d.get("direction") or "street",
        "cameras": cams,
        "is_new_vehicle": int(d.get("is_new_vehicle") or 0),
        "match_score": round(float(d.get("match_score") or 0.0), 4),
        "match_method": d.get("match_method") or "new",
    }


def _format_sighting_row(s_row):
    d = dict(s_row)
    attrs = {}
    try:
        attrs = json.loads(d.get("attrs_json") or "{}")
    except Exception:
        attrs = {}
    bbox = None
    try:
        if d.get("bbox"):
            bbox = json.loads(d["bbox"])
    except Exception:
        bbox = None
    return {
        "sighting_id": int(d["sighting_id"]),
        "doc_id": d.get("doc_id"),
        "evt_id": d.get("evt_id"),
        "pass_id": int(d["pass_id"]) if d.get("pass_id") is not None else None,
        "vehicle_id": d.get("vehicle_id"),
        "timestamp_s": float(d.get("timestamp_s") or 0.0),
        "timestamp_iso": d.get("timestamp_iso"),
        "cam": d.get("cam"),
        "track_id": d.get("track_id"),
        "cls": d.get("cls") or "car",
        "conf": round(float(d.get("conf") or 0.0), 3),
        "crop_path": d.get("crop_path"),
        "crop_url": _img_url(d.get("crop_path")),
        "full_frame_path": d.get("full_frame_path"),
        "full_frame_url": _img_url(d.get("full_frame_path")),
        "sub_snapshot_path": d.get("sub_snapshot_path"),
        "sub_snapshot_url": _img_url(d.get("sub_snapshot_path")),
        "bbox": bbox,
        "crop_w": int(d.get("crop_w") or 0),
        "crop_h": int(d.get("crop_h") or 0),
        "vlm_description": d.get("vlm_description") or "",
        "attrs": attrs,
        "best_sim": round(float(d.get("best_sim") or 0.0), 4),
        "best_match_vehicle_id": d.get("best_match_vehicle_id"),
        "is_unfamiliar": int(d.get("is_unfamiliar") or 0),
    }


def _parse_ts_param(val: str) -> Optional[float]:
    if not val:
        return None
    val = val.strip()
    try:
        return float(val)
    except ValueError:
        return parse_local_dt(val)


@app.get("/api/vehicles")
def api_list_vehicles(
    filter: str = "all",     # all | unfamiliar | familiar
    cls: str = "all",        # all | car | bicycle ...
    cam: str = "",           # optional cam3, cam4, cam5, cam6 (or 3, 4, 5, 6)
    q: str = "",             # text search
    from_ts: str = "",
    to_ts: str = "",
    sort: str = "last_seen", # last_seen | first_seen | sightings | passes
):
    conn = _get_vdb_conn()
    try:
        # Global summary stats across active (non-merged) vehicles
        all_active = conn.execute("SELECT * FROM vehicles WHERE merged_into IS NULL").fetchall()
        total_vehicles = len(all_active)
        familiar_count = sum(1 for r in all_active if int(r["is_familiar"] or 0) == 1)
        unfamiliar_count = total_vehicles - familiar_count
        total_passes = conn.execute(
            "SELECT COUNT(*) AS c FROM passes p JOIN vehicles v ON p.vehicle_id = v.vehicle_id WHERE v.merged_into IS NULL"
        ).fetchone()["c"]
        total_sightings = conn.execute(
            "SELECT COUNT(*) AS c FROM sightings s JOIN vehicles v ON s.vehicle_id = v.vehicle_id WHERE v.merged_into IS NULL"
        ).fetchone()["c"]
        recent_unfamiliar_passes = conn.execute(
            "SELECT COUNT(*) AS c FROM passes p JOIN vehicles v ON p.vehicle_id = v.vehicle_id WHERE v.merged_into IS NULL AND v.is_familiar = 0"
        ).fetchone()["c"]

        # Load recent passes per vehicle
        passes_rows = conn.execute(
            "SELECT * FROM passes ORDER BY start_ts DESC, pass_id DESC"
        ).fetchall()
        passes_by_vid: Dict[str, List[Dict[str, Any]]] = {}
        for pr in passes_rows:
            vid = pr["vehicle_id"]
            if len(passes_by_vid.setdefault(vid, [])) < 5:
                passes_by_vid[vid].append(_format_pass_row(pr))

        cam_norm = ""
        if cam.strip():
            c_s = cam.strip().lower()
            cam_norm = c_s if c_s.startswith("cam") else f"cam{c_s}"

        ts_lo = _parse_ts_param(from_ts)
        ts_hi = _parse_ts_param(to_ts)
        q_norm = q.strip().lower()

        filtered = []
        for r in all_active:
            if filter == "unfamiliar" and int(r["is_familiar"] or 0) != 0:
                continue
            if filter == "familiar" and int(r["is_familiar"] or 0) != 1:
                continue
            if cls and cls != "all" and (r["cls"] or "car").lower() != cls.lower():
                continue
            v_obj = _format_vehicle_row(r, recent_passes=passes_by_vid.get(r["vehicle_id"], []))
            if cam_norm and cam_norm not in [c.lower() for c in v_obj["cameras_seen"]]:
                continue
            if ts_lo is not None and v_obj["last_seen_ts"] < ts_lo:
                continue
            if ts_hi is not None and v_obj["first_seen_ts"] > ts_hi:
                continue
            if q_norm:
                hay = " ".join([
                    str(v_obj.get("vehicle_id") or ""),
                    str(v_obj.get("label") or ""),
                    str(v_obj.get("color") or ""),
                    str(v_obj.get("body_type") or ""),
                    str(v_obj.get("roof_trim") or ""),
                    str(v_obj.get("make_model") or ""),
                    str(v_obj.get("plate_partial") or ""),
                    str(v_obj.get("notes") or ""),
                ]).lower()
                if q_norm not in hay:
                    continue
            filtered.append(v_obj)

        if sort == "first_seen":
            filtered.sort(key=lambda x: x["first_seen_ts"], reverse=True)
        elif sort == "sightings":
            filtered.sort(key=lambda x: (x["sighting_count"], x["last_seen_ts"]), reverse=True)
        elif sort == "passes":
            filtered.sort(key=lambda x: (x["pass_count"], x["last_seen_ts"]), reverse=True)
        else:
            filtered.sort(key=lambda x: x["last_seen_ts"], reverse=True)

        return {
            "summary": {
                "total_vehicles": total_vehicles,
                "familiar_count": familiar_count,
                "unfamiliar_count": unfamiliar_count,
                "total_passes": int(total_passes),
                "total_sightings": int(total_sightings),
                "recent_unfamiliar_passes": int(recent_unfamiliar_passes),
            },
            "count": len(filtered),
            "vehicles": filtered,
        }
    finally:
        conn.close()


@app.get("/api/vehicles/{vehicle_id}")
def api_get_vehicle(vehicle_id: str):
    conn = _get_vdb_conn()
    try:
        v_row = conn.execute("SELECT * FROM vehicles WHERE vehicle_id = ?", (vehicle_id,)).fetchone()
        if v_row is None:
            raise HTTPException(status_code=404, detail=f"Vehicle {vehicle_id} not found")

        p_rows = conn.execute(
            "SELECT * FROM passes WHERE vehicle_id = ? ORDER BY start_ts DESC, pass_id DESC",
            (vehicle_id,),
        ).fetchall()
        s_rows = conn.execute(
            "SELECT * FROM sightings WHERE vehicle_id = ? ORDER BY timestamp_s DESC, sighting_id DESC",
            (vehicle_id,),
        ).fetchall()

        passes_fmt = [_format_pass_row(p) for p in p_rows]
        sightings_fmt = [_format_sighting_row(s) for s in s_rows]

        # Attach sightings inside each pass as well for chronological pass-by-pass UI rendering
        s_by_pid: Dict[int, List[Dict[str, Any]]] = {}
        for s in reversed(sightings_fmt):  # chronological within a pass
            if s["pass_id"] is not None:
                s_by_pid.setdefault(int(s["pass_id"]), []).append(s)
        for p in passes_fmt:
            p["sightings"] = s_by_pid.get(int(p["pass_id"]), [])

        # Compute top similar other active vehicles by mean_embedding cosine similarity
        similar_vehicles = []
        if v_row["mean_embedding"]:
            try:
                target_emb = json.loads(v_row["mean_embedding"])
                others = conn.execute(
                    "SELECT * FROM vehicles WHERE merged_into IS NULL AND vehicle_id != ?",
                    (vehicle_id,),
                ).fetchall()
                scored = []
                for o in others:
                    if not o["mean_embedding"]:
                        continue
                    sim = vrw.cosine_sim(target_emb, json.loads(o["mean_embedding"]))
                    scored.append((sim, o))
                scored.sort(key=lambda x: x[0], reverse=True)
                for sim, o in scored[:8]:
                    vo = _format_vehicle_row(o)
                    vo["similarity"] = round(float(sim), 4)
                    similar_vehicles.append(vo)
            except Exception:
                pass

        # Also list any vehicles that were previously merged into this vehicle
        merged_children = [
            r["vehicle_id"]
            for r in conn.execute("SELECT vehicle_id FROM vehicles WHERE merged_into = ?", (vehicle_id,)).fetchall()
        ]

        veh_dict = _format_vehicle_row(v_row, recent_passes=passes_fmt[:5])
        veh_dict["merged_children"] = merged_children
        return {
            "vehicle": veh_dict,
            "passes": passes_fmt,
            "sightings": sightings_fmt,
            "similar_vehicles": similar_vehicles,
        }
    finally:
        conn.close()


@app.get("/api/vehicle_sightings")
def api_list_vehicle_sightings(
    cam: str = "",
    vehicle_id: str = "",
    cls: str = "all",
    unfamiliar_only: int = 0,
    multi_cam_only: int = 0,
    q: str = "",
    from_ts: str = "",
    to_ts: str = "",
    limit: int = 100,
):
    conn = _get_vdb_conn()
    try:
        rows = conn.execute(
            """
            SELECT s.*, v.label AS vehicle_label, v.is_familiar AS vehicle_is_familiar,
                   v.color AS vehicle_color, v.body_type AS vehicle_body_type,
                   v.make_model AS vehicle_make_model, p.cameras AS pass_cameras,
                   p.direction AS pass_direction, p.match_method AS pass_match_method,
                   p.is_new_vehicle AS pass_is_new_vehicle, p.start_ts AS pass_start_ts, p.end_ts AS pass_end_ts
            FROM sightings s
            LEFT JOIN vehicles v ON s.vehicle_id = v.vehicle_id
            LEFT JOIN passes p ON s.pass_id = p.pass_id
            ORDER BY s.timestamp_s DESC, s.sighting_id DESC
            """
        ).fetchall()

        cam_norm = ""
        if cam.strip():
            c_s = cam.strip().lower()
            cam_norm = c_s if c_s.startswith("cam") else f"cam{c_s}"

        ts_lo = _parse_ts_param(from_ts)
        ts_hi = _parse_ts_param(to_ts)
        q_norm = q.strip().lower()

        out = []
        for r in rows:
            s_obj = _format_sighting_row(r)
            p_cams = []
            try:
                p_cams = json.loads(r["pass_cameras"] or "[]")
            except Exception:
                p_cams = [s_obj["cam"]] if s_obj["cam"] else []
            s_obj["vehicle_label"] = r["vehicle_label"] or s_obj["vehicle_id"]
            s_obj["vehicle_is_familiar"] = int(r["vehicle_is_familiar"] or 0)
            s_obj["pass_cameras"] = p_cams
            s_obj["pass_direction"] = r["pass_direction"] or "street"
            s_obj["pass_match_method"] = r["pass_match_method"] or "new"
            s_obj["pass_is_new_vehicle"] = int(r["pass_is_new_vehicle"] or 0)

            if cam_norm and (s_obj["cam"] or "").lower() != cam_norm:
                continue
            if vehicle_id.strip() and (s_obj["vehicle_id"] or "").lower() != vehicle_id.strip().lower():
                continue
            if cls and cls != "all" and (s_obj["cls"] or "car").lower() != cls.lower():
                continue
            if int(unfamiliar_only) == 1 and int(s_obj["vehicle_is_familiar"]) != 0:
                continue
            if int(multi_cam_only) == 1 and len(p_cams) < 2:
                continue
            if ts_lo is not None and s_obj["timestamp_s"] < ts_lo:
                continue
            if ts_hi is not None and s_obj["timestamp_s"] > ts_hi:
                continue
            if q_norm:
                hay = " ".join([
                    str(s_obj.get("vehicle_id") or ""),
                    str(s_obj.get("vehicle_label") or ""),
                    str(s_obj.get("cam") or ""),
                    str(s_obj.get("vlm_description") or ""),
                    json.dumps(s_obj.get("attrs") or {}),
                ]).lower()
                if q_norm not in hay:
                    continue
            out.append(s_obj)
            if len(out) >= max(1, min(limit, 500)):
                break

        return {
            "count": len(out),
            "sightings": out,
        }
    finally:
        conn.close()


@app.post("/api/vehicles/{vehicle_id}/update")
def api_update_vehicle(vehicle_id: str, payload: Dict[str, Any] = Body(...)):
    worker = _get_vdb_worker()
    try:
        updated = worker.update_vehicle_details(vehicle_id, payload)
        return {"ok": True, "vehicle": _format_vehicle_row(updated)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        worker.conn.close()


@app.post("/api/vehicles/merge")
def api_merge_vehicles(payload: Dict[str, Any] = Body(...)):
    source_id = (payload.get("source_id") or "").strip()
    target_id = (payload.get("target_id") or "").strip()
    if not source_id or not target_id:
        raise HTTPException(status_code=400, detail="Both source_id and target_id are required")
    worker = _get_vdb_worker()
    try:
        updated = worker.merge_vehicles(source_id, target_id)
        return {
            "ok": True,
            "source_id": source_id,
            "target_id": target_id,
            "vehicle": _format_vehicle_row(updated),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        worker.conn.close()


@app.post("/api/vehicles/split")
def api_split_vehicle(payload: Dict[str, Any] = Body(...)):
    sighting_ids = payload.get("sighting_ids") or []
    pass_ids = payload.get("pass_ids") or []
    target_vehicle_id = payload.get("target_vehicle_id") or "new"
    worker = _get_vdb_worker()
    try:
        res = worker.split_vehicle_items(
            sighting_ids=sighting_ids,
            pass_ids=pass_ids,
            target_vehicle_id=target_vehicle_id,
        )
        return {
            "ok": True,
            "target_vehicle": _format_vehicle_row(res["target_vehicle"]),
            "updated_source_vehicles": [_format_vehicle_row(v) for v in res["updated_source_vehicles"] if v],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        worker.conn.close()


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
