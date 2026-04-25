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
COLLECTION_NAME = "vision_events"
OLLAMA_MODEL   = "nomic-embed-text"
SNAPSHOT_DIR   = "/workspace/snapshots"
SEARCH_HTML    = "/workspace/search.html"

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

app = FastAPI(title="Vision Query API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"],
)
app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")

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


def fmt(r, doc_id, doc, meta, distance=None):
    return {
        "id":          doc_id,
        "wall_time":   meta.get("wall_time"),
        "camera_id":   meta.get("camera_id"),
        "label":       meta.get("label"),
        "confidence":  round(float(meta.get("confidence") or 0), 3),
        "document":    doc,
        "distance":    round(float(distance), 3) if distance is not None else None,
        "image_url":   meta.get("image_path"),
        "wall_time_s": round(float(meta.get("wall_time_s") or 0), 3),
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
):
    where = build_where(start_time, end_time, label, camera_id)

    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ChromaDB error: {e}")

    output = []

    if text.strip():
        try:
            resp = ollama.embeddings(model=OLLAMA_MODEL, prompt=text.strip())
            embedding = resp["embedding"]
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Ollama error: {e}")

        kwargs = {"query_embeddings": [embedding], "n_results": n}
        if where:
            kwargs["where"] = where
        try:
            results = collection.query(**kwargs)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"ChromaDB query error: {e}")

        for i, doc_id in enumerate(results["ids"][0]):
            output.append(fmt(
                None,
                doc_id,
                results["documents"][0][i],
                results["metadatas"][0][i],
                results["distances"][0][i],
            ))
    else:
        kwargs = {"limit": n, "include": ["documents", "metadatas"]}
        if where:
            kwargs["where"] = where
        try:
            results = collection.get(**kwargs)
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"ChromaDB get error: {e}")

        for i, doc_id in enumerate(results["ids"]):
            output.append(fmt(None, doc_id, results["documents"][i], results["metadatas"][i]))

    if sort_by == "time_desc":
        output.sort(key=lambda x: x["wall_time_s"] or 0, reverse=True)
    elif sort_by == "time_asc":
        output.sort(key=lambda x: x["wall_time_s"] or 0)

    return {"query": text, "count": len(output), "results": output}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
