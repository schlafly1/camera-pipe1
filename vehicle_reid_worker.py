#!/usr/bin/env python3
"""
vehicle_reid_worker.py — Phase 2 Vehicle Re-ID & Cross-Camera Pass Grouping Worker.

Runs on thor2 alongside pipeline_multi.py and query_server.py (zero GPU load on thor2).
Tails logs/vehicle_events.jsonl, calls reid_server.py on gx10 (REID_URL) for:
  - YOLO11m refined vehicle crop + FastReID sbs_R50-ibn 2048-d L2-normalized embedding
  - Structured Gemma-4-12B vehicle attributes (color, body_type, roof_trim, make_model, plate_partial)
Maintains SQLite database data/vehicles.db (tables: vehicles, passes, sightings),
writes enriched sighting records to logs/vehicle_sightings.jsonl, and updates
ChromaDB (vision_events_v2) metadata with vehicle_id, vehicle_label, pass_id, and is_unfamiliar.
"""

import argparse
import base64
import datetime
import json
import logging
import os
import signal
import sqlite3
import sys
import time
import urllib.request
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("vehicle_reid_worker")

REID_URL = os.environ.get("REID_URL", "http://gx10-2ea8:8090").rstrip("/")
VEHICLE_EVENTS_LOG = os.environ.get("VEHICLE_EVENTS_LOG", os.path.join("logs", "vehicle_events.jsonl"))
VEHICLE_SIGHTINGS_LOG = os.environ.get("VEHICLE_SIGHTINGS_LOG", os.path.join("logs", "vehicle_sightings.jsonl"))
VEHICLE_DB_PATH = os.environ.get("VEHICLE_DB_PATH", os.path.join("data", "vehicles.db"))
VEHICLE_SNAPSHOT_DIR = os.environ.get("VEHICLE_SNAPSHOT_DIR", os.path.join("snapshots", "vehicles"))
CHROMA_HOST = os.environ.get("CHROMA_HOST", "127.0.0.1")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "vision_events_v2")

# Matching & Pass-Linking Thresholds (from Phase 0 empirical validation)
PASS_WINDOW_S = float(os.environ.get("REID_PASS_WINDOW_S", "60.0"))
CROSS_CAM_MIN_SIM = float(os.environ.get("REID_CROSS_CAM_MIN_SIM", "0.45"))
CROSS_CAM_STRONG_ATTR_MIN_SIM = float(os.environ.get("REID_CROSS_CAM_STRONG_ATTR_MIN_SIM", "0.28"))
REID_HIGH_SIM = float(os.environ.get("REID_HIGH_SIM", "0.62"))
REID_VERY_HIGH_SIM = float(os.environ.get("REID_VERY_HIGH_SIM", "0.68"))
REID_GRAY_MIN_SIM = float(os.environ.get("REID_GRAY_MIN_SIM", "0.48"))


# ── SQLite Schema & Helpers ───────────────────────────────────────────────────
SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS vehicles (
    vehicle_id TEXT PRIMARY KEY,
    cls TEXT NOT NULL DEFAULT 'car',
    label TEXT,
    color TEXT,
    body_type TEXT,
    roof_trim TEXT,
    make_model TEXT,
    plate_partial TEXT,
    first_seen_ts REAL,
    first_seen_iso TEXT,
    last_seen_ts REAL,
    last_seen_iso TEXT,
    sighting_count INTEGER DEFAULT 0,
    pass_count INTEGER DEFAULT 0,
    cameras_seen TEXT DEFAULT '[]',
    is_familiar INTEGER DEFAULT 0,
    rep_crop_path TEXT,
    mean_embedding TEXT,
    merged_into TEXT NULL,
    custom_label INTEGER DEFAULT 0,
    notes TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS passes (
    pass_id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id TEXT NOT NULL,
    start_ts REAL,
    end_ts REAL,
    start_iso TEXT,
    direction TEXT,
    cameras TEXT DEFAULT '[]',
    is_new_vehicle INTEGER DEFAULT 0,
    match_score REAL DEFAULT 0.0,
    match_method TEXT
);

CREATE TABLE IF NOT EXISTS sightings (
    sighting_id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT UNIQUE NOT NULL,
    evt_id INTEGER,
    pass_id INTEGER,
    vehicle_id TEXT,
    timestamp_s REAL,
    timestamp_iso TEXT,
    cam TEXT,
    track_id INTEGER,
    cls TEXT,
    conf REAL,
    crop_path TEXT,
    full_frame_path TEXT,
    sub_snapshot_path TEXT,
    bbox TEXT,
    crop_w INTEGER,
    crop_h INTEGER,
    vlm_description TEXT,
    attrs_json TEXT,
    embedding TEXT,
    best_sim REAL DEFAULT 0.0,
    best_match_vehicle_id TEXT,
    is_unfamiliar INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_sightings_vehicle_id ON sightings(vehicle_id);
CREATE INDEX IF NOT EXISTS idx_sightings_pass_id ON sightings(pass_id);
CREATE INDEX IF NOT EXISTS idx_sightings_ts ON sightings(timestamp_s);
CREATE INDEX IF NOT EXISTS idx_passes_vehicle_id ON passes(vehicle_id);
CREATE INDEX IF NOT EXISTS idx_passes_end_ts ON passes(end_ts);
CREATE INDEX IF NOT EXISTS idx_vehicles_last_seen ON vehicles(last_seen_ts);
"""


def init_db(db_path=VEHICLE_DB_PATH):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(vehicles)").fetchall()}
    if "custom_label" not in cols:
        conn.execute("ALTER TABLE vehicles ADD COLUMN custom_label INTEGER DEFAULT 0")
    if "notes" not in cols:
        conn.execute("ALTER TABLE vehicles ADD COLUMN notes TEXT DEFAULT ''")
    conn.commit()
    return conn


# ── Attribute Compatibility & Cosine Similarity Helpers ───────────────────────
_COLOR_GROUPS = (
    frozenset({"white", "silver", "light gray"}),
    frozenset({"silver", "gray", "dark gray"}),
    frozenset({"black", "dark gray", "dark blue"}),
    frozenset({"blue", "dark blue", "light blue"}),
    frozenset({"red", "maroon", "dark red", "orange", "burgundy"}),
    frozenset({"tan", "brown", "gold", "beige", "silver"}),
    frozenset({"green", "dark gray", "black"}),
)

_BODY_GROUPS = (
    frozenset({"sedan", "coupe", "hatchback", "wagon"}),
    frozenset({"suv", "crossover", "wagon", "van"}),
    frozenset({"pickup", "truck", "suv"}),
    frozenset({"van", "minivan", "suv", "truck"}),
    frozenset({"mountain_bike", "road_bike", "ebike", "bicycle"}),
)

_HARD_BODY_CONFLICTS = (
    frozenset({"sedan", "pickup"}),
    frozenset({"sedan", "truck"}),
    frozenset({"sedan", "van"}),
    frozenset({"coupe", "pickup"}),
    frozenset({"coupe", "truck"}),
    frozenset({"hatchback", "pickup"}),
    frozenset({"hatchback", "truck"}),
)


def colors_compatible(c1, c2):
    c1 = (c1 or "unknown").strip().lower()
    c2 = (c2 or "unknown").strip().lower()
    if c1 in ("unknown", "") or c2 in ("unknown", ""):
        return True
    if c1 == c2:
        return True
    for g in _COLOR_GROUPS:
        if c1 in g and c2 in g:
            return True
    return False


def body_types_compatible(b1, b2):
    b1 = (b1 or "unknown").strip().lower()
    b2 = (b2 or "unknown").strip().lower()
    if b1 in ("unknown", "") or b2 in ("unknown", ""):
        return True
    if b1 == b2:
        return True
    pair = frozenset({b1, b2})
    if pair in _HARD_BODY_CONFLICTS:
        return False
    for g in _BODY_GROUPS:
        if b1 in g and b2 in g:
            return True
    return False


def roof_trim_conflicts(r1, r2):
    r1 = (r1 or "none").strip().lower()
    r2 = (r2 or "none").strip().lower()
    if r1 in ("none", "unknown", "") or r2 in ("none", "unknown", ""):
        return False
    if ("black roof" in r1 and "white roof" in r2) or ("white roof" in r1 and "black roof" in r2):
        return True
    if ("camper shell" in r1 and "open bed" in r2) or ("open bed" in r1 and "camper shell" in r2):
        return True
    return False


def make_models_match(m1, m2):
    m1 = (m1 or "unknown").strip().lower()
    m2 = (m2 or "unknown").strip().lower()
    if m1 in ("unknown", "", "bicycle", "mountain bike", "road bike") or m2 in ("unknown", "", "bicycle", "mountain bike", "road bike"):
        return False
    if m1 == m2:
        return True
    t1 = set(m1.replace("-", " ").split())
    t2 = set(m2.replace("-", " ").split())
    common = t1 & t2 - {"car", "suv", "truck", "sedan", "pickup", "van"}
    return len(common) >= 1


def attrs_compatible(a1, a2):
    if not colors_compatible(a1.get("color"), a2.get("color")):
        return False
    if not body_types_compatible(a1.get("body_type"), a2.get("body_type")):
        return False
    if roof_trim_conflicts(a1.get("roof_trim"), a2.get("roof_trim")):
        return False
    return True


def attrs_strongly_match(a1, a2):
    if not attrs_compatible(a1, a2):
        return False
    c1 = (a1.get("color") or "unknown").lower()
    c2 = (a2.get("color") or "unknown").lower()
    b1 = (a1.get("body_type") or "unknown").lower()
    b2 = (a2.get("body_type") or "unknown").lower()
    if c1 == "unknown" or c2 == "unknown" or b1 == "unknown" or b2 == "unknown":
        return False
    if c1 == c2 and b1 == b2:
        return True
    if make_models_match(a1.get("make_model"), a2.get("make_model")):
        return True
    # Compatible color + identical specific body_type
    if colors_compatible(c1, c2) and b1 == b2:
        return True
    return False


def cosine_sim(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    a = np.asarray(v1, dtype=np.float32)
    b = np.asarray(v2, dtype=np.float32)
    if a.shape != b.shape or a.size == 0:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-6 or nb <= 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def l2_normalize(vec):
    arr = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    if n > 1e-6:
        arr = arr / n
    return [round(float(x), 6) for x in arr.tolist()]


def _infer_direction(cam_str, attrs, vlm_desc=""):
    motion = (attrs.get("motion") or "").lower()
    if "park" in motion:
        return "parked"
    desc = (vlm_desc or "").lower()
    if "into a driveway" in desc or "entering" in desc or "down a paved driveway" in desc:
        return "in"
    if "away from" in desc or "exiting" in desc or "leaving" in desc:
        return "out"
    if "to the right" in desc or "toward the right" in desc or "towards the right" in desc:
        return "right"
    if "to the left" in desc or "toward the left" in desc or "towards the left" in desc:
        return "left"
    return "street" if cam_str in ("cam4", "cam6") else "driveway"


def _build_vehicle_label(cls, color, body_type, roof_trim, make_model):
    color_s = (color or "unknown").strip()
    body_s = (body_type or "unknown").strip()
    roof_s = (roof_trim or "none").strip()
    mm_s = (make_model or "unknown").strip()

    parts = []
    if color_s and color_s.lower() != "unknown":
        parts.append(color_s.capitalize())
    if mm_s and mm_s.lower() not in ("unknown", "none", "bicycle", "mountain bike", "road bike"):
        parts.append(mm_s)
    if body_s and body_s.lower() != "unknown":
        b_clean = body_s.replace("_", " ")
        if not parts or b_clean.lower() not in " ".join(parts).lower():
            parts.append(b_clean.upper() if b_clean.lower() == "suv" else b_clean)
    else:
        parts.append("bicycle" if cls == "bicycle" else "vehicle")

    if roof_s and roof_s.lower() not in ("none", "unknown", "parked"):
        parts.append(f"w/ {roof_s}")
    return " ".join(parts)


# ── Core Worker ───────────────────────────────────────────────────────────────
class VehicleReidWorker:
    def __init__(self, db_path=VEHICLE_DB_PATH, reid_url=REID_URL, update_chroma=True):
        self.db_path = db_path
        self.reid_url = reid_url.rstrip("/")
        self.update_chroma = update_chroma
        self.conn = init_db(db_path)
        self._chroma_col = None
        if update_chroma:
            self._init_chroma()

    def _init_chroma(self):
        try:
            import chromadb
            client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            self._chroma_col = client.get_collection(CHROMA_COLLECTION)
            log.info(f"Connected to ChromaDB collection '{CHROMA_COLLECTION}' on {CHROMA_HOST}:{CHROMA_PORT}")
        except Exception as e:
            log.warning(f"ChromaDB connection optional/unavailable ({e}); proceeding with SQLite + JSONL")
            self._chroma_col = None

    def is_doc_processed(self, doc_id):
        row = self.conn.execute(
            "SELECT sighting_id FROM sightings WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return row is not None

    def _next_vehicle_id(self):
        rows = self.conn.execute("SELECT vehicle_id FROM vehicles").fetchall()
        max_n = 0
        for r in rows:
            vid = r["vehicle_id"] or ""
            if vid.startswith("veh_"):
                try:
                    max_n = max(max_n, int(vid.split("_", 1)[1]))
                except ValueError:
                    pass
        return f"veh_{max_n + 1:03d}"

    def _read_b64(self, rel_path):
        if not rel_path:
            return None
        p = rel_path.lstrip("/")
        if not os.path.exists(p):
            return None
        try:
            with open(p, "rb") as fh:
                return base64.b64encode(fh.read()).decode("ascii")
        except Exception:
            return None

    def call_reid_server(self, ev):
        payload = {
            "cls": ev.get("cls") or "car",
            "cam": ev.get("cam"),
            "main_crop_b64": self._read_b64(ev.get("main_crop_path")),
            "sub_crop_b64": self._read_b64(ev.get("sub_crop_path")),
            "main_frame_b64": self._read_b64(ev.get("main_frame_path")),
            "main_bbox": ev.get("main_bbox"),
            "sub_bbox": ev.get("sub_bbox"),
            "sub_bbox_norm": ev.get("sub_bbox_norm"),
            "vlm_description": ev.get("vlm_description") or "",
            "extract_attrs": True,
            "return_refined_crop": True,
        }
        if not payload["main_crop_b64"] and not payload["sub_crop_b64"] and not payload["main_frame_b64"]:
            payload["sub_crop_b64"] = self._read_b64(ev.get("sub_snapshot_path"))

        req = urllib.request.Request(
            f"{self.reid_url}/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())

    def _find_historical_match(self, cls, emb, attrs, exclude_vid=None, before_ts=None):
        q = "SELECT * FROM vehicles WHERE merged_into IS NULL AND cls = ?"
        params = [cls]
        if exclude_vid:
            q += " AND vehicle_id != ?"
            params.append(exclude_vid)
        if before_ts is not None:
            q += " AND first_seen_ts < ?"
            params.append(before_ts)
        vehicles = self.conn.execute(q, tuple(params)).fetchall()
        best_vid = None
        best_sim = -1.0
        best_method = None
        for v in vehicles:
            vid = v["vehicle_id"]
            sims = []
            if v["mean_embedding"]:
                sims.append(cosine_sim(emb, json.loads(v["mean_embedding"])))
            ex_rows = self.conn.execute(
                """
                SELECT embedding FROM sightings
                WHERE vehicle_id = ? AND embedding IS NOT NULL
                ORDER BY (COALESCE(crop_w, 0) * COALESCE(crop_h, 0)) DESC
                LIMIT 5
                """,
                (vid,),
            ).fetchall()
            for er in ex_rows:
                sims.append(cosine_sim(emb, json.loads(er["embedding"])))
            sim = max(sims) if sims else 0.0
            v_attrs = {
                "color": v["color"],
                "body_type": v["body_type"],
                "roof_trim": v["roof_trim"],
                "make_model": v["make_model"],
            }
            method = None
            if (sim >= REID_HIGH_SIM and attrs_compatible(attrs, v_attrs)) or (
                sim >= REID_VERY_HIGH_SIM
                and body_types_compatible(attrs.get("body_type"), v_attrs.get("body_type"))
                and not roof_trim_conflicts(attrs.get("roof_trim"), v_attrs.get("roof_trim"))
            ):
                method = "reid_high"
            elif REID_GRAY_MIN_SIM <= sim < REID_HIGH_SIM and attrs_strongly_match(attrs, v_attrs):
                method = "reid_vlm_gate"
            if method is not None and sim > best_sim:
                best_sim = sim
                best_vid = vid
                best_method = method
        return best_vid, best_sim, best_method

    def _promote_provisional_pass(self, pass_id, old_vid, target_vid, match_score, match_method):
        # Move sightings & pass from provisional old_vid to target_vid
        prior_sights = self.conn.execute(
            "SELECT * FROM sightings WHERE pass_id = ? ORDER BY timestamp_s",
            (pass_id,),
        ).fetchall()
        self.conn.execute(
            "UPDATE passes SET vehicle_id = ?, is_new_vehicle = 0, match_score = ?, match_method = ? WHERE pass_id = ?",
            (target_vid, round(float(match_score), 4), match_method, pass_id),
        )
        self.conn.execute(
            "UPDATE sightings SET vehicle_id = ?, is_unfamiliar = 0, best_match_vehicle_id = ? WHERE pass_id = ?",
            (target_vid, target_vid, pass_id),
        )
        self.conn.execute("DELETE FROM vehicles WHERE vehicle_id = ?", (old_vid,))
        for idx, s in enumerate(prior_sights):
            s_emb = json.loads(s["embedding"]) if s["embedding"] else []
            s_attrs = json.loads(s["attrs_json"]) if s["attrs_json"] else {}
            lbl = self._update_vehicle_rollups(
                vehicle_id=target_vid,
                ts_s=float(s["timestamp_s"]),
                ts_iso=s["timestamp_iso"],
                cam_str=s["cam"],
                crop_path=s["crop_path"],
                crop_w=int(s["crop_w"] or 0),
                crop_h=int(s["crop_h"] or 0),
                emb=s_emb,
                attrs=s_attrs,
                new_pass=(idx == 0),
            )
            self._update_chroma_metadata(s["doc_id"], target_vid, lbl, pass_id, 0)

    def recluster_existing_sightings(self):
        """Replay all existing sightings in chronological order using cached embeddings & attributes."""
        rows = self.conn.execute("SELECT * FROM sightings ORDER BY timestamp_s ASC, sighting_id ASC").fetchall()
        if not rows:
            return 0
        self.conn.execute("DELETE FROM sightings")
        self.conn.execute("DELETE FROM passes")
        self.conn.execute("DELETE FROM vehicles")
        self.conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('passes', 'sightings')")
        self.conn.commit()
        if os.path.exists(VEHICLE_SIGHTINGS_LOG):
            os.remove(VEHICLE_SIGHTINGS_LOG)
        for s in rows:
            emb = json.loads(s["embedding"])
            attrs = json.loads(s["attrs_json"] or "{}")
            cam_str = s["cam"]
            cls = s["cls"] or "car"
            ts_s = float(s["timestamp_s"])
            ts_iso = s["timestamp_iso"]
            direction = _infer_direction(cam_str, attrs, s["vlm_description"] or "")
            match = self.match_or_assign(cls, cam_str, ts_s, ts_iso, emb, attrs, direction)
            vehicle_id = match["vehicle_id"]
            pass_id = match["pass_id"]
            is_unfamiliar = match["is_unfamiliar"]
            best_sim = match["best_sim"]
            best_match_vid = match["best_match_vehicle_id"]
            match_method = match["match_method"]
            crop_w = int(s["crop_w"] or 0)
            crop_h = int(s["crop_h"] or 0)
            self.conn.execute(
                """
                INSERT INTO sightings (
                    doc_id, evt_id, pass_id, vehicle_id, timestamp_s, timestamp_iso,
                    cam, track_id, cls, conf, crop_path, full_frame_path, sub_snapshot_path,
                    bbox, crop_w, crop_h, vlm_description, attrs_json, embedding,
                    best_sim, best_match_vehicle_id, is_unfamiliar
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    s["doc_id"], s["evt_id"], pass_id, vehicle_id, ts_s, ts_iso,
                    cam_str, s["track_id"], cls, s["conf"], s["crop_path"],
                    s["full_frame_path"], s["sub_snapshot_path"], s["bbox"],
                    crop_w, crop_h, s["vlm_description"], s["attrs_json"], s["embedding"],
                    best_sim, best_match_vid, is_unfamiliar,
                ),
            )
            vehicle_label = self._update_vehicle_rollups(
                vehicle_id, ts_s, ts_iso, cam_str, s["crop_path"],
                crop_w, crop_h, emb, attrs, match["new_pass"]
            )
            self.conn.commit()
            self._update_chroma_metadata(s["doc_id"], vehicle_id, vehicle_label, pass_id, is_unfamiliar)
            summary = {
                "timestamp": ts_iso, "timestamp_s": ts_s, "doc_id": s["doc_id"],
                "evt_id": s["evt_id"], "cam": cam_str, "cls": cls,
                "vehicle_id": vehicle_id, "vehicle_label": vehicle_label,
                "pass_id": pass_id, "is_unfamiliar": is_unfamiliar,
                "match_method": match_method, "best_sim": best_sim,
                "best_match_vehicle_id": best_match_vid, "crop_path": s["crop_path"],
                "crop_shape": [crop_h, crop_w], "attrs": attrs,
            }
            with open(VEHICLE_SIGHTINGS_LOG, "a") as fh:
                fh.write(json.dumps(summary) + "\n")
        return len(rows)

    def match_or_assign(self, cls, cam_str, ts_s, ts_iso, emb, attrs, direction):
        """
        Returns dict:
          vehicle_id, pass_id, is_unfamiliar, best_sim, best_match_vehicle_id, match_method
        """
        # 1. Check active multi-camera passes within PASS_WINDOW_S (60s)
        recent_passes = self.conn.execute(
            """
            SELECT p.*, v.cls AS v_cls, v.color AS v_color, v.body_type AS v_body_type,
                   v.roof_trim AS v_roof_trim, v.make_model AS v_make_model,
                   v.mean_embedding AS v_mean_emb
            FROM passes p
            JOIN vehicles v ON p.vehicle_id = v.vehicle_id
            WHERE v.merged_into IS NULL
              AND v.cls = ?
              AND p.start_ts <= ?
              AND p.end_ts >= ?
            ORDER BY p.end_ts DESC
            """,
            (cls, ts_s + 10.0, ts_s - PASS_WINDOW_S),
        ).fetchall()

        best_pass = None
        best_pass_sim = -1.0
        for p in recent_passes:
            p_sightings = self.conn.execute(
                "SELECT cam, embedding, attrs_json FROM sightings WHERE pass_id = ?",
                (p["pass_id"],),
            ).fetchall()
            sims = []
            if p["v_mean_emb"]:
                sims.append(cosine_sim(emb, json.loads(p["v_mean_emb"])))
            for s in p_sightings:
                if s["embedding"]:
                    sims.append(cosine_sim(emb, json.loads(s["embedding"])))
            sim = max(sims) if sims else 0.0

            v_attrs = {
                "color": p["v_color"],
                "body_type": p["v_body_type"],
                "roof_trim": p["v_roof_trim"],
                "make_model": p["v_make_model"],
            }
            if not attrs_compatible(attrs, v_attrs):
                continue

            cams_in_pass = json.loads(p["cameras"] or "[]")
            same_cam_only = (cams_in_pass == [cam_str])
            dt = abs(ts_s - float(p["end_ts"]))

            linked = False
            if not same_cam_only:
                # Cross-camera pass linking within 60s window
                if sim >= CROSS_CAM_MIN_SIM:
                    linked = True
                elif sim >= CROSS_CAM_STRONG_ATTR_MIN_SIM and attrs_strongly_match(attrs, v_attrs) and dt <= 35.0:
                    linked = True
                elif dt <= 12.0 and attrs_strongly_match(attrs, v_attrs):
                    linked = True
            else:
                # Same camera within short window (e.g., driveway re-trigger within 30s)
                if dt <= 30.0 and (sim >= 0.44 or (sim >= 0.35 and attrs_strongly_match(attrs, v_attrs))):
                    linked = True

            if linked and sim > best_pass_sim:
                best_pass_sim = sim
                best_pass = p

        if best_pass is not None:
            pass_id = int(best_pass["pass_id"])
            vehicle_id = best_pass["vehicle_id"]
            is_unfamiliar = int(best_pass["is_new_vehicle"])
            cams = json.loads(best_pass["cameras"] or "[]")
            if cam_str not in cams:
                cams.append(cam_str)
            new_start = min(float(best_pass["start_ts"]), ts_s)
            new_end = max(float(best_pass["end_ts"]), ts_s)

            # Check if this new camera angle in a provisional new pass (is_new_vehicle == 1)
            # strongly matches an older historical vehicle seen on a prior pass!
            if is_unfamiliar == 1:
                hist_vid, hist_sim, hist_method = self._find_historical_match(
                    cls, emb, attrs, exclude_vid=vehicle_id, before_ts=float(best_pass["start_ts"]) - 1.0
                )
                if hist_vid is not None:
                    log.info(
                        f"[ReID] Promoting provisional pass {pass_id} ({vehicle_id} -> {hist_vid}) "
                        f"via {cam_str} match ({hist_method}, sim={hist_sim:.3f})"
                    )
                    self._promote_provisional_pass(pass_id, vehicle_id, hist_vid, hist_sim, hist_method)
                    vehicle_id = hist_vid
                    is_unfamiliar = 0
                    best_pass_sim = hist_sim

            self.conn.execute(
                "UPDATE passes SET start_ts = ?, end_ts = ?, cameras = ? WHERE pass_id = ?",
                (new_start, new_end, json.dumps(cams), pass_id),
            )
            return {
                "vehicle_id": vehicle_id,
                "pass_id": pass_id,
                "is_unfamiliar": is_unfamiliar,
                "best_sim": round(float(best_pass_sim), 4),
                "best_match_vehicle_id": vehicle_id,
                "match_method": "cross_cam_pass",
                "new_pass": False,
            }

        # 2. Historical vehicle re-identification across all known vehicles of the same class
        vehicles = self.conn.execute(
            "SELECT * FROM vehicles WHERE merged_into IS NULL AND cls = ?",
            (cls,),
        ).fetchall()

        best_raw_sim = 0.0
        best_raw_vid = None
        best_cand_vid = None
        best_cand_sim = -1.0
        best_cand_method = None

        for v in vehicles:
            vid = v["vehicle_id"]
            sims = []
            if v["mean_embedding"]:
                sims.append(cosine_sim(emb, json.loads(v["mean_embedding"])))
            # Compare against top 5 highest-res sightings of this vehicle as exemplars
            ex_rows = self.conn.execute(
                """
                SELECT embedding FROM sightings
                WHERE vehicle_id = ? AND embedding IS NOT NULL
                ORDER BY (COALESCE(crop_w, 0) * COALESCE(crop_h, 0)) DESC
                LIMIT 5
                """,
                (vid,),
            ).fetchall()
            for er in ex_rows:
                sims.append(cosine_sim(emb, json.loads(er["embedding"])))
            sim = max(sims) if sims else 0.0

            if sim > best_raw_sim:
                best_raw_sim = sim
                best_raw_vid = vid

            v_attrs = {
                "color": v["color"],
                "body_type": v["body_type"],
                "roof_trim": v["roof_trim"],
                "make_model": v["make_model"],
            }

            method = None
            if (sim >= REID_HIGH_SIM and attrs_compatible(attrs, v_attrs)) or (
                sim >= REID_VERY_HIGH_SIM
                and body_types_compatible(attrs.get("body_type"), v_attrs.get("body_type"))
                and not roof_trim_conflicts(attrs.get("roof_trim"), v_attrs.get("roof_trim"))
            ):
                method = "reid_high"
            elif REID_GRAY_MIN_SIM <= sim < REID_HIGH_SIM and attrs_strongly_match(attrs, v_attrs):
                method = "reid_vlm_gate"

            if method is not None and sim > best_cand_sim:
                best_cand_sim = sim
                best_cand_vid = vid
                best_cand_method = method

        if best_cand_vid is not None:
            # Matched an existing historical vehicle on a new pass!
            cur = self.conn.execute(
                """
                INSERT INTO passes (vehicle_id, start_ts, end_ts, start_iso, direction, cameras, is_new_vehicle, match_score, match_method)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (
                    best_cand_vid,
                    ts_s,
                    ts_s,
                    ts_iso,
                    direction,
                    json.dumps([cam_str]),
                    round(float(best_cand_sim), 4),
                    best_cand_method,
                ),
            )
            return {
                "vehicle_id": best_cand_vid,
                "pass_id": int(cur.lastrowid),
                "is_unfamiliar": 0,
                "best_sim": round(float(best_cand_sim), 4),
                "best_match_vehicle_id": best_cand_vid,
                "match_method": best_cand_method,
                "new_pass": True,
            }

        # 3. Unfamiliar / brand-new vehicle!
        new_vid = self._next_vehicle_id()
        color = attrs.get("color") or "unknown"
        body_type = attrs.get("body_type") or ("bicycle" if cls == "bicycle" else "unknown")
        roof_trim = attrs.get("roof_trim") or "none"
        make_model = attrs.get("make_model") or "unknown"
        plate_partial = attrs.get("plate_partial") or ""
        label = attrs.get("label") or _build_vehicle_label(cls, color, body_type, roof_trim, make_model)

        self.conn.execute(
            """
            INSERT INTO vehicles (
                vehicle_id, cls, label, color, body_type, roof_trim, make_model, plate_partial,
                first_seen_ts, first_seen_iso, last_seen_ts, last_seen_iso,
                sighting_count, pass_count, cameras_seen, is_familiar, rep_crop_path, mean_embedding, merged_into
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, 0, NULL, ?, NULL)
            """,
            (
                new_vid,
                cls,
                label,
                color,
                body_type,
                roof_trim,
                make_model,
                plate_partial,
                ts_s,
                ts_iso,
                ts_s,
                ts_iso,
                json.dumps([cam_str]),
                json.dumps(emb),
            ),
        )
        cur = self.conn.execute(
            """
            INSERT INTO passes (vehicle_id, start_ts, end_ts, start_iso, direction, cameras, is_new_vehicle, match_score, match_method)
            VALUES (?, ?, ?, ?, ?, ?, 1, ?, 'new')
            """,
            (
                new_vid,
                ts_s,
                ts_s,
                ts_iso,
                direction,
                json.dumps([cam_str]),
                round(float(best_raw_sim), 4),
            ),
        )
        return {
            "vehicle_id": new_vid,
            "pass_id": int(cur.lastrowid),
            "is_unfamiliar": 1,
            "best_sim": round(float(best_raw_sim), 4),
            "best_match_vehicle_id": best_raw_vid,
            "match_method": "new",
            "new_pass": True,
        }

    def _update_vehicle_rollups(self, vehicle_id, ts_s, ts_iso, cam_str, crop_path, crop_w, crop_h, emb, attrs, new_pass):
        v = self.conn.execute("SELECT * FROM vehicles WHERE vehicle_id = ?", (vehicle_id,)).fetchone()
        if v is None:
            return None

        sighting_count = int(v["sighting_count"] or 0) + 1
        pass_count = int(v["pass_count"] or 0) + (1 if new_pass else 0)
        is_familiar = 1 if (pass_count >= 2 or int(v["is_familiar"] or 0) == 1) else 0

        cams = json.loads(v["cameras_seen"] or "[]")
        if cam_str not in cams:
            cams.append(cam_str)

        first_ts = min(float(v["first_seen_ts"] or ts_s), ts_s)
        first_iso = v["first_seen_iso"] if float(v["first_seen_ts"] or ts_s) <= ts_s else ts_iso
        last_ts = max(float(v["last_seen_ts"] or ts_s), ts_s)
        last_iso = ts_iso if ts_s >= float(v["last_seen_ts"] or 0.0) else v["last_seen_iso"]

        # Running L2-normalized mean embedding
        if v["mean_embedding"] and sighting_count > 1:
            old_mean = np.asarray(json.loads(v["mean_embedding"]), dtype=np.float32)
            new_vec = np.asarray(emb, dtype=np.float32)
            # Weight high-res crops slightly higher in running mean
            w_new = 1.5 if (crop_w * crop_h >= 120000) else 1.0
            combined = old_mean * float(sighting_count - 1) + new_vec * w_new
            mean_emb = l2_normalize(combined)
        else:
            mean_emb = l2_normalize(emb)

        # Representative crop: pick highest resolution crop seen for this vehicle
        best_s = self.conn.execute(
            """
            SELECT crop_path, (COALESCE(crop_w, 0) * COALESCE(crop_h, 0)) AS area
            FROM sightings
            WHERE vehicle_id = ? AND crop_path IS NOT NULL
            ORDER BY area DESC
            LIMIT 1
            """,
            (vehicle_id,),
        ).fetchone()
        rep_crop = v["rep_crop_path"] or crop_path
        if best_s and (crop_w * crop_h) >= int(best_s["area"] or 0):
            rep_crop = crop_path
        elif best_s and best_s["crop_path"]:
            rep_crop = best_s["crop_path"]

        # Enrich attributes if previously unknown
        color = v["color"] if (v["color"] and v["color"] != "unknown") else (attrs.get("color") or "unknown")
        body_type = v["body_type"] if (v["body_type"] and v["body_type"] != "unknown") else (attrs.get("body_type") or "unknown")
        roof_trim = v["roof_trim"] if (v["roof_trim"] and v["roof_trim"] not in ("none", "unknown")) else (attrs.get("roof_trim") or "none")
        make_model = v["make_model"] if (v["make_model"] and v["make_model"] != "unknown") else (attrs.get("make_model") or "unknown")
        plate_partial = v["plate_partial"] if v["plate_partial"] else (attrs.get("plate_partial") or "")

        label = v["label"]
        has_custom_label = int(v["custom_label"] or 0) == 1 if "custom_label" in v.keys() else False
        if not has_custom_label:
            if not label or "unknown" in label.lower() or (v["make_model"] == "unknown" and make_model != "unknown"):
                label = attrs.get("label") or _build_vehicle_label(v["cls"], color, body_type, roof_trim, make_model)

        self.conn.execute(
            """
            UPDATE vehicles
            SET label = ?, color = ?, body_type = ?, roof_trim = ?, make_model = ?, plate_partial = ?,
                first_seen_ts = ?, first_seen_iso = ?, last_seen_ts = ?, last_seen_iso = ?,
                sighting_count = ?, pass_count = ?, cameras_seen = ?, is_familiar = ?,
                rep_crop_path = ?, mean_embedding = ?
            WHERE vehicle_id = ?
            """,
            (
                label, color, body_type, roof_trim, make_model, plate_partial,
                first_ts, first_iso, last_ts, last_iso,
                sighting_count, pass_count, json.dumps(cams), is_familiar,
                rep_crop, json.dumps(mean_emb),
                vehicle_id,
            ),
        )
        return label

    def _update_chroma_metadata(self, doc_id, vehicle_id, vehicle_label, pass_id, is_unfamiliar):
        if self._chroma_col is None:
            return
        try:
            got = self._chroma_col.get(ids=[doc_id], include=["metadatas"])
            if not got or not got.get("ids"):
                return
            meta = dict((got.get("metadatas") or [{}])[0] or {})
            meta["vehicle_id"] = str(vehicle_id)
            meta["vehicle_label"] = str(vehicle_label or "")
            meta["pass_id"] = int(pass_id)
            meta["is_unfamiliar"] = int(is_unfamiliar)
            self._chroma_col.update(ids=[doc_id], metadatas=[meta])
        except Exception as e:
            log.debug(f"Chroma metadata update skipped for {doc_id}: {e}")

    def process_event_record(self, ev):
        doc_id = ev.get("doc_id")
        if not doc_id or self.is_doc_processed(doc_id):
            return None

        reid_res = self.call_reid_server(ev)
        if not reid_res.get("ok"):
            raise RuntimeError(f"reid_server error for {doc_id}: {reid_res.get('error')}")

        emb = reid_res["embedding"]
        attrs = reid_res.get("attrs") or {}
        crop_shape = reid_res.get("crop_shape") or ev.get("main_crop_shape") or ev.get("sub_crop_shape") or [0, 0]
        crop_h, crop_w = int(crop_shape[0]), int(crop_shape[1])
        crop_source = reid_res.get("crop_source") or "raw"

        # Save YOLO-refined crop if reid_server refined a main crop / main frame
        crop_path = ev.get("main_crop_path") or ev.get("sub_crop_path") or ev.get("sub_snapshot_path")
        refined_b64 = reid_res.get("refined_crop_b64")
        if refined_b64 and crop_source in ("main_crop_yolo", "main_frame_yolo", "sub_crop_vlm_rescued"):
            os.makedirs(VEHICLE_SNAPSHOT_DIR, exist_ok=True)
            refined_rel = f"snapshots/vehicles/{doc_id}_reid_crop.jpg"
            try:
                with open(refined_rel, "wb") as fh:
                    fh.write(base64.b64decode(refined_b64))
                crop_path = refined_rel
            except Exception as e:
                log.warning(f"Failed saving refined crop for {doc_id}: {e}")
        elif crop_source in ("sub_crop_fallback", "sub_crop_no_main_yolo") and ev.get("sub_crop_path"):
            crop_path = ev.get("sub_crop_path")

        cam_raw = ev.get("cam")
        cam_str = cam_raw if str(cam_raw).startswith("cam") else f"cam{cam_raw}"
        cls = ev.get("cls") or "car"
        ts_s = float(ev.get("timestamp_s") or time.time())
        ts_iso = ev.get("timestamp") or datetime.datetime.fromtimestamp(ts_s).isoformat()
        direction = _infer_direction(cam_str, attrs, ev.get("vlm_description") or "")

        match = self.match_or_assign(
            cls=cls,
            cam_str=cam_str,
            ts_s=ts_s,
            ts_iso=ts_iso,
            emb=emb,
            attrs=attrs,
            direction=direction,
        )

        vehicle_id = match["vehicle_id"]
        pass_id = match["pass_id"]
        is_unfamiliar = match["is_unfamiliar"]
        best_sim = match["best_sim"]
        best_match_vid = match["best_match_vehicle_id"]
        match_method = match["match_method"]

        bbox_used = reid_res.get("refined_bbox") or ev.get("main_bbox") or ev.get("sub_bbox")
        full_frame_path = ev.get("main_frame_path") or ev.get("sub_snapshot_path")

        self.conn.execute(
            """
            INSERT INTO sightings (
                doc_id, evt_id, pass_id, vehicle_id, timestamp_s, timestamp_iso,
                cam, track_id, cls, conf, crop_path, full_frame_path, sub_snapshot_path,
                bbox, crop_w, crop_h, vlm_description, attrs_json, embedding,
                best_sim, best_match_vehicle_id, is_unfamiliar
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                ev.get("evt_id"),
                pass_id,
                vehicle_id,
                ts_s,
                ts_iso,
                cam_str,
                ev.get("track_id"),
                cls,
                float(ev.get("best_conf") or ev.get("conf") or 0.0),
                crop_path,
                full_frame_path,
                ev.get("sub_snapshot_path"),
                json.dumps(bbox_used) if bbox_used is not None else None,
                crop_w,
                crop_h,
                ev.get("vlm_description"),
                json.dumps(attrs),
                json.dumps(emb),
                best_sim,
                best_match_vid,
                is_unfamiliar,
            ),
        )

        vehicle_label = self._update_vehicle_rollups(
            vehicle_id=vehicle_id,
            ts_s=ts_s,
            ts_iso=ts_iso,
            cam_str=cam_str,
            crop_path=crop_path,
            crop_w=crop_w,
            crop_h=crop_h,
            emb=emb,
            attrs=attrs,
            new_pass=match["new_pass"],
        )
        self.conn.commit()

        # Update ChromaDB metadata if doc exists
        self._update_chroma_metadata(
            doc_id=doc_id,
            vehicle_id=vehicle_id,
            vehicle_label=vehicle_label,
            pass_id=pass_id,
            is_unfamiliar=is_unfamiliar,
        )

        # Append enriched summary record to logs/vehicle_sightings.jsonl
        summary = {
            "timestamp": ts_iso,
            "timestamp_s": ts_s,
            "doc_id": doc_id,
            "evt_id": ev.get("evt_id"),
            "cam": cam_str,
            "cls": cls,
            "vehicle_id": vehicle_id,
            "vehicle_label": vehicle_label,
            "pass_id": pass_id,
            "is_unfamiliar": is_unfamiliar,
            "match_method": match_method,
            "best_sim": best_sim,
            "best_match_vehicle_id": best_match_vid,
            "crop_path": crop_path,
            "crop_source": crop_source,
            "crop_shape": [crop_h, crop_w],
            "attrs": {
                "color": attrs.get("color"),
                "body_type": attrs.get("body_type"),
                "roof_trim": attrs.get("roof_trim"),
                "make_model": attrs.get("make_model"),
                "plate_partial": attrs.get("plate_partial"),
                "motion": attrs.get("motion"),
            },
        }
        os.makedirs(os.path.dirname(VEHICLE_SIGHTINGS_LOG) or ".", exist_ok=True)
        with open(VEHICLE_SIGHTINGS_LOG, "a") as fh:
            fh.write(json.dumps(summary) + "\n")

        log.info(
            f"[ReID] {doc_id} ({cam_str} {cls}) -> {vehicle_id} [{vehicle_label}] "
            f"pass={pass_id} method={match_method} sim={best_sim:.3f} unfamiliar={is_unfamiliar} "
            f"crop={crop_source}({crop_w}x{crop_h})"
        )
        return summary


    def recompute_vehicle_from_db(self, vehicle_id):
        """Recompute all rollup fields and L2-normalized mean_embedding for vehicle_id from its current passes & sightings."""
        v = self.conn.execute("SELECT * FROM vehicles WHERE vehicle_id = ?", (vehicle_id,)).fetchone()
        if v is None:
            return None
        sights = self.conn.execute(
            "SELECT * FROM sightings WHERE vehicle_id = ? ORDER BY timestamp_s ASC, sighting_id ASC",
            (vehicle_id,),
        ).fetchall()
        passes = self.conn.execute(
            "SELECT * FROM passes WHERE vehicle_id = ? ORDER BY start_ts ASC, pass_id ASC",
            (vehicle_id,),
        ).fetchall()
        if not sights:
            self.conn.execute(
                """
                UPDATE vehicles
                SET sighting_count = 0, pass_count = 0, cameras_seen = '[]'
                WHERE vehicle_id = ?
                """,
                (vehicle_id,),
            )
            self.conn.commit()
            return dict(self.conn.execute("SELECT * FROM vehicles WHERE vehicle_id = ?", (vehicle_id,)).fetchone())

        sighting_count = len(sights)
        pass_count = len(passes)
        first_ts = float(sights[0]["timestamp_s"])
        first_iso = sights[0]["timestamp_iso"]
        last_ts = float(sights[-1]["timestamp_s"])
        last_iso = sights[-1]["timestamp_iso"]

        cams = []
        vecs = []
        weights = []
        best_area = -1
        rep_crop = v["rep_crop_path"]

        color = v["color"] or "unknown"
        body_type = v["body_type"] or "unknown"
        roof_trim = v["roof_trim"] or "none"
        make_model = v["make_model"] or "unknown"
        plate_partial = v["plate_partial"] or ""

        for s in sights:
            c = s["cam"]
            if c and c not in cams:
                cams.append(c)
            area = int(s["crop_w"] or 0) * int(s["crop_h"] or 0)
            if s["crop_path"] and area >= best_area:
                best_area = area
                rep_crop = s["crop_path"]
            if s["embedding"]:
                try:
                    vec = np.asarray(json.loads(s["embedding"]), dtype=np.float32)
                    if vec.size > 0:
                        vecs.append(vec)
                        weights.append(1.5 if area >= 120000 else 1.0)
                except Exception:
                    pass
            if s["attrs_json"]:
                try:
                    at = json.loads(s["attrs_json"])
                    if color in ("unknown", "") and at.get("color") and at.get("color") != "unknown":
                        color = at["color"]
                    if body_type in ("unknown", "") and at.get("body_type") and at.get("body_type") != "unknown":
                        body_type = at["body_type"]
                    if roof_trim in ("none", "unknown", "") and at.get("roof_trim") and at.get("roof_trim") not in ("none", "unknown"):
                        roof_trim = at["roof_trim"]
                    if make_model in ("unknown", "") and at.get("make_model") and at.get("make_model") != "unknown":
                        make_model = at["make_model"]
                    if not plate_partial and at.get("plate_partial"):
                        plate_partial = at["plate_partial"]
                except Exception:
                    pass

        mean_emb_json = v["mean_embedding"]
        if vecs:
            w_arr = np.asarray(weights, dtype=np.float32)[:, None]
            stacked = np.stack(vecs, axis=0)
            mean_emb = l2_normalize(np.sum(stacked * w_arr, axis=0))
            mean_emb_json = json.dumps(mean_emb)

        # Refresh each pass start/end/cameras from its sightings
        for idx, p_row in enumerate(passes):
            pid = p_row["pass_id"]
            p_sights = [s for s in sights if int(s["pass_id"] or 0) == int(pid)]
            if not p_sights:
                self.conn.execute("DELETE FROM passes WHERE pass_id = ?", (pid,))
                continue
            p_cams = []
            for s in p_sights:
                if s["cam"] and s["cam"] not in p_cams:
                    p_cams.append(s["cam"])
            p_start = min(float(s["timestamp_s"]) for s in p_sights)
            p_end = max(float(s["timestamp_s"]) for s in p_sights)
            p_iso = p_sights[0]["timestamp_iso"]
            is_new = 1 if idx == 0 else 0
            self.conn.execute(
                "UPDATE passes SET start_ts = ?, end_ts = ?, start_iso = ?, cameras = ?, is_new_vehicle = ? WHERE pass_id = ?",
                (p_start, p_end, p_iso, json.dumps(p_cams), is_new, pid),
            )

        pass_count = self.conn.execute("SELECT COUNT(*) AS c FROM passes WHERE vehicle_id = ?", (vehicle_id,)).fetchone()["c"]
        is_familiar = 1 if (pass_count >= 2 or int(v["is_familiar"] or 0) == 1) else 0

        label = v["label"]
        has_custom = int(v["custom_label"] or 0) == 1 if "custom_label" in v.keys() else False
        if not has_custom and (not label or "unknown" in label.lower()):
            label = _build_vehicle_label(v["cls"], color, body_type, roof_trim, make_model)

        self.conn.execute(
            """
            UPDATE vehicles
            SET label = ?, color = ?, body_type = ?, roof_trim = ?, make_model = ?, plate_partial = ?,
                first_seen_ts = ?, first_seen_iso = ?, last_seen_ts = ?, last_seen_iso = ?,
                sighting_count = ?, pass_count = ?, cameras_seen = ?, is_familiar = ?,
                rep_crop_path = ?, mean_embedding = ?
            WHERE vehicle_id = ?
            """,
            (
                label, color, body_type, roof_trim, make_model, plate_partial,
                first_ts, first_iso, last_ts, last_iso,
                sighting_count, pass_count, json.dumps(cams), is_familiar,
                rep_crop, mean_emb_json, vehicle_id,
            ),
        )
        self.conn.commit()
        return dict(self.conn.execute("SELECT * FROM vehicles WHERE vehicle_id = ?", (vehicle_id,)).fetchone())

    def update_vehicle_details(self, vehicle_id, updates):
        v = self.conn.execute("SELECT * FROM vehicles WHERE vehicle_id = ?", (vehicle_id,)).fetchone()
        if v is None:
            raise ValueError(f"Unknown vehicle_id: {vehicle_id}")

        label = v["label"]
        custom_label = int(v["custom_label"] or 0) if "custom_label" in v.keys() else 0
        if "label" in updates and updates["label"] is not None:
            new_lbl = str(updates["label"]).strip()
            if new_lbl:
                label = new_lbl
                custom_label = int(updates.get("custom_label", 1))
        elif "custom_label" in updates and updates["custom_label"] is not None:
            custom_label = int(updates["custom_label"])

        is_familiar = int(v["is_familiar"] or 0)
        if "is_familiar" in updates and updates["is_familiar"] is not None:
            is_familiar = 1 if int(updates["is_familiar"]) else 0

        color = str(updates.get("color", v["color"] or "unknown")).strip() or "unknown"
        body_type = str(updates.get("body_type", v["body_type"] or "unknown")).strip() or "unknown"
        roof_trim = str(updates.get("roof_trim", v["roof_trim"] or "none")).strip() or "none"
        make_model = str(updates.get("make_model", v["make_model"] or "unknown")).strip() or "unknown"
        plate_partial = str(updates.get("plate_partial", v["plate_partial"] or "")).strip()
        notes = str(updates.get("notes", v["notes"] if "notes" in v.keys() and v["notes"] else "")).strip()

        self.conn.execute(
            """
            UPDATE vehicles
            SET label = ?, custom_label = ?, is_familiar = ?, color = ?, body_type = ?,
                roof_trim = ?, make_model = ?, plate_partial = ?, notes = ?
            WHERE vehicle_id = ?
            """,
            (label, custom_label, is_familiar, color, body_type, roof_trim, make_model, plate_partial, notes, vehicle_id),
        )
        # If familiarity was explicitly toggled, also reflect on sightings.is_unfamiliar
        if "is_familiar" in updates and updates["is_familiar"] is not None:
            unfam = 0 if is_familiar else 1
            self.conn.execute("UPDATE sightings SET is_unfamiliar = ? WHERE vehicle_id = ?", (unfam, vehicle_id))
        self.conn.commit()

        sights = self.conn.execute("SELECT doc_id, pass_id, is_unfamiliar FROM sightings WHERE vehicle_id = ?", (vehicle_id,)).fetchall()
        for s in sights:
            self._update_chroma_metadata(s["doc_id"], vehicle_id, label, s["pass_id"], s["is_unfamiliar"])

        return dict(self.conn.execute("SELECT * FROM vehicles WHERE vehicle_id = ?", (vehicle_id,)).fetchone())

    def merge_vehicles(self, source_id, target_id):
        if source_id == target_id:
            raise ValueError("source_id and target_id must be different")
        src = self.conn.execute("SELECT * FROM vehicles WHERE vehicle_id = ?", (source_id,)).fetchone()
        tgt = self.conn.execute("SELECT * FROM vehicles WHERE vehicle_id = ?", (target_id,)).fetchone()
        if src is None:
            raise ValueError(f"Unknown source vehicle_id: {source_id}")
        if tgt is None:
            raise ValueError(f"Unknown target vehicle_id: {target_id}")
        if tgt["merged_into"]:
            raise ValueError(f"Target vehicle {target_id} is already merged into {tgt['merged_into']}")

        src_sights = self.conn.execute("SELECT doc_id, pass_id FROM sightings WHERE vehicle_id = ?", (source_id,)).fetchall()

        # Reassign passes and sightings from source_id to target_id
        self.conn.execute("UPDATE passes SET vehicle_id = ? WHERE vehicle_id = ?", (target_id, source_id))
        self.conn.execute("UPDATE sightings SET vehicle_id = ? WHERE vehicle_id = ?", (target_id, source_id))
        # Also redirect any older vehicle that had merged_into = source_id
        self.conn.execute("UPDATE vehicles SET merged_into = ? WHERE merged_into = ?", (target_id, source_id))
        # Mark source_id as merged into target_id
        was_familiar = 1 if (int(src["is_familiar"] or 0) == 1 or int(tgt["is_familiar"] or 0) == 1) else 0
        if was_familiar:
            self.conn.execute("UPDATE vehicles SET is_familiar = 1 WHERE vehicle_id = ?", (target_id,))
        # Preserve custom label if source had one and target did not
        if int(src["custom_label"] or 0) == 1 and int(tgt["custom_label"] or 0) == 0:
            self.conn.execute(
                "UPDATE vehicles SET label = ?, custom_label = 1 WHERE vehicle_id = ?",
                (src["label"], target_id),
            )
        self.conn.execute(
            "UPDATE vehicles SET merged_into = ?, sighting_count = 0, pass_count = 0 WHERE vehicle_id = ?",
            (target_id, source_id),
        )
        self.conn.commit()

        updated_tgt = self.recompute_vehicle_from_db(target_id)
        if int(updated_tgt.get("is_familiar") or 0) == 1:
            self.conn.execute("UPDATE sightings SET is_unfamiliar = 0 WHERE vehicle_id = ?", (target_id,))
            self.conn.commit()

        all_tgt_sights = self.conn.execute(
            "SELECT doc_id, pass_id, is_unfamiliar FROM sightings WHERE vehicle_id = ?",
            (target_id,),
        ).fetchall()
        for s in all_tgt_sights:
            self._update_chroma_metadata(
                s["doc_id"], target_id, updated_tgt["label"], s["pass_id"], s["is_unfamiliar"]
            )
        return updated_tgt

    def split_vehicle_items(self, sighting_ids=None, pass_ids=None, target_vehicle_id="new"):
        sighting_ids = [int(x) for x in (sighting_ids or []) if x is not None]
        pass_ids = [int(x) for x in (pass_ids or []) if x is not None]
        if not sighting_ids and not pass_ids:
            raise ValueError("Provide at least one sighting_id or pass_id to split")

        rows = []
        if pass_ids:
            ph = ",".join("?" for _ in pass_ids)
            rows.extend(self.conn.execute(f"SELECT * FROM sightings WHERE pass_id IN ({ph})", tuple(pass_ids)).fetchall())
        if sighting_ids:
            ph = ",".join("?" for _ in sighting_ids)
            rows.extend(self.conn.execute(f"SELECT * FROM sightings WHERE sighting_id IN ({ph})", tuple(sighting_ids)).fetchall())

        by_sid = {int(r["sighting_id"]): r for r in rows}
        if not by_sid:
            raise ValueError("No matching sightings found for split")

        affected_Sightings = sorted(by_sid.values(), key=lambda r: float(r["timestamp_s"]))
        source_vids = sorted({r["vehicle_id"] for r in affected_Sightings if r["vehicle_id"]})

        first_s = affected_Sightings[0]
        cls = first_s["cls"] or "car"
        attrs = json.loads(first_s["attrs_json"] or "{}")
        emb = json.loads(first_s["embedding"] or "[]")
        ts_s = float(first_s["timestamp_s"])
        ts_iso = first_s["timestamp_iso"]
        cam_str = first_s["cam"]

        created_new = False
        if not target_vehicle_id or target_vehicle_id == "new":
            target_vid = self._next_vehicle_id()
            created_new = True
            color = attrs.get("color") or "unknown"
            body_type = attrs.get("body_type") or ("bicycle" if cls == "bicycle" else "unknown")
            roof_trim = attrs.get("roof_trim") or "none"
            make_model = attrs.get("make_model") or "unknown"
            plate_partial = attrs.get("plate_partial") or ""
            label = attrs.get("label") or _build_vehicle_label(cls, color, body_type, roof_trim, make_model)
            self.conn.execute(
                """
                INSERT INTO vehicles (
                    vehicle_id, cls, label, color, body_type, roof_trim, make_model, plate_partial,
                    first_seen_ts, first_seen_iso, last_seen_ts, last_seen_iso,
                    sighting_count, pass_count, cameras_seen, is_familiar, rep_crop_path, mean_embedding, merged_into,
                    custom_label, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, 0, ?, ?, NULL, 0, '')
                """,
                (
                    target_vid, cls, label, color, body_type, roof_trim, make_model, plate_partial,
                    ts_s, ts_iso, ts_s, ts_iso, json.dumps([cam_str]), first_s["crop_path"], json.dumps(emb),
                ),
            )
        else:
            target_vid = str(target_vehicle_id).strip()
            t_row = self.conn.execute("SELECT * FROM vehicles WHERE vehicle_id = ?", (target_vid,)).fetchone()
            if t_row is None:
                raise ValueError(f"Unknown target_vehicle_id: {target_vid}")
            if t_row["merged_into"]:
                raise ValueError(f"Target vehicle {target_vid} is merged into {t_row['merged_into']}")

        # Move whole passes when all sightings of that pass are included; otherwise create a dedicated pass for split sightings
        sids_set = set(by_sid.keys())
        moved_pass_ids = set(pass_ids)
        for pid in list({int(r["pass_id"]) for r in affected_Sightings if r["pass_id"] is not None}):
            all_in_pass = self.conn.execute("SELECT sighting_id FROM sightings WHERE pass_id = ?", (pid,)).fetchall()
            if all(int(x["sighting_id"]) in sids_set for x in all_in_pass):
                moved_pass_ids.add(pid)

        for pid in moved_pass_ids:
            self.conn.execute(
                "UPDATE passes SET vehicle_id = ?, match_method = 'manual_split' WHERE pass_id = ?",
                (target_vid, pid),
            )
            self.conn.execute(
                "UPDATE sightings SET vehicle_id = ? WHERE pass_id = ?",
                (target_vid, pid),
            )

        orphan_sights = [r for r in affected_Sightings if int(r["pass_id"] or 0) not in moved_pass_ids]
        if orphan_sights:
            o_cams = []
            for r in orphan_sights:
                if r["cam"] and r["cam"] not in o_cams:
                    o_cams.append(r["cam"])
            o_start = min(float(r["timestamp_s"]) for r in orphan_sights)
            o_end = max(float(r["timestamp_s"]) for r in orphan_sights)
            o_iso = orphan_sights[0]["timestamp_iso"]
            direction = _infer_direction(orphan_sights[0]["cam"], attrs, orphan_sights[0]["vlm_description"] or "")
            cur = self.conn.execute(
                """
                INSERT INTO passes (vehicle_id, start_ts, end_ts, start_iso, direction, cameras, is_new_vehicle, match_score, match_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1.0, 'manual_split')
                """,
                (target_vid, o_start, o_end, o_iso, direction, json.dumps(o_cams), 1 if created_new else 0),
            )
            new_pid = int(cur.lastrowid)
            for r in orphan_sights:
                self.conn.execute(
                    "UPDATE sightings SET vehicle_id = ?, pass_id = ? WHERE sighting_id = ?",
                    (target_vid, new_pid, int(r["sighting_id"])),
                )

        self.conn.commit()
        updated_target = self.recompute_vehicle_from_db(target_vid)
        updated_sources = []
        for svid in source_vids:
            if svid != target_vid:
                rem = self.conn.execute("SELECT COUNT(*) AS c FROM sightings WHERE vehicle_id = ?", (svid,)).fetchone()["c"]
                if rem == 0:
                    self.conn.execute("UPDATE vehicles SET merged_into = ?, sighting_count = 0, pass_count = 0 WHERE vehicle_id = ?", (target_vid, svid))
                    self.conn.commit()
                else:
                    updated_sources.append(self.recompute_vehicle_from_db(svid))

        for r in self.conn.execute("SELECT doc_id, pass_id, is_unfamiliar FROM sightings WHERE vehicle_id = ?", (target_vid,)).fetchall():
            self._update_chroma_metadata(r["doc_id"], target_vid, updated_target["label"], r["pass_id"], r["is_unfamiliar"])

        return {
            "target_vehicle": updated_target,
            "updated_source_vehicles": updated_sources,
        }

    def sync_all_pending(self):
        if not os.path.exists(VEHICLE_EVENTS_LOG):
            return 0
        with open(VEHICLE_EVENTS_LOG, "r") as fh:
            lines = [line.strip() for line in fh if line.strip()]
        events = []
        for line in lines:
            try:
                events.append(json.loads(line))
            except Exception:
                pass
        # Sort chronologically by timestamp_s so passes group accurately
        events.sort(key=lambda x: float(x.get("timestamp_s") or 0.0))
        processed = 0
        for ev in events:
            doc_id = ev.get("doc_id")
            if not doc_id or self.is_doc_processed(doc_id):
                continue
            try:
                self.process_event_record(ev)
                processed += 1
            except Exception as e:
                log.warning(f"Failed processing {doc_id}: {e}")
        return processed


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Vehicle Re-ID Worker")
    parser.add_argument("--once", action="store_true", help="Backfill pending events and exit")
    parser.add_argument("--recluster", action="store_true", help="Recluster existing sightings in DB and exit")
    parser.add_argument("--db", default=VEHICLE_DB_PATH, help="SQLite DB path")
    parser.add_argument("--reid-url", default=REID_URL, help="reid_server URL on gx10")
    parser.add_argument("--poll-interval", type=float, default=1.5, help="Poll interval in seconds")
    parser.add_argument("--no-chroma", action="store_true", help="Skip ChromaDB metadata updates")
    args = parser.parse_args()

    worker = VehicleReidWorker(
        db_path=args.db,
        reid_url=args.reid_url,
        update_chroma=not args.no_chroma,
    )

    stop_flag = [False]
    def _handle_sig(signum, frame):
        log.info(f"Received signal {signum}, shutting down vehicle_reid_worker...")
        stop_flag[0] = True

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    if args.recluster:
        n = worker.recluster_existing_sightings()
        log.info(f"Reclustered {n} sightings.")
        return
    n = worker.sync_all_pending()
    log.info(f"Initial backfill complete: {n} events processed.")
    if args.once:
        return

    log.info(f"Tailing {VEHICLE_EVENTS_LOG} (poll={args.poll_interval}s, reid_url={args.reid_url}) ...")
    while not stop_flag[0]:
        try:
            worker.sync_all_pending()
        except Exception as e:
            log.warning(f"Poll error: {e}")
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
