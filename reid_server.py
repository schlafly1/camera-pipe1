"-^Welcome, Roger!"
#!/usr/bin/env python3
"""
reid_server.py — FastReID (sbs_R50-ibn) + YOLO11m crop refinement + Gemma-4-12B
structured vehicle attribute extraction service running on gx10.

Exposes:
  GET  /health  -> JSON status, device, loaded models, uptime
  POST /embed   -> Accepts base64 vehicle crop(s) and optional full main frame,
                   refines the vehicle crop using YOLO11m, computes the 2048-d
                   L2-normalized FastReID embedding, and extracts structured
                   vehicle attributes via local vLLM (google/gemma-4-12B-it).

Standing rule: runs on gx10 so zero heavy ReID / YOLO / VLM GPU load is placed
on Jetson Thor (thor2).
"""

import os
# Must be set before importing torch on GB10 (gx10)
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import argparse
import base64
import json
import logging
import re
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("reid_server")

REID_SCRATCH = os.path.expanduser(os.environ.get("REID_SCRATCH", "~/reid_scratch"))
FASTREID_DIR = os.environ.get("FASTREID_DIR", os.path.join(REID_SCRATCH, "fast-reid"))
FASTREID_CFG = os.environ.get(
    "FASTREID_CFG",
    os.path.join(FASTREID_DIR, "configs", "VeRi", "sbs_R50-ibn.yml"),
)
FASTREID_WEIGHTS = os.environ.get(
    "FASTREID_WEIGHTS",
    os.path.join(REID_SCRATCH, "veri_sbs_R50-ibn.pth"),
)
YOLO_WEIGHTS = os.environ.get(
    "YOLO_WEIGHTS",
    os.path.join(REID_SCRATCH, "yolo11m.pt"),
)
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000").rstrip("/")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "google/gemma-4-12B-it")
REID_HOST = os.environ.get("REID_HOST", "0.0.0.0")
REID_PORT = int(os.environ.get("REID_PORT", "8090"))

# COCO class IDs in yolo11m.pt
YOLO_CAR_CLASSES = [2, 5, 7]       # car, bus, truck
YOLO_BIKE_CLASSES = [1, 3, 0]      # bicycle, motorcycle, person (cyclist)

_VLM_ATTR_PROMPT_CAR = (
    "This is a cropped security camera image of a vehicle. Focus ONLY on the primary vehicle in this crop. "
    "Do not guess license plate characters unless clearly legible. "
    "Reply in this exact key=value format on a single line separated by semicolons:\n"
    "PRESENT=<yes|no>; COLOR=<white|black|silver|gray|dark gray|red|maroon|blue|dark blue|green|tan|brown|yellow|orange|unknown>; "
    "BODY=<sedan|suv|crossover|pickup|van|truck|coupe|hatchback|wagon|unknown>; "
    "ROOF=<none|roof rack|black roof|white roof|camper shell|open bed|emergency lights|side steps>; "
    "MAKE_MODEL=<best make/model estimate e.g. Ram 1500, Toyota RAV4, Jeep Wrangler, Ford Explorer, Toyota Corolla, or unknown>; "
    "PLATE=<legible plate characters or NONE>; MOTION=<moving|parked>; "
    "LABEL=<concise 3 to 6 word vehicle description e.g. Gray Ram 1500 pickup, Silver compact sedan, Dark blue SUV>"
)

_VLM_ATTR_PROMPT_BIKE = (
    "This is a cropped security camera image of a bicycle/cyclist. Focus ONLY on the bicycle and rider in this crop. "
    "Reply in this exact key=value format on a single line separated by semicolons:\n"
    "PRESENT=<yes|no>; COLOR=<black|gray|silver|red|blue|white|green|yellow|orange|unknown>; "
    "BODY=<mountain_bike|road_bike|ebike|bicycle>; "
    "ROOF=<rider helmet/clothing e.g. white helmet light shirt, black helmet, or parked>; "
    "MAKE_MODEL=<mountain bike, road bike, or bicycle>; "
    "PLATE=NONE; MOTION=<moving|parked>; "
    "LABEL=<concise 3 to 6 word description e.g. Red mountain bike, Dark mountain bike w/ rider>"
)


def _decode_b64_img(b64_str):
    if not b64_str:
        return None
    try:
        raw = base64.b64decode(b64_str)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img if (img is not None and img.size > 0) else None
    except Exception:
        return None


def _encode_b64_img(img, quality=92):
    if img is None or img.size == 0:
        return None
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _bbox_iou(a, b):
    if not a or not b or len(a) < 4 or len(b) < 4:
        return 0.0
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0.0 else 0.0


def _bbox_center_dist(a, b):
    if not a or not b or len(a) < 4 or len(b) < 4:
        return 1.0
    cax = 0.5 * (float(a[0]) + float(a[2]))
    cay = 0.5 * (float(a[1]) + float(a[3]))
    cbx = 0.5 * (float(b[0]) + float(b[2]))
    cby = 0.5 * (float(b[1]) + float(b[3]))
    return float(((cax - cbx) ** 2 + (cay - cby) ** 2) ** 0.5)


def _pad_and_crop(img, box, pad_frac=0.06):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    px = bw * pad_frac
    py = bh * pad_frac
    ix1 = max(0, min(w - 1, int(round(x1 - px))))
    iy1 = max(0, min(h - 1, int(round(y1 - py))))
    ix2 = max(ix1 + 1, min(w, int(round(x2 + px))))
    iy2 = max(iy1 + 1, min(h, int(round(y2 + py))))
    return img[iy1:iy2, ix1:ix2], [ix1, iy1, ix2, iy2]


def _normalize_color(c):
    c = (c or "").strip().lower()
    if not c or c in ("none", "n/a", "unknown"):
        return "unknown"
    for tok in (
        "dark gray", "dark grey", "light gray", "light grey", "dark blue",
        "light blue", "dark red", "white", "black", "silver", "gray", "grey",
        "red", "maroon", "blue", "green", "tan", "beige", "brown", "gold",
        "yellow", "orange", "charcoal", "burgundy",
    ):
        if tok in c:
            if tok == "grey":
                return "gray"
            if tok == "dark grey":
                return "dark gray"
            if tok == "light grey" or tok == "light gray":
                return "silver"
            if tok == "charcoal":
                return "dark gray"
            if tok in ("maroon", "burgundy", "dark red"):
                return "red"
            if tok == "beige":
                return "tan"
            return tok
    return c.split()[0] if c.split() else "unknown"


def _normalize_body(b, cls="car"):
    b = (b or "").strip().lower()
    if cls == "bicycle":
        if "mountain" in b or "mtb" in b:
            return "mountain_bike"
        if "road" in b:
            return "road_bike"
        if "ebike" in b or "e-bike" in b or "electric" in b:
            return "ebike"
        return "bicycle"
    for tok in ("pickup", "crossover", "suv", "sedan", "hatchback", "coupe", "van", "minivan", "wagon", "truck"):
        if tok in b:
            if tok == "minivan":
                return "van"
            return tok
    if "jeep" in b or "4x4" in b:
        return "suv"
    return "unknown"


def parse_vlm_kv(raw_text, cls="car", fallback_desc=""):
    text = (raw_text or "").strip()
    def _get(k):
        m = re.search(rf"\b{k}\s*=\s*([^;\n]+)", text, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    present_s = _get("PRESENT").lower()
    present = False if present_s in ("no", "false", "0") else True
    color = _normalize_color(_get("COLOR"))
    body_type = _normalize_body(_get("BODY"), cls=cls)
    roof_trim = _get("ROOF").strip().lower() or "none"
    make_model = _get("MAKE_MODEL").strip() or "unknown"
    if make_model.lower() in ("none", "n/a", "unknown", "unidentifiable"):
        make_model = "unknown"
    plate = _get("PLATE").strip().upper()
    if not plate or plate in ("NONE", "N/A", "UNKNOWN", "NO", "NOT VISIBLE", "NOT LEGIBLE"):
        plate = ""
    motion = _get("MOTION").strip().lower() or "moving"
    is_parked = 1 if "park" in motion else 0
    label = _get("LABEL").strip().strip('"\'.')

    # Fallback extraction from fallback_desc if VLM omitted fields
    if (color == "unknown" or body_type == "unknown") and fallback_desc:
        fd = fallback_desc.lower()
        if color == "unknown":
            color = _normalize_color(fd)
        if body_type == "unknown":
            body_type = _normalize_body(fd, cls=cls)

    if not label or label.lower() in ("none", "unknown"):
        parts = []
        if color != "unknown":
            parts.append(color.capitalize())
        if make_model != "unknown" and make_model.lower() not in ("bicycle", "mountain bike", "road bike"):
            parts.append(make_model)
        if body_type != "unknown":
            parts.append(body_type.replace("_", " "))
        else:
            parts.append("bicycle" if cls == "bicycle" else "vehicle")
        if roof_trim not in ("none", "unknown", ""):
            parts.append(f"w/ {roof_trim}")
        label = " ".join(parts)

    return {
        "present": present,
        "color": color,
        "body_type": body_type,
        "roof_trim": roof_trim,
        "make_model": make_model,
        "plate_partial": plate,
        "motion": motion,
        "is_parked": is_parked,
        "label": label,
        "raw": text,
    }


class ReidEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self.started_at = time.time()
        self.embed_calls = 0

        log.info(f"Loading YOLO11m from {YOLO_WEIGHTS} ...")
        from ultralytics import YOLO
        self.yolo = YOLO(YOLO_WEIGHTS)

        log.info(f"Loading FastReID sbs_R50-ibn from {FASTREID_CFG} / {FASTREID_WEIGHTS} ...")
        if FASTREID_DIR not in sys.path:
            sys.path.insert(0, FASTREID_DIR)
        from fastreid.config import get_cfg
        from fastreid.modeling.meta_arch import build_model
        from fastreid.utils.checkpoint import Checkpointer

        cfg = get_cfg()
        cfg.merge_from_file(FASTREID_CFG)
        cfg.MODEL.BACKBONE.PRETRAIN = False
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.MODEL.WEIGHTS = FASTREID_WEIGHTS
        self.reid_model = build_model(cfg)
        Checkpointer(self.reid_model).load(cfg.MODEL.WEIGHTS)
        self.reid_model.eval()
        self.input_h, self.input_w = cfg.INPUT.SIZE_TEST
        self.device = cfg.MODEL.DEVICE

        # Warmup both models
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        with self._lock:
            _ = self._embed_batch_unlocked([dummy])
            _ = self.yolo.predict(dummy, classes=YOLO_CAR_CLASSES, conf=0.3, verbose=False, imgsz=640)
        log.info(
            f"ReidEngine ready on {self.device} "
            f"(FastReID input={self.input_w}x{self.input_h}, YOLO={os.path.basename(YOLO_WEIGHTS)})"
        )

    def _embed_batch_unlocked(self, bgr_images):
        if not bgr_images:
            return np.zeros((0, 2048), dtype=np.float32)
        batch = []
        for im in bgr_images:
            rgb = im[:, :, ::-1]
            resized = cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
            batch.append(resized)
        x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).float()
        if self.device == "cuda":
            x = x.cuda()
        with torch.no_grad():
            feats = self.reid_model(x)
            feats = torch.nn.functional.normalize(feats, dim=1)
        return feats.cpu().numpy().astype(np.float32)

    def _yolo_detect_unlocked(self, img, cls="car", conf=0.25, imgsz=960):
        if img is None or img.size == 0:
            return []
        target_cls = YOLO_BIKE_CLASSES if cls == "bicycle" else YOLO_CAR_CLASSES
        res = self.yolo.predict(
            img,
            classes=target_cls,
            conf=conf,
            verbose=False,
            imgsz=imgsz,
            half=(self.device == "cuda"),
        )[0]
        h, w = img.shape[:2]
        out = []
        for bx, c, cf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            bw = bx[2] - bx[0]
            bh = bx[3] - bx[1]
            if bw < 28 or bh < 24:
                continue
            out.append({
                "box": [int(round(v)) for v in bx],
                "norm_box": [bx[0] / w, bx[1] / h, bx[2] / w, bx[3] / h],
                "cls_id": int(c),
                "cls_name": res.names[int(c)],
                "conf": float(cf),
                "area_frac": float((bw * bh) / max(1.0, w * h)),
            })
        return out

    def query_vlm_attrs(self, crop_img, cls="car", fallback_desc=""):
        prompt = _VLM_ATTR_PROMPT_BIKE if cls == "bicycle" else _VLM_ATTR_PROMPT_CAR
        b64 = _encode_b64_img(crop_img, quality=90)
        if not b64:
            return parse_vlm_kv("", cls=cls, fallback_desc=fallback_desc)
        payload = {
            "model": VLLM_MODEL,
            "max_tokens": 140,
            "temperature": 0.0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }
        try:
            req = urllib.request.Request(
                f"{VLLM_URL}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read())
            raw = data["choices"][0]["message"]["content"].strip()
            return parse_vlm_kv(raw, cls=cls, fallback_desc=fallback_desc)
        except Exception as e:
            log.warning(f"VLM attribute call failed: {e}")
            return parse_vlm_kv("", cls=cls, fallback_desc=fallback_desc)

    def process_event(self, payload):
        t0 = time.time()
        cls = payload.get("cls") or "car"
        main_crop_img = _decode_b64_img(payload.get("main_crop_b64") or payload.get("crop_b64") or payload.get("image_b64"))
        sub_crop_img = _decode_b64_img(payload.get("sub_crop_b64"))
        main_frame_img = _decode_b64_img(payload.get("main_frame_b64"))
        main_bbox_hint = payload.get("main_bbox")          # [x1, y1, x2, y2] in main_frame
        sub_bbox_norm = payload.get("sub_bbox_norm")       # [nx1, ny1, nx2, ny2] in [0,1]
        fallback_desc = payload.get("vlm_description") or ""
        extract_attrs = bool(payload.get("extract_attrs", True))
        return_refined = bool(payload.get("return_refined_crop", True))

        if main_crop_img is None and sub_crop_img is None and main_frame_img is None:
            raise ValueError("At least one of main_crop_b64, sub_crop_b64, or main_frame_b64 must be provided")

        with self._lock:
            self.embed_calls += 1

            # 1. Prepare sub_crop reference if available
            sub_ref_img = sub_crop_img
            if sub_crop_img is not None:
                sh, sw = sub_crop_img.shape[:2]
                if sw >= 96 and sh >= 72:
                    sdets = self._yolo_detect_unlocked(sub_crop_img, cls=cls, conf=0.25, imgsz=640)
                    if sdets:
                        best_s = max(sdets, key=lambda d: d["area_frac"] * d["conf"])
                        if best_s["area_frac"] >= 0.20:
                            sub_ref_img, _ = _pad_and_crop(sub_crop_img, best_s["box"], pad_frac=0.08)

            # 2. Gather high-res candidates from main_crop_img and main_frame_img
            candidates = []  # list of dict(source, img, box_in_main, norm_box, conf)
            if main_crop_img is not None:
                mc_h, mc_w = main_crop_img.shape[:2]
                mdets = self._yolo_detect_unlocked(main_crop_img, cls=cls, conf=0.22, imgsz=960)
                for d in mdets:
                    if d["area_frac"] < 0.04:
                        continue
                    cimg, cbox = _pad_and_crop(main_crop_img, d["box"], pad_frac=0.06)
                    if main_bbox_hint and len(main_bbox_hint) == 4:
                        gx1 = main_bbox_hint[0] + cbox[0]
                        gy1 = main_bbox_hint[1] + cbox[1]
                        gx2 = main_bbox_hint[0] + cbox[2]
                        gy2 = main_bbox_hint[1] + cbox[3]
                        gbox = [gx1, gy1, gx2, gy2]
                    else:
                        gbox = cbox
                    candidates.append({
                        "source": "main_crop_yolo",
                        "img": cimg,
                        "box": gbox,
                        "norm_box": sub_bbox_norm,
                        "conf": d["conf"],
                        "area_frac": d["area_frac"],
                    })
                # Always include main_crop_raw as a candidate in case YOLO missed a motion-blurred/small vehicle inside main_crop
                if mc_w <= 1200 and mc_h <= 900:
                    candidates.append({
                        "source": "main_crop_raw",
                        "img": main_crop_img,
                        "box": main_bbox_hint,
                        "norm_box": sub_bbox_norm,
                        "conf": 0.60,
                        "area_frac": 1.0,
                    })

            if main_frame_img is not None:
                mf_h, mf_w = main_frame_img.shape[:2]
                fdets = self._yolo_detect_unlocked(main_frame_img, cls=cls, conf=0.28, imgsz=1280)
                for d in fdets:
                    cimg, cbox = _pad_and_crop(main_frame_img, d["box"], pad_frac=0.06)
                    candidates.append({
                        "source": "main_frame_yolo",
                        "img": cimg,
                        "box": cbox,
                        "norm_box": d["norm_box"],
                        "conf": d["conf"],
                        "area_frac": d["area_frac"],
                    })

            # 3. Embed sub_ref_img + all high-res candidates in one batch
            to_embed = []
            has_sub = sub_ref_img is not None
            if has_sub:
                to_embed.append(sub_ref_img)
            for c in candidates:
                to_embed.append(c["img"])
            if not to_embed and main_crop_img is not None:
                to_embed.append(main_crop_img)

            embs = self._embed_batch_unlocked(to_embed)

        emb_sub = embs[0] if has_sub else None
        cand_embs = embs[1:] if has_sub else embs[:len(candidates)]

        chosen_img = None
        chosen_emb = None
        chosen_source = None
        chosen_box = None
        sim_main_sub = None

        if candidates:
            # Score each candidate against sub_ref_img and spatial hints
            main_hint_norm = None
            if main_bbox_hint and main_frame_img is not None:
                mh, mw = main_frame_img.shape[:2]
                main_hint_norm = [
                    main_bbox_hint[0] / mw, main_bbox_hint[1] / mh,
                    main_bbox_hint[2] / mw, main_bbox_hint[3] / mh,
                ]
            best_score = -1e9
            best_idx = -1
            for idx, c in enumerate(candidates):
                ce = cand_embs[idx]
                sim_sub = float(np.dot(ce, emb_sub)) if emb_sub is not None else 0.65
                c["sim_to_sub"] = sim_sub

                # Spatial proximity to sub_bbox_norm or main_hint_norm
                prox = 0.75
                if c["norm_box"] and sub_bbox_norm:
                    d_sub = _bbox_center_dist(c["norm_box"], sub_bbox_norm)
                    iou_sub = _bbox_iou(c["norm_box"], sub_bbox_norm)
                    prox_sub = max(0.0, 1.0 - d_sub * 1.4) + 0.4 * iou_sub
                    prox = prox_sub
                if c["norm_box"] and main_hint_norm:
                    d_main = _bbox_center_dist(c["norm_box"], main_hint_norm)
                    iou_main = _bbox_iou(c["norm_box"], main_hint_norm)
                    prox_main = max(0.0, 1.0 - d_main * 1.4) + 0.4 * iou_main
                    prox = max(prox, prox_main)

                src_bonus = 0.14 if c["source"] == "main_crop_yolo" else (0.10 if c["source"] == "main_crop_raw" else 0.0)
                score = 0.65 * sim_sub + 0.22 * min(1.2, prox) + 0.13 * c["conf"] + src_bonus
                c["score"] = score
                if score > best_score:
                    best_score = score
                    best_idx = idx

            best_c = candidates[best_idx]
            best_ce = cand_embs[best_idx]
            sim_main_sub = best_c["sim_to_sub"]

            # Guard against the case where main_crop/main_frame picked an unrelated parked
            # vehicle while sub_crop had the actual moving vehicle:
            if emb_sub is not None and sim_main_sub < 0.38:
                chosen_img = sub_ref_img
                chosen_emb = emb_sub
                chosen_source = "sub_crop_fallback"
                chosen_box = payload.get("sub_bbox")
            else:
                chosen_img = best_c["img"]
                chosen_source = best_c["source"]
                chosen_box = best_c["box"]
                if emb_sub is not None and sim_main_sub >= 0.42:
                    fused = 0.80 * best_ce + 0.20 * emb_sub
                    norm = float(np.linalg.norm(fused))
                    chosen_emb = (fused / norm) if norm > 1e-6 else best_ce
                else:
                    chosen_emb = best_ce
        elif has_sub:
            # No YOLO vehicle detected in main_crop or main_frame -> use sub_ref_img
            chosen_img = sub_ref_img
            chosen_emb = emb_sub
            chosen_source = "sub_crop_no_main_yolo"
            chosen_box = payload.get("sub_bbox")
        else:
            chosen_img = main_crop_img
            chosen_emb = embs[0]
            chosen_source = "main_crop_raw"
            chosen_box = main_bbox_hint

        attrs = {}
        if extract_attrs and chosen_img is not None:
            attrs = self.query_vlm_attrs(chosen_img, cls=cls, fallback_desc=fallback_desc)
            # If the chosen crop had PRESENT=no and we have sub_ref_img, retry on sub_ref_img
            if not attrs.get("present", True) and has_sub and chosen_img is not sub_ref_img:
                attrs_sub = self.query_vlm_attrs(sub_ref_img, cls=cls, fallback_desc=fallback_desc)
                if attrs_sub.get("present", True):
                    attrs = attrs_sub
                    chosen_img = sub_ref_img
                    chosen_emb = emb_sub
                    chosen_source = "sub_crop_vlm_rescued"

        ch, cw = chosen_img.shape[:2]
        resp = {
            "ok": True,
            "embedding": [round(float(x), 6) for x in chosen_emb.tolist()],
            "embedding_dim": int(chosen_emb.shape[0]),
            "crop_source": chosen_source,
            "crop_shape": [int(ch), int(cw)],
            "refined_bbox": chosen_box,
            "sim_main_sub": round(float(sim_main_sub), 4) if sim_main_sub is not None else None,
            "num_main_candidates": len(candidates),
            "attrs": attrs,
            "elapsed_ms": round((time.time() - t0) * 1000.0, 1),
        }
        if return_refined and chosen_img is not None:
            resp["refined_crop_b64"] = _encode_b64_img(chosen_img, quality=92)
        return resp


ENGINE = None


class ReidRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        log.info("%s - %s" % (self.address_string(), fmt % args))

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send_json(200, {
                "ok": True,
                "device": ENGINE.device if ENGINE else "uninitialized",
                "reid_model": "sbs_R50-ibn (VeRi-Wild)",
                "yolo_model": os.path.basename(YOLO_WEIGHTS),
                "vllm_url": VLLM_URL,
                "vllm_model": VLLM_MODEL,
                "embed_calls": ENGINE.embed_calls if ENGINE else 0,
                "uptime_s": round(time.time() - ENGINE.started_at, 1) if ENGINE else 0,
            })
        else:
            self._send_json(404, {"ok": False, "error": "Not found"})

    def do_POST(self):
        if not self.path.startswith("/embed"):
            self._send_json(404, {"ok": False, "error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            res = ENGINE.process_event(payload)
            self._send_json(200, res)
        except Exception as e:
            log.exception("Error handling /embed")
            self._send_json(400, {"ok": False, "error": str(e)})


def main():
    global ENGINE
    parser = argparse.ArgumentParser(description="FastReID + YOLO11m + VLM attribute server on gx10")
    parser.add_argument("--host", default=REID_HOST)
    parser.add_argument("--port", type=int, default=REID_PORT)
    args = parser.parse_args()

    ENGINE = ReidEngine()
    srv = ThreadingHTTPServer((args.host, args.port), ReidRequestHandler)
    log.info(f"reid_server listening on http://{args.host}:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
