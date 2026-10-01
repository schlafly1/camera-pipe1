#!/usr/bin/env python3
"""Offline RT-DETR (TrafficCamNet Transformer Lite) detector for saved JPEGs.

Runs a *copy* of the TensorRT engine in its own process (batch 1, one CUDA
stream) so it never touches the live pipeline's engine file or nvinfer
context. Pre/post-processing mirrors pgie_config_rtdetr.txt +
NvDsInferParseCustomDDETRTAO: RGB, /255, aspect-preserving resize into
960x544 with right/bottom padding; per query argmax over ALL output slots,
sigmoid, threshold, then drop the background slot (0) and road sign (4).

    python3 tools/rtdetr_offline.py [--engine E] [--all] img1.jpg [img2.jpg ...]

prints, per image, the max probability per output slot and the kept boxes.
The RTDETR class is importable for offline re-detection of saved snapshots.
Needs system TensorRT (the .venv sees system site-packages) and libcudart;
no pycuda/torch.
"""
import argparse
import ctypes
import glob
import os
import shutil
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(HERE, "models", "trafficcamnet_transformer_lite", "model")
LABELS_FILE = os.path.join(MODEL_DIR, "detector_labels.txt")
NET_W, NET_H = 960, 544


def _load_cudart():
    cands = ["libcudart.so"] + sorted(glob.glob("/usr/local/cuda/lib64/libcudart.so*")) \
        + sorted(glob.glob("/usr/local/cuda*/targets/*/lib/libcudart.so*"))
    for c in cands:
        try:
            return ctypes.CDLL(c)
        except OSError:
            continue
    raise OSError("libcudart not found")


def read_labels(path=LABELS_FILE):
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def default_engine_copy():
    """Copy the newest engine built by nvinfer to /tmp so the live pipeline
    rebuilding it on restart can't change it under us."""
    src = sorted(glob.glob(os.path.join(MODEL_DIR, "*.engine")), key=os.path.getmtime)
    if not src:
        raise FileNotFoundError("no .engine in " + MODEL_DIR)
    dst = "/tmp/rtdetr_offline_%s" % os.path.basename(src[-1])
    if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src[-1]):
        shutil.copy2(src[-1], dst + ".part")
        os.replace(dst + ".part", dst)
    return dst


class RTDETR:
    def __init__(self, engine_path=None, threshold=0.4, drop_slots=(0, 4)):
        import tensorrt as trt
        self.trt = trt
        self.threshold = threshold
        self.drop_slots = set(drop_slots)
        self.cudart = _load_cudart()
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")
        engine_path = engine_path or default_engine_copy()
        with open(engine_path, "rb") as fh:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(fh.read())
        if self.engine is None:
            raise RuntimeError("could not deserialize " + engine_path)
        self.ctx = self.engine.create_execution_context()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inp = [n for n in self.names
                    if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
        self.ctx.set_input_shape(self.inp, (1, 3, NET_H, NET_W))
        self.bufs, self.host = {}, {}
        for n in self.names:
            shape = tuple(self.ctx.get_tensor_shape(n))
            dt = np.dtype(trt.nptype(self.engine.get_tensor_dtype(n)))
            self.host[n] = np.empty(shape, dtype=dt)
            ptr = ctypes.c_void_p()
            self._ck(self.cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(self.host[n].nbytes)))
            self.bufs[n] = ptr
            self.ctx.set_tensor_address(n, ptr.value)
        self.stream = ctypes.c_void_p()
        self._ck(self.cudart.cudaStreamCreate(ctypes.byref(self.stream)))
        self.num_slots = int(self.host["pred_logits"].shape[-1])

    @staticmethod
    def _ck(rc):
        if rc != 0:
            raise RuntimeError(f"CUDA error {rc}")

    def _memcpy(self, dst, src, n, kind):
        self._ck(self.cudart.cudaMemcpy(ctypes.c_void_p(dst), ctypes.c_void_p(src),
                                        ctypes.c_size_t(n), ctypes.c_int(kind)))

    @staticmethod
    def preprocess(img):
        im = img.convert("RGB")
        s = min(NET_W / im.width, NET_H / im.height)
        nw, nh = int(round(im.width * s)), int(round(im.height * s))
        canvas = Image.new("RGB", (NET_W, NET_H))
        canvas.paste(im.resize((nw, nh), Image.BILINEAR), (0, 0))
        x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        return np.ascontiguousarray(x), s

    def raw(self, img):
        """Return (logits[300, slots], boxes[300, 4] cxcywh-normalized, scale)."""
        x, s = self.preprocess(img)
        h_in = self.host[self.inp]
        h_in[...] = x.astype(h_in.dtype)
        self._memcpy(self.bufs[self.inp].value, h_in.ctypes.data, h_in.nbytes, 1)
        if not self.ctx.execute_async_v3(self.stream.value):
            raise RuntimeError("TensorRT execute failed")
        self._ck(self.cudart.cudaStreamSynchronize(self.stream))
        for n in self.names:
            if n != self.inp:
                self._memcpy(self.host[n].ctypes.data, self.bufs[n].value, self.host[n].nbytes, 2)
        return (self.host["pred_logits"][0].astype(np.float32).copy(),
                self.host["pred_boxes"][0].astype(np.float32).copy(), s)

    def detect(self, img, threshold=None, keep_dropped=False):
        """List of dicts {slot, conf, box:(x1,y1,x2,y2) in ORIGINAL pixels},
        highest confidence first. Mirrors NvDsInferParseCustomDDETRTAO."""
        if isinstance(img, str):
            img = Image.open(img)
        thr = self.threshold if threshold is None else threshold
        logits, boxes, s = self.raw(img)
        slot = logits.argmax(1)
        conf = 1.0 / (1.0 + np.exp(-logits[np.arange(len(slot)), slot]))
        out = []
        for k in np.argsort(-conf):
            if conf[k] < thr:
                break
            c = int(slot[k])
            if c in self.drop_slots and not keep_dropped:
                continue
            cx, cy, w, h = boxes[k]
            x1, y1 = (cx - w / 2) * NET_W / s, (cy - h / 2) * NET_H / s
            x2, y2 = (cx + w / 2) * NET_W / s, (cy + h / 2) * NET_H / s
            out.append({"slot": c, "conf": float(conf[k]),
                        "box": (round(float(x1)), round(float(y1)),
                                round(float(x2)), round(float(y2)))})
        return out

    def slot_max(self, img):
        """Max sigmoid probability per output slot over all 300 queries."""
        if isinstance(img, str):
            img = Image.open(img)
        logits, _, _ = self.raw(img)
        return (1.0 / (1.0 + np.exp(-logits))).max(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine")
    ap.add_argument("--thr", type=float, default=0.4)
    ap.add_argument("--all", action="store_true", help="also list slot 0/4 boxes")
    ap.add_argument("--draw", help="write annotated copies to this dir")
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    det = RTDETR(a.engine, threshold=a.thr)
    try:
        labels = read_labels()
    except OSError:
        labels = []
    print(f"engine output slots={det.num_slots} labels file={labels}")
    for f in a.images:
        img = Image.open(f)
        sm = det.slot_max(img)
        dets = det.detect(img, keep_dropped=a.all)
        print(f"{os.path.basename(f)} {img.width}x{img.height} slot max prob:",
              " ".join(f"{i}:{p:.2f}" for i, p in enumerate(sm)))
        for d in dets[:12]:
            name = labels[d["slot"]] if d["slot"] < len(labels) else "?"
            print(f"    slot{d['slot']}({name}) conf={d['conf']:.2f} box={d['box']}")
        if a.draw:
            from PIL import ImageDraw
            os.makedirs(a.draw, exist_ok=True)
            im = img.convert("RGB")
            dr = ImageDraw.Draw(im)
            cols = ["white", "red", "yellow", "lime", "cyan"]
            for d in dets[:12]:
                dr.rectangle(d["box"], outline=cols[d["slot"] % 5], width=3)
                dr.text((d["box"][0] + 3, d["box"][1] + 3),
                        f"slot{d['slot']} {d['conf']:.2f}", fill=cols[d["slot"] % 5])
            im.save(os.path.join(a.draw, os.path.basename(f)))


if __name__ == "__main__":
    sys.exit(main())
