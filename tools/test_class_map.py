#!/usr/bin/env python3
"""Offline test for the detector class map (2026-09-30 fix): labels file ->
slot names, app classes keyed by NAME, refuse-to-start checks, DROP_CLASSES
name resolution, unique ids / never-overwrite snapshots. Touches nothing
live: LOG_DIR is pointed away from logs/, temp configs live in a temp dir.

    LOG_DIR=/tmp/vlm_selftest .venv/bin/python3 tools/test_class_map.py [--model]

--model also parses the real ONNX with TensorRT (CPU only, ~1 s) and checks
that its pred_logits width equals the labels line count.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("LOG_DIR", "/tmp/vlm_selftest")

import pipeline_multi as pm  # noqa: E402


def check(cond, msg, fails):
    if not cond:
        fails.append(msg)


def write_cfg(d, labels, ncls, filt="0;4", name="cfg.txt"):
    with open(os.path.join(d, "labels.txt"), "w") as fh:
        fh.write("\n".join(labels) + "\n")
    path = os.path.join(d, name)
    with open(path, "w") as fh:
        fh.write("# test\n[property]\nlabelfile-path=labels.txt\n"
                 f"num-detected-classes={ncls}\nfilter-out-class-ids={filt}\n"
                 "[class-attrs-all]\npre-cluster-threshold=0.4\n")
    return path


def raises(fn):
    try:
        fn()
    except pm.ClassMapError:
        return True
    return False


def main():
    f = []
    # The live map, from the repo's own pgie config + labels file.
    cm = pm.load_class_map("pgie_config_rtdetr.txt")
    check(cm["names"] == ["bg", "bicycle", "car", "person", "road_sign"],
          f"labels file slots {cm['names']}", f)
    check(cm["slots"] == {"person": 3, "car": 2, "bicycle": 1}, f"app slots {cm['slots']}", f)
    check(cm["filtered"] == {0, 4}, f"filtered {cm['filtered']}", f)
    check(pm.DETECT_CLASSES == {3: "person", 2: "car", 1: "bicycle"},
          f"DETECT_CLASSES {pm.DETECT_CLASSES}", f)
    for table in ("DETECT_MIN_CONF", "VLM_PROMPTS", "_VLM_SUBJECT_NOUNS", "_VLM_ABSENT_RX"):
        check(set(getattr(pm, table)) == set(pm.APP_CLASSES), f"{table} keyed by app names", f)
    check(set(pm._VLM_PRESENT_RX) <= set(pm.APP_CLASSES), "_VLM_PRESENT_RX keys", f)
    for name, prompt in pm.VLM_PROMPTS.items():
        check("reply exactly NONE" in prompt, f"{name} prompt offers NONE", f)
    check("vehicle" in pm.VLM_PROMPTS["car"] and "person" in pm.VLM_PROMPTS["person"]
          and "bicycle" in pm.VLM_PROMPTS["bicycle"], "prompt matches class", f)

    # Refuse-to-start checks.
    with tempfile.TemporaryDirectory() as d:
        ok5 = ["BG", "bicycle", "car", "person", "road_sign"]
        check(not raises(lambda: pm.load_class_map(write_cfg(d, ok5, 5))), "valid 5-slot cfg", f)
        check(raises(lambda: pm.load_class_map(write_cfg(d, ok5[:4], 5))), "4 labels vs 5 -> refuse", f)
        check(raises(lambda: pm.load_class_map(write_cfg(d, ok5, 4))), "5 labels vs 4 -> refuse", f)
        old4 = ["Car", "RoadSign", "Person", "Bicycle"]
        check(raises(lambda: pm.load_class_map(write_cfg(d, old4, 5, "1"))),
              "old 4-line labels file vs 5-slot config -> refuse", f)
        check(raises(lambda: pm.load_class_map(write_cfg(d, ["BG", "bicycle", "truck", "person", "rs"], 5))),
              "missing car -> refuse", f)
        check(raises(lambda: pm.load_class_map(write_cfg(d, ok5, 5, "0;3;4"))),
              "app class filtered -> refuse", f)
        check(raises(lambda: pm.load_class_map(write_cfg(d, ["BG", "car", "car", "person", "bicycle"], 5))),
              "duplicate names -> refuse", f)
        # Names are matched case-insensitively; old display name aliases to bicycle.
        cm2 = pm.load_class_map(write_cfg(d, ["Background", "Motorcycle", "Car", "Person", "RoadSign"], 5))
        check(cm2["slots"] == {"person": 3, "car": 2, "bicycle": 1}, f"aliases {cm2['slots']}", f)

    # DROP_CLASSES resolution (names preferred, numeric slots accepted).
    r = pm.resolve_class_tokens
    check(r({"bicycle"}) == {1}, "drop bicycle -> slot 1", f)
    check(r({"motorcycle"}) == {1}, "old name motorcycle -> bicycle slot", f)
    check(r({3}) == {3}, "numeric 3 is now person", f)
    check(r({"person", "car"}) == {2, 3}, "two names", f)
    check(r({"nope", 99}) == frozenset(), "unknown tokens ignored", f)
    import streams_config as sc
    check(r(sc._parse_class_list("Bicycle")) == {1}, ".env DROP_CLASSES_CAMn=Bicycle", f)

    # Event names: unique, never overwrite an existing snapshot.
    with tempfile.TemporaryDirectory() as d:
        n1 = pm._claim_event_name("cam3_src2_car_evt5_1790000000123", b"first", None, d)
        n2 = pm._claim_event_name("cam3_src2_car_evt5_1790000000123", b"second", None, d)
        n3 = pm._claim_event_name("cam3_src2_car_evt5_1790000000123", b"third", None, d)
        check((n1, n2, n3) == ("cam3_src2_car_evt5_1790000000123",
                               "cam3_src2_car_evt5_1790000000123_1",
                               "cam3_src2_car_evt5_1790000000123_2"), f"suffixes {n1} {n2} {n3}", f)
        check(open(os.path.join(d, n1 + ".jpg"), "rb").read() == b"first", "first file not overwritten", f)

        class FakeCol:
            def get(self, ids, include):
                return {"ids": [i for i in ids if i == "cam1_src0_person_evt9_1"]}
        n4 = pm._claim_event_name("cam1_src0_person_evt9_1", b"x", FakeCol(), d)
        check(n4 == "cam1_src0_person_evt9_1_1", f"id taken in Chroma -> suffix ({n4})", f)
        # The startup seeder must still see evt numbers in the new names.
        for nm in ("cam3_src2_car_evt812_1790000000123.jpg", "cam3_src2_person_evt900_1790000000999_1.jpg",
                   "cam3_src2_person_evt3469.jpg"):
            open(os.path.join(d, nm), "wb").close()
        old = pm.SNAPSHOT_DIR
        try:
            pm.SNAPSHOT_DIR = d
            seeds = pm.ObjectDetector._seed_event_ids()
        finally:
            pm.SNAPSHOT_DIR = old
        check(seeds.get(3) == 3469, f"seed max evt {seeds}", f)

    check(pm._mask_url("rtsp://u:pw@10.1.2.3:554/x?user=a&password=b") == "rtsp://10.1.2.3:554/…",
          "RTSP URL masked in logs", f)

    if "--model" in sys.argv:
        w = pm.model_output_width(cm["onnx_path"])
        check(w == len(cm["names"]), f"model pred_logits width {w} == labels {len(cm['names'])}", f)

    for m in f:
        print("FAIL", m)
    print("class map: " + ("all passed" if not f else f"{len(f)} failed"))
    return 1 if f else 0


if __name__ == "__main__":
    sys.exit(main())
