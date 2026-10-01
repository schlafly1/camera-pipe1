#!/usr/bin/env python3
"""Offline test for the per-camera env parsing in streams_config: the person
gate (MIN_CONF_PERSON_CAMn / MIN_CONF_PERSON_HOURS_CAMn) and DROP_CLASSES_CAMn
(class names or numeric slots). Pure Python, no DeepStream, nothing live
touched. Name -> slot resolution is covered by tools/test_class_map.py.

    .venv/bin/python3 tools/test_person_gate.py
"""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

for k in list(os.environ):          # isolate from any real .env in the shell
    if k.startswith(("RTSP_URL_CAM", "MIN_CONF_", "STREAM_URLS", "DROP_CLASSES")):
        del os.environ[k]

import streams_config as sc  # noqa: E402


def check(cond, msg, fails):
    if not cond:
        fails.append(msg)


def main():
    f = []
    w = sc._parse_hours("19:15-06:45", "k")
    check(w == (1155, 405), f"wrap window parse {w}", f)
    check(sc.in_hours(19 * 60 + 15, w), "19:15 in (start inclusive)", f)
    check(sc.in_hours(0, w), "00:00 in", f)
    check(sc.in_hours(6 * 60 + 44, w), "06:44 in", f)
    check(not sc.in_hours(6 * 60 + 45, w), "06:45 out (end exclusive)", f)
    check(not sc.in_hours(12 * 60, w), "12:00 out", f)
    d = sc._parse_hours("08:00-17:30", "k")
    check(d == (480, 1050) and sc.in_hours(600, d) and not sc.in_hours(1100, d),
          "day window", f)
    check(sc._parse_hours("", "k") is None, "blank hours -> None", f)
    check(sc._parse_hours("7pm-6am", "k") is False, "bad hours -> False", f)
    check(sc._parse_hours("10:00-10:00", "k") is False, "empty window -> False", f)
    check(sc._parse_min_conf("0.6", "k") == 0.6, "conf parse", f)
    check(sc._parse_min_conf("", "k") is None, "blank conf -> None", f)
    check(sc._parse_min_conf("abc", "k") is None, "bad conf -> None", f)
    check(sc._parse_min_conf("1.5", "k") is None, "out of range -> None", f)
    check(sc._parse_hours("19:15-06:45\r", "k") == (1155, 405), "CRLF tolerated", f)

    pm = sc.person_min_conf
    night = (0.6, (1155, 405))
    check(pm(0.4, None, 600) == 0.4, "no gate -> base", f)
    check(pm(0.4, (0.6, None), 600) == 0.6, "all-day gate", f)
    check(pm(0.4, (0.3, None), 600) == 0.4, "gate never lowers base", f)
    check(pm(0.4, night, 20 * 60) == 0.6, "night gate at 20:00", f)
    check(pm(0.4, night, 3 * 60) == 0.6, "night gate at 03:00", f)
    check(pm(0.4, night, 12 * 60) == 0.4, "night gate off at 12:00", f)

    os.environ["RTSP_URL_CAM1"] = "rtsp://example.invalid/1"
    os.environ["RTSP_URL_CAM2"] = "rtsp://example.invalid/2"
    os.environ["RTSP_URL_CAM3"] = "rtsp://example.invalid/3"
    os.environ["MIN_CONF_PERSON_CAM2"] = "0.6"
    os.environ["MIN_CONF_PERSON_HOURS_CAM2"] = "19:15-06:45"
    os.environ["MIN_CONF_PERSON_CAM3"] = "0.7"
    os.environ["MIN_CONF_PERSON_HOURS_CAM3"] = "junk"
    s = {x["camera_id"]: x for x in sc.load_streams()}
    check(s[1]["min_conf_person"] is None and s[1]["min_conf_person_hours"] is None,
          "unset camera unchanged", f)
    check(s[2]["min_conf_person"] == 0.6 and s[2]["min_conf_person_hours"] == (1155, 405),
          "cam2 gate + window", f)
    check(s[3]["min_conf_person"] is None, "bad window disables gate", f)
    os.environ["MIN_CONF_PERSON_HOURS_CAM1"] = "19:15-06:45"
    s = {x["camera_id"]: x for x in sc.load_streams()}
    check(s[1]["min_conf_person"] is None and s[1]["min_conf_person_hours"] is None,
          "window without threshold ignored", f)

    # The same gate for other classes: MIN_CONF_CAR_CAMn (+ _HOURS_).
    os.environ["MIN_CONF_CAR_CAM2"] = "0.8"
    os.environ["MIN_CONF_CAR_HOURS_CAM2"] = "19:15-06:45"
    os.environ["MIN_CONF_BICYCLE_CAM3"] = "0.55"
    s = {x["camera_id"]: x for x in sc.load_streams()}
    check(s[2]["min_conf_gates"] == {"person": (0.6, (1155, 405)), "car": (0.8, (1155, 405))},
          f"cam2 person + car gates {s[2]['min_conf_gates']}", f)
    check(s[3]["min_conf_gates"] == {"bicycle": (0.55, None)}, "cam3 all-day bicycle gate", f)
    check(s[1]["min_conf_gates"] == {}, "cam1 no gates", f)
    check(sc.class_min_conf(0.5, s[2]["min_conf_gates"]["car"], 23 * 60) == 0.8, "car gate at night", f)
    check(sc.class_min_conf(0.5, s[2]["min_conf_gates"]["car"], 12 * 60) == 0.5, "car gate off by day", f)

    # DROP_CLASSES_CAMn: names (preferred) and/or numeric slots.
    pc = sc._parse_class_list
    check(pc("bicycle") == frozenset({"bicycle"}), "name", f)
    check(pc(" Bicycle ") == frozenset({"bicycle"}), "name is case/space-insensitive", f)
    check(pc("1, road_sign") == frozenset({1, "road_sign"}), "mixed name + slot", f)
    check(pc("1;4") == frozenset({1, 4}), "semicolon list", f)
    check(pc("") == frozenset(), "blank -> empty", f)
    check(pc("3") == frozenset({3}), "numeric still accepted", f)
    os.environ["DROP_CLASSES_CAM1"] = "bicycle"
    os.environ["DROP_CLASSES_CAM2"] = "bicycle,road_sign"
    s = {x["camera_id"]: x for x in sc.load_streams()}
    check(s[1]["drop_classes"] == frozenset({"bicycle"}), "cam1 drop bicycle", f)
    check(s[2]["drop_classes"] == frozenset({"bicycle", "road_sign"}), "cam2 drop list", f)
    check(s[3]["drop_classes"] == frozenset(), "cam3 drops nothing", f)

    for m in f:
        print("FAIL", m)
    print("person-gate + drop-class parsing: " + ("all passed" if not f else f"{len(f)} failed"))
    return 1 if f else 0


if __name__ == "__main__":
    sys.exit(main())