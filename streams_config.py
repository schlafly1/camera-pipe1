"""Load RTSP stream definitions from environment variables."""

import os
import sys


def _warn(msg):
    print(f"[streams_config] WARNING: {msg}", file=sys.stderr, flush=True)


def _parse_class_list(raw):
    """DROP_CLASSES_CAMn value -> frozenset of tokens: class NAMES (lowercased
    str, e.g. 'bicycle') and/or numeric detector slots (int). 'bicycle' ->
    {'bicycle'}; '1,road_sign' -> {1, 'road_sign'}; blank -> empty.
    Tokens are resolved against the detector's class map (labels file) by
    pipeline_multi.resolve_class_tokens(), which logs what each one became."""
    out = set()
    for part in raw.replace(" ", "").replace(";", ",").split(","):
        part = part.strip().lower()
        if not part:
            continue
        out.add(int(part) if part.lstrip("-").isdigit() else part)
    return frozenset(out)


def _parse_min_conf(raw, key):
    """'0.6' -> 0.6; blank -> None. Invalid or out of (0, 1] -> warn, None
    (feature off = pipeline behaves as if the key were unset)."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        v = -1.0
    if not 0.0 < v <= 1.0:
        _warn(f"ignoring {key}={raw!r} (need a number in (0, 1])")
        return None
    return v


def _parse_hours(raw, key):
    """'19:15-06:45' -> (1155, 405) minutes after local midnight; blank -> None.
    The window may wrap past midnight. Invalid -> warn, returns False."""
    raw = "".join(raw.split())      # also drops tabs / CRLF "\r"
    if not raw:
        return None
    try:
        a, b = raw.split("-")
        out = []
        for t in (a, b):
            hh, mm = t.split(":") if ":" in t else (t, "0")
            hh, mm = int(hh), int(mm)
            if not (0 <= hh <= 23 and 0 <= mm <= 59):
                raise ValueError
            out.append(hh * 60 + mm)
        if out[0] == out[1]:
            raise ValueError
        return tuple(out)
    except ValueError:
        _warn(f"ignoring {key}={raw!r} (need HH:MM-HH:MM, start != end)")
        return False


def in_hours(minute_of_day, window):
    """True if minute_of_day (0..1439) is inside (start, end); wraps midnight.
    Start is inclusive, end exclusive."""
    start, end = window
    if start < end:
        return start <= minute_of_day < end
    return minute_of_day >= start or minute_of_day < end


def person_min_conf(base, gate, minute_of_day):
    """Effective confidence gate for one camera + class right now (named for
    its first use, the person gate; class_min_conf is the same function).

    base: DETECT_MIN_CONF[class]; gate: (min_conf, window-or-None) or None.
    Returns base unless a gate is set and (no window or inside it); a gate
    can only raise base, never lower it."""
    if not gate:
        return base
    conf, window = gate
    if window is not None and not in_hours(minute_of_day, window):
        return base
    return max(base, conf)


class_min_conf = person_min_conf

# Classes that accept MIN_CONF_<CLASS>_CAMn / MIN_CONF_<CLASS>_HOURS_CAMn.
GATE_CLASSES = ("person", "car", "bicycle")


def _parse_gate(cam, cname):
    """(min_conf, window-or-None) for MIN_CONF_<CLASS>_CAMn (+ _HOURS_), or
    (None, None) if unset/invalid (an invalid window disables the gate)."""
    k = cname.upper()
    key, hkey = f"MIN_CONF_{k}_CAM{cam}", f"MIN_CONF_{k}_HOURS_CAM{cam}"
    conf = _parse_min_conf(os.environ.get(key, ""), key)
    hours = _parse_hours(os.environ.get(hkey, ""), hkey)
    if hours is False:
        return None, None               # bad window -> gate off, not all-day
    if hours and not os.environ.get(key, "").strip():
        _warn(f"{hkey} is set but {key} is not; ignoring the window")
        hours = None
    if conf is None:
        hours = None
    return conf, hours


def load_streams():
    """
    Load cameras from RTSP_URL_CAM1, RTSP_URL_CAM2, ... in .env.

    Each camera may set:
      RTSP_TRANSPORT_CAMn=4 for TCP
      CAM_TYPE_CAMn=office or street (default street)
        - street: always send detections to VLM (no SAVE_INTERVAL throttle)
        - office: throttle with SAVE_INTERVAL; drop if VLM queue full
      DROP_CLASSES_CAMn=bicycle (comma list of class NAMES from
        detector_labels.txt, or numeric detector slots) to ignore those
        classes on that camera only, e.g. bicycle on the office cams where it
        fires on empty rooms. Names are preferred: numeric slots changed with
        the 2026-09-30 5-slot class map (old "3" meant bicycle, now person)
      MIN_CONF_PERSON_CAMn=0.60 raises the person confidence gate on that
        camera (never lowers it below DETECT_MIN_CONF["person"]); unset =
        unchanged
      MIN_CONF_PERSON_HOURS_CAMn=19:15-06:45 limits that gate to a local-time
        window (may wrap midnight); unset = all day. An invalid value for
        either key disables the gate for that camera (with a warning).
      MIN_CONF_CAR_CAMn / MIN_CONF_CAR_HOURS_CAMn and MIN_CONF_BICYCLE_CAMn /
        MIN_CONF_BICYCLE_HOURS_CAMn: the same gate for the other classes
        (e.g. cam6's night-IR false "car" detections).
    Fallback: comma-separated STREAM_URLS for quick tests (treated as street).
    """
    streams = []
    cam = 1
    while True:
        url = os.environ.get(f"RTSP_URL_CAM{cam}", "").strip()
        if not url:
            break
        transport = int(
            os.environ.get(
                f"RTSP_TRANSPORT_CAM{cam}",
                os.environ.get("RTSP_TRANSPORT", "0"),
            )
        )
        cam_type = os.environ.get(f"CAM_TYPE_CAM{cam}", "street").lower()
        is_office = cam_type == "office"
        drop_classes = _parse_class_list(
            os.environ.get(f"DROP_CLASSES_CAM{cam}", "")
        )
        gates = {}
        for cname in GATE_CLASSES:
            conf, hours = _parse_gate(cam, cname)
            if conf is not None:
                gates[cname] = (conf, hours)
        min_conf_person, min_conf_person_hours = gates.get("person", (None, None))
        streams.append({
            "camera_id": cam,
            "source_index": cam - 1,
            "url": url,
            "rtsp_transport": transport,
            "is_office": is_office,
            "cam_type": cam_type,
            "drop_classes": drop_classes,
            "min_conf_person": min_conf_person,
            "min_conf_person_hours": min_conf_person_hours,
            "min_conf_gates": gates,     # {class name: (conf, window-or-None)}
        })
        cam += 1

    if not streams:
        raw = os.environ.get("STREAM_URLS", "")
        for idx, url in enumerate(raw.split(","), start=1):
            url = url.strip()
            if url:
                streams.append({
                    "camera_id": idx,
                    "source_index": idx - 1,
                    "url": url,
                    "rtsp_transport": 0,
                    "is_office": False,
                    "cam_type": "street",
                })

    if not streams:
        raise ValueError(
            "No streams configured. Set RTSP_URL_CAM1..N or STREAM_URLS in .env"
        )

    return streams