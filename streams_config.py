"""Load RTSP stream definitions from environment variables."""

import os


def _parse_class_list(raw):
    """'3' or '0,3' -> frozenset({3}) / frozenset({0, 3}); blank -> empty."""
    out = set()
    for part in raw.replace(" ", "").split(","):
        if part:
            out.add(int(part))
    return frozenset(out)


def load_streams():
    """
    Load cameras from RTSP_URL_CAM1, RTSP_URL_CAM2, ... in .env.

    Each camera may set:
      RTSP_TRANSPORT_CAMn=4 for TCP
      CAM_TYPE_CAMn=office or street (default street)
        - street: always send detections to VLM (no SAVE_INTERVAL throttle)
        - office: throttle with SAVE_INTERVAL; drop if VLM queue full
      DROP_CLASSES_CAMn=3 (comma list of detector class ids) to ignore those
        classes on that camera only, e.g. RT-DETR class 3 (Bicycle, app label
        "motorcycle") on office cams where it fires on empty rooms
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
        streams.append({
            "camera_id": cam,
            "source_index": cam - 1,
            "url": url,
            "rtsp_transport": transport,
            "is_office": is_office,
            "cam_type": cam_type,
            "drop_classes": drop_classes,
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