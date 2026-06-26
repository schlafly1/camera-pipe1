"""Load RTSP stream definitions from environment variables."""

import os


def load_streams():
    """
    Load cameras from RTSP_URL_CAM1, RTSP_URL_CAM2, ... in .env.

    Each camera may set RTSP_TRANSPORT_CAMn=4 for TCP (Dahua, etc.).
    Fallback: comma-separated STREAM_URLS for quick tests.
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
        streams.append({
            "camera_id": cam,
            "source_index": cam - 1,
            "url": url,
            "rtsp_transport": transport,
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
                })

    if not streams:
        raise ValueError(
            "No streams configured. Set RTSP_URL_CAM1..N or STREAM_URLS in .env"
        )

    return streams