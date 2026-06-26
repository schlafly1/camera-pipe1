# Adding Cameras — Multi-Stream Pipeline

With `pipeline_multi.py` + `cam_multi.yml`, adding a camera requires changes to
**two files** only. No compose or Python changes.

## File 1 — `.env`

Add the next numbered camera URL:

```
RTSP_URL_CAM5=rtsp://user:password@192.168.x.x:554/sub
# RTSP_TRANSPORT_CAM5=4   # uncomment if camera requires TCP
```

`streams_config.py` reads `RTSP_URL_CAM1`, `RTSP_URL_CAM2`, … until the first
gap. Do not skip numbers.

## File 2 — `search.html`

Add one `<option>` in the camera filter (around line 73):

```html
<option value="5">Camera 5</option>
```

## Restart

```bash
# Restart pipeline inside the single deepstream container
# Ctrl+C pipeline_multi.py, then:
python3 pipeline_multi.py
```

TensorRT rebuilds the batched engine if batch-size changes (~5 min one-time).

## Notes

- `pgie_config_multi.yml` batch-size in the YAML is a default; runtime uses N streams.
- `monitor.py` writes per-camera stats to `./stats/cam{N}_stats.json` automatically.
- `query_server.py` and ChromaDB need no changes.
- For many cameras (8+), lower `FRAME_W`/`FRAME_H` to substream resolution.