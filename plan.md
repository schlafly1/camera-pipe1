# Adding More Cameras

Adding a camera requires changes to exactly three files, then one shell command.
No changes to pipeline2.py, query_server.py, or pgie_config.yml.

---

## File 1 — .env

Add one URL per camera. Comment out alternatives you are not using — Docker Compose
uses the **first** occurrence of a duplicate key, not the last.

```
# CAM5 — choose one stream
# RTSP_URL_CAM5=rtsp://user:password@192.168.x.x:554/main
RTSP_URL_CAM5=rtsp://user:password@192.168.x.x:554/sub
```

To switch VLM model for all cameras at once, edit this block:
```
# VLM_MODEL=gemma4:26b
VLM_MODEL=nemotron3:33b
```

---

## File 2 — cam1.yml

Add one service block per camera, following the same pattern as the existing ones.
If the camera requires TCP (i.e. ffplay only works with -rtsp_transport tcp), add
`RTSP_TRANSPORT: "4"`.

```yaml
  deepstream-cam5:
    <<: *deepstream
    environment:
      DISPLAY: "${DISPLAY:-:1}"
      XAUTHORITY: /root/.Xauthority
      NVIDIA_DRIVER_CAPABILITIES: all
      CAMERA_ID: "5"
      RTSP_URL: "${RTSP_URL_CAM5}"
      VLM_MODEL: "${VLM_MODEL:-gemma4:26b}"
      # RTSP_TRANSPORT: "4"   # uncomment if camera requires TCP
```

---

## File 3 — search.html

Find the camera `<select>` block (around line 73) and add one `<option>` per new camera:

```html
<option value="5">Camera 5</option>
```

---

## Startup

```bash
# Rebuild is NOT needed — pipeline2.py is already generic.
# Just bring up the new containers:
docker compose -f cam1.yml up -d

# First run only: TRT engine builds inside each new container (~5 min each).
# You will see "deserializing trt engine" in the log when it is ready.

# Start each camera pipeline in its own terminal (DISPLAY is set in .env):
docker exec -it cam1-deepstream-cam5-1 bash
python3 pipeline2.py
```

---

## Notes

- Each camera writes snapshots named `cam{N}_src0_label_evt{N}.jpg` — no collisions.
- All cameras share the same ChromaDB collection. The `camera_id` metadata field
  lets you filter by camera in the search UI.
- Each new container builds its own TRT engine on first use (~5 min, one-time cost).
  The engine lives inside the container; do not `docker compose down` unless you want
  to rebuild it next time. Use `docker compose stop` instead.
- The query server and ChromaDB require no changes and handle N cameras automatically.
- GPU load: each camera runs nvinfer at ~6 fps (interval=4 in pgie_config.yml).
  A Jetson Orin handles 4–6 cameras comfortably at this rate. If you see lag,
  increase the interval value or raise SAVE_INTERVAL in pipeline2.py.
- The VLM model is shared across all cameras via the VLM_MODEL env var in .env.
  Each pipeline process has its own queue and its own VLM calls — they run
  independently and may compete for GPU memory. Monitor with `tegrastats`.
- TCP transport (RTSP_TRANSPORT=4) is needed for cameras where only
  `ffplay -rtsp_transport tcp` works. Dahua cameras often require this.
