# Adding More Cameras

Adding a camera requires changes to exactly two files, then one shell command.
No changes to pipeline2.py, query_server.py, or pgie_config.yml.

---

## File 1 — cam1.yml

Add one service block per camera, following the same pattern as cam1 and cam2.
Replace the IP address and camera number throughout.

```yaml
  deepstream-cam3:
    <<: *deepstream
    environment:
      DISPLAY: "${DISPLAY:-:1}"
      XAUTHORITY: /root/.Xauthority
      NVIDIA_DRIVER_CAPABILITIES: all
      CAMERA_ID: "3"
      RTSP_URL: "rtsp://192.168.9.XXX:554/11"

  deepstream-cam4:
    <<: *deepstream
    environment:
      DISPLAY: "${DISPLAY:-:1}"
      XAUTHORITY: /root/.Xauthority
      NVIDIA_DRIVER_CAPABILITIES: all
      CAMERA_ID: "4"
      RTSP_URL: "rtsp://192.168.9.XXX:554/11"
```

---

## File 2 — search.html

Find the camera `<select>` block (around line 73) and add one `<option>` per new camera:

```html
<option value="3">Camera 3</option>
<option value="4">Camera 4</option>
```

---

## Startup

```bash
# Rebuild is NOT needed — pipeline2.py is already generic.
# Just bring up the new containers:
docker compose -f cam1.yml up -d

# First run only: TRT engine builds inside each new container (~5 min each).
# You will see "deserializing trt engine" in the log when it is ready.

# Start each camera pipeline in its own terminal:
docker exec -it cam1-deepstream-cam3-1 bash
DISPLAY=:1 python3 pipeline2.py

docker exec -it cam1-deepstream-cam4-1 bash
DISPLAY=:1 python3 pipeline2.py
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
- The VLM (Gemma4:26b) is shared across all cameras via the event queue in each
  pipeline process. Each process has its own queue and its own VLM calls — they
  run independently and may compete for GPU memory. Monitor with `tegrastats`.
