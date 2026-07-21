# Spark/SBSA image bundles the Jetson multimedia libs (libnvbufsurface etc.)
# that pyservicemaker needs. The generic 9.1-triton-multiarch image ships them
# as dangling symlinks (expects a host mount the Spark doesn't provide), so it
# is NOT usable here — use the dgx-spark variant, mirroring the 9.0 setup.
FROM nvcr.io/nvidia/deepstream:9.1-triton-sbsa-dgx-spark

# tzdata for America/Los_Angeles timezone support
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

# Python packages — pyservicemaker wheel lives inside the base image
RUN pip3 install --break-system-packages \
    /opt/nvidia/deepstream/deepstream/service-maker/python/pyservicemaker*.whl \
    pyyaml chromadb ollama fastapi uvicorn Pillow

# Ensure DeepStream libs (including libnvds_service_maker + nvbufsurface) are found
# by pyservicemaker and GStreamer on Spark/Jetson containers.
ENV LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream-9.1/lib:/opt/nvidia/deepstream/deepstream/lib:${LD_LIBRARY_PATH}
ENV GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream-9.1/lib/gst-plugins:/opt/nvidia/deepstream/deepstream/lib/gst-plugins:${GST_PLUGIN_PATH}
