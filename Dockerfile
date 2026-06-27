FROM nvcr.io/nvidia/deepstream:9.0-triton-sbsa-dgx-spark

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
ENV LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream-9.0/lib:/opt/nvidia/deepstream/deepstream/lib:${LD_LIBRARY_PATH}
ENV GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream-9.0/lib/gst-plugins:/opt/nvidia/deepstream/deepstream/lib/gst-plugins:${GST_PLUGIN_PATH}
