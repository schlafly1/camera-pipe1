FROM nvcr.io/nvidia/deepstream:9.0-triton-multiarch

# tzdata for America/Los_Angeles timezone support
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

# Python packages — pyservicemaker wheel lives inside the base image
RUN pip3 install --break-system-packages \
    /opt/nvidia/deepstream/deepstream/service-maker/python/pyservicemaker*.whl \
    pyyaml chromadb ollama fastapi uvicorn Pillow
