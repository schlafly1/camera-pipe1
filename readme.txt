
I have Jetpack 7.1 and Deepstream 9 installed. I have a camera at rtsp://192.168.8.130:554/11 .

I have ollama natively installed, with the model gemma4:26b.

I can run this Deepstream container:
docker run -it --rm \
    --runtime nvidia \
    --network host \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /etc/X11:/etc/X11 \
    -v ~/VisionHistory:/opt/nvidia/deepstream/deepstream-9.0/sources/apps/VisionHistory \
    -w /opt/nvidia/deepstream/deepstream-9.0/sources/apps/VisionHistory \
    nvcr.io/nvidia/deepstream:9.0-triton-multiarch

In the container, I can install:
pip3 install pyservicemaker chromadb ollama

Create a cam1.yml file for docker compose. It starts a DeepStream 9.0 container and a ChromaDB container. Then, generate a Python script using pyservicemaker that connects to my RTSP camera and prints a message every time it detects a person.

Generate a pyservicemaker pipeline for the container. It should take my RTSP stream, detect objects, and save embeddings to a local ChromaDB collection.

Create a FastAPI web server inside my container. It should have one endpoint called /query that takes a text string, converts it to an embedding using Ollama, searches ChromaDB, and returns the timestamp of the matching video event.

I want to scale to 6 cameras eventually. Based on the @example_prompts/multi_stream_tracker.md logic, create a single DeepStream pipeline that handles multiple RTSP inputs but saves all their embeddings into one ChromaDB collection with a source_id tag in the metadata.


Try to make everything simple and easy to implement. Give me explicit instructions on what to do.

