# TrafficCamNet Transformer Lite (RT-DETR) — PGIE model

Modern transformer (RT-DETR, resnet50 backbone) traffic detector from NVIDIA
NGC TAO. Replaces the resnet18 TrafficCamNet as the primary detector. Used by
`../../pgie_config_rtdetr.txt` (referenced from `pipeline_multi.py`).

Classes (NGC order): `0=Car 1=RoadSign 2=Person 3=Bicycle`. RoadSign is
filtered out in the pgie config; Bicycle is surfaced with the app's
"motorcycle" label/prompt.

## The large binaries are gitignored — regenerate them like this

The ONNX (167 MB), the TensorRT engine (built on first pipeline run), and the
compiled parser `.so` are not committed. To reproduce on a fresh checkout:

```bash
# 1. Download the ONNX from NGC (no auth needed for this deployable model)
cd models/trafficcamnet_transformer_lite/model
wget --content-disposition \
  'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/trafficcamnet_transformer_lite/deployable_v1.0/files?redirect=true&path=resnet50_trafficamnet_rtdetr.onnx' \
  -O resnet50_trafficamnet_rtdetr.onnx

# 2. Build the DDETR/RT-DETR bbox parser inside the DeepStream container
docker exec <deepstream-container> bash -c \
  'cd /workspace/models/trafficcamnet_transformer_lite/parser && make'

# 3. Start the pipeline — nvinfer builds the TensorRT engine on first run
#    (~45s; a one-time "kFP16 flag" ERROR is expected and auto-retried, see
#    the note in ../../pgie_config_rtdetr.txt).
```

`detector_labels.txt`, the parser source (`nvdsinfer_custombboxparser_tao.cpp`,
from the DS 9.1 RTDETR sample) and its `Makefile` ARE committed.
