#!/usr/bin/env python3
"""Print the class-slot width (last dim of pred_logits) of an RT-DETR ONNX
by parsing it with TensorRT's ONNX parser (CPU only; no engine build, no GPU
inference). Used by pipeline_multi.py's startup class-map check, which runs
it in a subprocess so the pipeline process never imports tensorrt itself.

    python3 tools/model_width.py models/.../resnet50_trafficamnet_rtdetr.onnx [pred_logits]
"""
import sys


def output_width(onnx_path, tensor="pred_logits"):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        errs = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError("ONNX parse failed: " + errs)
    for i in range(network.num_outputs):
        t = network.get_output(i)
        if t.name == tensor:
            return int(t.shape[-1])
    raise KeyError(f"no output named {tensor}")


if __name__ == "__main__":
    print(output_width(sys.argv[1], *(sys.argv[2:3] or [])))
