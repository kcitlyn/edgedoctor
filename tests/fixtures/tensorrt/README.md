# TensorRT parser test fixtures

⚠️ **These logs are SYNTHETIC.** They are hand-assembled from *verified real
error strings* (sourced from onnx-tensorrt's ModelImporter.cpp /
errorHelpers.hpp and real logs pasted in NVIDIA/TensorRT GitHub issues
#3346, #2625, #3610, #4736, #4658, and forum threads #187181, #84205), but the
surrounding log structure is reconstructed, not captured from a real run.

They exist so the parser has tests before the first real ThinkPad logs land.
As real logs arrive in `corpus/tensorrt/`, add tests against those too — and
if a real log ever contradicts a synthetic fixture, the real log wins and the
fixture gets fixed.

| fixture | scenario |
|---|---|
| `unsupported_op_trt8.log` | TRT 8.x: unsupported op (GridSample) → plugin fallback fails → parse error |
| `parse_error_trt10.log` | TRT 10.x: new-format node parse error (INVALID_NODE) |
| `build_failure_trt10.log` | TRT 10.x: parse OK, builder fails (no implementation for LayerNormalization) |
| `success.log` | clean FP16 build — parser must NOT invent failures here |
