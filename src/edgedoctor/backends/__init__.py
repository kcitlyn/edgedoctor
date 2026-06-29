"""Per-backend modules.

Each backend (tensorrt, coreml, onnxruntime, …) implements the `Backend`
interface from `base.py`. Only one is fully built at a time; the rest are stubs
that prove the seam exists. See docs/DESIGN.md.
"""
