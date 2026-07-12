"""TensorRT backend — the beachhead (STUB).

This is the first backend edgedoctor will fully implement (Phases 1–2). For now
it's an honest stub: it implements the `Backend` interface so the seam is real
and import-able, but every method raises `NotImplementedError` rather than
pretending to work.

When built out:
  - `convert()` will drive PyTorch→ONNX→TensorRT (via trtexec/Polygraphy)
  - `parse()` will turn captured trtexec/Polygraphy logs into `Facts` for:
      (A) op-support failures / build failures
      (B) accuracy divergence (FP32 vs INT8, layer-level SQNR)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import Backend, Facts


class TensorRTBackend(Backend):
    name = "tensorrt"

    def convert(self, model_path: Path, **options: Any) -> list[Path]:
        raise NotImplementedError(
            "TensorRT conversion is not implemented yet (Phase 1). See ROADMAP.md."
        )

    def parse(self, artifact_path: Path) -> Facts:
        raise NotImplementedError(
            "TensorRT artifact parsing is not implemented yet (Phase 2). See ROADMAP.md."
        )
