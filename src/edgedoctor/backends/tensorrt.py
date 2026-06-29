"""TensorRT backend — the beachhead (STUB).

This is the first backend edgedoctor will fully implement (Phases 1–2). For now
it's an honest stub: it implements the `Backend` interface so the seam is real
and import-able, but every method raises `NotImplementedError` rather than
pretending to work. Honesty over fake results — see the project guardrails.

When built out, `convert()` will drive PyTorch→ONNX→TensorRT (via trtexec /
Polygraphy) and `parse()` will turn the captured logs into `Facts` for failure
classes (A) op-support/CPU-fallback and (B) accuracy divergence.
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
