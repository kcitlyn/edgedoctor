"""Verify an ONNX model against its PyTorch source — the golden-baseline check.

Purpose: confirm the ONNX export didn't corrupt the model by running the same
input through both PyTorch and ONNX Runtime and comparing outputs. This is
exactly what Polygraphy does (`polygraphy run --trt --onnxrt`) except here we
compare PyTorch-native vs ONNX-Runtime-on-CPU, which is the "golden" reference
comparison before any hardware-specific backend enters the picture.

If this check FAILS, the export is already wrong — don't waste time debugging
TensorRT or the edge target. Fix the export first.

HOW TO RUN:
    uv run python scripts/verify_onnx.py [--model mobilenet|resnet18] [--atol 1e-5]

WHAT THIS TEACHES:
    1. onnxruntime.InferenceSession loads and runs the graph on CPU.
    2. Comparison uses numpy.allclose (abs + rel tolerance) — the same semantic
       as Polygraphy's default CompareFunc.simple.
    3. "Max absolute difference" and "cosine similarity" are the two complementary
       metrics: abs diff catches magnitude errors; cosine catches shape distortion.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def verify(model_name: str = "mobilenet", atol: float = 1e-5, out_dir: Path = Path("artifacts")) -> None:
    # numpy is imported HERE, not at module scope, for the same reason torch and
    # onnxruntime are: `--help` must work on a machine that can't run the script.
    # A generator that can't even explain itself without the full ML stack
    # installed is needlessly hostile, and CI (which installs no heavy deps)
    # checks exactly this.
    import numpy as np

    try:
        import torch
        import onnxruntime as ort
    except ImportError as e:
        raise SystemExit(
            f"Missing dependency: {e.name}.\n"
            "Install with: uv pip install torch torchvision onnxruntime --extra-index-url "
            "https://download.pytorch.org/whl/cpu"
        ) from e

    import torchvision.models as models

    # ── 1. Load PyTorch model ────────────────────────────────────────────
    print(f"Loading PyTorch {model_name}...")
    if model_name == "mobilenet":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        onnx_path = out_dir / "mobilenetv3_small.onnx"
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        onnx_path = out_dir / "resnet18.onnx"
    else:
        raise SystemExit(f"Unknown model: {model_name}")

    model.eval()

    if not onnx_path.exists():
        raise SystemExit(
            f"ONNX file not found: {onnx_path}\n"
            f"Run `uv run python scripts/export_onnx.py --model {model_name}` first."
        )

    # ── 2. Create deterministic input ────────────────────────────────────
    # Use a fixed seed so results are reproducible across runs.
    rng = np.random.default_rng(seed=42)
    input_np = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)
    input_torch = torch.from_numpy(input_np)

    # ── 3. Run PyTorch inference ─────────────────────────────────────────
    with torch.no_grad():
        pt_output = model(input_torch).numpy()

    # ── 4. Run ONNX Runtime inference ────────────────────────────────────
    # InferenceSession loads the ONNX protobuf and picks the best available
    # ExecutionProvider. On this machine that's CPUExecutionProvider — which is
    # the trusted baseline (identical math to PyTorch float32).
    print(f"Running ONNX Runtime on {onnx_path}...")
    session = ort.InferenceSession(str(onnx_path))
    ort_output = session.run(None, {"input": input_np})[0]

    # ── 5. Compare ───────────────────────────────────────────────────────
    max_abs_diff = float(np.max(np.abs(pt_output - ort_output)))
    # Cosine similarity: measures whether the output *shape* (direction) is
    # preserved, independent of scale. 1.0 = identical direction; values near
    # 0.997+ are considered passing by Polygraphy.
    cos_sim = float(
        np.dot(pt_output.flatten(), ort_output.flatten())
        / (np.linalg.norm(pt_output) * np.linalg.norm(ort_output) + 1e-12)
    )
    matches = bool(np.allclose(pt_output, ort_output, atol=atol, rtol=1e-5))

    print(f"\n{'─' * 60}")
    print(f"  Model:                {model_name}")
    print(f"  ONNX file:            {onnx_path}")
    print(f"  Output shape:         {pt_output.shape}")
    print(f"  Max absolute diff:    {max_abs_diff:.2e}")
    print(f"  Cosine similarity:    {cos_sim:.8f}")
    print(f"  numpy.allclose(atol={atol}): {'PASS ✓' if matches else 'FAIL ✗'}")
    print(f"{'─' * 60}")

    if not matches:
        print("\n⚠️  ONNX output diverges from PyTorch beyond tolerance!")
        print("    This means the export introduced an error — fix the export")
        print("    before attempting any hardware-specific conversion.")
        raise SystemExit(1)
    else:
        print("\n✓ ONNX Runtime output matches PyTorch — export is correct.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ONNX export against PyTorch.")
    parser.add_argument("--model", default="mobilenet", choices=["mobilenet", "resnet18"])
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    verify(model_name=args.model, atol=args.atol, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
