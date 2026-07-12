"""Export a torchvision model to ONNX — the first real slice of Phase 1.

Purpose: get hands-on with the PyTorch → ONNX export step, understand what the
exporter does, and produce a real .onnx artifact that we'll later feed to
TensorRT. We export two models (MobileNetV3-Small, ResNet18) because they
behave very differently under quantization and TensorRT:

  - MobileNetV3-Small is designed for edge (1.5M params, 58 MB ONNX); it uses
    depthwise convolutions that are fragile under INT8.
  - ResNet18 is a classic baseline (11.7M params); its simple Conv→BN→ReLU
    blocks quantize smoothly.

HOW TO RUN:
    uv run python scripts/export_onnx.py [--model mobilenet|resnet18] [--opset 17]

WHAT THIS SCRIPT TEACHES:
    1. torch.onnx.export() traces the model and writes an ONNX protobuf graph.
       Since PyTorch 2.9 the default exporter is Dynamo-based (dynamo=True),
       which captures the graph via torch.export rather than TorchScript tracing.
       We use the legacy path (dynamo=False) here for wider compatibility with
       older TensorRT versions you'll encounter on the ThinkPad.
    2. opset_version fixes which version of each ONNX operator's spec the graph
       uses. TensorRT only supports a range of opsets — too new or too old and
       conversion fails. We default to opset 17 (widely supported by TRT 8.6+).
    3. dynamic_axes lets you mark batch size (or other dims) as variable, so one
       ONNX file works for batch=1 inference and batch=8 evaluation.
    4. After export we validate the graph (onnx.checker) and print a summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def export(model_name: str = "mobilenet", opset: int = 17, out_dir: Path = Path("artifacts")) -> Path:
    """Export a torchvision model to ONNX and validate the result."""
    # Imports are inside the function so the script prints useful errors if
    # torch/onnx aren't installed, rather than failing with a bare ImportError
    # at the top.
    try:
        import torch
        import onnx
    except ImportError as e:
        raise SystemExit(
            f"Missing dependency: {e.name}.\n"
            "Install with: uv pip install torch torchvision onnx --index-url "
            "https://download.pytorch.org/whl/cpu"
        ) from e

    import torchvision.models as models

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load the pretrained model ─────────────────────────────────────
    # `weights=DEFAULT` fetches the best available pretrained weights. The
    # model is set to eval() mode — critical before export, because:
    #   - BatchNorm layers switch from tracking running stats to using them
    #   - Dropout layers become pass-through
    # Exporting a model in training mode bakes in training-time behavior.
    print(f"Loading torchvision.{model_name} (pretrained)...")
    if model_name == "mobilenet":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        input_shape = (1, 3, 224, 224)
        filename = "mobilenetv3_small.onnx"
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        input_shape = (1, 3, 224, 224)
        filename = "resnet18.onnx"
    else:
        raise SystemExit(f"Unknown model: {model_name}. Choose 'mobilenet' or 'resnet18'.")

    model.eval()

    # ── 2. Create a dummy input ──────────────────────────────────────────
    # The exporter needs an example tensor to trace the computation graph.
    # Shape = (batch=1, channels=3, height=224, width=224) — ImageNet standard.
    dummy_input = torch.randn(*input_shape)

    # ── 3. Export to ONNX ────────────────────────────────────────────────
    onnx_path = out_dir / filename
    print(f"Exporting to ONNX (opset {opset}) → {onnx_path}")

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=opset,
        # dynamo=False uses the TorchScript-tracing exporter. We choose this
        # deliberately: it's what TensorRT 8.x/10.x documentation and examples
        # target, and it produces ONNX graphs without the torch.export IR
        # differences that can confuse older parsers.
        dynamo=False,
        # input_names/output_names label the graph's I/O tensors so tools like
        # trtexec, Polygraphy, and Netron display meaningful names.
        input_names=["input"],
        output_names=["output"],
        # dynamic_axes marks batch (dim 0) as variable-length. Without this,
        # the ONNX graph hard-codes batch=1 and you'd need a separate export
        # per batch size. TensorRT uses this info for its optimization profiles
        # (--minShapes / --optShapes / --maxShapes).
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    # ── 4. Validate ──────────────────────────────────────────────────────
    # onnx.checker verifies the protobuf is well-formed and conforms to the
    # opset spec (types match, required attributes present, shapes consistent).
    # It's cheap and catches export bugs early.
    print("Validating ONNX graph...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # ── 5. Print summary ─────────────────────────────────────────────────
    graph = onnx_model.graph
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"\n{'─' * 60}")
    print(f"  Model:       {model_name}")
    print(f"  Opset:       {opset}")
    print(f"  File:        {onnx_path}  ({size_mb:.1f} MB)")
    print(f"  Inputs:      {[f'{i.name}: {[d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]}' for i in graph.input]}")
    print(f"  Outputs:     {[o.name for o in graph.output]}")
    print(f"  Nodes:       {len(graph.node)}")
    print(f"  Initializers (weights): {len(graph.initializer)}")
    print(f"{'─' * 60}")
    print("✓ Export and validation succeeded.")

    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a torchvision model to ONNX.")
    parser.add_argument("--model", default="mobilenet", choices=["mobilenet", "resnet18"])
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    export(model_name=args.model, opset=args.opset, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
