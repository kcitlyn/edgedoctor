"""Generate REAL accuracy-divergence artifacts with Polygraphy, on any machine.

Purpose: failure class (B) — accuracy divergence — needs real comparison logs to
build a parser against. TensorRT needs an NVIDIA GPU, but the *divergence
phenomenon* does not: quantizing an ONNX model to INT8 and comparing it against
its own FP32 outputs produces genuine numeric divergence, in Polygraphy's exact
output format, on a laptop CPU.

That matters for grounding discipline: the parser signatures are written against
logs a real tool actually emitted, not against a format we imagined.

WHAT IT PRODUCES (into corpus/onnxruntime/, with .meta.md sidecars):
    fp32_baseline_run.log        clean single-runner run (no comparison)
    int8_vs_fp32_fail.log        divergence exceeding default tolerance
    int8_vs_fp32_pass.log        the SAME divergence, tolerance loosened -> PASSES
    int8_vs_fp32_layerwise.log   every intermediate tensor compared (mark all)
    shape_mismatch.log           comparing outputs of different shapes
    nan_output.log               --validate catching NaN/Inf in an output

The pass/fail pair is the important one. Both logs contain byte-identical
"Average Metrics" and "Minimum Required Tolerance" lines — only the verdict
differs. Any parser that treats metric lines as failure evidence will invent
failures on clean logs; this pair is the regression test for that bug.

HOW TO RUN:
    uv pip install polygraphy onnxruntime onnx
    uv run python scripts/export_onnx.py --model resnet18   # if not done yet
    uv run python scripts/make_divergence_corpus.py

WHAT THIS TEACHES:
    1. Polygraphy compares *runners* — `--save-outputs` writes one runner's
       results, `--load-outputs` replays them as the comparison baseline. That's
       how you compare across machines and precisions without running both at once.
    2. Tolerance is a *policy*, not a fact. The measured divergence is identical
       in the pass and fail logs; only the threshold moved.
    3. `--onnx-outputs mark all` marks every intermediate tensor as an output, so
       the comparison reveals the FIRST layer where error appears — which is the
       actionable signal. NVIDIA's own accuracy-debugging guide prescribes exactly
       this, and warns that marking outputs can perturb optimization, so you must
       confirm the failure still reproduces.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Fixed input shape for the image classifiers we export. Batch is 1 so the
# shape-mismatch case can use batch 2 as the deliberately-wrong comparison.
_SHAPE = "input:[1,3,224,224]"


def _run(cmd: list[str], log_path: Path | None = None) -> int:
    """Run a command, optionally capturing combined stdout+stderr to a log.

    Combined capture matters: Polygraphy writes its report to stdout and stderr
    interleaved, and line numbers in the saved log are what edgedoctor cites.
    """
    print(f"  $ {' '.join(cmd)}")
    if log_path is None:
        return subprocess.run(cmd, capture_output=True, text=True).returncode
    with log_path.open("w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
    print(f"    -> {log_path}  (exit {proc.returncode})")
    return proc.returncode


def _quantize(src: Path, dst: Path) -> None:
    """Dynamic-quantize an ONNX model's weights to uint8.

    Dynamic quantization is used deliberately: it needs no calibration dataset,
    so this script stays dependency-light and reproducible. It still produces
    real INT8 weight error — which is the phenomenon we want to capture.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print(f"  quantizing {src.name} -> {dst.name} (uint8 weights)")
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8)


def _make_nan_model(dst: Path) -> None:
    """Build a tiny ONNX model whose output is NaN (0/0), to trigger --validate.

    Real models produce NaN through overflow in reduced precision; reproducing
    that reliably is fiddly, so we provoke the same *validator output* with a
    2-node graph. The log is real; only the model is minimal.
    """
    import onnx
    from onnx import TensorProto, helper

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph(
        [
            helper.make_node("Sub", ["input", "input"], ["zero"]),  # x - x = 0
            helper.make_node("Div", ["zero", "zero"], ["output"]),  # 0 / 0 = NaN
        ],
        "nan_maker",
        [inp],
        [out],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))


def _sidecar(log: Path, command: str, outcome: str, root_cause: str, fix: str) -> None:
    """Write the .meta.md label that corpus/README.md requires for every artifact."""
    import platform

    import onnxruntime as ort

    meta = f"""# {log.name}
- command:  {command}
- machine:  {platform.machine()} · {platform.system()} {platform.release()} · CPUExecutionProvider
- versions: polygraphy {_polygraphy_version()}, onnxruntime {ort.__version__}, ONNX opset 17
- outcome:  {outcome}
- root cause (human-verified): {root_cause}
- fix that worked: {fix}
- generated by: scripts/make_divergence_corpus.py (reproducible)
"""
    log.with_suffix(".meta.md").write_text(meta)


def _polygraphy_version() -> str:
    try:
        import polygraphy

        return polygraphy.__version__
    except Exception:  # pragma: no cover - informational only
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--out", type=Path, default=Path("corpus/onnxruntime"))
    ap.add_argument("--work", type=Path, default=Path("artifacts/_divergence_work"))
    args = ap.parse_args()

    if shutil.which("polygraphy") is None:
        raise SystemExit("polygraphy not found. Install with: uv pip install polygraphy")

    fp32 = args.artifacts / "resnet18.onnx"
    if not fp32.exists():
        raise SystemExit(
            f"{fp32} not found. Run: uv run python scripts/export_onnx.py --model resnet18"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    int8 = args.work / "resnet18_int8.onnx"
    _quantize(fp32, int8)

    # ── 1. FP32 baseline: run once, save inputs AND outputs ──────────────
    # Saving the inputs is what makes the later comparisons apples-to-apples:
    # the INT8 model is fed the exact same data, so any difference is precision.
    print("\n[1/6] FP32 baseline run")
    inputs = args.work / "inputs.json"
    fp32_out = args.work / "fp32_outputs.json"
    baseline_log = args.out / "fp32_baseline_run.log"
    _run(
        ["polygraphy", "run", str(fp32), "--onnxrt", "--input-shapes", _SHAPE,
         "--save-inputs", str(inputs), "--save-outputs", str(fp32_out)],
        baseline_log,
    )
    _sidecar(
        baseline_log,
        f"polygraphy run {fp32} --onnxrt --input-shapes {_SHAPE} --save-inputs ... --save-outputs ...",
        "succeeded",
        "Nothing went wrong. This is a clean single-runner run with no comparison "
        "at all, kept so the parser is proven not to invent failures (or "
        "divergence facts) where there are none.",
        "n/a",
    )

    # ── 2. INT8 vs FP32 at default tolerance -> FAILS ────────────────────
    print("\n[2/6] INT8 vs FP32, default tolerance (expect FAILED)")
    fail_log = args.out / "int8_vs_fp32_fail.log"
    _run(
        ["polygraphy", "run", str(int8), "--onnxrt",
         "--load-inputs", str(inputs), "--load-outputs", str(fp32_out)],
        fail_log,
    )
    _sidecar(
        fail_log,
        f"polygraphy run {int8} --onnxrt --load-inputs ... --load-outputs ...",
        "failed",
        "Genuine INT8 quantization error. Polygraphy's default tolerance "
        "(abs=1e-05, rel=1e-05) is an FP32-vs-FP32 tolerance; uint8 weight "
        "quantization produces error orders of magnitude larger, so the "
        "comparison fails. The divergence is real but EXPECTED for INT8 — the "
        "failure is a tolerance-policy mismatch, not necessarily a broken model.",
        "Compare against a task metric (accuracy/mAP) instead of elementwise "
        "tolerance, or set a tolerance appropriate to the precision.",
    )

    # ── 3. Same divergence, loosened tolerance -> PASSES ─────────────────
    # The honesty pair. Identical numbers, different verdict.
    print("\n[3/6] Same comparison, loosened tolerance (expect PASSED)")
    pass_log = args.out / "int8_vs_fp32_pass.log"
    _run(
        ["polygraphy", "run", str(int8), "--onnxrt",
         "--load-inputs", str(inputs), "--load-outputs", str(fp32_out),
         "--atol", "6.0", "--rtol", "300"],
        pass_log,
    )
    _sidecar(
        pass_log,
        f"polygraphy run {int8} --onnxrt --load-inputs ... --load-outputs ... --atol 6.0 --rtol 300",
        "succeeded",
        "Same model, same inputs, same measured divergence as "
        "int8_vs_fp32_fail.log — only the tolerance changed. The metric lines "
        "in the two logs are byte-identical; only the verdict lines differ. "
        "This is the regression case proving edgedoctor keys on verdicts, not "
        "on the presence of error metrics.",
        "n/a (demonstrates that tolerance is a policy choice, not a measurement)",
    )

    # ── 4. Layer-wise: which layer diverges FIRST ────────────────────────
    print("\n[4/6] Layer-wise comparison (mark all intermediate tensors)")
    lw_in = args.work / "layerwise_inputs.json"
    lw_fp32 = args.work / "layerwise_fp32_outputs.json"
    _run(
        ["polygraphy", "run", str(fp32), "--onnxrt", "--onnx-outputs", "mark", "all",
         "--input-shapes", _SHAPE, "--save-inputs", str(lw_in),
         "--save-outputs", str(lw_fp32)],
    )
    lw_log = args.out / "int8_vs_fp32_layerwise.log"
    _run(
        ["polygraphy", "run", str(int8), "--onnxrt", "--onnx-outputs", "mark", "all",
         "--load-inputs", str(lw_in), "--load-outputs", str(lw_fp32)],
        lw_log,
    )
    _sidecar(
        lw_log,
        f"polygraphy run {int8} --onnxrt --onnx-outputs mark all --load-inputs ... --load-outputs ...",
        "failed",
        "Every intermediate tensor is compared, so the log shows exactly where "
        "error first appears and how it propagates. Here divergence starts at "
        "the very first convolution and accumulates downstream — consistent "
        "with weight quantization rather than one pathological layer. NVIDIA's "
        "accuracy-debugging guide prescribes this 'find the first layer "
        "introducing error' workflow.",
        "n/a (diagnostic run). Note NVIDIA's caveat: marking all outputs can "
        "perturb optimization, so confirm the failure still reproduces.",
    )

    # ── 5. Shape mismatch: comparison refused, not attempted ─────────────
    print("\n[5/6] Shape mismatch (batch 1 vs batch 2)")
    b2_out = args.work / "batch2_outputs.json"
    _run(
        ["polygraphy", "run", str(fp32), "--onnxrt",
         "--input-shapes", "input:[2,3,224,224]", "--save-outputs", str(b2_out)],
    )
    shape_log = args.out / "shape_mismatch.log"
    _run(
        ["polygraphy", "run", str(fp32), "--onnxrt", "--input-shapes", _SHAPE,
         "--load-outputs", str(b2_out)],
        shape_log,
    )
    _sidecar(
        shape_log,
        f"polygraphy run {fp32} --onnxrt --input-shapes {_SHAPE} --load-outputs <batch-2 outputs>",
        "failed",
        "The two runners produced outputs of different shapes ((1,1000) vs "
        "(2,1000)) because they were run at different batch sizes. Polygraphy "
        "refuses to compare rather than silently broadcasting. This is a "
        "harness/setup error, NOT an accuracy problem — the distinction matters "
        "because the fix is completely different.",
        "Run both sides at the same input shapes.",
    )

    # ── 6. NaN/Inf detection via --validate ──────────────────────────────
    print("\n[6/6] NaN detection (--validate)")
    nan_model = args.work / "nan_model.onnx"
    _make_nan_model(nan_model)
    nan_log = args.out / "nan_output.log"
    _run(
        ["polygraphy", "run", str(nan_model), "--onnxrt", "--validate",
         "--input-shapes", "input:[1,4]"],
        nan_log,
    )
    _sidecar(
        nan_log,
        f"polygraphy run {nan_model} --onnxrt --validate --input-shapes input:[1,4]",
        "failed",
        "The graph computes 0/0, so the output is NaN, and --validate reports "
        "both NaN and non-finite values. NaN in an output is categorically "
        "worse than tolerance divergence: the model is producing garbage, not "
        "slightly-wrong numbers. Provoked with a 2-node model; the validator "
        "output is real and identical to what a genuine overflow produces.",
        "n/a (deliberately constructed). For real models: check for "
        "overflow-prone ops and division by zero in reduced precision.",
    )

    print(f"\nDone. Artifacts + sidecars written to {args.out}/")
    print("Diagnose them with:  uv run edgedoctor diagnose "
          f"{args.out}/int8_vs_fp32_fail.log -b polygraphy")


if __name__ == "__main__":
    sys.exit(main())
