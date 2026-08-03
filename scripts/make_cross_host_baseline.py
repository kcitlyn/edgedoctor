"""Capture a portable INT8 baseline for cross-host (x86 vs ARM) comparison.

WHAT THIS IS FOR
The Pi milestone claims INT8 accuracy divergence appears on x86 but not on ARM.
Demonstrating that needs the SAME model and the SAME inputs run on two different
machines, with the outputs compared numerically. You cannot run both at once, so
one host saves a baseline and the other loads it — exactly the workflow
Polygraphy's --save-outputs / --load-outputs was built for.

RUN IT TWICE:

  1. On the x86 host (ThinkPad), which must be AVX2/AVX512 WITHOUT VNNI:
         uv run python scripts/host_capability.py      # must say YES
         uv run python scripts/make_cross_host_baseline.py --save
     Commit the resulting corpus/cross_host/ files.

  2. On the ARM host (Raspberry Pi):
         uv run python scripts/make_cross_host_baseline.py --compare
     This loads the x86 outputs and compares them against ARM's own.

WHY --save WRITES THE INPUTS TOO
Both hosts must see byte-identical inputs, or any difference could be the input
data rather than the architecture — which would make the whole comparison
worthless. Saving inputs (not just outputs) is what makes the two runs
apples-to-apples across machines.

WHY IT REFUSES TO RUN ON AN UNSUITABLE HOST
An Apple Silicon Mac is ARM64 — the same side of the saturation issue as the Pi.
A Mac-vs-Pi comparison is ARM vs ARM and will show no divergence, which looks
like a successful experiment but is a null setup. --save therefore checks
host_capability first and refuses unless the host can actually exhibit the
effect. Override with --force only if you know why.

WHAT THE COMPARISON MEANS (and doesn't)
A difference here is evidence about ARITHMETIC, not about model quality. Read it
with the same discipline as the rest of the corpus: tolerance is a policy, so the
useful output is the measured divergence and the minimum tolerance that would
have passed — not a pass/fail verdict.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _corpus_paths import scrub_log  # noqa: E402
from host_capability import describe_host  # noqa: E402

_SHAPE = "input:[1,3,224,224]"


def _run(cmd: list[str], log_path: Path) -> int:
    print(f"  $ {' '.join(cmd)}")
    with log_path.open("w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
    changed = scrub_log(log_path)
    print(f"    -> {log_path} (exit {proc.returncode}"
          f"{f', {changed} line(s) path-normalized' if changed else ''})")
    return proc.returncode


def _quantize(src: Path, dst: Path, *, reduce_range: bool) -> None:
    """Dynamic-quantize to U8S8 — the format with the x86 saturation issue.

    `reduce_range` quantizes weights to 7 bits, which is ORT's documented
    workaround: it keeps accumulations inside the 16-bit range. Capturing both
    variants is what isolates the cause — if reduce_range removes the divergence,
    saturation is confirmed as the mechanism rather than merely suspected.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(src), str(dst),
        weight_type=QuantType.QInt8,   # int8 weights + uint8 activations = U8S8
        reduce_range=reduce_range,
    )
    print(f"  quantized -> {dst.name} (reduce_range={reduce_range})")


def _host_tag() -> str:
    """Short host label used in filenames, e.g. 'x86_64' or 'arm64'."""
    return platform.machine().lower().replace("-", "_")


def _write_host_meta(out: Path, tag: str) -> None:
    """Record the capability verdict alongside the data.

    Without this, a saved baseline is unattributable months later: whether it
    demonstrates anything depends entirely on the CPU that produced it.
    """
    info = describe_host()
    try:
        import onnxruntime as ort
        info["onnxruntime_version"] = ort.__version__
    except ImportError:
        pass
    (out / f"host_{tag}.json").write_text(json.dumps(info, indent=2))
    print(f"  host capability -> host_{tag}.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--save", action="store_true",
                    help="Capture this host's outputs as the baseline (run on x86).")
    ap.add_argument("--compare", action="store_true",
                    help="Compare this host against a saved baseline (run on ARM).")
    ap.add_argument("--force", action="store_true",
                    help="Save even if this host cannot exhibit the effect.")
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--out", type=Path, default=Path("corpus/cross_host"))
    ap.add_argument("--work", type=Path, default=Path("artifacts/_cross_host_work"))
    args = ap.parse_args()

    if not (args.save or args.compare):
        ap.error("choose --save (on x86) or --compare (on ARM)")

    info = describe_host()
    print(f"Host: {info['machine']} · saturation possible: "
          f"{info['u8s8_saturation_possible']}")
    print(f"  {info['reason']}\n")

    if args.save and not info["suitable_as_x86_baseline"] and not args.force:
        raise SystemExit(
            "Refusing to save a baseline from this host.\n\n"
            f"  {info['reason']}\n\n"
            "A comparison against this baseline cannot demonstrate the x86-vs-ARM\n"
            "effect — a 'no divergence' result would be a null setup, not a\n"
            "finding. Run --save on an AVX2/AVX512 x86 host without VNNI.\n"
            "Use --force to override if you know why you want this."
        )

    fp32 = args.artifacts / "resnet18.onnx"
    if not fp32.exists():
        raise SystemExit(
            f"{fp32} not found. Run: "
            "uv run python scripts/export_onnx.py --model resnet18"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    tag = _host_tag()

    # Two quantizations: the plain U8S8 that can saturate, and the reduce_range
    # variant that shouldn't. Comparing both isolates the mechanism.
    plain = args.work / "resnet18_u8s8.onnx"
    reduced = args.work / "resnet18_u8s8_reduced.onnx"
    _quantize(fp32, plain, reduce_range=False)
    _quantize(fp32, reduced, reduce_range=True)

    inputs = args.out / "shared_inputs.json"

    if args.save:
        print(f"\n[save] capturing baseline on {tag}")
        # Save inputs alongside outputs so the other host runs identical data.
        for label, model in (("u8s8", plain), ("u8s8_reduced", reduced)):
            out_json = args.out / f"outputs_{label}_{tag}.json"
            _run(
                ["polygraphy", "run", str(model), "--onnxrt",
                 "--input-shapes", _SHAPE,
                 "--save-inputs", str(inputs),
                 "--save-outputs", str(out_json)],
                args.out / f"baseline_{label}_{tag}.log",
            )
        _write_host_meta(args.out, tag)
        print(f"\nDone. Commit {args.out}/ and run --compare on the other host.")
        return

    # --compare
    print(f"\n[compare] comparing {tag} against the saved baseline")
    if not inputs.exists():
        raise SystemExit(
            f"{inputs} not found — run --save on the x86 host first and commit "
            f"{args.out}/."
        )

    baselines = sorted(args.out.glob("outputs_u8s8_*.json"))
    baselines = [b for b in baselines if tag not in b.name]
    if not baselines:
        raise SystemExit(
            f"No baseline from a DIFFERENT host in {args.out}/. "
            "Comparing a host against itself proves nothing."
        )

    for baseline in baselines:
        other = baseline.stem.replace("outputs_u8s8_", "")
        label = "u8s8"
        model = plain
        print(f"\n  {other} vs {tag} ({label})")
        _run(
            ["polygraphy", "run", str(model), "--onnxrt",
             "--input-shapes", _SHAPE,
             "--load-inputs", str(inputs),
             "--load-outputs", str(baseline)],
            args.out / f"compare_{label}_{other}_vs_{tag}.log",
        )
    _write_host_meta(args.out, tag)
    print(f"\nDone. Diagnose with: uv run edgedoctor diagnose "
          f"{args.out}/compare_*.log -b polygraphy")


if __name__ == "__main__":
    main()
