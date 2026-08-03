"""Generate REAL ONNX Runtime placement/fallback artifacts, on any machine.

Purpose: failure class (A) *CPU fallback* — the class that lives in ONNX Runtime,
not TensorRT. TensorRT refuses to build when it hits an op it can't handle, so
"fell back to CPU" is not a TensorRT failure mode at all; ORT silently *succeeds*
while quietly running part of your graph somewhere slow. That silence is the
whole problem, and it is why this class belongs here.

This is also the Raspberry Pi 5 milestone's backend. A Pi has no CUDA, so
TensorRT can never run there — but ORT installs via pip on Pi OS aarch64 and the
SAME parser handles its logs. Building it on the Mac first makes Pi day a host
swap rather than a build day.

WHAT IT PRODUCES (into corpus/onnxruntime/, with .meta.md sidecars):
    ort_all_nodes_one_ep.log     every node on one EP — the clean case
    ort_partial_fallback.log     graph SPLIT across CoreML and CPU (the bug)
    ort_cpu_only.log             CPU-only session, nothing to fall back from
    ort_missing_provider.log     a requested EP that isn't available in this build

The partial-fallback log is the valuable one. ORT reports both the node counts
per EP *and* the number of partitions the graph was cut into — and partitions
matter more than counts, because each boundary is a synchronization point. Three
unsupported ops scattered through a graph hurt far more than three adjacent ones.

HOW TO RUN:
    uv run python scripts/export_onnx.py --model resnet18   # if not done yet
    uv run python scripts/make_ort_corpus.py

WHAT THIS TEACHES:
    1. ORT logs node placement only at log_severity_level=0 (VERBOSE) via
       `VerifyEachNodeIsAssignedToAnEp`. At default severity you get NO placement
       information — which is exactly why so much CPU fallback goes unnoticed.
    2. Providers are a PRIORITY LIST, not a request. ORT walks it in order asking
       each EP `GetCapability`, and anything nobody claims lands on CPU. CPU is
       the guaranteed fallback, so a session almost never fails outright.
    3. An EP silently missing from the build is a distinct failure: you think you
       are testing accelerated execution and you are measuring CPU.
    4. Partition count is the performance-relevant fact, not just node count.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _corpus_paths import scrub_log  # noqa: E402

# The child-process driver. ORT's verbose log goes to the process's stderr from
# C++, so it cannot be captured from inside the same process with contextlib
# redirects — it has to be a subprocess with stderr redirected at the OS level.
# That is also how a user would capture it, which keeps line numbers honest.
_DRIVER = '''
import sys
import onnxruntime as ort

model, providers = sys.argv[1], sys.argv[2].split(",")
so = ort.SessionOptions()
so.log_severity_level = 0    # VERBOSE — required for placement logging
so.log_verbosity_level = 1
try:
    sess = ort.InferenceSession(model, so, providers=providers)
    print("SESSION_PROVIDERS: " + ",".join(sess.get_providers()))
except Exception as exc:
    print("SESSION_FAILED: %s: %s" % (type(exc).__name__, exc))
    sys.exit(1)
'''


def _run_session(model: Path, providers: list[str], log_path: Path) -> int:
    """Create an ORT session in a subprocess, capturing its verbose log.

    Combined stdout+stderr capture, as with the Polygraphy corpus: the line
    numbers in the saved file are what edgedoctor will cite, so they must match
    what a user would see.
    """
    cmd = [sys.executable, "-c", _DRIVER, str(model), ",".join(providers)]
    print(f"  $ python -c <driver> {model.name} {','.join(providers)}")
    with log_path.open("w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
    # Mask this machine's absolute paths. Done at capture time so committed logs
    # are portable by construction; preserves line numbers, which edgedoctor
    # cites. See scripts/_corpus_paths.py.
    changed = scrub_log(log_path)
    print(f"    -> {log_path}  (exit {proc.returncode}"
          f"{f', {changed} line(s) path-normalized' if changed else ''})")
    return proc.returncode


def _make_partial_model(dst: Path) -> None:
    """Build a graph that deliberately SPLITS across two execution providers.

    Conv/Relu are well supported by the CoreML EP; Erf and Round are not, and
    they sit in the MIDDLE of the graph on purpose. Putting them mid-graph forces
    ORT to cut the graph into multiple partitions rather than lopping off a tail,
    which is what produces the interesting multi-partition log.

    Tiny (8x8) because the artifact is a log, not a benchmark.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8, 8])
    rng = np.random.default_rng(0)  # seeded: the corpus must be reproducible
    weight = helper.make_tensor(
        "w", TensorProto.FLOAT, [3, 3, 3, 3],
        rng.standard_normal(3 * 3 * 3 * 3).astype("float32").ravel(),
    )
    nodes = [
        helper.make_node("Conv", ["input", "w"], ["c1"], name="Conv_0", pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["c1"], ["r1"], name="Relu_0"),
        helper.make_node("Erf", ["r1"], ["e1"], name="Erf_0"),
        helper.make_node("Round", ["e1"], ["rd"], name="Round_0"),
        helper.make_node("Relu", ["rd"], ["output"], name="Relu_1"),
    ]
    graph = helper.make_graph(nodes, "partial_fallback", [inp], [out],
                             initializer=[weight])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))


def _sidecar(log: Path, command: str, outcome: str, root_cause: str, fix: str) -> None:
    """Write the .meta.md label corpus/README.md requires for every artifact."""
    import onnxruntime as ort

    providers = ", ".join(ort.get_available_providers())
    meta = f"""# {log.name}
- command:  {command}
- machine:  {platform.machine()} · {platform.system()} {platform.release()}
- versions: onnxruntime {ort.__version__}, ONNX opset 17
- available EPs: {providers}
- outcome:  {outcome}
- root cause (human-verified): {textwrap.fill(root_cause, 76,
                                              subsequent_indent="  ")}
- fix that worked: {fix}
- generated by: scripts/make_ort_corpus.py (reproducible)
"""
    log.with_suffix(".meta.md").write_text(meta)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    ap.add_argument("--out", type=Path, default=Path("corpus/onnxruntime"))
    ap.add_argument("--work", type=Path, default=Path("artifacts/_ort_work"))
    args = ap.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        raise SystemExit(
            "onnxruntime not found. Install with: uv pip install onnxruntime onnx"
        ) from None

    available = ort.get_available_providers()
    print(f"Available execution providers: {', '.join(available)}")

    # The accelerator EP differs by platform: CoreML on macOS, and on a Pi there
    # may be none beyond CPU. Pick whatever real accelerator this host offers so
    # the script produces a genuine split wherever it runs.
    accel = next(
        (p for p in ("CoreMLExecutionProvider", "CUDAExecutionProvider",
                     "XnnpackExecutionProvider") if p in available),
        None,
    )

    resnet = args.artifacts / "resnet18.onnx"
    if not resnet.exists():
        raise SystemExit(
            f"{resnet} not found. Run: "
            "uv run python scripts/export_onnx.py --model resnet18"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)

    # ── 1. Everything on one EP: the clean case ───────────────────────────
    # A parser that reports fallback here is hallucinating. This is the
    # ORT-side equivalent of the divergence corpus's clean baseline.
    if accel:
        print(f"\n[1/4] All nodes on one EP ({accel})")
        log = args.out / "ort_all_nodes_one_ep.log"
        _run_session(resnet, [accel, "CPUExecutionProvider"], log)
        _sidecar(
            log,
            f"InferenceSession(resnet18.onnx, providers=[{accel}, CPUExecutionProvider]) "
            "with log_severity_level=0",
            "succeeded",
            f"Nothing wrong. Every node was claimed by {accel}, so the graph runs "
            "as one partition with no host/device boundary mid-graph. Included as "
            "the negative control: any rule that reports CPU fallback on this log "
            "is inventing a failure.",
            "n/a — this is the desired state.",
        )
    else:
        print("\n[1/4] SKIPPED — no accelerator EP available on this host")

    # ── 2. Partial fallback: the actual bug ──────────────────────────────
    if accel:
        print(f"\n[2/4] Partial fallback ({accel} + CPU)")
        partial = args.work / "partial_fallback.onnx"
        _make_partial_model(partial)
        log = args.out / "ort_partial_fallback.log"
        _run_session(partial, [accel, "CPUExecutionProvider"], log)
        _sidecar(
            log,
            f"InferenceSession(partial_fallback.onnx, providers=[{accel}, "
            "CPUExecutionProvider]) with log_severity_level=0",
            "succeeded-with-warnings",
            f"Erf and Round are not supported by the {accel} EP, and they sit in "
            "the middle of the graph. ORT therefore cut the graph into multiple "
            "partitions, running the Conv/Relu sections on the accelerator and "
            "the unsupported ops on CPU. The session SUCCEEDS, so nothing alerts "
            "the user — but every partition boundary is a synchronization point, "
            "and the cost is dominated by the number of boundaries rather than "
            "the number of CPU nodes.",
            "Replace or remove the unsupported ops, or accept the split "
            "knowingly. Check partition count, not just node count.",
        )

    # ── 3. CPU-only: nothing to fall back FROM ───────────────────────────
    # Distinct from case 1: all-on-CPU is only a problem if you expected
    # acceleration. A rule must not treat a deliberate CPU session as a failure.
    print("\n[3/4] CPU-only session")
    log = args.out / "ort_cpu_only.log"
    _run_session(resnet, ["CPUExecutionProvider"], log)
    _sidecar(
        log,
        "InferenceSession(resnet18.onnx, providers=[CPUExecutionProvider]) "
        "with log_severity_level=0",
        "succeeded",
        "A deliberate CPU-only session. Every node is on CPU, but no accelerator "
        "was ever requested, so this is not fallback — it is the stated intent. "
        "Distinguishing this from case 2 is the point: identical 'all nodes on "
        "CPUExecutionProvider' placement means something completely different "
        "depending on what was requested.",
        "n/a — intended configuration.",
    )

    # ── 4. Requested EP not in this build ────────────────────────────────
    # Silent and expensive: you believe you are measuring accelerated
    # execution and you are measuring CPU.
    print("\n[4/4] Missing/unavailable provider")
    log = args.out / "ort_missing_provider.log"
    _run_session(resnet, ["TensorrtExecutionProvider", "CPUExecutionProvider"], log)
    _sidecar(
        log,
        "InferenceSession(resnet18.onnx, providers=[TensorrtExecutionProvider, "
        "CPUExecutionProvider]) with log_severity_level=0",
        "succeeded-with-warnings",
        "TensorrtExecutionProvider is not present in this onnxruntime build (a "
        "CPU/CoreML wheel on macOS). ORT warns and drops it rather than failing, "
        "so the session silently runs entirely on CPU. This is the most "
        "misleading case in the set: the code looks like it requested "
        "acceleration and the run reports success.",
        "Install the matching onnxruntime-gpu build, or assert on "
        "session.get_providers() after construction.",
    )

    print(f"\nDone. Artifacts in {args.out}/")
    print("Inspect with: uv run edgedoctor parse <log> -b onnxruntime")


if __name__ == "__main__":
    main()
