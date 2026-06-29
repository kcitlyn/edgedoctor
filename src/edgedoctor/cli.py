"""edgedoctor command-line interface.

A *thin* surface over the (future) library core. Today it parses arguments and
honestly reports what's implemented — it does not pretend to diagnose. We use
the standard-library `argparse` so the CLI runs with zero dependencies; heavier
machinery arrives per-phase.

Run it:  `edgedoctor diagnose model.onnx --backend tensorrt`
"""

from __future__ import annotations

import argparse
import sys

from . import __version__

# Backends edgedoctor knows the *name* of. Only "tensorrt" has even a stub
# implementation; the rest are declared so --help honestly shows the planned
# cross-vendor surface without claiming they work.
KNOWN_BACKENDS = ["tensorrt", "coreml", "onnxruntime", "tflite", "executorch"]
IMPLEMENTED_BACKENDS: set[str] = set()  # none fully built yet — honest status


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edgedoctor",
        description="The universal, cross-vendor edge-AI deployment diagnostician.",
    )
    parser.add_argument("--version", action="version", version=f"edgedoctor {__version__}")

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    diagnose = sub.add_parser(
        "diagnose",
        help="Diagnose why a model fails or underperforms on an edge backend.",
    )
    diagnose.add_argument("model", help="Path to the model (e.g. model.onnx).")
    diagnose.add_argument(
        "--backend",
        default="tensorrt",
        choices=KNOWN_BACKENDS,
        help="Target edge backend (default: tensorrt).",
    )

    return parser


def _cmd_diagnose(model: str, backend: str) -> int:
    """Honest stub: report status instead of faking a diagnosis."""
    print(f"edgedoctor v{__version__}")
    print(f"  model:   {model}")
    print(f"  backend: {backend}")
    print()
    if backend not in IMPLEMENTED_BACKENDS:
        print("⚠️  Not implemented yet.")
        print(
            f"    The '{backend}' diagnosis pipeline is still being built "
            "(Phases 1–2)."
        )
        print("    See ROADMAP.md for what's now / next / later.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (referenced by [project.scripts] in pyproject.toml)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "diagnose":
        return _cmd_diagnose(args.model, args.backend)

    # No subcommand → show help and exit non-zero so scripts notice.
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
