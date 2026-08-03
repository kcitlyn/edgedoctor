"""Report whether THIS host can exhibit x86 U8S8 quantization saturation.

WHY THIS EXISTS
The Pi milestone's headline scientific claim is that INT8 accuracy divergence
appears on x86 but not on ARM. That claim is real but narrower than it sounds,
and the narrowing is easy to get wrong in a way that would invalidate the whole
experiment.

ONNX Runtime's own documentation states it exactly:

    "There is no such issue on other CPU architectures (x64 with VNNI and Arm)."
    -- https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

So saturation requires ALL of:
  1. x86-64, AND
  2. AVX2 or AVX512, AND
  3. NOT VNNI  (VNNI accumulates into 32-bit lanes, so nothing overflows), AND
  4. U8S8 quantization (uint8 activations, int8 weights) without reduce_range.

The mechanism: on AVX2/AVX512 without VNNI, ORT uses VPMADDUBSW, which
accumulates uint8 x int8 products into 16-BIT lanes. A large dot product
overflows and is clamped, so the result is not merely rounded — it is wrong.
VNNI hardware has no 16-bit intermediate, and ARM's dot-product instructions
behave like VNNI.

THE TRAP THIS SCRIPT EXISTS TO PREVENT
An Apple Silicon Mac is ARM64 — the SAME side of this issue as the Pi's
Cortex-A76. Comparing a Mac against a Pi is ARM vs ARM, expected to show no
divergence, and proving nothing. A "no divergence found!" result from that
comparison would look like a successful experiment while actually being a null
setup. So run this on any machine BEFORE trusting a divergence comparison from
it.

USAGE
    uv run python scripts/host_capability.py           # human summary
    uv run python scripts/host_capability.py --json     # machine-readable
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from typing import Any


def _cpu_flags() -> set[str]:
    """CPU feature flags, lower-cased, best-effort across platforms.

    Returns an empty set when flags can't be read (notably on Apple Silicon,
    where there are no x86 flags to find) — callers must treat "unknown" as
    "cannot confirm", never as "absent".
    """
    machine = platform.machine().lower()

    # Linux: /proc/cpuinfo is authoritative.
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("flags"):
                    return set(line.split(":", 1)[1].lower().split())
    except OSError:
        pass

    # macOS on Intel: sysctl exposes leaf7 features (where AVX512/VNNI live).
    if platform.system() == "Darwin" and machine in ("x86_64", "amd64"):
        flags: set[str] = set()
        for key in ("machdep.cpu.features", "machdep.cpu.leaf7_features"):
            try:
                out = subprocess.run(["sysctl", "-n", key], capture_output=True,
                                     text=True, timeout=5)
                flags |= set(out.stdout.lower().split())
            except (OSError, subprocess.SubprocessError):
                pass
        return flags

    return set()


def _has(flags: set[str], *needles: str) -> bool:
    """True if any flag contains any needle.

    Substring matching because AVX512 appears under many names
    (avx512f, avx512_vnni, avx512vnni) that differ by kernel and vendor.
    """
    return any(n in f for f in flags for n in needles)


def describe_host() -> dict[str, Any]:
    """Classify this host's exposure to U8S8 saturation."""
    machine = platform.machine().lower()
    flags = _cpu_flags()

    is_x86 = machine in ("x86_64", "amd64", "i386", "i686")
    is_arm = machine in ("arm64", "aarch64", "armv7l", "armv8l")

    has_avx2 = _has(flags, "avx2")
    has_avx512 = _has(flags, "avx512")
    # VNNI is spelled avx512_vnni / avx512vnni / avx_vnni depending on the
    # source, so match the substring rather than an exact token.
    has_vnni = _has(flags, "vnni")

    flags_readable = bool(flags)

    if is_arm:
        saturation_possible = False
        reason = (
            "ARM: ONNX Runtime documents no saturation issue on Arm. Its "
            "dot-product instructions accumulate into 32-bit lanes, like VNNI."
        )
    elif not is_x86:
        saturation_possible = False
        reason = f"architecture '{machine}' is neither x86 nor ARM; not applicable."
    elif not flags_readable:
        saturation_possible = None  # genuinely unknown
        reason = (
            "x86 host, but CPU feature flags could not be read, so AVX2/VNNI "
            "support is unknown. Treat this as UNVERIFIED, not as absent."
        )
    elif has_vnni:
        saturation_possible = False
        reason = (
            "x86 with VNNI: accumulation is 32-bit, so there is nothing to "
            "saturate. ORT's docs say reduce_range is not needed here."
        )
    elif has_avx2 or has_avx512:
        saturation_possible = True
        reason = (
            "x86 with AVX2/AVX512 but WITHOUT VNNI: ORT uses VPMADDUBSW, which "
            "accumulates uint8 x int8 into 16-bit lanes and must clamp on "
            "overflow. This is the configuration where U8S8 saturation occurs."
        )
    else:
        saturation_possible = False
        reason = (
            "x86 without AVX2/AVX512: ORT does not take the VPMADDUBSW path, "
            "so this specific saturation issue does not arise."
        )

    return {
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "system": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "is_x86": is_x86,
        "is_arm": is_arm,
        "cpu_flags_readable": flags_readable,
        "has_avx2": has_avx2 if flags_readable else None,
        "has_avx512": has_avx512 if flags_readable else None,
        "has_vnni": has_vnni if flags_readable else None,
        "u8s8_saturation_possible": saturation_possible,
        "reason": reason,
        "suitable_as_x86_baseline": saturation_possible is True,
    }


def _onnxruntime_info() -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError:
        return {"installed": False}
    return {
        "installed": True,
        "version": ort.__version__,
        "providers": ort.get_available_providers(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="Machine-readable output.")
    args = ap.parse_args()

    info = describe_host()
    info["onnxruntime"] = _onnxruntime_info()

    if args.json:
        print(json.dumps(info, indent=2))
        return

    print(f"Host:         {info['machine']} · {info['system']}")
    print(f"Processor:    {info['processor']}")
    if info["cpu_flags_readable"]:
        print(f"AVX2 {info['has_avx2']} · AVX512 {info['has_avx512']} · "
              f"VNNI {info['has_vnni']}")
    else:
        print("CPU flags:    not readable on this platform")

    verdict = info["u8s8_saturation_possible"]
    label = {True: "YES", False: "NO", None: "UNKNOWN"}[verdict]
    print(f"\nU8S8 saturation possible here: {label}")
    print(f"  {info['reason']}")

    if verdict is True:
        print("\n=> Valid x86 baseline host. Capture the INT8 comparison here.")
    elif verdict is False:
        print("\n=> NOT a valid x86 baseline host. A divergence comparison from")
        print("   this machine cannot demonstrate the x86-vs-ARM effect; a")
        print("   'no divergence' result here would be a null setup, not a finding.")
    else:
        print("\n=> Capability unverified. Do not claim either result from this host.")

    ort_info = info["onnxruntime"]
    if ort_info["installed"]:
        print(f"\nonnxruntime {ort_info['version']} · "
              f"{', '.join(ort_info['providers'])}")
    else:
        print("\nonnxruntime not installed (needed to capture a comparison).")


if __name__ == "__main__":
    main()
