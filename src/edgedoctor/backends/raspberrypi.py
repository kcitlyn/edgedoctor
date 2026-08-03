"""Raspberry Pi host-health backend — the fact source that validates benchmarks.

WHY A BENCHMARK TOOL NEEDS THIS
Every other backend answers "why is my model wrong or slow?" This one answers a
prior question: "is this machine currently capable of producing a trustworthy
number at all?" A Pi under thermal throttling or marginal power silently runs its
CPU slower, so a latency measurement taken then is not a property of your model —
it is a property of your power supply. Reporting such a number as a model
benchmark is the most common way edge performance work goes wrong, and nothing
in the ML stack will tell you it happened.

That makes this a first-class diagnostic source, not a footnote: it can
INVALIDATE a measurement the rest of the tool would otherwise explain.

WHAT IT PARSES
  1. `vcgencmd get_throttled` — a bitfield of undervoltage / frequency-capping /
     throttling / soft-temperature-limit state, in two halves (see below).
  2. `vcgencmd measure_temp` and `measure_clock arm` — supporting context.
  3. Kernel OOM-killer messages from dmesg/journalctl — why a model died with no
     Python traceback.

THE TWO HALVES OF THE BITFIELD, AND WHY THE DISTINCTION IS THE WHOLE POINT
Bits 0-3 are live ("is it happening right now"); bits 16-19 are sticky ("did it
happen earlier"). They matter differently and the difference is easy to get
backwards:

  - A live bit means any measurement you are taking RIGHT NOW is suspect.
  - A sticky bit with its live counterpart clear means the machine is fine now,
    but something already happened — so any number collected EARLIER in this
    session may be worthless, and you cannot tell which ones from this fact
    alone.

Reporting a sticky bit as if it were live would cry wolf on a healthy machine;
reporting a live bit as merely historical would let a corrupted benchmark
through. So the parser records the two halves as separate fact kinds and the
rules treat them as different severities.

GROUNDING NOTE ON THE OUTPUT FORMAT
The bit meanings below are from Raspberry Pi's official documentation. The exact
output STRING is not: `vcgencmd`'s own source only does `printf("%s\\n", result)`
— the `throttled=0x...` text is produced by the closed VideoCore firmware, so it
cannot be verified from source the way the TensorRT and Polygraphy signatures
were. This parser therefore accepts both the widely-reported `throttled=0x50005`
form and a bare hex value, rather than betting on one spelling. When the Pi
arrives, the real output goes into corpus/raspberrypi/ and this note gets
replaced by a citation.

Reference: https://www.raspberrypi.com/documentation/computers/os.html
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from .base import Backend, Fact, Facts

# ── The bitfield ──────────────────────────────────────────────────────────
# Official meanings. Bits 4-15 are unassigned; we deliberately do NOT guess at
# them — an undocumented bit is reported as unknown rather than invented.

#: bit -> (short key, human description). Live conditions.
LIVE_BITS: dict[int, tuple[str, str]] = {
    0: ("undervoltage", "undervoltage detected"),
    1: ("freq_capped", "Arm frequency capped"),
    2: ("throttled", "currently throttled"),
    3: ("soft_temp_limit", "soft temperature limit active"),
}

#: bit -> (short key, human description). Sticky "has occurred" records.
STICKY_BITS: dict[int, tuple[str, str]] = {
    16: ("undervoltage_occurred", "undervoltage has occurred"),
    17: ("freq_capped_occurred", "Arm frequency capping has occurred"),
    18: ("throttled_occurred", "throttling has occurred"),
    19: ("soft_temp_limit_occurred", "soft temperature limit has occurred"),
}

#: Bits with no documented meaning. Present so unknown bits can be reported
#: honestly instead of silently dropped or guessed at.
_DOCUMENTED = set(LIVE_BITS) | set(STICKY_BITS)


def decode_throttled(value: int) -> dict[str, Any]:
    """Decode a `get_throttled` bitfield into its documented components.

    Returns live conditions and sticky records separately — they mean different
    things (see the module docstring) and collapsing them loses the distinction
    that decides whether a measurement is currently trustworthy.

    Undocumented set bits are reported under `unknown_bits` rather than ignored:
    a bit we can't explain is itself worth surfacing, and inventing a meaning for
    it would violate the grounding discipline.
    """
    live = [key for bit, (key, _) in LIVE_BITS.items() if value & (1 << bit)]
    sticky = [key for bit, (key, _) in STICKY_BITS.items() if value & (1 << bit)]
    unknown = [
        bit for bit in range(value.bit_length())
        if value & (1 << bit) and bit not in _DOCUMENTED
    ]
    return {
        "raw": value,
        "hex": f"0x{value:X}",
        "live": live,
        "sticky": sticky,
        "unknown_bits": unknown,
        "healthy": not live and not sticky and not unknown,
    }


def describe(keys: list[str]) -> str:
    """Human-readable list for a set of decoded bit keys."""
    lookup = {k: d for k, d in list(LIVE_BITS.values()) + list(STICKY_BITS.values())}
    return ", ".join(lookup.get(k, k) for k in keys)


_Sig = tuple[str, re.Pattern[str], Callable[[re.Match[str]], str]]

_SIGNATURES: list[_Sig] = [
    # ── vcgencmd get_throttled ────────────────────────────────────────────
    # The `throttled=` key is REQUIRED, not optional. An earlier version also
    # accepted a bare `0x...` value, to be tolerant about a format that comes
    # from closed firmware — but that matched `gfp_mask=0x140dca` in a kernel OOM
    # log and decoded it as a throttle bitfield, inventing undervoltage and
    # throttling out of a memory error. Tolerance in a signature is not free:
    # a pattern loose enough to match anything hex will eventually fabricate a
    # hardware fault. Both the hex and decimal spellings of the VALUE are still
    # accepted, since only the value's formatting is genuinely uncertain.
    # Regression: tests/fixtures/raspberrypi/oom_kill.log.
    (
        "throttled_bitfield",
        re.compile(
            r"throttled\s*=\s*(?:0x(?P<hex>[0-9a-fA-F]+)|(?P<dec>\d+))\b"
        ),
        lambda m: (
            f"vcgencmd get_throttled = 0x{m['hex'].upper()}"
            if m["hex"] else f"vcgencmd get_throttled = {m['dec']}"
        ),
    ),
    # ── Supporting context ────────────────────────────────────────────────
    (
        "soc_temperature",
        re.compile(r"temp\s*=\s*(?P<celsius>[\d.]+)'?C"),
        lambda m: f"SoC temperature {m['celsius']}°C",
    ),
    (
        "arm_clock",
        re.compile(r"frequency\((?P<clock_id>\d+)\)\s*=\s*(?P<hertz>\d+)"),
        lambda m: f"Arm clock {int(m['hertz']) // 1_000_000} MHz",
    ),
    # ── Kernel OOM killer ─────────────────────────────────────────────────
    # Why a process vanished with no Python traceback: the kernel killed it.
    # Format from mm/oom_kill.c.
    (
        "oom_kill",
        re.compile(
            r"Out of memory: Killed process (?P<pid>\d+) \((?P<process>[^)]+)\)"
        ),
        lambda m: f"Kernel OOM-killed process '{m['process']}' (pid {m['pid']})",
    ),
    (
        "oom_invoked",
        re.compile(
            r"(?P<invoker>[\w./-]+) invoked oom-killer: "
            r"gfp_mask=(?P<gfp_mask>\S+?),.*?order=(?P<order>-?\d+)"
        ),
        lambda m: f"'{m['invoker']}' triggered the kernel OOM killer",
    ),
    (
        "oom_memory_summary",
        re.compile(
            r"Mem-Info:|"
            r"Total swap = (?P<total_swap>\d+)kB|"
            r"(?P<pages>\d+) pages RAM"
        ),
        lambda m: "Kernel memory summary at OOM time",
    ),
    # ORT/allocator-level allocation failure — distinct from a kernel OOM kill:
    # here the process survived and raised, so there IS a traceback.
    (
        "allocation_failed",
        re.compile(
            r"Failed to allocate memory for requested buffer of size (?P<bytes>\d+)"
            r"|bad_alloc|std::bad_alloc"
        ),
        lambda m: (
            f"Allocator failed to reserve {int(m['bytes']):,} bytes"
            if m.groupdict().get("bytes") else "Memory allocation failed (bad_alloc)"
        ),
    ),
]


class RaspberryPiBackend(Backend):
    """Parses Raspberry Pi host-health output into benchmark-validity Facts."""

    name = "raspberrypi"

    def convert(self, model_path: Path, **options: Any) -> list[Path]:
        """Not applicable: this backend describes the HOST, not a model.

        There is nothing to convert — the artifacts are `vcgencmd` output and
        kernel logs. Kept explicit rather than silently inherited so the seam
        stays honest about what each backend does.
        """
        raise NotImplementedError(
            "raspberrypi is a host-health fact source, not a model converter. "
            "Capture artifacts with: vcgencmd get_throttled, dmesg."
        )

    def parse(self, artifact_path: Path) -> Facts:
        text = artifact_path.read_text(errors="replace")
        return self.parse_text(text, artifact_name=artifact_path.name)

    def parse_text(self, text: str, artifact_name: str = "<string>") -> Facts:
        facts: list[Fact] = []

        for lineno, line in enumerate(text.splitlines(), start=1):
            for kind, pattern, summarize in _SIGNATURES:
                m = pattern.search(line)
                if m is None:
                    continue
                groups = {k: v for k, v in m.groupdict().items() if v is not None}

                if kind == "throttled_bitfield":
                    value = (
                        int(groups["hex"], 16) if "hex" in groups
                        else int(groups["dec"])
                    )
                    decoded = decode_throttled(value)
                    groups.update(decoded)
                    # Emit the raw reading, then one fact per meaningful state.
                    # Splitting them lets a rule require "live throttling" without
                    # also matching a machine that merely throttled an hour ago.
                    facts.append(
                        _fact(facts, kind, summarize(m), artifact_name, lineno,
                              line, groups)
                    )
                    # `conditions` is a prose string, not the raw list: rule
                    # messages interpolate it with {conditions}, and a Python
                    # repr like "['undervoltage', 'throttled']" in a
                    # user-facing sentence reads like a leaked internal. The
                    # machine-readable list stays available as `condition_keys`.
                    if decoded["live"]:
                        facts.append(_fact(
                            facts, "throttle_active",
                            f"Live: {describe(decoded['live'])}",
                            artifact_name, lineno, line,
                            {"conditions": describe(decoded["live"]),
                             "condition_keys": decoded["live"],
                             "count": len(decoded["live"]), "hex": decoded["hex"]},
                        ))
                    if decoded["sticky"]:
                        facts.append(_fact(
                            facts, "throttle_occurred",
                            f"Previously: {describe(decoded['sticky'])}",
                            artifact_name, lineno, line,
                            {"conditions": describe(decoded["sticky"]),
                             "condition_keys": decoded["sticky"],
                             "count": len(decoded["sticky"]), "hex": decoded["hex"]},
                        ))
                    if decoded["unknown_bits"]:
                        bits = decoded["unknown_bits"]
                        facts.append(_fact(
                            facts, "throttle_unknown_bits",
                            f"Undocumented bit(s) set: "
                            f"{', '.join(str(b) for b in bits)}",
                            artifact_name, lineno, line,
                            {"bits": ", ".join(str(b) for b in bits),
                             "bit_numbers": bits, "hex": decoded["hex"]},
                        ))
                    if decoded["healthy"]:
                        facts.append(_fact(
                            facts, "throttle_clear",
                            "No throttling or undervoltage, now or earlier",
                            artifact_name, lineno, line, {"hex": decoded["hex"]},
                        ))
                    break

                if kind == "soc_temperature":
                    groups["celsius"] = float(groups["celsius"])
                if kind == "arm_clock":
                    groups["hertz"] = int(groups["hertz"])
                    groups["mhz"] = groups["hertz"] // 1_000_000
                if kind == "allocation_failed" and "bytes" in groups:
                    groups["bytes"] = int(groups["bytes"])
                    groups["mib"] = round(groups["bytes"] / 1024 / 1024, 1)

                facts.append(
                    _fact(facts, kind, summarize(m), artifact_name, lineno, line,
                          groups)
                )
                break

        return Facts(backend=self.name, artifact_path=artifact_name, facts=facts)


def _fact(existing: list[Fact], kind: str, summary: str, artifact: str,
          lineno: int, line: str, data: dict[str, Any]) -> Fact:
    return Fact(
        id=f"f{len(existing) + 1}",
        kind=kind,
        summary=summary,
        source=f"{artifact}:{lineno}",
        excerpt=line.strip(),
        data=data,
    )
