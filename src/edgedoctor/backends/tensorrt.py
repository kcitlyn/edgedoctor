"""TensorRT backend — the beachhead.

`parse()` is real: it deterministically extracts `Facts` from trtexec /
TensorRT build logs. `convert()` (driving trtexec) is still a stub — it needs
the NVIDIA machine.

Every signature below matches a REAL error string, verified against the
TensorRT/onnx-tensorrt source code (ModelImporter.cpp, errorHelpers.hpp) and
real logs pasted in NVIDIA GitHub issues. Two generations of parser-error
formats exist (the wording was rewritten in TensorRT 10.x), so several
failure modes need two patterns. Sources are cited on each signature.

Grounding note: the parser records only what is IN the log — op names, node
names, line numbers, verbatim excerpts. Interpretation (cause, fix) happens
later, in the diagnoser, and may only build on these Facts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from .base import Backend, Fact, Facts

# ── Signature table ────────────────────────────────────────────────────────
# Each entry: (kind, compiled_pattern, summary_builder).
# The summary_builder receives the regex match and returns a one-line,
# human-readable statement of the OBSERVATION (not the cause).

_Sig = tuple[str, re.Pattern[str], Callable[[re.Match[str]], str]]

_SIGNATURES: list[_Sig] = [
    # Unsupported op → plugin-fallback attempt. Op name appears bracketed
    # ([LayerNorm]) or bare (grid_sampler) depending on TRT version.
    # Source: onnx-tensorrt ModelImporter.cpp; NVIDIA/TensorRT issues #3346, #2625.
    (
        "unsupported_op",
        re.compile(
            r"No importer registered for op:\s*\[?(?P<op>[^\].\s]+)\]?\.?"
            r"\s*Attempting to import as plugin"
        ),
        lambda m: f"ONNX parser has no importer for op '{m['op']}'; trying plugin fallback",
    ),
    # Plugin lookup failed (both punctuation variants: with and without colons).
    # Source: NVIDIA forums #187181, #84205; TRT 8.6 issue #3346.
    (
        "plugin_not_found",
        re.compile(
            r"getPluginCreator could not find plugin:?\s+(?P<plugin>\S+)\s+version:?\s+(?P<ver>\d+)"
        ),
        lambda m: f"No plugin named '{m['plugin']}' (version {m['ver']}) in the plugin registry",
    ),
    # Parser error, TensorRT <=8.x format.
    # Source: real log in SegmentAnything-TensorRT issue #2.
    (
        "parse_error_node",
        re.compile(
            r"While parsing node number (?P<idx>\d+) \[(?P<op>\w+) -> \"(?P<output>[^\"]+)\"\]"
        ),
        lambda m: f"Parse failed at node {m['idx']} (op {m['op']}, output '{m['output']}')",
    ),
    # Parser error, TensorRT 10.x format. Error code is delimited by ': ',
    # not bracketed. Name/operator segment is optional in some emissions.
    # Source: onnx-tensorrt parserErrorStr(); errorHelpers.hpp code list.
    (
        "parse_error_node",
        re.compile(
            r"In node (?P<idx>-?\d+)"
            r"(?: with name: (?P<name>\S+) and operator: (?P<op>\S+))?"
            r" \((?P<func>\w+)\): (?P<code>[A-Z_]+):"
        ),
        lambda m: (
            f"Parse failed at node {m['idx']}"
            + (f" ('{m['name']}', op {m['op']})" if m["name"] else "")
            + f" with error {m['code']}"
        ),
    ),
    # Builder found no kernel implementation for a layer — the single most
    # reliable "this layer broke the build" line.
    # Source: NVIDIA/TensorRT issue #4736.
    (
        "no_implementation",
        re.compile(r"Could not find any implementation for node (?P<node>.+?)[.)]*$"),
        lambda m: f"Builder found no kernel implementation for node '{m['node']}'",
    ),
    # Numbered TensorRT error codes (10 = computeCosts/no-implementation,
    # 4 = workspace/shape, 2 = build-serialize wrapper, 9 = tactic skip).
    # Source: real clusters in issues #3610, #4736.
    (
        "trt_error_code",
        re.compile(r"Error Code (?P<code>\d+): (?P<detail>.+)$"),
        lambda m: f"TensorRT reported error code {m['code']}: {m['detail'][:120]}",
    ),
    # Tactic skipped during autotuning (OOM or internal assertion).
    # Source: issues #4658, #4736.
    (
        "tactic_skipped",
        re.compile(
            r"Skipping tactic (?P<tactic>0x[0-9a-fA-F]+|\d+) due to (?P<reason>.+)$"
        ),
        lambda m: f"Autotuner skipped tactic {m['tactic']}: {m['reason'][:120]}",
    ),
    # trtexec final verdict banner (standard trtexec output).
    (
        "run_verdict",
        re.compile(r"&&&& (?P<verdict>PASSED|FAILED) TensorRT\.trtexec"),
        lambda m: f"trtexec run verdict: {m['verdict']}",
    ),
    # Version banner: "TensorRT version: 10.7.0" — anchors every other fact
    # to a toolchain version (essential for version-mismatch diagnosis).
    (
        "trt_version",
        re.compile(r"TensorRT version:\s*(?P<version>[\d.]+)"),
        lambda m: f"TensorRT version {m['version']}",
    ),
]


class TensorRTBackend(Backend):
    name = "tensorrt"

    def convert(self, model_path: Path, **options: Any) -> list[Path]:
        raise NotImplementedError(
            "TensorRT conversion is not implemented yet (needs the NVIDIA "
            "machine — see ROADMAP.md)."
        )

    def parse(self, artifact_path: Path) -> Facts:
        """Extract Facts from a trtexec / TensorRT build log.

        Pure and deterministic: same log in → same Facts out. Every Fact
        carries `source` = "<filename>:<line>" and the verbatim line as
        `excerpt`, so downstream claims stay traceable.
        """
        text = artifact_path.read_text(errors="replace")
        return self.parse_text(text, artifact_name=artifact_path.name)

    def parse_text(self, text: str, artifact_name: str = "<string>") -> Facts:
        """Parse log text directly (separated from file I/O for testability)."""
        facts: list[Fact] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            for kind, pattern, summarize in _SIGNATURES:
                m = pattern.search(line)
                if m is None:
                    continue
                facts.append(
                    Fact(
                        id=f"f{len(facts) + 1}",
                        kind=kind,
                        summary=summarize(m),
                        source=f"{artifact_name}:{lineno}",
                        excerpt=line.strip(),
                        # Named groups become the structured payload, so the
                        # diagnoser can reason over op/plugin/code fields
                        # without re-parsing strings.
                        data={k: v for k, v in m.groupdict().items() if v is not None},
                    )
                )
                # One fact per line: signatures are ordered most-specific
                # first, so the first match wins and we avoid double-counting
                # (e.g. an "Error Code" line that also mentions a node).
                break
        return Facts(
            backend=self.name,
            artifact_path=artifact_name,
            facts=facts,
        )
