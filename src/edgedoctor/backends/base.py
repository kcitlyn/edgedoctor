"""The pluggable Backend / Parser interface.

This is the architectural seam that lets edgedoctor grow from one backend
(TensorRT) to all of them (CoreML, ONNX-RT, TFLite, ExecuTorch, vendor NPUs)
*without* rewriting the core. We define the contract once here; each backend
fills it in its own module. Only one backend is fully implemented at a time —
but the seam exists from day one. See docs/DESIGN.md.

The data flow this file encodes:

    raw artifact  ──Backend.parse()──▶  Facts  ──▶  (diagnoser)  ──▶  Diagnosis

The two data contracts (`Facts`, `Diagnosis`) are deliberately separate. The
diagnoser is only ever handed `Facts`; it physically cannot see the raw artifact.
That separation is what *structurally* enforces the grounding discipline — the
LLM can't hallucinate over a blob it never receives.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Fact:
    """A single observed, traceable fact extracted from a raw artifact.

    The whole anti-hallucination design rests on this type: a Fact records only
    what was *observed*, plus a pointer back to where it came from. No
    interpretation, no cause, no fix — that's the diagnoser's job, and it may
    only build on these.

    Attributes:
        kind:    A short machine-readable tag, e.g. "unsupported_op",
                 "cpu_fallback", "layer_sqnr", "version_mismatch".
        summary: A human-readable one-liner describing the observation.
        source:  Where this came from — e.g. "trtexec.log:412". This is the
                 traceability anchor; every downstream claim cites it.
        data:    Structured payload for this fact (op name, measured value, …).
    """

    kind: str
    summary: str
    source: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Facts:
    """The structured, JSON-serializable output of a parser.

    This is everything the diagnoser is allowed to know. If it isn't in here,
    the diagnoser must say "I don't have enough info" rather than guess.
    """

    backend: str
    artifact_path: str
    facts: list[Fact] = field(default_factory=list)


@dataclass
class Diagnosis:
    """The grounded explanation the diagnoser produces.

    Every field must trace back to one or more `Fact`s. When the evidence is
    insufficient, `insufficient_info` is set and `root_cause`/`fix` stay empty —
    an honest non-answer is a valid, often correct, result.

    Attributes:
        root_cause:        Plain-English cause, grounded in the cited evidence.
        fix:               Concrete, actionable remediation.
        evidence:          The Facts that support the above (the traceability).
        confidence:        0.0–1.0, how well the evidence supports the claim.
        insufficient_info: True when the Facts don't support any confident claim.
    """

    root_cause: str = ""
    fix: str = ""
    evidence: list[Fact] = field(default_factory=list)
    confidence: float = 0.0
    insufficient_info: bool = False


class Backend(ABC):
    """Abstract base every backend module implements.

    A backend owns two responsibilities:
      1. `convert()` — drive the vendor toolchain (export → convert) and capture
         the raw artifacts (logs, profiler dumps, numeric traces).
      2. `parse()`   — deterministically turn one raw artifact into `Facts`.

    Note what is *not* here: explaining the facts. Explanation is backend-
    agnostic and lives in the diagnoser, so adding a new backend never means
    re-writing the LLM layer — you only teach edgedoctor how to *parse* a new
    vendor's output. That's the whole point of the seam.
    """

    #: Short identifier, e.g. "tensorrt". Used by the CLI's --backend flag.
    name: str = "base"

    @abstractmethod
    def convert(self, model_path: Path, **options: Any) -> list[Path]:
        """Run the vendor toolchain; return paths to the raw artifacts produced.

        Implementations should capture *everything* (stdout/stderr logs,
        profiler output, layer dumps) — those artifacts are both the diagnosis
        input and the project's growing test corpus.
        """

    @abstractmethod
    def parse(self, artifact_path: Path) -> Facts:
        """Deterministically extract structured `Facts` from one raw artifact.

        Must be pure and deterministic (no LLM, no network): same artifact in →
        same Facts out. This is the firewall that keeps the LLM grounded.
        """
