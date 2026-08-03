"""The pluggable Backend / Parser interface.

This is the architectural seam that lets edgedoctor grow from one backend
(TensorRT) to all of them (ONNX Runtime, CoreML, TFLite, ExecuTorch, vendor
NPUs) *without* rewriting the core. We define the contract once here; each
backend fills it in its own module. See docs/DESIGN.md.

The data flow this file encodes:

    raw artifact  ──Backend.parse()──▶  Facts  ──▶  (diagnoser)  ──▶  Diagnosis

The two data contracts (`Facts`, `Diagnosis`) are deliberately separate. The
diagnoser is only ever handed `Facts`; it physically cannot see the raw
artifact. That separation is what *structurally* enforces the grounding
discipline — the LLM can't hallucinate over a blob it never receives.

These are pydantic models (not plain dataclasses) for three reasons:
  1. Validation at the boundary — a parser bug that produces a malformed Fact
     fails loudly at construction, not silently three layers later.
  2. Free JSON: `model_dump_json()` powers the CLI's `--json` output.
  3. Free JSON Schema: `model_json_schema()` is later handed to the LLM as its
     required output shape (structured outputs), so the diagnosis the model
     returns is *the same contract* the rest of the tool speaks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Fact(BaseModel):
    """A single observed, traceable fact extracted from a raw artifact.

    The whole anti-hallucination design rests on this type: a Fact records only
    what was *observed*, plus a pointer back to where it came from. No
    interpretation, no cause, no fix — that's the diagnoser's job, and it may
    only build on these.
    """

    id: str = Field(description="Stable id other records cite, e.g. 'f1'.")
    kind: str = Field(
        description="Machine-readable tag, e.g. 'unsupported_op', "
        "'cpu_fallback', 'layer_sqnr', 'version_mismatch'."
    )
    summary: str = Field(description="Human-readable one-liner of the observation.")
    source: str = Field(
        description="Traceability anchor — where this came from, e.g. "
        "'trtexec.log:412'. Every downstream claim cites it."
    )
    excerpt: str = Field(
        default="",
        description="The verbatim artifact line(s) this fact was extracted "
        "from. Shown to the user unmodified — never paraphrased.",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured payload (op name, measured value, ...).",
    )


class Facts(BaseModel):
    """The structured output of a parser.

    This is everything the diagnoser is allowed to know. If it isn't in here,
    the diagnoser must say "I don't have enough info" rather than guess.
    """

    backend: str
    artifact_path: str
    facts: list[Fact] = Field(default_factory=list)


class Suggestion(BaseModel):
    """One concrete remediation, with a machine-readable safety level.

    `applicability` follows the rustc convention: 'machine-applicable' means an
    agent may run `command` unattended; 'maybe-incorrect' means a human should
    review first.
    """

    summary: str
    command: str = ""
    applicability: str = Field(
        default="maybe-incorrect",
        description="'machine-applicable' | 'maybe-incorrect'",
    )


class Diagnosis(BaseModel):
    """The grounded explanation the diagnoser produces.

    Every field must trace back to one or more `Fact`s (via `evidence` ids).
    When the evidence is insufficient, `insufficient_info` is set and the
    causal fields stay empty — an honest non-answer is a valid, often correct,
    result.
    """

    code: str = Field(default="", description="Stable rule code, e.g. 'ED0042'.")
    severity: str = Field(default="error", description="'error' | 'warning' | 'info'")
    message: str = Field(default="", description="One-line plain-English headline.")
    root_cause: str = Field(default="", description="Why this happened (1-3 lines).")
    suggestions: list[Suggestion] = Field(default_factory=list)
    evidence: list[str] = Field(
        default_factory=list,
        description="Ids of the Facts that support this diagnosis.",
    )
    confidence: str = Field(default="low", description="'high' | 'medium' | 'low'")
    insufficient_info: bool = False
    origin: str = Field(
        default="rules",
        description="'rules' = matched a curated, human-reviewed rule; "
        "'llm' = synthesized from facts no rule covered. Defaults to 'rules' "
        "so the deterministic path and every existing snapshot are unaffected. "
        "A user must always be able to tell a reviewed diagnosis from a "
        "generated one, so this is surfaced in the report.",
    )


class Backend(ABC):
    """Abstract base every backend module implements.

    A backend owns two responsibilities:
      1. `convert()` — drive the vendor toolchain (export → convert) and capture
         the raw artifacts (logs, profiler output, numeric traces).
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
