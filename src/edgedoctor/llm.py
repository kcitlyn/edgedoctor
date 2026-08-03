"""The optional, grounded LLM synthesis layer.

This is the *second* diagnoser, and it is deliberately the weaker one. The rules
engine in `diagnoser.py` handles everything it knows; this module only looks at
the facts no rule matched, and tries to explain them. See
docs/adr/0001-llm-synthesis-layer.md for the trade-offs behind each choice here.

Three properties this module guarantees, in priority order:

1. IT CANNOT BREAK THE TOOL. Every failure path — anthropic not installed, no
   API key, network timeout, malformed response, ungrounded output — returns an
   empty list. The caller's rules-based diagnoses are never touched. The LLM can
   only ever ADD to a diagnosis set.

2. IT CANNOT INVENT EVIDENCE. The model receives Facts only — never the raw
   artifact (`Facts` is the firewall; see backends/base.py). Any diagnosis it
   returns citing a fact id that wasn't in its input is DROPPED, not shown. A
   prompt instruction is a request; the validation in `_ground()` is the
   guarantee.

3. IT CANNOT IMPERSONATE A CURATED RULE. Synthesized diagnoses are marked
   `origin="llm"` and clamped to at most `medium` confidence. The model is
   handed a narrow schema (`SynthesizedDiagnosis`) that has no `origin` field at
   all, so it is structurally unable to label its own output as trusted.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from .backends.base import Diagnosis, Facts, Suggestion

# Haiku: the plan's default. Roughly half a cent per diagnosis, and this is a
# constrained extraction task over a handful of pre-parsed facts, not open-ended
# reasoning — a bigger model would cost more for little gain.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# A diagnosis is a few hundred tokens. This is a runaway guard, not a target.
MAX_TOKENS = 2048

# Synthesized output may never claim the confidence a reviewed rule earns.
MAX_LLM_CONFIDENCE = "medium"
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


class SynthesizedSuggestion(BaseModel):
    """A remediation proposed by the model."""

    summary: str = Field(description="One concrete, actionable fix.")
    command: str = Field(
        default="",
        description="An exact shell command if one applies, else empty. "
        "Never invent flags you are unsure of.",
    )


class SynthesizedDiagnosis(BaseModel):
    """The wire schema handed to the model — deliberately narrower than `Diagnosis`.

    Note what is ABSENT: `origin`, `code`, and `confidence` are set by this
    module, not by the model. Excluding them from the schema means the model
    cannot mislabel its own output as a curated rule, or award itself high
    confidence, no matter what it generates. Structural beats instructional.
    """

    message: str = Field(
        description="One-line plain-English headline of what went wrong."
    )
    root_cause: str = Field(
        description="Why this happened, 1-3 sentences, grounded strictly in the "
        "supplied facts. If the facts don't support a cause, say so."
    )
    severity: str = Field(
        default="warning",
        description="'error' if it certainly breaks the deployment, "
        "'warning' if it may, 'info' if it is merely notable.",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="The exact fact ids (e.g. 'f3') this diagnosis rests on. "
        "Every id MUST come from the supplied facts. Never invent one.",
    )
    suggestions: list[SynthesizedSuggestion] = Field(default_factory=list)
    insufficient_info: bool = Field(
        default=False,
        description="True if the facts are too thin to support any conclusion. "
        "This is a CORRECT and valued answer — prefer it over speculation.",
    )


class SynthesisResult(BaseModel):
    """What the model returns: zero or more diagnoses over the unmatched facts."""

    diagnoses: list[SynthesizedDiagnosis] = Field(
        default_factory=list,
        description="Leave empty if the facts warrant no diagnosis at all.",
    )


SYSTEM_PROMPT = """\
You are the synthesis layer of edgedoctor, an edge-AI deployment diagnostician.

You are given FACTS extracted from a vendor tool's log by a deterministic parser.
You never see the raw log. Facts already matched by edgedoctor's curated rule
knowledge base have been removed — you are looking only at what the rules could
not explain.

YOUR ABSOLUTE CONSTRAINTS:

1. Ground every claim in the supplied facts. Cite the fact ids you used in
   `evidence`. Never cite an id that is not in the input.
2. Never invent log lines, error messages, version numbers, op names, or
   measurements. If a detail is not in the facts, you do not know it.
3. "I don't have enough information" is a CORRECT answer. Set
   `insufficient_info: true` and return no cause rather than speculating. A
   confident wrong diagnosis is far more damaging to a user than an honest
   non-answer — they will act on it and lose hours.
4. Do not restate a fact as if it were a diagnosis. A fact says WHAT was
   observed; a diagnosis explains WHY it happened and what to do. If you can
   only restate, return no diagnoses.
5. Prefer returning nothing over returning something weak. An empty
   `diagnoses` list is a perfectly good response.

On distinguishing measurements from verdicts: a large numeric difference is not
by itself a failure. Tolerance is a policy set by whoever ran the tool, not a
property of the model. Only treat something as failing if a fact says it failed.

Be specific and technical. The reader is an ML engineer debugging a deployment.\
"""


def _facts_payload(facts: list[Any]) -> str:
    """Render facts for the prompt.

    Deliberately includes each fact's verbatim `excerpt`: the model needs the
    real text to reason about, and this is still grounded — the excerpt came
    from the parser, not from the model.
    """
    lines = []
    for f in facts:
        lines.append(
            f"- id: {f.id}\n"
            f"  kind: {f.kind}\n"
            f"  observed: {f.summary}\n"
            f"  source: {f.source}\n"
            f"  log line: {f.excerpt}"
        )
        if f.data:
            lines.append(f"  data: {f.data}")
    return "\n".join(lines)


def unmatched_facts(facts: Facts, diagnoses: list[Diagnosis]) -> list[Any]:
    """The facts no rule-based diagnosis cited.

    This is the LLM's entire input surface, and the reason the layer is additive
    rather than competitive: it physically cannot revisit ground a curated rule
    already covered, so it can never contradict one.
    """
    cited = {eid for d in diagnoses for eid in d.evidence}
    return [f for f in facts.facts if f.id not in cited]


def _clamp_confidence(value: str) -> str:
    """Cap synthesized confidence at MAX_LLM_CONFIDENCE."""
    rank = _CONFIDENCE_RANK.get(value, 0)
    if rank > _CONFIDENCE_RANK[MAX_LLM_CONFIDENCE]:
        return MAX_LLM_CONFIDENCE
    return value


def _ground(
    result: SynthesisResult, valid_ids: set[str]
) -> list[Diagnosis]:
    """Convert model output into Diagnoses, dropping anything ungrounded.

    THIS FUNCTION IS THE GUARANTEE. The system prompt asks the model to cite
    real fact ids; this is what happens when it doesn't. A diagnosis is dropped
    outright if it:
      - cites any id that wasn't in the input (a hallucinated citation), or
      - cites nothing at all (an unfalsifiable claim), or
      - declares insufficient_info (an honest non-answer needs no report entry).

    Dropping rather than repairing is intentional: a diagnosis whose evidence is
    partly fabricated has demonstrated it isn't grounded, and salvaging its
    remaining citations would launder that.
    """
    grounded: list[Diagnosis] = []
    for d in result.diagnoses:
        if d.insufficient_info:
            continue
        cited = set(d.evidence)
        if not cited or not cited.issubset(valid_ids):
            continue
        grounded.append(
            Diagnosis(
                # No ED code: those are reserved for curated rules. An LLM
                # diagnosis is not a stable, documented failure class.
                code="ED9001",
                severity=d.severity if d.severity in ("error", "warning", "info")
                else "warning",
                message=d.message,
                root_cause=d.root_cause,
                suggestions=[
                    Suggestion(
                        summary=s.summary,
                        command=s.command,
                        # Never machine-applicable: an unreviewed generated
                        # command must not be run unattended by an agent.
                        applicability="maybe-incorrect",
                    )
                    for s in d.suggestions
                ],
                evidence=sorted(cited),
                confidence=_clamp_confidence("medium"),
                insufficient_info=False,
                origin="llm",
            )
        )
    return grounded


def synthesize(
    facts: Facts,
    rule_diagnoses: list[Diagnosis],
    *,
    client: Any | None = None,
    model: str = DEFAULT_MODEL,
) -> list[Diagnosis]:
    """Try to explain the facts no rule matched. Returns [] on ANY failure.

    `client` is injectable so tests run hermetically with no API key and no
    network (see tests/test_llm.py). In production it is constructed here.

    This function never raises. A diagnostic tool that crashes because an
    optional enhancement failed is worse than one that simply doesn't enhance.
    """
    leftover = unmatched_facts(facts, rule_diagnoses)
    if not leftover:
        # Everything was already explained by curated rules — the best possible
        # outcome, and it costs nothing.
        return []

    if client is None:
        client = _build_client()
        if client is None:
            return []

    valid_ids = {f.id for f in leftover}
    user_prompt = (
        f"Backend: {facts.backend}\n"
        f"Artifact: {facts.artifact_path}\n\n"
        f"{len(leftover)} fact(s) that edgedoctor's rules could not explain:\n\n"
        f"{_facts_payload(leftover)}\n\n"
        "Diagnose these if — and only if — the facts support a conclusion."
    )

    try:
        response = client.messages.parse(
            model=model,
            max_tokens=MAX_TOKENS,
            # Deterministic as the API allows: this is extraction, not writing.
            temperature=0,
            system=SYSTEM_PROMPT,
            output_format=SynthesisResult,
            messages=[{"role": "user", "content": user_prompt}],
        )
        result = _extract(response)
    except Exception:
        # Intentionally broad: timeouts, auth errors, rate limits, schema
        # refusals, SDK changes. Every one of them means "no synthesis", and
        # none of them may take the rules-based report down with it.
        return []

    if result is None:
        return []
    return _ground(result, valid_ids)


def _extract(response: Any) -> SynthesisResult | None:
    """Pull the parsed object out of a ParsedMessage.

    The SDK puts it on the content block as `parsed_output`. Read defensively —
    an SDK shape change must degrade to "no synthesis", not raise.
    """
    for block in getattr(response, "content", []) or []:
        parsed = getattr(block, "parsed_output", None)
        if isinstance(parsed, SynthesisResult):
            return parsed
    return None


def _build_client() -> Any | None:
    """Construct an Anthropic client, or None if that isn't possible.

    Both failure modes here are ordinary, expected states — the SDK is an
    optional extra and the key is user-supplied — so neither raises.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from anthropic import Anthropic
    except ImportError:
        return None
    try:
        # A short timeout: this is an optional enhancement to a CLI that is
        # otherwise instant. Waiting a minute for it would be a regression.
        return Anthropic(timeout=30.0, max_retries=1)
    except Exception:
        return None


def availability() -> tuple[bool, str]:
    """Whether synthesis can run, plus a human-readable reason if not.

    Used by the CLI to explain --llm doing nothing, instead of failing silently.
    """
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False, (
            "the anthropic SDK isn't installed — "
            'install it with: pip install "edgedoctor[llm]"'
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False, "ANTHROPIC_API_KEY isn't set in the environment"
    return True, ""
