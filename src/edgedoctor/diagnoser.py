"""The grounded diagnoser — matches Facts against the rule KB.

This is the rules-only path. It takes a `Facts` object from a parser and
produces a list of `Diagnosis` objects by checking which rule(s) match the
fact kinds present. Every Diagnosis produced here has zero LLM involvement —
it's deterministic and fast, and it works with no API key.

The LLM layer (not yet built) will be a separate, optional step that:
  1. Takes unmatched facts and tries to synthesize a diagnosis.
  2. Takes matched diagnoses and optionally renders them into richer prose.
It will import this module's results, not replace them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .backends.base import Diagnosis, Facts, Suggestion

# ── Rule loading ──────────────────────────────────────────────────────────

RULES_DIR = Path(__file__).parent / "rules"


def _load_rules(backend: str) -> list[dict[str, Any]]:
    """Load the YAML rule file for a backend. Returns [] if none exists."""
    path = RULES_DIR / f"{backend}.yaml"
    if not path.exists():
        return []
    with path.open() as f:
        rules = yaml.safe_load(f)
    return rules if isinstance(rules, list) else []


# ── Matching engine ───────────────────────────────────────────────────────

def diagnose(facts: Facts) -> list[Diagnosis]:
    """Match parsed Facts against the rule KB. Returns diagnoses, ranked by
    severity (errors first, then warnings, then info).

    Matching logic (deliberately simple — complexity goes in the rules, not here):
      - A rule fires if ALL of its `requires` fact kinds are present in `facts`.
      - A rule does NOT fire if any of its `absent` fact kinds are present.
        This is how a rule stays honest about context it must not ignore (e.g.
        "diverged" must not fire when the log also says "all outputs matched").
      - `optional` kinds never affect whether a rule fires; they are pulled in
        as extra evidence and placeholder values when present. This is what
        lets a rule cite a measured number it doesn't depend on.
      - Placeholders in `message` (e.g. `{op}`) are filled from the matched
        facts' `data` dicts (first fact wins per key).
      - Evidence is the list of ids of the facts that matched.

    Returns an empty list if no rules match — the tool then honestly reports
    "no known pattern matched" rather than guessing.
    """
    rules = _load_rules(facts.backend)
    if not rules:
        return []

    fact_kinds_present = {f.kind for f in facts.facts}
    results: list[Diagnosis] = []

    for rule in rules:
        required = set(rule.get("requires", []))
        if not required:
            continue
        if not required.issubset(fact_kinds_present):
            continue

        # Disqualifying context: any of these present means this rule is the
        # wrong explanation, regardless of what else matched.
        forbidden = set(rule.get("absent", []))
        if forbidden & fact_kinds_present:
            continue

        # Numeric thresholds on a fact's data field. Needed because presence
        # alone can't distinguish "one output diverged" from "forty-nine did",
        # and those warrant different explanations.
        if not _conditions_met(rule.get("conditions", []), facts):
            continue

        # Gather evidence: required kinds, plus any optional kinds that happen
        # to be present. Required facts come first so the report shows the
        # proof before the supporting measurements.
        #
        # `optional` is ordered by the rule author, and that order is preserved:
        # the report caps how many evidence blocks it prints, so the most
        # actionable fact must be citable first. (A rule listing the ops that
        # fell back before the raw placement counts gets the ops shown.)
        #
        # Deduplicated because a kind may legitimately appear in both lists, and
        # showing the user the same log line twice looks like a bug in the tool.
        optional = list(dict.fromkeys(rule.get("optional", [])))
        evidence_facts = [f for f in facts.facts if f.kind in required]
        for kind in optional:
            if kind in required:
                continue
            evidence_facts += [f for f in facts.facts if f.kind == kind]
        evidence_ids = list(dict.fromkeys(f.id for f in evidence_facts))

        # Build a placeholder dict from the matched facts' data fields.
        # First fact of each kind wins for placeholder resolution.
        placeholders: dict[str, str] = {}
        for f in evidence_facts:
            for k, v in f.data.items():
                placeholders.setdefault(k, str(v))

        # Render the message template with available placeholders.
        message = rule.get("message", "")
        try:
            message = message.format_map(
                _SafeDict(placeholders)
            )
        except (KeyError, ValueError):
            pass  # template had a placeholder we can't fill — leave it

        suggestions = [
            Suggestion(
                summary=s.get("summary", ""),
                command=s.get("command", ""),
                applicability=s.get("applicability", "maybe-incorrect"),
            )
            for s in rule.get("suggestions", [])
        ]

        results.append(
            Diagnosis(
                code=rule.get("id", ""),
                severity=rule.get("severity", "error"),
                message=message,
                root_cause=rule.get("cause", "").strip(),
                suggestions=suggestions,
                evidence=evidence_ids,
                confidence="high",
                insufficient_info=False,
            )
        )

    # Sort: errors > warnings > info.
    severity_rank = {"error": 0, "warning": 1, "info": 2}
    results.sort(key=lambda d: severity_rank.get(d.severity, 9))
    return results


def _conditions_met(conditions: list[dict[str, Any]], facts: Facts) -> bool:
    """Check numeric threshold conditions declared by a rule.

    Each condition looks like:
        {kind: mismatched_outputs, field: count, min: 2}

    Semantics: at least one fact of `kind` must have `field` satisfying the
    bound(s). A condition whose fact/field is missing or non-numeric FAILS —
    a rule may never fire on evidence that isn't actually there. Supported
    bounds: `min` (>=) and `max` (<=).
    """
    for cond in conditions:
        kind = cond.get("kind")
        field = cond.get("field")
        lo, hi = cond.get("min"), cond.get("max")
        if not any(
            _satisfies(f.data.get(field), lo, hi)
            for f in facts.facts
            if f.kind == kind
        ):
            return False
    return True


def _satisfies(value: Any, lo: Any, hi: Any) -> bool:
    """True if `value` is numeric and within the [lo, hi] bounds given."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return False
    if lo is not None and num < float(lo):
        return False
    if hi is not None and num > float(hi):
        return False
    return True


class _SafeDict(dict):
    """A dict that returns the key itself (in braces) for missing keys,
    so .format_map never raises on partial data."""

    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"
