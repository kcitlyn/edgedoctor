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
      - Placeholders in `message` (e.g. `{op}`) are filled from the first fact
        of the first matching kind's `data` dict.
      - Evidence is the list of fact ids whose kind matched a `requires` entry.

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

        # Gather evidence: all facts whose kind is in the required set.
        evidence_facts = [f for f in facts.facts if f.kind in required]
        evidence_ids = [f.id for f in evidence_facts]

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


class _SafeDict(dict):
    """A dict that returns the key itself (in braces) for missing keys,
    so .format_map never raises on partial data."""

    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"
