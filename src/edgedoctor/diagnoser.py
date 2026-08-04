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

import re
from pathlib import Path
from typing import Any

import yaml

from .backends.base import Diagnosis, Facts, Suggestion

# ── Rule loading ──────────────────────────────────────────────────────────

RULES_DIR = Path(__file__).parent / "rules"


def _load_rules(backend: str) -> list[dict[str, Any]]:
    """Load the YAML rule file for a backend. Returns [] if unusable.

    Rule files are hand-edited data, so malformed content is a realistic
    mistake — not a "can't happen". This function is deliberately defensive on
    three axes, because a diagnostic tool that crashes while explaining a crash
    is worse than useless:

      - Unparseable YAML yields [] rather than propagating yaml's ParserError.
      - A document that isn't a list of mappings yields [], and non-mapping
        entries are dropped individually, so one bad rule can't take out the
        file.
      - `backend` is validated as a bare name, so it can't traverse out of
        RULES_DIR (a Facts object could in principle carry `backend`
        "../../etc/passwd"; parsers set it, but nothing structurally stopped a
        crafted Facts JSON from reaching here via a future --load path).

    A malformed rule file therefore degrades to "no rules for this backend",
    which the caller already handles honestly as "no known pattern matched".
    """
    # Reject anything that isn't a simple identifier-ish name before touching
    # the filesystem. os.path.basename would not be enough: an absolute path
    # would still escape.
    if not backend or not re.fullmatch(r"[A-Za-z0-9_.-]+", backend):
        return []
    if backend in (".", ".."):
        return []

    path = RULES_DIR / f"{backend}.yaml"
    try:
        if not path.is_file():
            return []
        with path.open() as f:
            rules = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        # Unreadable or unparseable: no rules, no traceback.
        return []
    if not isinstance(rules, list):
        return []
    # Drop non-mapping entries rather than crashing on rule.get() later.
    return [r for r in rules if isinstance(r, dict)]


def _str_list(value: Any) -> list[str]:
    """Coerce a rule field to a list of strings, tolerating hand-edit slips.

    A bare string is wrapped, NOT iterated. This is the subtle one:
    `requires: k` (missing brackets) previously became `set("k") == {"k"}`,
    which happened to work for a one-character kind and silently matched the
    WRONG thing for any real multi-character kind — `requires: cpu_fallback`
    would become {'c','p','u','_',...} and match a fact of kind "c". A silent
    mismatch in rule matching is exactly the class of bug this tool exists to
    avoid, so it's normalized here instead.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | set):
        return [str(v) for v in value]
    return []


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
        # _str_list, not set(...): a hand-edited `requires: cpu_fallback`
        # (missing brackets) would otherwise become a set of CHARACTERS and
        # silently match the wrong fact kind. See _str_list.
        required = set(_str_list(rule.get("requires")))
        if not required:
            continue
        if not required.issubset(fact_kinds_present):
            continue

        # Disqualifying context: any of these present means this rule is the
        # wrong explanation, regardless of what else matched.
        forbidden = set(_str_list(rule.get("absent")))
        if forbidden & fact_kinds_present:
            continue

        # Numeric thresholds on a fact's data field. Needed because presence
        # alone can't distinguish "one output diverged" from "forty-nine did",
        # and those warrant different explanations.
        conditions = rule.get("conditions")
        if not isinstance(conditions, list):
            # A malformed `conditions:` must not silently disable the gate it
            # was written to enforce, so treat it as unsatisfiable.
            conditions = [] if conditions is None else [conditions]
        if not _conditions_met(conditions, facts):
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
        # Deduped here AND at evidence_ids below. The redundancy is deliberate:
        # this one keeps the KIND list clean (so `optional: [b, b]` doesn't scan
        # facts twice), the other guarantees no id is cited twice whatever the
        # rule says. Either alone suffices for current rules — mutation testing
        # confirms removing both is caught, removing one is not — but they guard
        # different mistakes and both are one cheap call.
        optional = list(dict.fromkeys(_str_list(rule.get("optional"))))
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
        message = rule.get("message") or ""
        if not isinstance(message, str):
            message = str(message)
        try:
            message = message.format_map(
                _SafeDict(placeholders)
            )
        except (KeyError, ValueError):
            pass  # template had a placeholder we can't fill — leave it

        # Only mapping-shaped suggestions are usable; a bare string (a common
        # hand-edit slip) is skipped rather than crashing the whole diagnosis.
        raw_suggestions = rule.get("suggestions")
        if not isinstance(raw_suggestions, list):
            raw_suggestions = []
        suggestions = [
            Suggestion(
                summary=str(s.get("summary", "")),
                command=str(s.get("command", "")),
                applicability=str(s.get("applicability", "maybe-incorrect")),
            )
            for s in raw_suggestions
            if isinstance(s, dict)
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

    A MALFORMED CONDITION ALSO FAILS, rather than raising. A rule file is
    hand-edited data, so a typo (`min: "two"`) is a realistic mistake; letting
    it crash the whole diagnosis would take down every OTHER rule's correct
    output alongside it. Failing closed means the buggy rule stays silent and
    the rest of the report still reaches the user.
    """
    for cond in conditions:
        if not isinstance(cond, dict):
            return False
        kind = cond.get("kind")
        field = cond.get("field")
        if kind is None or field is None:
            # An underspecified condition can't be checked, so it isn't met.
            return False
        lo, hi = cond.get("min"), cond.get("max")
        if not any(
            _satisfies(f.data.get(field), lo, hi)
            for f in facts.facts
            if f.kind == kind
        ):
            return False
    return True


def _satisfies(value: Any, lo: Any, hi: Any) -> bool:
    """True if `value` is a finite number within the [lo, hi] bounds given.

    Two non-obvious guards:

    NaN is rejected. Comparisons against NaN are all False, so a naive
    `if num < lo: return False` lets NaN through EVERY threshold — a NaN
    measurement would satisfy "at least 50%" and "at most 10%" simultaneously.
    Infinity is rejected for the same reason in reverse: it satisfies any `min`,
    so an unparsed sentinel could trip a threshold it has no business tripping.

    A non-numeric BOUND (from a typo in a rule file) makes the condition fail
    rather than raise — see _conditions_met.
    """
    try:
        num = float(value)
    except (TypeError, ValueError):
        return False
    # NaN and +/-inf are not measurements a threshold can meaningfully bound.
    if num != num or num in (float("inf"), float("-inf")):
        return False
    try:
        if lo is not None and num < float(lo):
            return False
        if hi is not None and num > float(hi):
            return False
    except (TypeError, ValueError):
        # Malformed bound in the rule file: fail closed, don't crash the run.
        return False
    return True


class _SafeDict(dict):
    """A dict that returns the key itself (in braces) for missing keys,
    so .format_map never raises on partial data."""

    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"
