"""Static integrity checks on the rule knowledge base itself.

The rules ARE the product — the parsers only supply them evidence. They're
hand-written YAML, so they can be wrong in ways no parser test would notice: a
duplicate code, a placeholder for a field that doesn't exist, a condition on a
kind the rule doesn't require, or a `requires`/`absent` pair that makes the rule
dead code that can never fire.

These checks are driven by a sweep of the rules directory and of every artifact
in the repo, so a NEW RULE IS CHECKED THE MOMENT IT IS WRITTEN.

Two of these caught real gaps: `session_failed` and `few_iterations` were
referenced by four rules but produced by no real artifact, meaning those rules
were only ever exercised against hand-built Facts. Real artifacts were added.
"""

import re
from pathlib import Path

import pytest
import yaml

from edgedoctor.backends import PARSER_REGISTRY, get_parser
from edgedoctor.diagnoser import RULES_DIR

REPO = Path(__file__).parent.parent
CORPUS = REPO / "corpus" / "onnxruntime"
FIXTURES = REPO / "tests" / "fixtures"

VALID_SEVERITIES = {"error", "warning", "info"}
VALID_APPLICABILITY = {"machine-applicable", "maybe-incorrect"}


def load(backend: str) -> list[dict]:
    path = RULES_DIR / f"{backend}.yaml"
    return yaml.safe_load(path.read_text()) or []


ALL_RULES = [(b, r) for b in sorted(PARSER_REGISTRY) for r in load(b)]
RULE_IDS = [f"{b}:{r.get('id')}" for b, r in ALL_RULES]


def _artifacts_for(backend: str) -> list[Path]:
    if backend == "tensorrt":
        return sorted((FIXTURES / "tensorrt").glob("*.log"))
    if backend == "raspberrypi":
        return sorted((FIXTURES / "raspberrypi").glob("*.log"))
    if backend == "onnxruntime":
        return sorted(CORPUS.glob("ort_*.log"))
    if backend == "polygraphy":
        return [p for p in sorted(CORPUS.glob("*.log"))
                if not p.name.startswith("ort_")]
    if backend == "ort_profile":
        return sorted(CORPUS.glob("*.json"))
    return []


def _kind_keys(backend: str) -> dict[str, set[str]]:
    """kind -> the data keys that kind actually carries, from real artifacts."""
    out: dict[str, set[str]] = {}
    parser = get_parser(backend)
    for path in _artifacts_for(backend):
        for fact in parser.parse(path).facts:
            out.setdefault(fact.kind, set()).update(fact.data.keys())
    return out


KIND_KEYS = {b: _kind_keys(b) for b in sorted(PARSER_REGISTRY)}


def test_rules_were_discovered():
    assert len(ALL_RULES) >= 20, f"only found {len(ALL_RULES)} rules"


class TestCodesAreUnique:
    def test_no_duplicate_codes_anywhere(self):
        # A duplicate code makes a diagnosis unattributable to a documented
        # cause, and ED codes are the tool's stable public identifiers.
        codes = [r.get("id") for _, r in ALL_RULES]
        dupes = {c for c in codes if codes.count(c) > 1}
        assert not dupes, f"duplicate rule codes: {dupes}"

    def test_codes_follow_the_documented_scheme(self):
        for backend, rule in ALL_RULES:
            assert re.fullmatch(r"ED\d{4}", rule["id"]), (
                f"{backend}/{rule['id']} does not match EDnnnn"
            )

    def test_each_backend_uses_its_own_code_range(self):
        # ED01xx tensorrt, ED02xx polygraphy, ED03xx onnxruntime,
        # ED04xx raspberrypi, ED05xx ort_profile. Overlapping ranges would make
        # a code ambiguous about which family it belongs to.
        prefixes: dict[str, set[str]] = {}
        for backend, rule in ALL_RULES:
            prefixes.setdefault(rule["id"][:4], set()).add(backend)
        for prefix, backends in prefixes.items():
            assert len(backends) == 1, f"{prefix}xx shared by {backends}"


@pytest.mark.parametrize("backend,rule", ALL_RULES, ids=RULE_IDS)
class TestEveryRuleIsWellFormed:
    def test_has_all_required_fields(self, backend, rule):
        for field in ("id", "requires", "message", "cause"):
            assert rule.get(field), f"{rule.get('id')} missing '{field}'"

    def test_severity_is_valid(self, backend, rule):
        assert rule.get("severity", "error") in VALID_SEVERITIES

    def test_requires_is_a_list(self, backend, rule):
        # A bare string would become a set of characters and match wrongly.
        assert isinstance(rule["requires"], list), (
            f"{rule['id']}: requires must be a list, got {type(rule['requires'])}"
        )

    def test_absent_and_optional_are_lists(self, backend, rule):
        for field in ("absent", "optional"):
            if field in rule:
                assert isinstance(rule[field], list), f"{rule['id']}: {field}"

    def test_is_not_dead_code(self, backend, rule):
        # requires and absent overlapping means the rule can never fire.
        overlap = set(rule["requires"]) & set(rule.get("absent") or [])
        assert not overlap, f"{rule['id']} requires AND forbids {overlap}"

    def test_suggestions_are_well_formed(self, backend, rule):
        for s in rule.get("suggestions") or []:
            assert isinstance(s, dict), f"{rule['id']}: suggestion is not a mapping"
            assert s.get("summary"), f"{rule['id']}: suggestion without a summary"
            assert s.get("applicability", "maybe-incorrect") in VALID_APPLICABILITY

    def test_has_at_least_one_suggestion(self, backend, rule):
        # A diagnosis with no suggested action leaves the user stuck knowing
        # what broke but not what to do.
        assert rule.get("suggestions"), f"{rule['id']} offers no suggestions"

    def test_has_reference_links(self, backend, rule):
        # Every claim should be checkable against vendor documentation.
        refs = rule.get("refs") or []
        assert refs, f"{rule['id']} cites no references"
        for ref in refs:
            assert ref.startswith("http"), f"{rule['id']}: bad ref {ref!r}"

    def test_machine_applicable_suggestions_have_a_command(self, backend, rule):
        # "machine-applicable" means an agent may run it unattended; with no
        # command there is nothing to run, so the label is meaningless.
        for s in rule.get("suggestions") or []:
            if s.get("applicability") == "machine-applicable":
                assert s.get("command"), (
                    f"{rule['id']}: machine-applicable suggestion has no command"
                )

    def test_conditions_are_grounded_in_the_rules_own_kinds(self, backend, rule):
        available = set(rule["requires"]) | set(rule.get("optional") or [])
        for cond in rule.get("conditions") or []:
            assert cond.get("kind") in available, (
                f"{rule['id']}: condition on '{cond.get('kind')}', which the "
                "rule neither requires nor lists as optional"
            )

    def test_conditions_declare_a_bound(self, backend, rule):
        # A condition with neither min nor max checks nothing but silently
        # requires the field to be numeric — almost certainly a mistake.
        for cond in rule.get("conditions") or []:
            assert "min" in cond or "max" in cond, (
                f"{rule['id']}: condition on {cond.get('field')} has no bound"
            )

    def test_condition_fields_are_really_emitted(self, backend, rule):
        keys = KIND_KEYS.get(backend, {})
        for cond in rule.get("conditions") or []:
            kind, field = cond.get("kind"), cond.get("field")
            if kind in keys:  # only checkable when we have an artifact for it
                assert field in keys[kind], (
                    f"{rule['id']}: condition field '{field}' is never emitted "
                    f"by kind '{kind}'"
                )

    def test_message_placeholders_come_from_the_rules_own_evidence(self, backend, rule):
        """A placeholder must be fillable from the rule's OWN kinds.

        The diagnoser resolves placeholders across all matched facts, so a rule
        can appear to work because an unrelated fact happened to supply the key.
        That is luck, not correctness: on a log without that other fact, the
        user would see a literal "{count}".
        """
        keys = KIND_KEYS.get(backend, {})
        if not keys:
            pytest.skip(f"no artifacts for {backend}")
        own: set[str] = set()
        for kind in set(rule["requires"]) | set(rule.get("optional") or []):
            own |= keys.get(kind, set())
        for placeholder in re.findall(r"\{(\w+)\}", rule["message"]):
            assert placeholder in own, (
                f"{rule['id']}: {{{placeholder}}} is not provided by any of its "
                "own required/optional fact kinds"
            )


class TestEveryFactKindHasRealArtifactCoverage:
    """A rule exercised only by hand-built Facts is a rule tested against an
    assumption rather than against reality.

    This caught `session_failed` and `few_iterations`, which four rules
    referenced but no real artifact produced; real artifacts were then added to
    the corpus.
    """

    def test_every_referenced_kind_appears_in_a_real_artifact(self):
        missing = []
        for backend, rule in ALL_RULES:
            observed = set(KIND_KEYS.get(backend, {}))
            referenced = (
                set(rule["requires"])
                | set(rule.get("absent") or [])
                | set(rule.get("optional") or [])
            )
            for kind in referenced - observed:
                missing.append(f"{backend}/{rule['id']}: '{kind}'")
        assert not missing, (
            "fact kinds referenced by rules but produced by no real artifact "
            f"(add a corpus artifact that exercises them): {missing}"
        )

    def test_every_rule_fires_on_at_least_one_artifact(self):
        """A rule that never fires anywhere is untested in practice."""
        from edgedoctor.diagnoser import diagnose

        fired: set[str] = set()
        for backend in sorted(PARSER_REGISTRY):
            parser = get_parser(backend)
            for path in _artifacts_for(backend):
                for d in diagnose(parser.parse(path)):
                    fired.add(d.code)
        never = {r["id"] for _, r in ALL_RULES} - fired
        assert not never, (
            f"rules that fire on no artifact in the repo: {sorted(never)}"
        )
