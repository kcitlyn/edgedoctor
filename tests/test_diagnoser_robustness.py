"""Tests that the rule engine fails CLOSED on malformed input.

Rule files are hand-edited YAML, so mistakes in them are realistic, not
hypothetical. The design principle these tests pin:

    A broken rule must go silent, never crash the run and never misfire.

Crashing is unacceptable because one typo would take down every OTHER rule's
correct output — a diagnostic tool that dies while explaining a failure is worse
than useless. Misfiring is unacceptable because a confident wrong diagnosis
sends the user hours in the wrong direction.

Six crashes and one silent misfire were found by probing this way; all are
pinned below.
"""

import pathlib
import tempfile

import pytest

from edgedoctor import diagnoser
from edgedoctor.backends.base import Fact, Facts
from edgedoctor.diagnoser import _conditions_met, _load_rules, _satisfies, diagnose


@pytest.fixture
def rules_dir(monkeypatch):
    """Point the diagnoser at a temp rules dir we can fill with bad YAML."""
    tmp = pathlib.Path(tempfile.mkdtemp())
    monkeypatch.setattr(diagnoser, "RULES_DIR", tmp)
    return tmp


def facts_for(backend="probe", kind="cpu_fallback", **data) -> Facts:
    return Facts(
        backend=backend,
        artifact_path="t.log",
        facts=[Fact(id="f1", kind=kind, summary="s", source="t.log:1",
                    excerpt="line", data=data or {"n": 5})],
    )


class TestMalformedRuleFilesFailClosed:
    """Every one of these crashed the diagnoser before being fixed."""

    @pytest.mark.parametrize(
        "label,content",
        [
            ("unparseable_yaml", "- id: ED1\n  requires: [k\n   bad: {{{"),
            ("mapping_not_list", "id: ED1\nrequires: [cpu_fallback]"),
            ("empty_file", ""),
            ("null_document", "null"),
            ("list_of_strings", "- just a string\n- another"),
            ("suggestions_not_a_list",
             "- id: ED1\n  requires: [cpu_fallback]\n  suggestions: nope\n  message: m"),
            ("suggestion_is_a_string",
             "- id: ED1\n  requires: [cpu_fallback]\n  suggestions: ['str']\n  message: m"),
            ("conditions_not_a_list",
             "- id: ED1\n  requires: [cpu_fallback]\n  conditions: nope\n  message: m"),
            ("condition_bound_not_numeric",
             "- id: ED1\n  requires: [cpu_fallback]\n  message: m\n"
             "  conditions: [{kind: cpu_fallback, field: n, min: two}]"),
            ("condition_not_a_mapping",
             "- id: ED1\n  requires: [cpu_fallback]\n  message: m\n  conditions: ['x']"),
            ("message_is_a_number",
             "- id: ED1\n  requires: [cpu_fallback]\n  message: 42"),
            ("deeply_nested_garbage",
             "- id: ED1\n  requires: [cpu_fallback]\n  message: m\n  extra: {a: {b: {c: [1,2]}}}"),
        ],
    )
    def test_does_not_raise(self, rules_dir, label, content):
        (rules_dir / "probe.yaml").write_text(content)
        # The contract is simply: no exception escapes.
        diagnose(facts_for())

    def test_one_bad_rule_does_not_suppress_a_good_one(self, rules_dir):
        # The whole point of failing closed per-rule rather than per-file.
        (rules_dir / "probe.yaml").write_text(
            "- a bare string that is not a rule\n"
            "- id: ED_GOOD\n  requires: [cpu_fallback]\n  message: still works\n"
        )
        assert [d.code for d in diagnose(facts_for())] == ["ED_GOOD"]


class TestBareStringRequiresDoesNotMisfire:
    """The silent misfire, which is subtler than any of the crashes.

    `requires: cpu_fallback` (missing brackets) used to become
    `set("cpu_fallback")` — a set of CHARACTERS. It matched a fact of kind "c",
    so the rule fired on completely unrelated evidence.
    """

    def test_bare_string_matches_the_whole_kind(self, rules_dir):
        (rules_dir / "probe.yaml").write_text(
            "- id: ED1\n  requires: cpu_fallback\n  message: m\n"
        )
        assert "ED1" in [d.code for d in diagnose(facts_for(kind="cpu_fallback"))]

    def test_bare_string_does_not_match_a_single_character_kind(self, rules_dir):
        # The actual bug: this must NOT fire on a fact of kind "c".
        (rules_dir / "probe.yaml").write_text(
            "- id: ED1\n  requires: cpu_fallback\n  message: m\n"
        )
        assert diagnose(facts_for(kind="c")) == []

    def test_bare_string_absent_is_also_normalized(self, rules_dir):
        (rules_dir / "probe.yaml").write_text(
            "- id: ED1\n  requires: [cpu_fallback]\n  absent: cpu_fallback\n  message: m\n"
        )
        # absent names the kind that IS present, so the rule must stay silent.
        assert diagnose(facts_for(kind="cpu_fallback")) == []


class TestRuleFileLookupIsConfined:
    """`backend` reaches _load_rules from data, so it must not select a path."""

    @pytest.mark.parametrize(
        "backend",
        ["../../../etc/passwd", "..", ".", "", "/etc/passwd",
         "tensorrt/../tensorrt", "a b", "$(whoami)", "sub/dir"],
    )
    def test_suspicious_backend_names_load_nothing(self, backend):
        assert _load_rules(backend) == []

    def test_a_real_backend_still_loads(self):
        assert len(_load_rules("tensorrt")) > 0


class TestSatisfiesRejectsNonMeasurements:
    """NaN and infinity must not satisfy thresholds.

    All comparisons against NaN are False, so a naive `if num < lo: return
    False` lets NaN pass EVERY bound — it would satisfy "at least 50%" and "at
    most 10%" at the same time. A NaN reaching a threshold means a measurement
    failed to parse, which is the opposite of evidence.
    """

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_never_satisfies_a_minimum(self, value):
        assert _satisfies(value, 2, None) is False

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_never_satisfies_a_maximum(self, value):
        assert _satisfies(value, None, 100) is False

    def test_nan_string_is_also_rejected(self):
        # float("nan") succeeds on the STRING "nan", so this path needs the
        # same guard.
        assert _satisfies("nan", 2, None) is False
        assert _satisfies("inf", 2, None) is False

    @pytest.mark.parametrize("value", [5, 5.0, "5", "5.0", " 5 "])
    def test_real_numbers_still_work(self, value):
        assert _satisfies(value, 2, None) is True

    @pytest.mark.parametrize("value", [None, "abc", [1], {"a": 1}, "0x10", ""])
    def test_non_numeric_fails(self, value):
        assert _satisfies(value, 2, None) is False

    def test_bounds_are_inclusive(self):
        assert _satisfies(2, 2, None) is True
        assert _satisfies(2, None, 2) is True

    def test_malformed_bound_fails_closed(self):
        # A typo in a rule file must not raise from deep inside matching.
        assert _satisfies(5, "two", None) is False
        assert _satisfies(5, None, "ten") is False


class TestConditionsFailClosed:
    def test_missing_kind_or_field_is_not_met(self):
        facts = facts_for(n=5)
        assert _conditions_met([{}], facts) is False
        assert _conditions_met([{"kind": "cpu_fallback"}], facts) is False
        assert _conditions_met([{"field": "n"}], facts) is False

    def test_non_mapping_condition_is_not_met(self):
        assert _conditions_met(["nope"], facts_for(n=5)) is False

    def test_unknown_kind_is_not_met(self):
        assert _conditions_met(
            [{"kind": "nope", "field": "n", "min": 1}], facts_for(n=5)
        ) is False

    def test_absent_field_is_not_met(self):
        assert _conditions_met(
            [{"kind": "cpu_fallback", "field": "missing", "min": 1}], facts_for(n=5)
        ) is False

    def test_all_conditions_must_hold(self):
        facts = facts_for(n=5)
        assert _conditions_met(
            [{"kind": "cpu_fallback", "field": "n", "min": 1},
             {"kind": "cpu_fallback", "field": "n", "min": 99}], facts
        ) is False

    def test_empty_conditions_are_trivially_met(self):
        assert _conditions_met([], facts_for(n=5)) is True


class TestDiagnoseWithOddFacts:
    """Facts arriving from an unexpected shape must not break matching."""

    def test_no_facts_yields_no_diagnoses(self):
        assert diagnose(Facts(backend="tensorrt", artifact_path="t.log")) == []

    def test_unknown_backend_yields_no_diagnoses(self):
        facts = Facts(
            backend="nonexistent_backend", artifact_path="t.log",
            facts=[Fact(id="f1", kind="k", summary="s", source="t.log:1")],
        )
        assert diagnose(facts) == []

    def test_fact_with_empty_data_does_not_break_placeholders(self, rules_dir):
        (rules_dir / "probe.yaml").write_text(
            "- id: ED1\n  requires: [cpu_fallback]\n  message: 'value is {missing}'\n"
        )
        facts = Facts(
            backend="probe", artifact_path="t.log",
            facts=[Fact(id="f1", kind="cpu_fallback", summary="s", source="t.log:1")],
        )
        d = diagnose(facts)
        # Unfilled placeholders are left visible rather than crashing — the
        # reader can see the rule is incomplete.
        assert d and "{missing}" in d[0].message

    def test_many_facts_do_not_slow_matching(self):
        import time

        facts = Facts(
            backend="onnxruntime", artifact_path="t.log",
            facts=[
                Fact(id=f"f{i}", kind="node_placement", summary="s",
                     source=f"t.log:{i}",
                     data={"provider": "CPUExecutionProvider", "count": 1})
                for i in range(20000)
            ],
        )
        start = time.monotonic()
        diagnose(facts)
        assert time.monotonic() - start < 5.0


class TestSeverityOrderingIsEnforced:
    """Errors must be listed before warnings before info.

    A reader skims top-down, so the most serious finding has to come first. No
    committed artifact exercises this today — every rule file happens to be
    AUTHORED in severity order, so removing the sort changes nothing about
    current output. That makes it a latent regression: the day someone appends an
    error-severity rule below an info one, the report silently starts burying it.

    Found by mutation testing: replacing the sort key with a constant broke no
    test.
    """

    def _rules(self, rules_dir, body: str):
        (rules_dir / "probe.yaml").write_text(body)

    def test_error_is_sorted_before_info_regardless_of_file_order(self, rules_dir):
        # Deliberately declared info-first, the opposite of every real file.
        self._rules(rules_dir, """
- id: ED_INFO
  severity: info
  requires: [cpu_fallback]
  message: an informational note
- id: ED_ERROR
  severity: error
  requires: [cpu_fallback]
  message: a serious error
""")
        severities = [d.severity for d in diagnose(facts_for())]
        assert severities == ["error", "info"], (
            f"got {severities}; the error must be reported first"
        )

    def test_full_ordering_is_error_warning_info(self, rules_dir):
        self._rules(rules_dir, """
- id: ED_INFO
  severity: info
  requires: [cpu_fallback]
  message: note
- id: ED_WARN
  severity: warning
  requires: [cpu_fallback]
  message: warn
- id: ED_ERR
  severity: error
  requires: [cpu_fallback]
  message: err
""")
        assert [d.severity for d in diagnose(facts_for())] == [
            "error", "warning", "info"
        ]

    def test_unknown_severity_sorts_last(self, rules_dir):
        # It must not displace a real error from the top of the report.
        self._rules(rules_dir, """
- id: ED_ODD
  severity: bizarre
  requires: [cpu_fallback]
  message: odd
- id: ED_ERR
  severity: error
  requires: [cpu_fallback]
  message: err
""")
        assert [d.severity for d in diagnose(facts_for())][0] == "error"

    def test_ordering_is_stable_within_a_severity(self, rules_dir):
        # Equal severities keep their declaration order, so a rule author can
        # control which of two errors reads first.
        self._rules(rules_dir, """
- id: ED_FIRST
  severity: error
  requires: [cpu_fallback]
  message: first
- id: ED_SECOND
  severity: error
  requires: [cpu_fallback]
  message: second
""")
        assert [d.code for d in diagnose(facts_for())] == ["ED_FIRST", "ED_SECOND"]


class TestEvidenceDeduplication:
    """A fact must never be cited twice by one diagnosis.

    Showing the user the same log line twice reads as a bug in the tool. Two
    routes produce it, and neither is exercised by any real rule file today —
    found by mutation testing:

      - a repeated entry in `optional` (a plausible hand-edit slip)
      - a kind listed in both `requires` and `optional`
    """

    def test_repeated_optional_entry_cites_once(self, rules_dir):
        (rules_dir / "probe.yaml").write_text(
            "- id: ED1\n  requires: [a]\n  optional: [b, b, b]\n  message: m\n"
        )
        facts = Facts(
            backend="probe", artifact_path="t.log",
            facts=[
                Fact(id="f1", kind="a", summary="s", source="t.log:1"),
                Fact(id="f2", kind="b", summary="s", source="t.log:2"),
            ],
        )
        evidence = diagnose(facts)[0].evidence
        assert evidence == ["f1", "f2"], f"duplicated evidence: {evidence}"

    def test_kind_in_both_requires_and_optional_cites_once(self, rules_dir):
        (rules_dir / "probe.yaml").write_text(
            "- id: ED1\n  requires: [a]\n  optional: [a, b]\n  message: m\n"
        )
        facts = Facts(
            backend="probe", artifact_path="t.log",
            facts=[
                Fact(id="f1", kind="a", summary="s", source="t.log:1"),
                Fact(id="f2", kind="b", summary="s", source="t.log:2"),
            ],
        )
        evidence = diagnose(facts)[0].evidence
        assert len(evidence) == len(set(evidence))
        assert evidence == ["f1", "f2"]

    def test_many_facts_of_one_optional_kind_are_all_cited_once_each(self, rules_dir):
        # Dedup must not collapse DISTINCT facts that share a kind.
        (rules_dir / "probe.yaml").write_text(
            "- id: ED1\n  requires: [a]\n  optional: [b, b]\n  message: m\n"
        )
        facts = Facts(
            backend="probe", artifact_path="t.log",
            facts=[Fact(id="f1", kind="a", summary="s", source="t.log:1")]
            + [Fact(id=f"g{i}", kind="b", summary="s", source=f"t.log:{i}")
               for i in range(5)],
        )
        evidence = diagnose(facts)[0].evidence
        assert len(evidence) == 6, "each distinct fact must still be cited"
        assert len(evidence) == len(set(evidence))

    def test_optional_order_is_preserved_after_dedup(self, rules_dir):
        # The report caps evidence blocks, so the author's ordering decides what
        # a reader actually sees. Dedup must not reshuffle it.
        (rules_dir / "probe.yaml").write_text(
            "- id: ED1\n  requires: [a]\n  optional: [c, c, b]\n  message: m\n"
        )
        facts = Facts(
            backend="probe", artifact_path="t.log",
            facts=[
                Fact(id="f1", kind="a", summary="s", source="t.log:1"),
                Fact(id="f2", kind="b", summary="s", source="t.log:2"),
                Fact(id="f3", kind="c", summary="s", source="t.log:3"),
            ],
        )
        # required first, then optional in declared order: c before b.
        assert diagnose(facts)[0].evidence == ["f1", "f3", "f2"]
