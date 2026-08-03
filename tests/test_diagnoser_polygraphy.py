"""Tests for the diagnoser against the Polygraphy (ED02xx) rule family.

These exercise the whole rule engine end-to-end on real corpus logs, plus the
three mechanisms the accuracy rules needed the engine to grow: `absent` (a
disqualifying fact kind), `optional` (evidence that doesn't gate firing), and
`conditions` (a numeric threshold on a fact field).

The regression that matters most is the pass/fail pair: the same measured
divergence must produce an error on the failing log and only an info on the
passing one. If the diagnoser ever keys off magnitudes, that test breaks.
"""

from pathlib import Path

from edgedoctor.backends.base import Fact, Facts
from edgedoctor.backends.polygraphy import PolygraphyBackend
from edgedoctor.diagnoser import diagnose

CORPUS = Path(__file__).parent.parent / "corpus" / "onnxruntime"

parser = PolygraphyBackend()


def diagnose_log(name: str):
    return diagnose(parser.parse(CORPUS / name))


def codes(diagnoses) -> list[str]:
    return [d.code for d in diagnoses]


class TestRuleFiring:
    def test_divergence_fires_ed0201(self):
        assert "ED0201" in codes(diagnose_log("int8_vs_fp32_fail.log"))

    def test_layerwise_fires_ed0202_naming_first_tensor(self):
        diagnoses = diagnose_log("int8_vs_fp32_layerwise.log")
        ed0202 = next(d for d in diagnoses if d.code == "ED0202")
        assert "/conv1/Conv_output_0" in ed0202.message
        assert "49" in ed0202.message

    def test_nan_fires_ed0203(self):
        assert "ED0203" in codes(diagnose_log("nan_output.log"))

    def test_shape_mismatch_fires_ed0204(self):
        assert "ED0204" in codes(diagnose_log("shape_mismatch.log"))

    def test_pass_fires_ed0205_info(self):
        diagnoses = diagnose_log("int8_vs_fp32_pass.log")
        ed0205 = next(d for d in diagnoses if d.code == "ED0205")
        assert ed0205.severity == "info"


class TestPassFailRegression:
    """The tolerance-is-policy invariant, enforced at the diagnosis layer."""

    def test_failing_log_reports_an_error(self):
        diagnoses = diagnose_log("int8_vs_fp32_fail.log")
        assert any(d.severity == "error" for d in diagnoses)
        assert "ED0201" in codes(diagnoses)

    def test_passing_log_reports_no_error(self):
        # Same model, same divergence — only the tolerance differs. This must
        # NOT be an error.
        diagnoses = diagnose_log("int8_vs_fp32_pass.log")
        assert not any(d.severity == "error" for d in diagnoses)
        assert "ED0201" not in codes(diagnoses)

    def test_clean_baseline_yields_nothing_actionable(self):
        # A single-runner run did no comparison; no ED020x error may appear.
        diagnoses = diagnose_log("fp32_baseline_run.log")
        assert not any(d.severity == "error" for d in diagnoses)


class TestAbsentMechanism:
    def test_absent_kind_blocks_a_rule(self):
        # ED0201 requires output_mismatch but is disqualified by
        # all_outputs_matched. A (contrived) log with both present must not fire
        # ED0201 — the "all matched" verdict wins.
        facts = Facts(
            backend="polygraphy",
            artifact_path="mixed.log",
            facts=[
                Fact(id="f1", kind="output_mismatch", summary="s",
                     source="mixed.log:1", data={"output": "out"}),
                Fact(id="f2", kind="all_outputs_matched", summary="s",
                     source="mixed.log:2", data={"outputs": ["out"], "count": 1}),
            ],
        )
        assert "ED0201" not in codes(diagnose(facts))

    def test_rule_fires_when_forbidden_kind_absent(self):
        facts = Facts(
            backend="polygraphy",
            artifact_path="clean_fail.log",
            facts=[
                Fact(id="f1", kind="output_mismatch", summary="s",
                     source="clean_fail.log:1", data={"output": "out"}),
            ],
        )
        assert "ED0201" in codes(diagnose(facts))


class TestConditionsMechanism:
    def test_single_output_does_not_fire_ed0202(self):
        # count=1 fails ED0202's `min: 2` condition — it would just restate
        # ED0201, and "which layer came first" is meaningless for one tensor.
        facts = Facts(
            backend="polygraphy",
            artifact_path="single.log",
            facts=[
                Fact(id="f1", kind="mismatched_outputs", summary="s",
                     source="single.log:1",
                     data={"outputs": ["out"], "count": 1, "first_output": "out"}),
            ],
        )
        assert "ED0202" not in codes(diagnose(facts))

    def test_two_outputs_fire_ed0202(self):
        facts = Facts(
            backend="polygraphy",
            artifact_path="multi.log",
            facts=[
                Fact(id="f1", kind="mismatched_outputs", summary="s",
                     source="multi.log:1",
                     data={"outputs": ["a", "b"], "count": 2, "first_output": "a"}),
            ],
        )
        assert "ED0202" in codes(diagnose(facts))

    def test_missing_condition_field_fails_closed(self):
        # A rule may never fire on evidence that isn't actually there: if the
        # count field is absent, the condition must FAIL, not pass.
        facts = Facts(
            backend="polygraphy",
            artifact_path="broken.log",
            facts=[
                Fact(id="f1", kind="mismatched_outputs", summary="s",
                     source="broken.log:1", data={"outputs": ["a", "b"]}),
            ],
        )
        assert "ED0202" not in codes(diagnose(facts))


class TestOptionalMechanism:
    def test_optional_evidence_is_attached_when_present(self):
        # ED0201 requires output_mismatch and lists required_tolerance as
        # optional. When present, its fact id must be cited as evidence.
        facts = Facts(
            backend="polygraphy",
            artifact_path="withtol.log",
            facts=[
                Fact(id="f1", kind="output_mismatch", summary="s",
                     source="withtol.log:1", data={"output": "out"}),
                Fact(id="f2", kind="required_tolerance", summary="s",
                     source="withtol.log:2",
                     data={"req_abs": "4.5", "req_rel": "256", "stat": "elemwise"}),
            ],
        )
        ed0201 = next(d for d in diagnose(facts) if d.code == "ED0201")
        assert "f2" in ed0201.evidence

    def test_optional_evidence_absent_does_not_block_firing(self):
        # No required_tolerance present — ED0201 must still fire, since optional
        # kinds never gate.
        facts = Facts(
            backend="polygraphy",
            artifact_path="notol.log",
            facts=[
                Fact(id="f1", kind="output_mismatch", summary="s",
                     source="notol.log:1", data={"output": "out"}),
            ],
        )
        ed0201 = next(d for d in diagnose(facts) if d.code == "ED0201")
        assert ed0201.evidence == ["f1"]


class TestDiagnosisProperties:
    def test_evidence_ids_point_to_real_facts(self):
        for name in ("int8_vs_fp32_fail.log", "int8_vs_fp32_layerwise.log",
                     "nan_output.log", "shape_mismatch.log"):
            facts = parser.parse(CORPUS / name)
            fact_ids = {f.id for f in facts.facts}
            for d in diagnose(facts):
                for eid in d.evidence:
                    assert eid in fact_ids, f"{name}: evidence {eid} not in facts"

    def test_placeholders_resolved(self):
        for name in ("int8_vs_fp32_fail.log", "int8_vs_fp32_layerwise.log",
                     "shape_mismatch.log"):
            for d in diagnose(parser.parse(CORPUS / name)):
                assert "{" not in d.message, f"{name}: unresolved placeholder in {d.code}"

    def test_every_diagnosis_has_evidence(self):
        # A grounded diagnosis with no evidence is a contradiction in terms.
        for d in diagnose_log("int8_vs_fp32_fail.log"):
            assert d.evidence, f"{d.code} has no evidence"
