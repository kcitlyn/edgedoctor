"""Tests for the rule-based diagnoser engine."""

from pathlib import Path

from edgedoctor.backends.tensorrt import TensorRTBackend
from edgedoctor.diagnoser import diagnose

FIXTURES = Path(__file__).parent / "fixtures" / "tensorrt"


def facts_from(name: str):
    return TensorRTBackend().parse(FIXTURES / name)


class TestRuleMatching:
    def test_unsupported_op_fires_ed0101(self):
        diagnoses = diagnose(facts_from("unsupported_op_trt8.log"))
        codes = [d.code for d in diagnoses]
        assert "ED0101" in codes

    def test_no_implementation_fires_ed0102(self):
        diagnoses = diagnose(facts_from("build_failure_trt10.log"))
        codes = [d.code for d in diagnoses]
        assert "ED0102" in codes

    def test_tactic_skip_fires_ed0104(self):
        diagnoses = diagnose(facts_from("build_failure_trt10.log"))
        codes = [d.code for d in diagnoses]
        assert "ED0104" in codes

    def test_parse_error_fires_ed0103(self):
        diagnoses = diagnose(facts_from("parse_error_trt10.log"))
        codes = [d.code for d in diagnoses]
        assert "ED0103" in codes

    def test_plugin_not_found_fires_ed0105(self):
        diagnoses = diagnose(facts_from("unsupported_op_trt8.log"))
        codes = [d.code for d in diagnoses]
        assert "ED0105" in codes


class TestCleanLog:
    def test_no_rules_fire_on_success(self):
        diagnoses = diagnose(facts_from("success.log"))
        assert diagnoses == []


class TestDiagnosisProperties:
    def test_evidence_ids_point_to_real_facts(self):
        facts = facts_from("unsupported_op_trt8.log")
        diagnoses = diagnose(facts)
        fact_ids = {f.id for f in facts.facts}
        for d in diagnoses:
            for eid in d.evidence:
                assert eid in fact_ids, f"evidence {eid} not in facts"

    def test_errors_sorted_before_warnings(self):
        diagnoses = diagnose(facts_from("build_failure_trt10.log"))
        severities = [d.severity for d in diagnoses]
        assert severities.index("error") < severities.index("warning")

    def test_placeholders_resolved(self):
        diagnoses = diagnose(facts_from("unsupported_op_trt8.log"))
        ed0101 = next(d for d in diagnoses if d.code == "ED0101")
        assert "GridSample" in ed0101.message
        assert "{op}" not in ed0101.message
