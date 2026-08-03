"""Tests for the diagnoser against the ONNX Runtime (ED03xx) rule family.

The pair that matters most is TestIntentionalVsSilentCpu: two corpus logs with
identical placement lines must produce opposite diagnoses — no diagnosis for a
deliberate CPU session, an ERROR for a silently-degraded accelerator run. That's
the ORT analogue of the polygraphy pass/fail regression pair.
"""

from pathlib import Path

from edgedoctor.backends.base import Fact, Facts
from edgedoctor.backends.onnxruntime import OnnxRuntimeBackend
from edgedoctor.diagnoser import diagnose

CORPUS = Path(__file__).parent.parent / "corpus" / "onnxruntime"

parser = OnnxRuntimeBackend()


def diagnose_log(name: str):
    return diagnose(parser.parse(CORPUS / name))


def codes(diagnoses) -> list[str]:
    return [d.code for d in diagnoses]


def ort_facts(*facts: Fact) -> Facts:
    return Facts(backend="onnxruntime", artifact_path="t.log", facts=list(facts))


class TestRuleFiring:
    def test_partial_fallback_fires_ed0301(self):
        diagnoses = diagnose_log("ort_partial_fallback.log")
        assert "ED0301" in codes(diagnoses)
        ed = next(d for d in diagnoses if d.code == "ED0301")
        assert ed.severity == "warning"  # the session succeeded; it's slow, not broken

    def test_missing_provider_fires_ed0302_as_error(self):
        diagnoses = diagnose_log("ort_missing_provider.log")
        ed = next(d for d in diagnoses if d.code == "ED0302")
        # An error, not a warning: you believe you're testing an accelerator and
        # you are measuring CPU. Every number you collect is wrong.
        assert ed.severity == "error"
        assert "TensorrtExecutionProvider" in ed.message

    def test_clean_single_provider_fires_ed0305_info(self):
        diagnoses = diagnose_log("ort_all_nodes_one_ep.log")
        ed = next(d for d in diagnoses if d.code == "ED0305")
        assert ed.severity == "info"


class TestIntentionalVsSilentCpu:
    """Identical placement, opposite verdicts."""

    def test_deliberate_cpu_session_yields_no_diagnosis(self):
        # Nothing is wrong: CPU is what was asked for. Reporting fallback here
        # would be crying wolf, and would train users to ignore the tool.
        assert diagnose_log("ort_cpu_only.log") == []

    def test_silently_degraded_session_is_an_error(self):
        diagnoses = diagnose_log("ort_missing_provider.log")
        assert any(d.severity == "error" for d in diagnoses)

    def test_ed0305_does_not_fire_on_an_all_cpu_session(self):
        # ED0305 says "no fallback, ideal placement". For an all-CPU run that
        # would be a claim the log cannot support — CPU-only is only ideal if
        # CPU was the goal, and placement alone can't tell.
        assert "ED0305" not in codes(diagnose_log("ort_cpu_only.log"))
        assert "ED0305" not in codes(diagnose_log("ort_missing_provider.log"))


class TestBroadFallbackThreshold:
    def test_two_fallback_ops_do_not_fire_ed0303(self):
        # The real corpus log has exactly 2 fallback ops, below ED0303's min of
        # 3 — a couple of stray ops warrants different advice than half a graph.
        assert "ED0303" not in codes(diagnose_log("ort_partial_fallback.log"))

    def test_three_fallback_ops_fire_ed0303(self):
        facts = ort_facts(
            Fact(id="f1", kind="cpu_fallback_ops", summary="s", source="t.log:1",
                 excerpt="x",
                 data={"ops": ["Erf", "Round", "NonZero"], "count": 3,
                       "first_op": "Erf"}),
        )
        diagnoses = diagnose(facts)
        assert "ED0303" in codes(diagnoses)
        ed = next(d for d in diagnoses if d.code == "ED0303")
        assert "Erf" in ed.message

    def test_ed0303_suppressed_when_provider_was_unavailable(self):
        # If the accelerator never loaded, "this provider covers little of your
        # graph" is the wrong explanation — ED0302 is the right one.
        facts = ort_facts(
            Fact(id="f1", kind="cpu_fallback_ops", summary="s", source="t.log:1",
                 excerpt="x",
                 data={"ops": ["A", "B", "C"], "count": 3, "first_op": "A"}),
            Fact(id="f2", kind="silent_cpu_only", summary="s", source="t.log:2",
                 excerpt="y",
                 data={"requested": "CUDAExecutionProvider",
                       "actual": ["CPUExecutionProvider"]}),
        )
        assert "ED0303" not in codes(diagnose(facts))


class TestSessionFailure:
    def test_session_failure_fires_ed0304(self):
        facts = ort_facts(
            Fact(id="f1", kind="session_failed", summary="s", source="t.log:1",
                 excerpt="SESSION_FAILED: Fail: bad model",
                 data={"error": "Fail: bad model"}),
        )
        assert "ED0304" in codes(diagnose(facts))

    def test_session_failure_suppresses_the_clean_verdict(self):
        # A failed session did no placement, so ED0305 must not claim success.
        facts = ort_facts(
            Fact(id="f1", kind="session_failed", summary="s", source="t.log:1",
                 excerpt="SESSION_FAILED: x", data={"error": "x"}),
            Fact(id="f2", kind="all_nodes_one_provider", summary="s",
                 source="t.log:2", excerpt="y",
                 data={"provider": "CPUExecutionProvider", "count": 1}),
        )
        assert "ED0305" not in codes(diagnose(facts))


class TestEvidenceQuality:
    def test_no_duplicate_evidence(self):
        # A kind can appear in both `requires` and `optional`; showing the user
        # the same log line twice reads as a bug in the tool.
        for name in ("ort_partial_fallback.log", "ort_missing_provider.log",
                     "ort_all_nodes_one_ep.log"):
            for d in diagnose_log(name):
                assert len(d.evidence) == len(set(d.evidence)), (
                    f"{name} {d.code} cites a fact twice"
                )

    def test_actionable_evidence_survives_the_report_cap(self):
        # The report prints only the first MAX_EVIDENCE_SHOWN blocks, so the ops
        # that fell back must be cited early enough to actually be displayed.
        from edgedoctor.report import MAX_EVIDENCE_SHOWN

        facts = parser.parse(CORPUS / "ort_partial_fallback.log")
        ed0301 = next(d for d in diagnose(facts) if d.code == "ED0301")
        by_id = {f.id: f for f in facts.facts}
        shown_kinds = [by_id[e].kind for e in ed0301.evidence[:MAX_EVIDENCE_SHOWN]]
        assert "cpu_fallback_ops" in shown_kinds

    def test_evidence_ids_point_to_real_facts(self):
        for name in ("ort_partial_fallback.log", "ort_missing_provider.log",
                     "ort_all_nodes_one_ep.log", "ort_cpu_only.log"):
            facts = parser.parse(CORPUS / name)
            fact_ids = {f.id for f in facts.facts}
            for d in diagnose(facts):
                for eid in d.evidence:
                    assert eid in fact_ids

    def test_placeholders_resolved(self):
        for name in ("ort_partial_fallback.log", "ort_missing_provider.log",
                     "ort_all_nodes_one_ep.log"):
            for d in diagnose_log(name):
                assert "{" not in d.message, f"{name}: unresolved placeholder in {d.code}"

    def test_every_diagnosis_has_evidence(self):
        for name in ("ort_partial_fallback.log", "ort_missing_provider.log"):
            for d in diagnose_log(name):
                assert d.evidence, f"{name} {d.code} has no evidence"
