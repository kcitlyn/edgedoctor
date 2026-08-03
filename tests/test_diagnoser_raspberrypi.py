"""Tests for the diagnoser against the Raspberry Pi (ED04xx) rule family.

The distinction under test throughout: live throttling is an ERROR (the number
you're taking now is invalid) while historical throttling is a WARNING (the
machine is fine now, but earlier numbers are suspect). Getting that backwards
either cries wolf on a healthy host or lets a corrupted benchmark through.
"""

from pathlib import Path

from edgedoctor.backends.base import Fact, Facts
from edgedoctor.backends.raspberrypi import RaspberryPiBackend
from edgedoctor.diagnoser import diagnose

FIXTURES = Path(__file__).parent / "fixtures" / "raspberrypi"

parser = RaspberryPiBackend()


def diagnose_fixture(name: str):
    return diagnose(parser.parse(FIXTURES / name))


def codes(diagnoses) -> list[str]:
    return [d.code for d in diagnoses]


def pi_facts(*facts: Fact) -> Facts:
    return Facts(backend="raspberrypi", artifact_path="t.log", facts=list(facts))


class TestLiveVsHistorical:
    def test_live_throttling_is_an_error(self):
        diagnoses = diagnose_fixture("throttle_active.log")
        ed = next(d for d in diagnoses if d.code == "ED0401")
        # An error even though nothing crashed: a confidently-reported wrong
        # benchmark does more damage than a crash, because it gets published.
        assert ed.severity == "error"

    def test_historical_throttling_is_only_a_warning(self):
        diagnoses = diagnose_fixture("throttle_occurred.log")
        ed = next(d for d in diagnoses if d.code == "ED0402")
        assert ed.severity == "warning"
        assert "ED0401" not in codes(diagnoses)

    def test_live_throttling_suppresses_the_historical_rule(self):
        # 0x50005 sets both halves. Reporting both would double-report one event;
        # the live condition is the one that matters.
        diagnoses = diagnose_fixture("throttle_active.log")
        assert "ED0402" not in codes(diagnoses)

    def test_healthy_host_is_reported_as_trustworthy(self):
        diagnoses = diagnose_fixture("throttle_clear.log")
        ed = next(d for d in diagnoses if d.code == "ED0406")
        assert ed.severity == "info"

    def test_healthy_host_has_no_errors_or_warnings(self):
        diagnoses = diagnose_fixture("throttle_clear.log")
        assert all(d.severity == "info" for d in diagnoses)


class TestOom:
    def test_kernel_oom_fires_ed0403(self):
        diagnoses = diagnose_fixture("oom_kill.log")
        ed = next(d for d in diagnoses if d.code == "ED0403")
        assert ed.severity == "error"
        assert "python3" in ed.message

    def test_allocation_failure_fires_ed0404(self):
        assert "ED0404" in codes(diagnose_fixture("allocation_failed.log"))

    def test_kernel_oom_suppresses_the_allocator_rule(self):
        # If the kernel killed the process, "configure your arena" is the wrong
        # advice — ED0403's fix is about total RAM, ED0404's about one buffer.
        facts = pi_facts(
            Fact(id="f1", kind="oom_kill", summary="s", source="t.log:1",
                 excerpt="Out of memory: Killed process 1 (python3)",
                 data={"pid": "1", "process": "python3"}),
            Fact(id="f2", kind="allocation_failed", summary="s", source="t.log:2",
                 excerpt="bad_alloc", data={}),
        )
        result = codes(diagnose(facts))
        assert "ED0403" in result
        assert "ED0404" not in result

    def test_oom_log_does_not_report_throttling(self):
        # End-to-end version of the parser regression: a memory log must never
        # produce a hardware-throttling diagnosis.
        result = codes(diagnose_fixture("oom_kill.log"))
        assert "ED0401" not in result
        assert "ED0402" not in result
        assert "ED0405" not in result


class TestUndocumentedBits:
    def test_unknown_bit_fires_ed0405(self):
        diagnoses = diagnose_fixture("throttle_unknown_bit.log")
        ed = next(d for d in diagnoses if d.code == "ED0405")
        assert "8" in ed.message

    def test_unknown_bit_does_not_claim_a_healthy_host(self):
        # An unexplained hardware flag is not evidence of health.
        assert "ED0406" not in codes(diagnose_fixture("throttle_unknown_bit.log"))


class TestDiagnosisProperties:
    LOGS = ["throttle_active.log", "throttle_occurred.log", "throttle_clear.log",
            "throttle_unknown_bit.log", "oom_kill.log", "allocation_failed.log"]

    def test_placeholders_resolved(self):
        for name in self.LOGS:
            for d in diagnose_fixture(name):
                assert "{" not in d.message, f"{name}: unresolved placeholder in {d.code}"

    def test_no_python_reprs_leak_into_messages(self):
        # A message containing "['x', 'y']" means a raw list reached the user.
        for name in self.LOGS:
            for d in diagnose_fixture(name):
                assert "['" not in d.message, f"{name}: raw list in {d.code}"

    def test_no_duplicate_evidence(self):
        for name in self.LOGS:
            for d in diagnose_fixture(name):
                assert len(d.evidence) == len(set(d.evidence)), (
                    f"{name} {d.code} cites a fact twice"
                )

    def test_evidence_ids_point_to_real_facts(self):
        for name in self.LOGS:
            facts = parser.parse(FIXTURES / name)
            fact_ids = {f.id for f in facts.facts}
            for d in diagnose(facts):
                for eid in d.evidence:
                    assert eid in fact_ids

    def test_every_diagnosis_has_evidence(self):
        for name in self.LOGS:
            for d in diagnose_fixture(name):
                assert d.evidence, f"{name} {d.code} has no evidence"
