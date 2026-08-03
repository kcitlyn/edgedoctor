"""Tests for the Polygraphy accuracy-divergence parser.

These run against REAL logs in corpus/onnxruntime/ — generated on this machine
by scripts/make_divergence_corpus.py, not hand-written. That's the point: the
signatures were written to match what the tool actually emits.

The most important test in this file is TestToleranceIsPolicyNotFact. It pins
the invariant that the passing and failing logs of the SAME comparison contain
identical measurements, so the parser must never treat a large measured
difference as evidence of failure.
"""

from pathlib import Path

import pytest

from edgedoctor.backends.polygraphy import PolygraphyBackend

CORPUS = Path(__file__).parent.parent / "corpus" / "onnxruntime"

parser = PolygraphyBackend()


def parse_log(name: str):
    return parser.parse(CORPUS / name)


def kinds_in(facts) -> set[str]:
    return {f.kind for f in facts.facts}


class TestDivergenceFailure:
    LOG = "int8_vs_fp32_fail.log"

    def test_detects_output_mismatch(self):
        facts = parse_log(self.LOG)
        mismatches = [f for f in facts.facts if f.kind == "output_mismatch"]
        assert len(mismatches) == 1
        assert mismatches[0].data["output"] == "output"

    def test_records_the_tolerance_that_was_applied(self):
        facts = parse_log(self.LOG)
        tol = next(f for f in facts.facts if f.kind == "tolerance_setting")
        # Polygraphy's FP32-vs-FP32 default, which INT8 cannot meet.
        assert tol.data["used_abs"] == "1e-05"
        assert tol.data["used_rel"] == "1e-05"

    def test_records_the_tolerance_that_would_have_passed(self):
        # The single most actionable number in the log.
        facts = parse_log(self.LOG)
        req = next(f for f in facts.facts if f.kind == "required_tolerance")
        assert float(req.data["req_abs"]) > 0
        assert req.data["stat"] == "elemwise"

    def test_splits_metrics_into_individual_numbers(self):
        facts = parse_log(self.LOG)
        metrics = next(f for f in facts.facts if f.kind == "divergence_metrics")
        assert float(metrics.data["max_absdiff"]) > 0
        assert float(metrics.data["mean_absdiff"]) > 0

    def test_records_failed_verdict(self):
        facts = parse_log(self.LOG)
        verdict = next(f for f in facts.facts if f.kind == "run_verdict")
        assert verdict.data["verdict"] == "FAILED"


class TestToleranceIsPolicyNotFact:
    """The honesty invariant of this parser.

    Same model, same inputs, same measured divergence — only the tolerance
    differs. If any of these assertions ever break, the parser has started
    treating a measurement as a verdict, and it will invent failures.
    """

    def test_measurements_are_identical_in_pass_and_fail_logs(self):
        fail = parse_log("int8_vs_fp32_fail.log")
        passing = parse_log("int8_vs_fp32_pass.log")

        fail_metrics = next(f for f in fail.facts if f.kind == "divergence_metrics")
        pass_metrics = next(f for f in passing.facts if f.kind == "divergence_metrics")
        assert fail_metrics.data["max_absdiff"] == pass_metrics.data["max_absdiff"]
        assert fail_metrics.data["max_reldiff"] == pass_metrics.data["max_reldiff"]

    def test_required_tolerance_is_identical_too(self):
        fail = parse_log("int8_vs_fp32_fail.log")
        passing = parse_log("int8_vs_fp32_pass.log")
        fail_req = next(f for f in fail.facts if f.kind == "required_tolerance")
        pass_req = next(f for f in passing.facts if f.kind == "required_tolerance")
        assert fail_req.data["req_abs"] == pass_req.data["req_abs"]

    def test_only_the_verdicts_differ(self):
        fail = parse_log("int8_vs_fp32_fail.log")
        passing = parse_log("int8_vs_fp32_pass.log")
        assert "output_mismatch" in kinds_in(fail)
        assert "output_mismatch" not in kinds_in(passing)
        assert "all_outputs_matched" in kinds_in(passing)
        assert "all_outputs_matched" not in kinds_in(fail)


class TestPassingComparison:
    LOG = "int8_vs_fp32_pass.log"

    def test_detects_match(self):
        facts = parse_log(self.LOG)
        match = next(f for f in facts.facts if f.kind == "output_match")
        assert match.data["output"] == "output"

    def test_no_failure_verdict_facts(self):
        # A parser that hallucinates failures in a passing run is exactly as
        # broken as a diagnoser that invents causes.
        facts = parse_log(self.LOG)
        failure_kinds = {
            "output_mismatch",
            "mismatched_outputs",
            "shape_mismatch",
            "nan_detected",
            "inf_detected",
            "validation_failed",
        }
        assert not (kinds_in(facts) & failure_kinds)


class TestCleanBaselineRun:
    def test_single_runner_run_has_no_comparison_facts(self):
        # No comparison happened at all, so there must be no divergence facts
        # of ANY kind — not even measurements.
        facts = parse_log("fp32_baseline_run.log")
        comparison_kinds = {
            "output_mismatch",
            "output_match",
            "divergence_metrics",
            "required_tolerance",
            "compared_output",
        }
        assert not (kinds_in(facts) & comparison_kinds)

    def test_still_records_the_passing_verdict(self):
        facts = parse_log("fp32_baseline_run.log")
        verdict = next(f for f in facts.facts if f.kind == "run_verdict")
        assert verdict.data["verdict"] == "PASSED"


class TestLayerwiseComparison:
    LOG = "int8_vs_fp32_layerwise.log"

    def test_identifies_the_earliest_diverging_tensor(self):
        # The actionable signal: everything downstream may just inherit this.
        facts = parse_log(self.LOG)
        roll_up = next(f for f in facts.facts if f.kind == "mismatched_outputs")
        assert roll_up.data["first_output"] == "/conv1/Conv_output_0"

    def test_counts_all_diverging_outputs(self):
        facts = parse_log(self.LOG)
        roll_up = next(f for f in facts.facts if f.kind == "mismatched_outputs")
        assert roll_up.data["count"] == 49
        assert len(roll_up.data["outputs"]) == 49

    def test_tensor_names_with_slashes_survive_parsing(self):
        # ONNX tensor names look like '/layer1/layer1.0/conv1/Conv_output_0';
        # a naive split on '.' or '/' would mangle them.
        facts = parse_log(self.LOG)
        roll_up = next(f for f in facts.facts if f.kind == "mismatched_outputs")
        assert "/layer1/layer1.0/conv1/Conv_output_0" in roll_up.data["outputs"]

    def test_block_context_attributes_metrics_to_the_right_tensor(self):
        # "Minimum Required Tolerance" lines don't name their tensor; the parser
        # must attribute them to the enclosing "Comparing Output" block.
        facts = parse_log(self.LOG)
        reqs = [f for f in facts.facts if f.kind == "required_tolerance"]
        assert len(reqs) == 49
        # First block in the log is the first conv.
        assert reqs[0].data["output"] == "/conv1/Conv_output_0"
        # Distinct tensors, not the same name repeated 49 times.
        assert len({r.data["output"] for r in reqs}) == 49


class TestShapeMismatch:
    LOG = "shape_mismatch.log"

    def test_detects_refused_comparison(self):
        facts = parse_log(self.LOG)
        assert "shape_mismatch" in kinds_in(facts)

    def test_records_both_shapes(self):
        facts = parse_log(self.LOG)
        detail = next(f for f in facts.facts if f.kind == "shape_detail")
        assert detail.data["shape0"] == "1, 1000"
        assert detail.data["shape1"] == "2, 1000"

    def test_is_not_reported_as_divergence(self):
        # Nothing was measured, so an accuracy claim would be unfounded.
        facts = parse_log(self.LOG)
        assert "output_mismatch" not in kinds_in(facts)
        assert "divergence_metrics" not in kinds_in(facts)


class TestNaNDetection:
    LOG = "nan_output.log"

    def test_detects_nan_and_inf(self):
        facts = parse_log(self.LOG)
        assert "nan_detected" in kinds_in(facts)
        assert "inf_detected" in kinds_in(facts)

    def test_records_which_output_failed_validation(self):
        facts = parse_log(self.LOG)
        failed = next(f for f in facts.facts if f.kind == "validation_failed")
        assert failed.data["output"] == "output"


class TestGroundingContract:
    def test_every_fact_is_traceable(self):
        for name in ("int8_vs_fp32_fail.log", "nan_output.log", "shape_mismatch.log"):
            facts = parse_log(name)
            assert facts.facts, f"{name} produced no facts"
            for f in facts.facts:
                assert f.source.startswith(f"{name}:")
                assert f.excerpt != ""
                assert int(f.source.rsplit(":", 1)[1]) >= 1

    def test_excerpt_appears_in_the_source_line(self):
        # The excerpt must be the real line, not a reconstruction.
        name = "int8_vs_fp32_fail.log"
        lines = (CORPUS / name).read_text().splitlines()
        for f in parse_log(name).facts:
            lineno = int(f.source.rsplit(":", 1)[1])
            assert f.excerpt == lines[lineno - 1].strip()

    def test_fact_ids_are_unique(self):
        facts = parse_log("int8_vs_fp32_layerwise.log")
        ids = [f.id for f in facts.facts]
        assert len(ids) == len(set(ids))


class TestParserProperties:
    def test_deterministic(self):
        a = parse_log("int8_vs_fp32_fail.log")
        b = parse_log("int8_vs_fp32_fail.log")
        assert a.model_dump() == b.model_dump()

    def test_empty_input_yields_no_facts(self):
        assert parser.parse_text("", artifact_name="empty.log").facts == []

    def test_garbage_input_yields_no_facts(self):
        facts = parser.parse_text(
            "hello world\nnot a polygraphy log\n42\n", artifact_name="junk.log"
        )
        assert facts.facts == []

    def test_backend_name(self):
        assert parse_log("int8_vs_fp32_fail.log").backend == "polygraphy"

    def test_convert_is_honestly_unimplemented(self):
        # Polygraphy compares; it does not convert. The stub says so.
        with pytest.raises(NotImplementedError, match="comparison tool"):
            parser.convert(Path("model.onnx"))


@pytest.mark.parametrize(
    "log",
    [
        "fp32_baseline_run.log",
        "int8_vs_fp32_fail.log",
        "int8_vs_fp32_pass.log",
        "shape_mismatch.log",
        "nan_output.log",
    ],
)
def test_snapshot(log, snapshot):
    """Golden-file layer: full parsed output of each corpus log is pinned.

    The layer-wise log is excluded deliberately — its 248 facts would make an
    unreviewable snapshot, and the explicit tests above cover it better.
    Regenerate intentionally with: uv run pytest --snapshot-update
    """
    assert parse_log(log).model_dump() == snapshot
