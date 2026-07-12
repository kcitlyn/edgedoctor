"""Tests for the TensorRT log parser.

Two layers of protection:
  1. Explicit assertions — pin the *semantics* (this log yields an
     unsupported_op fact naming GridSample).
  2. syrupy snapshots — pin the *entire output* (any change to what gets
     extracted from a fixture shows up as a reviewable diff; regenerate
     deliberately with `pytest --snapshot-update`).

Fixtures are synthetic-but-verified (see tests/fixtures/tensorrt/README.md);
real ThinkPad logs will be added to the corpus and tested as they arrive.
"""

from pathlib import Path

import pytest

from edgedoctor.backends.tensorrt import TensorRTBackend

FIXTURES = Path(__file__).parent / "fixtures" / "tensorrt"

parser = TensorRTBackend()


def parse_fixture(name: str):
    return parser.parse(FIXTURES / name)


class TestUnsupportedOpTRT8:
    def test_detects_unsupported_op(self):
        facts = parse_fixture("unsupported_op_trt8.log")
        ops = [f for f in facts.facts if f.kind == "unsupported_op"]
        assert len(ops) == 1
        assert ops[0].data["op"] == "GridSample"

    def test_detects_plugin_not_found(self):
        facts = parse_fixture("unsupported_op_trt8.log")
        plugins = [f for f in facts.facts if f.kind == "plugin_not_found"]
        assert len(plugins) == 1
        assert plugins[0].data["plugin"] == "GridSample"

    def test_detects_old_format_parse_error(self):
        facts = parse_fixture("unsupported_op_trt8.log")
        errors = [f for f in facts.facts if f.kind == "parse_error_node"]
        assert len(errors) == 1
        assert errors[0].data["idx"] == "42"
        assert errors[0].data["op"] == "GridSample"

    def test_detects_failed_verdict(self):
        facts = parse_fixture("unsupported_op_trt8.log")
        verdicts = [f for f in facts.facts if f.kind == "run_verdict"]
        assert len(verdicts) == 1
        assert verdicts[0].data["verdict"] == "FAILED"

    def test_every_fact_is_traceable(self):
        # The grounding contract: every fact cites file:line and carries the
        # verbatim excerpt it was extracted from.
        facts = parse_fixture("unsupported_op_trt8.log")
        for f in facts.facts:
            assert f.source.startswith("unsupported_op_trt8.log:")
            assert f.excerpt != ""
            lineno = int(f.source.rsplit(":", 1)[1])
            assert lineno >= 1


class TestParseErrorTRT10:
    def test_detects_new_format_node_error(self):
        facts = parse_fixture("parse_error_trt10.log")
        errors = [f for f in facts.facts if f.kind == "parse_error_node"]
        assert len(errors) == 1
        assert errors[0].data["idx"] == "444"
        assert errors[0].data["op"] == "Clip"
        assert errors[0].data["code"] == "INVALID_NODE"


class TestBuildFailureTRT10:
    def test_detects_no_implementation(self):
        facts = parse_fixture("build_failure_trt10.log")
        no_impl = [f for f in facts.facts if f.kind == "no_implementation"]
        assert len(no_impl) == 1
        assert "LayerNormalization" in no_impl[0].data["node"]

    def test_detects_tactic_skip(self):
        facts = parse_fixture("build_failure_trt10.log")
        skips = [f for f in facts.facts if f.kind == "tactic_skipped"]
        assert len(skips) == 1
        assert skips[0].data["tactic"] == "0x0000000000000000"

    def test_detects_error_codes(self):
        # The "Error Code 10" line also contains the no-implementation string;
        # one-fact-per-line + most-specific-first means it surfaces as
        # no_implementation (more actionable), so only Error Code 2 remains
        # as a bare trt_error_code fact.
        facts = parse_fixture("build_failure_trt10.log")
        codes = {f.data["code"] for f in facts.facts if f.kind == "trt_error_code"}
        assert codes == {"2"}


class TestSuccessLog:
    def test_no_failure_facts_in_clean_log(self):
        # A parser that hallucinates failures in a passing run is exactly as
        # broken as a diagnoser that invents causes.
        facts = parse_fixture("success.log")
        failure_kinds = {
            "unsupported_op",
            "plugin_not_found",
            "parse_error_node",
            "no_implementation",
            "trt_error_code",
            "tactic_skipped",
        }
        failures = [f for f in facts.facts if f.kind in failure_kinds]
        assert failures == []

    def test_passed_verdict(self):
        facts = parse_fixture("success.log")
        verdicts = [f for f in facts.facts if f.kind == "run_verdict"]
        assert len(verdicts) == 1
        assert verdicts[0].data["verdict"] == "PASSED"

    def test_version_still_extracted(self):
        facts = parse_fixture("success.log")
        versions = [f for f in facts.facts if f.kind == "trt_version"]
        assert versions and versions[0].data["version"] == "10.7.0"


class TestParserProperties:
    def test_deterministic(self):
        # Same log in → same Facts out, byte-for-byte.
        a = parse_fixture("unsupported_op_trt8.log")
        b = parse_fixture("unsupported_op_trt8.log")
        assert a.model_dump() == b.model_dump()

    def test_empty_input_yields_no_facts(self):
        facts = parser.parse_text("", artifact_name="empty.log")
        assert facts.facts == []

    def test_garbage_input_yields_no_facts(self):
        facts = parser.parse_text(
            "hello world\nthis is not a trt log\n12345\n", artifact_name="junk.log"
        )
        assert facts.facts == []


@pytest.mark.parametrize(
    "fixture",
    [
        "unsupported_op_trt8.log",
        "parse_error_trt10.log",
        "build_failure_trt10.log",
        "success.log",
    ],
)
def test_snapshot(fixture, snapshot):
    """Golden-file layer: the full parsed output of every fixture is pinned.

    If a signature change alters ANY extraction, this fails with a diff.
    Regenerate intentionally with: uv run pytest --snapshot-update
    """
    facts = parse_fixture(fixture)
    assert facts.model_dump() == snapshot
