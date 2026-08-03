"""Tests for the ONNX Runtime profiling-JSON parser and its ED05xx rules.

Run against real traces in corpus/onnxruntime/, produced by
scripts/make_ort_corpus.py with enable_profiling=True.

The theme throughout: profiling data invites invented causation. A share is a
measurement; a bottleneck is a conclusion. These tests pin that the parser never
crosses that line, that every percentage travels with its denominator, and that
the tool declines to attribute cost from a warm-up-dominated sample.
"""

import json
from pathlib import Path

import pytest

from edgedoctor.backends.base import Fact, Facts
from edgedoctor.backends.ort_profile import (
    MIN_RELIABLE_ITERATIONS,
    OrtProfileBackend,
    _readable_op,
)
from edgedoctor.diagnoser import diagnose

CORPUS = Path(__file__).parent.parent / "corpus" / "onnxruntime"

parser = OrtProfileBackend()

CPU_TRACE = "ort_profile_cpu.json"
SPLIT_TRACE = "ort_profile_split.json"


def parse_trace(name: str):
    return parser.parse(CORPUS / name)


def kinds_in(facts) -> set[str]:
    return {f.kind for f in facts.facts}


def fact_of(facts, kind):
    return next(f for f in facts.facts if f.kind == kind)


def codes(diagnoses) -> list[str]:
    return [d.code for d in diagnoses]


class TestParsesRealTrace:
    def test_reports_session_phases(self):
        facts = parse_trace(CPU_TRACE)
        phases = [f for f in facts.facts if f.kind == "session_phase"]
        assert {p.data["phase"] for p in phases} >= {
            "model_loading_uri", "session_initialization"
        }

    def test_aggregates_op_cost(self):
        facts = parse_trace(CPU_TRACE)
        conv = fact_of(facts, "op_cost")
        # ResNet18 on CPU: convolution dominates, as expected.
        assert conv.data["op"] == "FusedConv"
        assert conv.data["share_pct"] > 50
        assert conv.data["calls"] > 1

    def test_reports_iteration_count(self):
        facts = parse_trace(CPU_TRACE)
        summary = fact_of(facts, "profile_summary")
        # The generator runs 5 iterations.
        assert summary.data["iterations"] == 5

    def test_identifies_the_slowest_node(self):
        facts = parse_trace(CPU_TRACE)
        slowest = fact_of(facts, "slowest_node")
        assert slowest.data["us"] > 0
        assert slowest.data["provider"]


class TestPercentagesCarryTheirDenominator:
    """A share without its base is uninterpretable, so it must travel together."""

    def test_op_cost_includes_the_total(self):
        facts = parse_trace(CPU_TRACE)
        for f in (x for x in facts.facts if x.kind == "op_cost"):
            assert "total_node_ms" in f.data
            assert f.data["total_node_ms"] > 0

    def test_slowest_node_includes_the_total(self):
        assert "total_node_ms" in fact_of(parse_trace(CPU_TRACE), "slowest_node").data

    def test_message_states_both_numbers(self):
        # The rendered diagnosis must show "X ms of Y ms", not a bare percentage.
        diag = next(d for d in diagnose(parse_trace(CPU_TRACE)) if d.code == "ED0501")
        assert " of " in diag.message
        assert "ms" in diag.message

    def test_shares_sum_to_at_most_100(self):
        facts = parse_trace(CPU_TRACE)
        total = sum(f.data["share_pct"] for f in facts.facts if f.kind == "op_cost")
        assert total <= 100.01  # float tolerance


class TestProviderSplit:
    def test_split_trace_reports_a_provider_breakdown(self):
        facts = parse_trace(SPLIT_TRACE)
        split = fact_of(facts, "provider_time_split")
        assert split.data["provider_count"] >= 2
        assert "cpu_share_pct" in split.data

    def test_single_provider_trace_has_no_split_fact(self):
        # Reporting a "split" for a one-provider run would be inventing one.
        assert "provider_time_split" not in kinds_in(parse_trace(CPU_TRACE))

    def test_small_cpu_share_does_not_warn(self):
        # On this trace CPU is <1% of node time. Warning about a 1% split would
        # be crying wolf; the threshold exists precisely for this.
        facts = parse_trace(SPLIT_TRACE)
        if fact_of(facts, "provider_time_split").data["cpu_share_pct"] < 10:
            assert "ED0503" not in codes(diagnose(facts))

    def test_large_cpu_share_does_warn(self):
        facts = Facts(
            backend="ort_profile", artifact_path="t.json",
            facts=[
                Fact(id="f1", kind="provider_time_split", summary="s",
                     source="t.json:summary", excerpt="x",
                     data={"provider_count": 2, "cpu_share_pct": 45.0,
                           "cpu_us": 4500, "total_node_ms": 10.0}),
            ],
        )
        assert "ED0503" in codes(diagnose(facts))


class TestDeclinesToAttributeFromWarmUp:
    """The core honesty rule of this family."""

    def _thin(self, iterations: int) -> Facts:
        return Facts(
            backend="ort_profile", artifact_path="t.json",
            facts=[
                Fact(id="f1", kind="few_iterations", summary="s",
                     source="t.json:summary", excerpt="x",
                     data={"iterations": iterations,
                           "minimum": MIN_RELIABLE_ITERATIONS}),
                Fact(id="f2", kind="profile_summary", summary="s",
                     source="t.json:summary", excerpt="x",
                     data={"iterations": iterations, "total_node_ms": 5.0}),
                Fact(id="f3", kind="op_cost", summary="s", source="t.json:op:Conv",
                     excerpt="x",
                     data={"op": "Conv", "share_pct": 90.0, "ms": 4.5,
                           "total_node_ms": 5.0, "calls": 1}),
            ],
        )

    def test_single_iteration_warns(self):
        assert "ED0502" in codes(diagnose(self._thin(1)))

    def test_single_iteration_suppresses_the_cost_attribution(self):
        # A 90% share from one warm-up run is not a finding, so ED0501 must not
        # fire even though its own conditions are met.
        assert "ED0501" not in codes(diagnose(self._thin(1)))

    def test_sufficient_iterations_allow_attribution(self):
        facts = parse_trace(CPU_TRACE)
        assert "few_iterations" not in kinds_in(facts)
        assert "ED0501" in codes(diagnose(facts))

    def test_parser_flags_thin_samples(self):
        trace = json.dumps([
            {"cat": "Node", "name": "/a/Conv_kernel_time", "dur": 100,
             "args": {"op_name": "Conv", "provider": "CPUExecutionProvider"}},
        ])
        facts = parser.parse_text(trace, artifact_name="t.json")
        assert "few_iterations" in kinds_in(facts)


class TestReadableOpNames:
    @pytest.mark.parametrize(
        "raw",
        [
            "7615378459790495232_CoreML_7615378459790495232_0",
            "CoreMLExecutionProvider_7615378459790495232_CoreML_7615378459790495232_1_1",
        ],
    )
    def test_hashed_subgraph_names_are_made_legible(self, raw):
        readable = _readable_op(raw)
        assert "CoreML" in readable
        assert "compiled subgraph" in readable
        assert "7615378459790495232" not in readable

    @pytest.mark.parametrize("real", ["FusedConv", "Conv", "/conv1/Conv", "Relu"])
    def test_real_op_names_pass_through_unchanged(self, real):
        assert _readable_op(real) == real

    def test_raw_name_is_retained_for_traceability(self):
        # The readable label is for humans; the original must stay checkable.
        facts = parse_trace(SPLIT_TRACE)
        costs = [f for f in facts.facts if f.kind == "op_cost"]
        assert any("raw_op" in f.data for f in costs)


class TestGroundingContract:
    def test_every_fact_cites_a_location(self):
        for name in (CPU_TRACE, SPLIT_TRACE):
            facts = parse_trace(name)
            assert facts.facts
            for f in facts.facts:
                assert f.source.startswith(f"{name}:")
                assert f.excerpt != ""

    def test_event_citations_point_at_real_events(self):
        # `events[N]` must be a valid index into the trace, or the citation is
        # decorative rather than checkable.
        name = CPU_TRACE
        events = json.loads((CORPUS / name).read_text())
        for f in parse_trace(name).facts:
            ref = f.source.split(":", 1)[1]
            if ref.startswith("events["):
                idx = int(ref[len("events["):-1])
                assert 0 <= idx < len(events)

    def test_fact_ids_are_unique(self):
        for name in (CPU_TRACE, SPLIT_TRACE):
            ids = [f.id for f in parse_trace(name).facts]
            assert len(ids) == len(set(ids))

    def test_diagnoses_have_resolved_placeholders(self):
        for name in (CPU_TRACE, SPLIT_TRACE):
            for d in diagnose(parse_trace(name)):
                assert "{" not in d.message

    def test_evidence_ids_point_to_real_facts(self):
        for name in (CPU_TRACE, SPLIT_TRACE):
            facts = parse_trace(name)
            ids = {f.id for f in facts.facts}
            for d in diagnose(facts):
                for eid in d.evidence:
                    assert eid in ids


class TestParserProperties:
    def test_deterministic(self):
        assert parse_trace(CPU_TRACE).model_dump() == parse_trace(CPU_TRACE).model_dump()

    def test_non_json_input_yields_no_facts(self):
        # Must not raise: a wrong-format artifact is "nothing matched".
        assert parser.parse_text("not json at all", artifact_name="x.json").facts == []

    def test_empty_input_yields_no_facts(self):
        assert parser.parse_text("", artifact_name="x.json").facts == []

    def test_empty_json_list_yields_no_facts(self):
        assert parser.parse_text("[]", artifact_name="x.json").facts == []

    def test_json_object_without_events_yields_no_facts(self):
        assert parser.parse_text('{"foo": 1}', artifact_name="x.json").facts == []

    def test_chrome_trace_envelope_is_accepted(self):
        trace = json.dumps({"traceEvents": [
            {"cat": "Session", "name": "session_initialization", "dur": 5000},
        ]})
        assert parser.parse_text(trace, artifact_name="x.json").facts

    def test_malformed_events_are_skipped_not_fatal(self):
        trace = json.dumps([
            "not a dict",
            {"cat": "Node"},                                  # no dur
            {"cat": "Node", "dur": "abc", "name": "x_kernel_time"},  # bad dur
            {"cat": "Node", "dur": 10, "name": "/a/Conv_kernel_time",
             "args": {"op_name": "Conv", "provider": "CPUExecutionProvider"}},
        ])
        facts = parser.parse_text(trace, artifact_name="x.json")
        assert "profile_summary" in kinds_in(facts)

    def test_backend_name(self):
        assert parse_trace(CPU_TRACE).backend == "ort_profile"

    def test_convert_is_honestly_unimplemented(self):
        with pytest.raises(NotImplementedError, match="doesn't create models"):
            parser.convert(Path("m.onnx"))

    def test_registered(self):
        from edgedoctor.backends import PARSER_REGISTRY, get_parser

        assert "ort_profile" in PARSER_REGISTRY
        assert get_parser("ort_profile").name == "ort_profile"


@pytest.mark.parametrize("trace", [CPU_TRACE, SPLIT_TRACE])
def test_snapshot(trace, snapshot):
    assert parse_trace(trace).model_dump() == snapshot
