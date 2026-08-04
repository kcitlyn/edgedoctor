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


def codes_of(facts) -> list[str]:
    """Rule codes produced for an already-parsed Facts object."""
    return codes(diagnose(facts))


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


class TestSingleIterationTrace:
    """A REAL one-iteration profile, not a hand-built one.

    ED0502 and its suppression of the cost rules previously had no real-artifact
    coverage — the threshold was only ever checked against a synthetic Facts
    object, so it was tested against an assumption about what ORT emits.
    """

    TRACE = "ort_profile_one_iteration.json"

    def test_reports_a_single_iteration(self):
        facts = parse_trace(self.TRACE)
        assert fact_of(facts, "profile_summary").data["iterations"] == 1

    def test_flags_the_thin_sample(self):
        assert "few_iterations" in kinds_in(parse_trace(self.TRACE))

    def test_warns_and_declines_to_attribute_cost(self):
        codes = codes_of(parse_trace(self.TRACE))
        assert "ED0502" in codes, "must warn about warm-up-dominated timings"
        # The whole point: no cost attribution from a one-run sample, even
        # though a dominant op is clearly present in the data.
        assert "ED0501" not in codes
        assert "ED0503" not in codes
        assert "ED0504" not in codes

    def test_the_dominant_op_is_still_measured_just_not_diagnosed(self):
        # The parser records what it saw; only the RULES decline to conclude.
        facts = parse_trace(self.TRACE)
        assert any(f.kind == "op_cost" for f in facts.facts)


class TestBroadFallbackProfile:
    """The measured counterpart to ort_many_fallback.log.

    Gives ED0503 (>=10% of node time on CPU) its first real-log coverage. The
    two-partition trace sits at ~1% CPU and correctly does NOT warn, so without
    this artifact the threshold was never exercised in the firing direction.
    """

    TRACE = "ort_many_fallback_profile.json"

    def test_a_material_share_of_time_is_on_cpu(self):
        facts = parse_trace(self.TRACE)
        split = fact_of(facts, "provider_time_split")
        assert split.data["cpu_share_pct"] >= 10, (
            f"expected a material CPU share, got {split.data['cpu_share_pct']}%"
        )

    def test_warns_about_the_measured_cost_of_the_split(self):
        assert "ED0503" in codes_of(parse_trace(self.TRACE))

    def test_both_providers_appear_in_the_trace(self):
        facts = parse_trace(self.TRACE)
        assert fact_of(facts, "provider_time_split").data["provider_count"] >= 2

    def test_contrasts_with_the_low_cpu_share_trace(self):
        # The pair is the point: same KIND of split, opposite verdicts, decided
        # by measured cost rather than by structure.
        heavy = fact_of(parse_trace(self.TRACE), "provider_time_split")
        light = fact_of(parse_trace(SPLIT_TRACE), "provider_time_split")
        assert heavy.data["cpu_share_pct"] > light.data["cpu_share_pct"]
        assert "ED0503" in codes_of(parse_trace(self.TRACE))
        assert "ED0503" not in codes_of(parse_trace(SPLIT_TRACE))


class TestReadableOpIsConservative:
    """Only genuine EP-subgraph hashes may be relabelled.

    An earlier heuristic ("contains a long digit run") rewrote the ordinary op
    name `a_1234567890_b` into "a compiled subgraph #?", inventing a fused
    subgraph the model doesn't contain. Renaming a real operator is worse than
    showing a hash: the user goes hunting for a node that isn't there.
    """

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1234567890_TensorRT_1234567890_2", "TensorRT compiled subgraph #2"),
            ("7615378459790495232_CoreML_7615378459790495232_0",
             "CoreML compiled subgraph #0"),
            ("CoreMLExecutionProvider_123456789_CoreML_123456789_0_0",
             "CoreML compiled subgraph #0"),
        ],
    )
    def test_genuine_subgraph_names_are_relabelled(self, raw, expected):
        assert _readable_op(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "Conv", "FusedConv", "Relu", "/conv1/Conv",
            "/model/layers.0/attn/MatMul",     # real ONNX naming, has digits+dots
            "node_1", "_1", "123",
            "a_1234567890_b",                  # the regression: NOT a subgraph
            "x_9999999_y_8888888_z",           # two hashes but wrong shape
            "",
        ],
    )
    def test_real_names_are_never_rewritten(self, raw):
        assert _readable_op(raw) == raw


class TestNonMeasurementsAreRejected:
    """A duration must be a real, non-negative measurement.

    A negative duration would pollute the total that every percentage share is
    computed against, quietly making all of them wrong.
    """

    def _trace(self, dur, cat="Node"):
        name = "/a_kernel_time" if cat == "Node" else "session_initialization"
        event = {"cat": cat, "name": name, "dur": dur}
        if cat == "Node":
            event["args"] = {"op_name": "Conv", "provider": "CPUExecutionProvider"}
        return json.dumps([event] * 5)

    @pytest.mark.parametrize("dur", [-1, -1000, -0.5])
    def test_negative_node_durations_are_skipped(self, dur):
        facts = parser.parse_text(self._trace(dur), artifact_name="t.json")
        assert facts.facts == []

    @pytest.mark.parametrize("dur", [-1, -100])
    def test_negative_session_durations_are_skipped(self, dur):
        facts = parser.parse_text(self._trace(dur, cat="Session"),
                                  artifact_name="t.json")
        assert facts.facts == []

    def test_boolean_durations_are_skipped(self):
        # bool is a subclass of int in Python, so `True` would otherwise be
        # accepted as a 1-microsecond measurement.
        facts = parser.parse_text(self._trace(True), artifact_name="t.json")
        assert facts.facts == []

    def test_zero_duration_produces_no_bogus_shares(self):
        # A total of zero would make every share a division by zero.
        facts = parser.parse_text(self._trace(0), artifact_name="t.json")
        assert "op_cost" not in kinds_in(facts)

    def test_positive_durations_still_work(self):
        facts = parser.parse_text(self._trace(10), artifact_name="t.json")
        assert "op_cost" in kinds_in(facts)


class TestMalformedTraceStructures:
    """Realistic corruption: a truncated write, a partial upload, an SDK change."""

    @pytest.mark.parametrize(
        "payload",
        [
            "[]",
            "{}",
            '{"traceEvents": []}',
            '{"foo": 1}',
            "null",
            "[null, null]",
            '["a string", 42]',
            '[{"cat": "Node"}]',                       # no dur, no name
            '[{"cat": "Node", "dur": "abc", "name": "x_kernel_time"}]',
            '[{"cat": "Unknown", "dur": 5, "name": "x"}]',
            '[{"name": "x_kernel_time", "dur": 5}]',   # no cat
        ],
    )
    def test_never_raises(self, payload):
        # The contract is total: any JSON shape returns a valid Facts object.
        facts = parser.parse_text(payload, artifact_name="t.json")
        assert facts.backend == "ort_profile"
        assert isinstance(facts.facts, list)

    def test_truncated_json_yields_no_facts(self):
        # A trace written by a process that was killed mid-write.
        assert parser.parse_text('[{"cat": "Node", "dur": 1', artifact_name="t.json").facts == []

    def test_nodes_without_args_are_still_counted(self):
        # args carries op_name/provider; without it the timing is still real, so
        # it must contribute to the total rather than being silently dropped.
        trace = json.dumps([
            {"cat": "Node", "name": "/a_kernel_time", "dur": 10} for _ in range(5)
        ])
        facts = parser.parse_text(trace, artifact_name="t.json")
        assert fact_of(facts, "profile_summary").data["total_node_us"] == 50
