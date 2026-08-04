"""Tests for the ONNX Runtime placement/fallback parser.

Run against REAL logs in corpus/onnxruntime/, generated on this machine by
scripts/make_ort_corpus.py against a genuine CoreML EP.

The most important test here is TestRequestedVsActual. ort_cpu_only.log and
ort_missing_provider.log contain the SAME placement line ("All nodes placed on
[CPUExecutionProvider]"), but one is a deliberate CPU session and the other is a
silently-degraded accelerator run. The parser must distinguish them, because no
rule can if the parser doesn't.
"""

from pathlib import Path

import pytest

from edgedoctor.backends.onnxruntime import OnnxRuntimeBackend

CORPUS = Path(__file__).parent.parent / "corpus" / "onnxruntime"

parser = OnnxRuntimeBackend()


def parse_log(name: str):
    return parser.parse(CORPUS / name)


def kinds_in(facts) -> set[str]:
    return {f.kind for f in facts.facts}


def fact_of(facts, kind):
    return next(f for f in facts.facts if f.kind == kind)


class TestPartialFallback:
    LOG = "ort_partial_fallback.log"

    def test_detects_split_execution(self):
        facts = parse_log(self.LOG)
        split = fact_of(facts, "split_execution")
        assert split.data["providers"] == [
            "CPUExecutionProvider", "CoreMLExecutionProvider"
        ]

    def test_names_the_ops_that_fell_back(self):
        # The single most actionable fact: WHICH ops to replace.
        facts = parse_log(self.LOG)
        ops = fact_of(facts, "cpu_fallback_ops")
        assert ops.data["ops"] == ["Erf", "Round"]
        assert ops.data["count"] == 2

    def test_records_partition_count(self):
        # Partition count matters more than node count — each boundary is a
        # synchronization point.
        facts = parse_log(self.LOG)
        cap = fact_of(facts, "provider_capability")
        assert cap.data["partitions"] == 2
        assert cap.data["supported"] == 3
        assert cap.data["graph_nodes"] == 5
        assert cap.data["unsupported"] == 2

    def test_records_both_placement_groups(self):
        facts = parse_log(self.LOG)
        placements = [f for f in facts.facts if f.kind == "node_placement"]
        providers = {f.data["provider"] for f in placements}
        assert providers == {"CoreMLExecutionProvider", "CPUExecutionProvider"}

    def test_skips_synthetic_ep_subgraph_names(self):
        # EP-compiled partitions are named with hashes like
        # '7615..._CoreML_7615..._0'. Reporting those as "the op that fell back"
        # would be noise, so they must not appear as ops.
        facts = parse_log(self.LOG)
        for f in facts.facts:
            if f.kind == "node_on_provider":
                assert not f.data["op"][0].isdigit()


class TestRequestedVsActual:
    """The honesty invariant: identical placement, opposite meanings.

    Both logs below end with every node on CPU. One asked for CPU; the other
    asked for TensorRT and silently didn't get it. If these two ever become
    indistinguishable, edgedoctor will either miss real degradation or cry wolf
    on a deliberate CPU run.
    """

    def test_deliberate_cpu_run_has_no_fallback_facts(self):
        facts = parse_log("ort_cpu_only.log")
        assert "silent_cpu_only" not in kinds_in(facts)
        assert "split_execution" not in kinds_in(facts)
        assert "provider_unavailable" not in kinds_in(facts)

    def test_silently_degraded_run_is_flagged(self):
        facts = parse_log("ort_missing_provider.log")
        silent = fact_of(facts, "silent_cpu_only")
        assert silent.data["requested"] == "TensorrtExecutionProvider"
        assert silent.data["actual"] == ["CPUExecutionProvider"]

    def test_both_logs_share_the_same_placement_line(self):
        # Proves the two cases really are indistinguishable from placement
        # alone — i.e. that the test above is testing something real.
        cpu = fact_of(parse_log("ort_cpu_only.log"), "all_nodes_one_provider")
        missing = fact_of(
            parse_log("ort_missing_provider.log"), "all_nodes_one_provider"
        )
        assert cpu.data["provider"] == missing.data["provider"] == "CPUExecutionProvider"

    def test_records_what_was_requested_and_available(self):
        facts = parse_log("ort_missing_provider.log")
        unavail = fact_of(facts, "provider_unavailable")
        assert unavail.data["requested"] == "TensorrtExecutionProvider"
        assert "CPUExecutionProvider" in unavail.data["available"]

    def test_records_the_providers_actually_obtained(self):
        facts = parse_log("ort_missing_provider.log")
        got = fact_of(facts, "session_providers")
        assert got.data["providers"] == ["CPUExecutionProvider"]


class TestCleanSingleProvider:
    LOG = "ort_all_nodes_one_ep.log"

    def test_detects_single_provider_placement(self):
        facts = parse_log(self.LOG)
        placed = fact_of(facts, "all_nodes_one_provider")
        assert placed.data["provider"] == "CoreMLExecutionProvider"

    def test_whole_graph_claimed_in_one_partition(self):
        facts = parse_log(self.LOG)
        cap = fact_of(facts, "provider_capability")
        assert cap.data["partitions"] == 1
        assert cap.data["supported"] == cap.data["graph_nodes"] == 49
        assert cap.data["unsupported"] == 0

    def test_no_fallback_facts_in_clean_log(self):
        # The honesty test: a parser that reports fallback on a fully-accelerated
        # graph is inventing a failure.
        facts = parse_log(self.LOG)
        failure_kinds = {
            "split_execution", "cpu_fallback_ops", "silent_cpu_only",
            "provider_unavailable", "session_failed",
        }
        assert not (kinds_in(facts) & failure_kinds)


class TestGroundingContract:
    LOGS = [
        "ort_all_nodes_one_ep.log",
        "ort_partial_fallback.log",
        "ort_cpu_only.log",
        "ort_missing_provider.log",
    ]

    def test_every_fact_is_traceable(self):
        for name in self.LOGS:
            facts = parse_log(name)
            assert facts.facts, f"{name} produced no facts"
            for f in facts.facts:
                assert f.source.startswith(f"{name}:")
                assert f.excerpt != ""
                assert int(f.source.rsplit(":", 1)[1]) >= 1

    def test_excerpt_matches_the_cited_line(self):
        # Including DERIVED facts: split_execution and cpu_fallback_ops are
        # computed in a second pass, but must still cite a real line whose text
        # the user can go and read.
        for name in self.LOGS:
            lines = (CORPUS / name).read_text(errors="replace").splitlines()
            for f in parse_log(name).facts:
                lineno = int(f.source.rsplit(":", 1)[1])
                assert f.excerpt == lines[lineno - 1].strip(), (
                    f"{name}:{lineno} excerpt does not match the source line"
                )

    def test_fact_ids_are_unique(self):
        for name in self.LOGS:
            ids = [f.id for f in parse_log(name).facts]
            assert len(ids) == len(set(ids)), f"{name} has duplicate fact ids"


class TestParserProperties:
    def test_deterministic(self):
        a = parse_log("ort_partial_fallback.log")
        b = parse_log("ort_partial_fallback.log")
        assert a.model_dump() == b.model_dump()

    def test_empty_input_yields_no_facts(self):
        assert parser.parse_text("", artifact_name="empty.log").facts == []

    def test_garbage_input_yields_no_facts(self):
        facts = parser.parse_text(
            "hello world\nnot an ort log\n42\n", artifact_name="junk.log"
        )
        assert facts.facts == []

    def test_backend_name(self):
        assert parse_log("ort_cpu_only.log").backend == "onnxruntime"

    def test_convert_is_honestly_unimplemented(self):
        with pytest.raises(NotImplementedError, match="no conversion"):
            parser.convert(Path("model.onnx"))

    def test_module_does_not_shadow_the_real_onnxruntime(self):
        # The module is named onnxruntime.py, same as the pip package. Relative
        # imports keep them separate, but a regression here would be subtle and
        # would break the corpus generator rather than the parser.
        #
        # Skipped when the real package is absent (it lives in the [corpus]
        # extra, which CI doesn't install): with nothing to shadow, the test has
        # nothing to prove. It still guards the machines that regenerate the
        # corpus, which is exactly where the bug would bite.
        real_ort = pytest.importorskip(
            "onnxruntime", reason="shadowing can only be checked when ORT is installed"
        )

        from edgedoctor.backends.onnxruntime import OnnxRuntimeBackend as Ours

        assert Ours().name == "onnxruntime"
        assert hasattr(real_ort, "InferenceSession")


class TestRegistry:
    def test_backend_resolves_through_the_registry(self):
        from edgedoctor.backends import PARSER_REGISTRY, get_parser

        assert "onnxruntime" in PARSER_REGISTRY
        assert get_parser("onnxruntime").name == "onnxruntime"


@pytest.mark.parametrize(
    "log",
    [
        "ort_all_nodes_one_ep.log",
        "ort_partial_fallback.log",
        "ort_cpu_only.log",
        "ort_missing_provider.log",
    ],
)
def test_snapshot(log, snapshot):
    """Golden-file layer. Regenerate deliberately: uv run pytest --snapshot-update"""
    assert parse_log(log).model_dump() == snapshot


class TestSessionFailure:
    """A real corrupt-model log — the one ORT case where the session fails.

    Added because ED0304 and ED0305 referenced `session_failed` while no real
    artifact produced it, so both rules were only ever tested against hand-built
    Facts.
    """

    LOG = "ort_session_failed.log"

    def test_detects_the_failure(self):
        facts = parse_log(self.LOG)
        assert "session_failed" in kinds_in(facts)

    def test_records_the_error_text(self):
        facts = parse_log(self.LOG)
        assert fact_of(facts, "session_failed").data["error"]

    def test_no_placement_facts_when_the_session_never_started(self):
        # Nothing was placed, so any provider or fallback claim would be
        # unfounded.
        facts = parse_log(self.LOG)
        placement = {"node_placement", "all_nodes_one_provider", "split_execution",
                     "cpu_fallback_ops", "silent_cpu_only"}
        assert not (kinds_in(facts) & placement)

    def test_diagnoses_as_ed0304_only(self):
        from edgedoctor.diagnoser import diagnose

        codes = [d.code for d in diagnose(parse_log(self.LOG))]
        assert "ED0304" in codes
        # Must NOT claim a clean single-provider run.
        assert "ED0305" not in codes


class TestBroadFallback:
    """A real log where five DISTINCT ops fall back across five partitions.

    This is what exercises ED0303's >=3-distinct-ops threshold against reality;
    previously only a hand-built Facts object did.
    """

    LOG = "ort_many_fallback.log"

    def test_names_every_op_that_fell_back(self):
        facts = parse_log(self.LOG)
        ops = fact_of(facts, "cpu_fallback_ops")
        assert ops.data["ops"] == ["Ceil", "Erf", "Floor", "Round", "Sign"]
        assert ops.data["count"] == 5

    def test_reports_many_partitions(self):
        # Partition count is the performance-relevant number, and five
        # boundaries is qualitatively worse than two.
        facts = parse_log(self.LOG)
        assert fact_of(facts, "provider_capability").data["partitions"] == 5

    def test_accelerator_claims_only_half_the_graph(self):
        facts = parse_log(self.LOG)
        cap = fact_of(facts, "provider_capability")
        assert cap.data["unsupported"] == 5
        assert cap.data["supported"] == 5

    def test_fires_the_broad_fallback_rule(self):
        from edgedoctor.diagnoser import diagnose

        codes = [d.code for d in diagnose(parse_log(self.LOG))]
        assert "ED0303" in codes, "broad fallback should be diagnosed"
        assert "ED0301" in codes, "and it is still a split graph"
