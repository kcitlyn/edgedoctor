"""Live-API smoke test for the LLM synthesis layer. SKIPPED by default.

Everything in tests/test_llm.py is hermetic and proves the logic we own: the
grounding gate, degradation, trust labelling. What it CANNOT prove is that our
call actually works against the real API — that the SDK surface we coded to
(`messages.parse(output_format=...)` returning `parsed_output` on a content
block) is real, and that a live model returns something our validation accepts.

That gap is what this file closes. It is skipped unless BOTH:
  - ANTHROPIC_API_KEY is set, and
  - EDGEDOCTOR_LIVE_TESTS=1

The second gate is deliberate: a developer with a key exported for unrelated
reasons must not start paying for test runs by surprise, and CI must never
silently begin making network calls. Opt-in twice, on purpose.

Run it:
    EDGEDOCTOR_LIVE_TESTS=1 uv run pytest tests/test_llm_live.py -v

Cost: a handful of Haiku calls on a few hundred tokens — well under a cent.
"""

import os

import pytest

from edgedoctor.backends.base import Diagnosis, Fact, Facts
from edgedoctor.llm import synthesize

pytestmark = pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("EDGEDOCTOR_LIVE_TESTS")),
    reason="live API test: set ANTHROPIC_API_KEY and EDGEDOCTOR_LIVE_TESTS=1",
)


@pytest.fixture
def anthropic_installed():
    pytest.importorskip("anthropic", reason='needs: pip install "edgedoctor[llm]"')


def facts_with_unexplained_error() -> Facts:
    """Facts describing a real failure class no ED rule covers yet.

    Deliberately a genuine TensorRT problem (a plugin ABI/version mismatch)
    rather than nonsense, so a sensible diagnosis is actually possible and the
    test measures grounding rather than the model's tolerance for gibberish.
    """
    return Facts(
        backend="tensorrt",
        artifact_path="live_smoke.log",
        facts=[
            Fact(
                id="f1",
                kind="plugin_version_mismatch",
                summary="Plugin 'MyCustomOp' version 2 requested, version 1 registered",
                source="live_smoke.log:88",
                excerpt="[E] [TRT] IPluginRegistry::getCreator: Error Code 4: "
                        "Plugin 'MyCustomOp' version '2' not found (version '1' "
                        "is registered)",
                data={"plugin": "MyCustomOp", "requested": "2", "registered": "1"},
            ),
            Fact(
                id="f2",
                kind="build_aborted",
                summary="Engine build aborted after plugin resolution failure",
                source="live_smoke.log:91",
                excerpt="[E] [TRT] ModelImporter.cpp:773: Failed to parse ONNX model",
                data={},
            ),
        ],
    )


class TestLiveSynthesis:
    """Proves the real call works end to end — the one thing fakes can't."""

    def test_returns_grounded_diagnoses(self, anthropic_installed):
        facts = facts_with_unexplained_error()
        out = synthesize(facts, [])

        # An empty result is a legitimate model answer (it may judge the facts
        # too thin), but it would also be what a silently-broken SDK call
        # returns — and that ambiguity is exactly what this test exists to
        # resolve. So we require output here.
        assert out, (
            "live synthesis returned nothing. Either the SDK surface changed "
            "(check messages.parse / parsed_output) or the request failed and "
            "was swallowed by synthesize()'s broad except."
        )

    def test_every_cited_fact_id_is_real(self, anthropic_installed):
        # The groundedness eval the plan called for, against a live model.
        facts = facts_with_unexplained_error()
        valid = {f.id for f in facts.facts}
        for d in synthesize(facts, []):
            assert d.evidence
            assert set(d.evidence).issubset(valid), f"fabricated id in {d.evidence}"

    def test_live_output_carries_the_llm_trust_label(self, anthropic_installed):
        for d in synthesize(facts_with_unexplained_error(), []):
            assert d.origin == "llm"
            assert d.confidence != "high"
            assert d.code == "ED9001"
            for sug in d.suggestions:
                assert sug.applicability == "maybe-incorrect"

    def test_message_and_cause_are_populated(self, anthropic_installed):
        # A diagnosis with an empty message would render as a blank report line.
        for d in synthesize(facts_with_unexplained_error(), []):
            assert d.message.strip()
            assert d.root_cause.strip()

    def test_skips_the_api_when_rules_covered_everything(self, anthropic_installed):
        # The no-spend guarantee, verified live: if this ever regressed, every
        # clean run would start costing money.
        facts = facts_with_unexplained_error()
        covered = [Diagnosis(code="ED0105", evidence=["f1", "f2"])]
        assert synthesize(facts, covered) == []


class TestLiveGroundingUnderPressure:
    """Does the model actually decline when the facts don't support a cause?

    Hermetic tests can only check that we HANDLE insufficient_info correctly.
    This checks whether a real model reaches for it — the behaviour the whole
    honesty design depends on.
    """

    def test_thin_facts_do_not_produce_invented_causes(self, anthropic_installed):
        # One near-contentless fact. A well-behaved response is either no
        # diagnosis at all, or one grounded strictly in f1.
        facts = Facts(
            backend="tensorrt",
            artifact_path="thin.log",
            facts=[
                Fact(
                    id="f1",
                    kind="unknown_line",
                    summary="Unrecognized log line",
                    source="thin.log:1",
                    excerpt="[I] [TRT] ---------- Layers Running on DLA ----------",
                    data={},
                )
            ],
        )
        out = synthesize(facts, [])
        for d in out:
            # Whatever it says must still be traceable to f1 and nothing else.
            assert set(d.evidence) == {"f1"}, (
                f"cited {d.evidence} but only f1 exists — the model invented "
                "evidence and the grounding gate should have dropped this"
            )
