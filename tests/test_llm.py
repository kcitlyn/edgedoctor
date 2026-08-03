"""Tests for the grounded LLM synthesis layer.

Hermetic by construction: every test injects a fake client, so the suite needs
no API key, makes no network call, and costs nothing. See
docs/adr/0001-llm-synthesis-layer.md section 5 for why this was chosen over
recorded HTTP cassettes.

What's actually under test is the part we own and the part that can hurt a user:
the grounding gate, the degradation guarantees, and the trust labelling. Whether
the SDK serializes JSON correctly is the SDK's problem.
"""

import pytest

from edgedoctor.backends.base import Diagnosis, Fact, Facts
from edgedoctor.llm import (
    DEFAULT_MODEL,
    SynthesisResult,
    SynthesizedDiagnosis,
    SynthesizedSuggestion,
    availability,
    synthesize,
    unmatched_facts,
)

# ── Fakes ─────────────────────────────────────────────────────────────────


class FakeBlock:
    """Mimics the SDK's content block, which carries `parsed_output`."""

    def __init__(self, parsed):
        self.parsed_output = parsed


class FakeResponse:
    def __init__(self, parsed):
        self.content = [FakeBlock(parsed)]


class FakeMessages:
    def __init__(self, parsed=None, error=None):
        self._parsed = parsed
        self._error = error
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return FakeResponse(self._parsed)


class FakeClient:
    def __init__(self, parsed=None, error=None):
        self.messages = FakeMessages(parsed=parsed, error=error)


def make_facts(*specs) -> Facts:
    """specs: (id, kind) tuples."""
    return Facts(
        backend="polygraphy",
        artifact_path="test.log",
        facts=[
            Fact(id=i, kind=k, summary=f"observed {i}", source=f"test.log:{n}",
                 excerpt=f"raw line {n}")
            for n, (i, k) in enumerate(specs, start=1)
        ],
    )


def one_diagnosis(**overrides) -> SynthesisResult:
    payload = {
        "message": "custom plugin failed to load",
        "root_cause": "the plugin library was not on the search path",
        "severity": "error",
        "evidence": ["f1"],
        "suggestions": [SynthesizedSuggestion(summary="set LD_LIBRARY_PATH")],
        "insufficient_info": False,
    }
    payload.update(overrides)
    return SynthesisResult(diagnoses=[SynthesizedDiagnosis(**payload)])


# ── The grounding gate: the core guarantee ────────────────────────────────


class TestGroundingGate:
    def test_grounded_diagnosis_is_kept(self):
        facts = make_facts(("f1", "mystery_error"))
        client = FakeClient(parsed=one_diagnosis(evidence=["f1"]))
        out = synthesize(facts, [], client=client)
        assert len(out) == 1
        assert out[0].evidence == ["f1"]

    def test_hallucinated_fact_id_is_dropped(self):
        # The model cites 'f99', which does not exist. This is THE failure mode
        # the whole layer is designed to make impossible for a user to see.
        facts = make_facts(("f1", "mystery_error"))
        client = FakeClient(parsed=one_diagnosis(evidence=["f99"]))
        assert synthesize(facts, [], client=client) == []

    def test_partially_hallucinated_evidence_drops_whole_diagnosis(self):
        # f1 is real, f99 is not. We drop the entire diagnosis rather than
        # salvaging the real citation — a source that fabricated one id has
        # shown it isn't grounded, and keeping the rest would launder that.
        facts = make_facts(("f1", "mystery_error"))
        client = FakeClient(parsed=one_diagnosis(evidence=["f1", "f99"]))
        assert synthesize(facts, [], client=client) == []

    def test_diagnosis_citing_nothing_is_dropped(self):
        # An unfalsifiable claim: no evidence means nothing to check it against.
        facts = make_facts(("f1", "mystery_error"))
        client = FakeClient(parsed=one_diagnosis(evidence=[]))
        assert synthesize(facts, [], client=client) == []

    def test_insufficient_info_produces_no_diagnosis(self):
        # The honest non-answer. Valued, but it needs no report entry.
        facts = make_facts(("f1", "mystery_error"))
        client = FakeClient(parsed=one_diagnosis(insufficient_info=True))
        assert synthesize(facts, [], client=client) == []

    def test_only_unmatched_fact_ids_are_citable(self):
        # f1 was already explained by a rule, so it is NOT in the LLM's input —
        # citing it must therefore be treated as ungrounded.
        facts = make_facts(("f1", "known_error"), ("f2", "mystery_error"))
        rule_diag = [Diagnosis(code="ED0101", evidence=["f1"])]
        client = FakeClient(parsed=one_diagnosis(evidence=["f1"]))
        assert synthesize(facts, rule_diag, client=client) == []


# ── Degradation: the LLM can never break the tool ─────────────────────────


class TestDegradation:
    @pytest.mark.parametrize(
        "error",
        [
            ConnectionError("network down"),
            TimeoutError("timed out"),
            ValueError("malformed response"),
            RuntimeError("rate limited"),
        ],
    )
    def test_any_client_error_yields_no_diagnoses(self, error):
        facts = make_facts(("f1", "mystery_error"))
        client = FakeClient(error=error)
        # Must not raise — a broken optional enhancement may not take down a
        # working diagnostic run.
        assert synthesize(facts, [], client=client) == []

    def test_malformed_response_shape_yields_no_diagnoses(self):
        class Empty:
            content = []

        class Weird:
            def __init__(self):
                self.messages = self

            def parse(self, **kwargs):
                return Empty()

        facts = make_facts(("f1", "mystery_error"))
        assert synthesize(facts, [], client=Weird()) == []

    def test_no_unmatched_facts_makes_no_api_call(self):
        # Everything already explained: the best outcome, and it must cost
        # nothing. A call here would be wasted spend on every clean run.
        facts = make_facts(("f1", "known_error"))
        rule_diag = [Diagnosis(code="ED0101", evidence=["f1"])]
        client = FakeClient(parsed=one_diagnosis())
        assert synthesize(facts, rule_diag, client=client) == []
        assert client.messages.calls == []

    def test_missing_key_and_sdk_reported_by_availability(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        ok, reason = availability()
        assert ok is False
        assert reason  # must explain itself, not fail silently

    def test_synthesize_without_key_returns_empty(self, monkeypatch):
        # No client injected and no key: must degrade, not raise.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        facts = make_facts(("f1", "mystery_error"))
        assert synthesize(facts, []) == []


# ── Trust labelling: a synthesis may not impersonate a curated rule ───────


class TestTrustLabelling:
    def test_marked_as_llm_origin(self):
        facts = make_facts(("f1", "mystery_error"))
        client = FakeClient(parsed=one_diagnosis())
        assert synthesize(facts, [], client=client)[0].origin == "llm"

    def test_confidence_is_capped_below_high(self):
        # A curated rule earns 'high'. An unreviewed synthesis never does.
        facts = make_facts(("f1", "mystery_error"))
        client = FakeClient(parsed=one_diagnosis())
        assert synthesize(facts, [], client=client)[0].confidence != "high"

    def test_suggestions_are_never_machine_applicable(self):
        # An agent must not run a generated command unattended.
        facts = make_facts(("f1", "mystery_error"))
        client = FakeClient(parsed=one_diagnosis())
        out = synthesize(facts, [], client=client)
        for sug in out[0].suggestions:
            assert sug.applicability == "maybe-incorrect"

    def test_does_not_claim_a_curated_ed_code(self):
        # ED01xx/ED02xx are documented, reviewed failure classes. Synthesis
        # gets its own reserved range so it can't pose as one.
        facts = make_facts(("f1", "mystery_error"))
        client = FakeClient(parsed=one_diagnosis())
        code = synthesize(facts, [], client=client)[0].code
        assert code == "ED9001"

    def test_model_schema_cannot_set_origin(self):
        # Structural, not instructional: the field simply isn't on the wire
        # schema, so no generated output can populate it.
        assert "origin" not in SynthesizedDiagnosis.model_fields
        assert "confidence" not in SynthesizedDiagnosis.model_fields

    def test_rules_diagnoses_default_to_rules_origin(self):
        assert Diagnosis(code="ED0101").origin == "rules"


# ── Request construction ──────────────────────────────────────────────────


class TestRequestConstruction:
    def test_sends_only_unmatched_facts(self):
        facts = make_facts(("f1", "known_error"), ("f2", "mystery_error"))
        rule_diag = [Diagnosis(code="ED0101", evidence=["f1"])]
        client = FakeClient(parsed=one_diagnosis(evidence=["f2"]))
        synthesize(facts, rule_diag, client=client)

        prompt = client.messages.calls[0]["messages"][0]["content"]
        assert "f2" in prompt
        # The matched fact must not be re-litigated by the LLM.
        assert "f1" not in prompt

    def test_uses_haiku_and_zero_temperature(self):
        facts = make_facts(("f1", "mystery_error"))
        client = FakeClient(parsed=one_diagnosis())
        synthesize(facts, [], client=client)
        call = client.messages.calls[0]
        assert call["model"] == DEFAULT_MODEL
        assert call["temperature"] == 0  # extraction, not creative writing

    def test_requests_the_narrow_schema(self):
        facts = make_facts(("f1", "mystery_error"))
        client = FakeClient(parsed=one_diagnosis())
        synthesize(facts, [], client=client)
        assert client.messages.calls[0]["output_format"] is SynthesisResult

    def test_system_prompt_states_the_honesty_contract(self):
        facts = make_facts(("f1", "mystery_error"))
        client = FakeClient(parsed=one_diagnosis())
        synthesize(facts, [], client=client)
        system = client.messages.calls[0]["system"]
        assert "insufficient_info" in system
        assert "Never cite an id that is not in the input" in system


class TestUnmatchedFacts:
    def test_returns_facts_no_diagnosis_cited(self):
        facts = make_facts(("f1", "a"), ("f2", "b"), ("f3", "c"))
        diags = [Diagnosis(code="ED0101", evidence=["f1", "f3"])]
        assert [f.id for f in unmatched_facts(facts, diags)] == ["f2"]

    def test_all_facts_unmatched_when_no_diagnoses(self):
        facts = make_facts(("f1", "a"), ("f2", "b"))
        assert len(unmatched_facts(facts, [])) == 2

    def test_empty_when_everything_matched(self):
        facts = make_facts(("f1", "a"))
        diags = [Diagnosis(code="ED0101", evidence=["f1"])]
        assert unmatched_facts(facts, diags) == []
