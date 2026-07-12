"""Tests for the core data contracts (Fact, Facts, Diagnosis, Suggestion).

These models are the spine of the tool: parsers produce Facts, the diagnoser
produces Diagnosis, the CLI serializes both. A silent change in their shape
breaks every layer, so we pin the behavior here.
"""

import json

import pytest
from pydantic import ValidationError

from edgedoctor.backends.base import Diagnosis, Fact, Facts, Suggestion


def make_fact(**overrides) -> Fact:
    defaults = dict(
        id="f1",
        kind="unsupported_op",
        summary="Op GridSample not supported",
        source="trtexec.log:412",
        excerpt="No importer registered for op: GridSample",
        data={"op": "GridSample", "opset": 13},
    )
    defaults.update(overrides)
    return Fact(**defaults)


class TestFact:
    def test_required_fields(self):
        # id, kind, summary, source are mandatory — a Fact without a
        # traceability anchor must be impossible to construct.
        with pytest.raises(ValidationError):
            Fact(id="f1", kind="unsupported_op", summary="x")  # no source

    def test_optional_fields_default(self):
        f = Fact(id="f1", kind="k", summary="s", source="log:1")
        assert f.excerpt == ""
        assert f.data == {}

    def test_round_trips_through_json(self):
        f = make_fact()
        restored = Fact.model_validate_json(f.model_dump_json())
        assert restored == f


class TestFacts:
    def test_holds_facts(self):
        facts = Facts(backend="tensorrt", artifact_path="build.log", facts=[make_fact()])
        assert facts.facts[0].id == "f1"

    def test_empty_facts_is_valid(self):
        # "The parser found nothing" is a legitimate result — the diagnoser
        # must then say insufficient_info, not invent causes.
        facts = Facts(backend="tensorrt", artifact_path="build.log")
        assert facts.facts == []


class TestDiagnosis:
    def test_insufficient_info_is_a_valid_result(self):
        # An honest non-answer must be constructible with no causal fields.
        d = Diagnosis(insufficient_info=True)
        assert d.insufficient_info
        assert d.root_cause == ""
        assert d.suggestions == []

    def test_full_diagnosis(self):
        d = Diagnosis(
            code="ED0042",
            severity="error",
            message="TensorRT cannot import GridSample at opset 13",
            root_cause="GridSample requires opset >= 16",
            suggestions=[
                Suggestion(
                    summary="Re-export with opset 17",
                    command="torch.onnx.export(..., opset_version=17)",
                    applicability="machine-applicable",
                )
            ],
            evidence=["f1"],
            confidence="high",
        )
        assert d.evidence == ["f1"]
        assert d.suggestions[0].applicability == "machine-applicable"

    def test_json_schema_is_generatable(self):
        # This schema is later handed to the LLM as its forced output shape;
        # it must always be valid JSON Schema with the Suggestion def inlined.
        schema = Diagnosis.model_json_schema()
        assert "Suggestion" in json.dumps(schema)
        assert schema["properties"]["evidence"]["items"]["type"] == "string"
