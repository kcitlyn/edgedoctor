"""Wire-contract tests: the real SDK, a mock transport, no network.

These sit between the two existing layers and close most of the gap the live
test was written for:

  tests/test_llm.py       fake client       — our logic (grounding, degradation)
  tests/test_llm_wire.py  REAL SDK + mock   — our *request* is well-formed  <-- here
  tests/test_llm_live.py  real API          — the model behaves (needs a key)

The fake-client tests can't catch a wrong kwarg name, an unserializable schema,
or reading the response off the wrong attribute, because the fake accepts
anything. These drive `anthropic`'s actual request-building and response-parsing
code against httpx.MockTransport, so the SDK contract is verified with no API
key, no spend, and no network — which means CI checks it on every push.

What still genuinely requires a live key: whether a real model's *output* is
sensible and grounded. That is behaviour, not contract, and it stays in
test_llm_live.py.
"""

import json

import pytest

from edgedoctor.backends.base import Fact, Facts
from edgedoctor.llm import DEFAULT_MODEL, MAX_TOKENS, synthesize

httpx = pytest.importorskip("httpx", reason="needs the anthropic SDK's httpx")
anthropic = pytest.importorskip("anthropic", reason='needs: pip install "edgedoctor[llm]"')


def make_facts() -> Facts:
    return Facts(
        backend="tensorrt",
        artifact_path="wire.log",
        facts=[
            Fact(id="f1", kind="mystery_error", summary="something unexplained",
                 source="wire.log:7", excerpt="[E] [TRT] unknown failure",
                 data={"detail": "x"}),
        ],
    )


def model_reply(payload: dict) -> dict:
    """A well-formed Messages API response carrying `payload` as the parsed output."""
    return {
        "id": "msg_wire",
        "type": "message",
        "role": "assistant",
        "model": DEFAULT_MODEL,
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


ONE_DIAGNOSIS = {
    "diagnoses": [{
        "message": "plugin ABI mismatch",
        "root_cause": "the registered plugin version differs from the requested one",
        "severity": "error",
        "evidence": ["f1"],
        "suggestions": [{"summary": "rebuild the plugin", "command": "make"}],
        "insufficient_info": False,
    }]
}


def client_returning(payload: dict, capture: dict | None = None):
    """A real Anthropic client whose transport is mocked — no socket is opened."""
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture["url"] = str(request.url)
            capture["headers"] = dict(request.headers)
            capture["body"] = json.loads(request.content)
        return httpx.Response(200, json=model_reply(payload))

    return anthropic.Anthropic(
        api_key="sk-ant-fake-for-tests",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


class TestRequestIsWellFormed:
    """Catches the class of bug a fake client cannot: a malformed request."""

    def test_full_round_trip_produces_a_diagnosis(self):
        # The headline check: our call goes through the real SDK's request
        # building and its response parsing, and comes back as a Diagnosis.
        out = synthesize(make_facts(), [], client=client_returning(ONE_DIAGNOSIS))
        assert len(out) == 1
        assert out[0].origin == "llm"
        assert out[0].evidence == ["f1"]

    def test_hits_the_messages_endpoint(self):
        cap: dict = {}
        synthesize(make_facts(), [], client=client_returning(ONE_DIAGNOSIS, cap))
        assert cap["url"].endswith("/v1/messages")

    def test_sends_the_parameters_we_intend(self):
        cap: dict = {}
        synthesize(make_facts(), [], client=client_returning(ONE_DIAGNOSIS, cap))
        body = cap["body"]
        assert body["model"] == DEFAULT_MODEL
        assert body["max_tokens"] == MAX_TOKENS
        assert body["temperature"] == 0
        assert body["messages"][0]["role"] == "user"

    def test_schema_is_serialized_into_the_request(self):
        # If our pydantic schema were unserializable or the wrong shape, this is
        # where it would surface — the SDK has to turn it into JSON to send it.
        cap: dict = {}
        synthesize(make_facts(), [], client=client_returning(ONE_DIAGNOSIS, cap))
        schema = cap["body"]["output_config"]["format"]["schema"]
        assert schema["type"] == "object"
        assert "diagnoses" in schema["properties"]
        # The nested model must resolve, or the API would reject the request.
        assert "SynthesizedDiagnosis" in schema["$defs"]

    def test_the_narrow_schema_is_what_goes_on_the_wire(self):
        # The trust guarantee, verified at the boundary rather than in-process:
        # the model is never *offered* a field it must not control.
        cap: dict = {}
        synthesize(make_facts(), [], client=client_returning(ONE_DIAGNOSIS, cap))
        sent = json.dumps(cap["body"]["output_config"])
        diag_props = cap["body"]["output_config"]["format"]["schema"]["$defs"][
            "SynthesizedDiagnosis"]["properties"]
        assert "origin" not in diag_props
        assert "confidence" not in diag_props
        assert "code" not in diag_props
        assert "insufficient_info" in sent  # the honest-answer escape hatch is offered

    def test_system_prompt_reaches_the_wire(self):
        cap: dict = {}
        synthesize(make_facts(), [], client=client_returning(ONE_DIAGNOSIS, cap))
        system = json.dumps(cap["body"]["system"])
        assert "Never cite an id that is not in the input" in system

    def test_only_unmatched_facts_are_transmitted(self):
        # Proven at the network boundary: a fact a rule already explained must
        # not leave the process.
        from edgedoctor.backends.base import Diagnosis

        facts = Facts(
            backend="tensorrt", artifact_path="wire.log",
            facts=[
                Fact(id="f1", kind="known", summary="covered", source="wire.log:1",
                     excerpt="SECRET_COVERED_LINE"),
                Fact(id="f2", kind="mystery", summary="uncovered",
                     source="wire.log:2", excerpt="UNCOVERED_LINE"),
            ],
        )
        cap: dict = {}
        synthesize(facts, [Diagnosis(code="ED0101", evidence=["f1"])],
                   client=client_returning(ONE_DIAGNOSIS, cap))
        body = json.dumps(cap["body"])
        assert "UNCOVERED_LINE" in body
        assert "SECRET_COVERED_LINE" not in body


class TestResponseHandlingAgainstTheRealSdk:
    def test_grounding_gate_applies_to_real_sdk_output(self):
        # Same rejection as the fake-client test, but the object under test is a
        # genuine ParsedMessage built by the SDK.
        bad = {"diagnoses": [dict(ONE_DIAGNOSIS["diagnoses"][0], evidence=["f99"])]}
        assert synthesize(make_facts(), [], client=client_returning(bad)) == []

    def test_empty_diagnoses_list_is_handled(self):
        assert synthesize(make_facts(), [], client=client_returning({"diagnoses": []})) == []

    def test_insufficient_info_from_the_real_sdk_is_dropped(self):
        payload = {"diagnoses": [
            dict(ONE_DIAGNOSIS["diagnoses"][0], insufficient_info=True)
        ]}
        assert synthesize(make_facts(), [], client=client_returning(payload)) == []

    @pytest.mark.parametrize("status", [401, 429, 500, 529])
    def test_api_errors_degrade_to_no_synthesis(self, status):
        # Real SDK exception types (AuthenticationError, RateLimitError, ...)
        # rather than stand-ins, so the broad except is proven to cover them.
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status, json={"error": {"message": "nope"}})

        client = anthropic.Anthropic(
            api_key="sk-ant-fake-for-tests",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        assert synthesize(make_facts(), [], client=client) == []

    def test_network_failure_degrades_to_no_synthesis(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        client = anthropic.Anthropic(
            api_key="sk-ant-fake-for-tests",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        assert synthesize(make_facts(), [], client=client) == []

    def test_non_json_response_body_degrades(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="not json at all")

        client = anthropic.Anthropic(
            api_key="sk-ant-fake-for-tests",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        assert synthesize(make_facts(), [], client=client) == []


class TestSdkSurfaceIsStillWhatWeCodedTo:
    """Fails loudly if an SDK upgrade moves what llm.py depends on."""

    def test_parse_accepts_every_kwarg_we_pass(self):
        import inspect

        params = inspect.signature(
            anthropic.Anthropic(api_key="sk-ant-fake").messages.parse
        ).parameters
        for kwarg in ("model", "max_tokens", "temperature", "system",
                      "output_format", "messages"):
            assert kwarg in params, f"SDK no longer accepts {kwarg}"

    def test_parsed_output_is_where_we_read_it(self):
        from anthropic.types import ParsedTextBlock

        assert "parsed_output" in ParsedTextBlock.model_fields
