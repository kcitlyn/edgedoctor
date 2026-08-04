"""The exit-code contract under --llm.

The exit code is an API: CI gates branch on it, so it has to mean something
stable. The documented contract is

    0 = clean · 1 = usage/tool error · 2 = errors found · 3 = warnings only

and a SYNTHESIZED finding may never produce 2. An LLM diagnosis is unreviewed
and capped at medium confidence; letting one fail a build would contradict this
layer's stated premise (docs/adr/0001) that `--llm` can only ever add to a run,
never degrade it. Exit 3 keeps such a finding visible to automation rather than
silently swallowing it, while reserving hard failure for curated rules.

Before this was fixed, a synthesized error escalated a perfectly clean run from
0 to 2, so a generated guess could break a build.

These tests patch edgedoctor.llm at the module level rather than mocking the API
client: the behaviour under test is the CLI's exit logic, not the SDK.
"""

import pytest
from typer.testing import CliRunner

from edgedoctor.backends.base import Diagnosis
from edgedoctor.cli import app

CLEAN_LOG = "tests/fixtures/tensorrt/success.log"
ERROR_LOG = "tests/fixtures/tensorrt/unsupported_op_trt8.log"

runner = CliRunner(env={
    "NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "100",
    "ANTHROPIC_API_KEY": "sk-ant-fake-never-used",
})


@pytest.fixture
def llm_returning(monkeypatch):
    """Force --llm to be 'available' and return the diagnoses we choose."""
    import edgedoctor.llm as llm_module

    def _configure(*diagnoses: Diagnosis):
        monkeypatch.setattr(llm_module, "availability", lambda: (True, ""))

        def fake_synthesize(facts, rule_diagnoses, **kwargs):
            # Attach real evidence so the renderer has something to cite.
            evidence = [facts.facts[0].id] if facts.facts else []
            return [d.model_copy(update={"evidence": evidence}) for d in diagnoses]

        monkeypatch.setattr(llm_module, "synthesize", fake_synthesize)

    return _configure


def synthesized(severity: str) -> Diagnosis:
    return Diagnosis(code="ED9001", severity=severity, message="a generated finding",
                     origin="llm", confidence="medium")


class TestSynthesizedFindingsNeverExitTwo:
    @pytest.mark.parametrize("severity", ["error", "warning", "info"])
    def test_clean_log_plus_synthesis_exits_3(self, llm_returning, severity):
        # Whatever severity the model claimed, an unreviewed finding caps at 3.
        llm_returning(synthesized(severity))
        result = runner.invoke(app, ["diagnose", CLEAN_LOG, "--llm"])
        assert result.exit_code == 3, (
            f"synthesized {severity} produced exit {result.exit_code}; a generated "
            "guess must not fail a CI gate"
        )

    def test_synthesized_finding_is_visible(self, llm_returning):
        # Capping the exit code must not HIDE the finding — that would be the
        # opposite failure, and worse.
        llm_returning(synthesized("error"))
        result = runner.invoke(app, ["diagnose", CLEAN_LOG, "--llm"])
        assert "ED9001" in result.output
        assert "synthesized" in result.output, "origin must be marked for the reader"

    def test_multiple_synthesized_findings_still_cap_at_3(self, llm_returning):
        llm_returning(synthesized("error"), synthesized("error"))
        assert runner.invoke(app, ["diagnose", CLEAN_LOG, "--llm"]).exit_code == 3


class TestRuleFindingsStillExitTwo:
    def test_rule_error_exits_2_even_alongside_synthesis(self, llm_returning):
        # A curated error is a real failure; synthesis must not soften it.
        llm_returning(synthesized("info"))
        assert runner.invoke(app, ["diagnose", ERROR_LOG, "--llm"]).exit_code == 2

    def test_rule_error_exits_2_without_llm(self):
        assert runner.invoke(app, ["diagnose", ERROR_LOG]).exit_code == 2

    def test_clean_log_exits_0_without_llm(self):
        assert runner.invoke(app, ["diagnose", CLEAN_LOG]).exit_code == 0


class TestNoSynthesisMeansNoChange:
    def test_clean_log_with_empty_synthesis_exits_0(self, llm_returning):
        llm_returning()  # synthesis returns nothing
        result = runner.invoke(app, ["diagnose", CLEAN_LOG, "--llm"])
        assert result.exit_code == 0
        assert "no additional grounded diagnoses" in result.output

    def test_unavailable_llm_leaves_the_exit_code_alone(self):
        # No key/SDK: the run must behave exactly as it would without --llm.
        keyless = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "100",
                                 "ANTHROPIC_API_KEY": ""})
        assert keyless.invoke(app, ["diagnose", ERROR_LOG, "--llm"]).exit_code == 2
        assert keyless.invoke(app, ["diagnose", CLEAN_LOG, "--llm"]).exit_code == 0


class TestJsonReflectsSynthesis:
    def test_json_marks_the_origin_of_each_diagnosis(self, llm_returning):
        import json

        llm_returning(synthesized("error"))
        result = runner.invoke(app, ["diagnose", CLEAN_LOG, "--llm", "--json"])
        data = json.loads(result.stdout)
        origins = {d["origin"] for d in data["diagnostics"]}
        assert "llm" in origins, "a consumer must be able to filter generated output"

    def test_json_exit_code_matches_the_human_run(self, llm_returning):
        # --json changes the FORMAT, never the verdict. The two paths must agree,
        # or a CI job piping to jq would see a different result than a human
        # reading the same report.
        llm_returning(synthesized("error"))
        human = runner.invoke(app, ["diagnose", CLEAN_LOG, "--llm"])
        as_json = runner.invoke(app, ["diagnose", CLEAN_LOG, "--llm", "--json"])
        assert human.exit_code == as_json.exit_code == 3
