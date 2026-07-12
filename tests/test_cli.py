"""CLI behavior tests, using typer's CliRunner (no subprocess needed).

These pin the CLI contract: exit codes, honest stub messaging, and help
surface. When the diagnose pipeline becomes real, the stub tests get replaced
by report-format tests.
"""

from typer.testing import CliRunner

from edgedoctor import __version__
from edgedoctor.cli import app

runner = CliRunner()


class TestVersion:
    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert f"edgedoctor {__version__}" in result.output


class TestHelp:
    def test_no_args_shows_help(self):
        # no_args_is_help=True: bare `edgedoctor` should show usage, not crash.
        result = runner.invoke(app, [])
        assert "Usage" in result.output

    def test_diagnose_in_help(self):
        result = runner.invoke(app, ["--help"])
        assert "diagnose" in result.output

    def test_diagnose_help_lists_backend_option(self):
        result = runner.invoke(app, ["diagnose", "--help"])
        assert "--backend" in result.output


class TestParse:
    FIXTURE = "tests/fixtures/tensorrt/unsupported_op_trt8.log"

    def test_parse_table_output(self):
        result = runner.invoke(app, ["parse", self.FIXTURE])
        assert result.exit_code == 0
        assert "unsupported_op" in result.output
        assert "GridSample" in result.output

    def test_parse_json_output_is_valid_facts(self):
        import json

        from edgedoctor.backends.base import Facts

        result = runner.invoke(app, ["parse", self.FIXTURE, "--json"])
        assert result.exit_code == 0
        # stdout must be parseable back into the Facts contract.
        facts = Facts.model_validate(json.loads(result.stdout))
        assert facts.backend == "tensorrt"
        assert any(f.kind == "unsupported_op" for f in facts.facts)

    def test_missing_file_exits_1(self):
        result = runner.invoke(app, ["parse", "nope.log"])
        assert result.exit_code == 1

    def test_unimplemented_parser_backend_exits_1(self):
        result = runner.invoke(app, ["parse", self.FIXTURE, "-b", "coreml"])
        assert result.exit_code == 1
        assert "no parser" in result.output


class TestDiagnose:
    FIXTURE = "tests/fixtures/tensorrt/unsupported_op_trt8.log"
    SUCCESS = "tests/fixtures/tensorrt/success.log"

    def test_error_log_exits_2(self):
        result = runner.invoke(app, ["diagnose", self.FIXTURE])
        assert result.exit_code == 2

    def test_success_log_exits_0(self):
        result = runner.invoke(app, ["diagnose", self.SUCCESS])
        assert result.exit_code == 0

    def test_shows_rule_code_and_message(self):
        result = runner.invoke(app, ["diagnose", self.FIXTURE])
        assert "ED0101" in result.output
        assert "GridSample" in result.output

    def test_shows_evidence(self):
        result = runner.invoke(app, ["diagnose", self.FIXTURE])
        assert "No importer registered" in result.output

    def test_json_output_is_valid(self):
        import json
        result = runner.invoke(app, ["diagnose", self.FIXTURE, "--json"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["schemaVersion"] == 1
        assert len(data["diagnostics"]) >= 1
        assert data["diagnostics"][0]["code"] == "ED0101"

    def test_missing_file_exits_1(self):
        result = runner.invoke(app, ["diagnose", "nope.log"])
        assert result.exit_code == 1

    def test_unimplemented_backend_exits_1(self):
        result = runner.invoke(app, ["diagnose", self.FIXTURE, "-b", "coreml"])
        assert result.exit_code == 1
        assert "Not implemented yet" in result.output

    def test_unknown_backend_rejected(self):
        result = runner.invoke(app, ["diagnose", self.FIXTURE, "-b", "notreal"])
        assert result.exit_code != 0
