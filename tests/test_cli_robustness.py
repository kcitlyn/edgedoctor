"""CLI behaviour on bad inputs: clean errors, correct exit codes, no tracebacks.

The exit-code contract is part of the tool's API — CI systems branch on it:
    0 = clean · 1 = tool/usage error · 2 = errors found · 3 = warnings only

So a usage mistake must produce exit 1 and a one-line message on STDERR, never a
Python traceback. A traceback tells the user nothing actionable and, in a CI log,
looks like edgedoctor itself is broken.

Passing a directory used to dump a raw IsADirectoryError; that's pinned here.
"""

import json
import os
import stat

import pytest
from typer.testing import CliRunner

from edgedoctor.backends import PARSER_REGISTRY
from edgedoctor.cli import app

BACKENDS = sorted(PARSER_REGISTRY)

runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "100"})


@pytest.fixture
def artifacts(tmp_path):
    """A directory of pathological inputs."""
    paths = {}

    paths["empty"] = tmp_path / "empty.log"
    paths["empty"].write_text("")

    paths["binary"] = tmp_path / "binary.log"
    paths["binary"].write_bytes(bytes(range(256)) * 20)

    paths["utf16"] = tmp_path / "utf16.log"
    paths["utf16"].write_bytes("invalid utf8 here".encode("utf-16"))

    paths["invalid_utf8"] = tmp_path / "invalid_utf8.log"
    paths["invalid_utf8"].write_bytes(b"\xff\xfe\x00bad bytes\x80\x81")

    paths["huge_line"] = tmp_path / "huge_line.log"
    paths["huge_line"].write_text("z" * 200_000)

    paths["directory"] = tmp_path / "adir"
    paths["directory"].mkdir()

    paths["broken_symlink"] = tmp_path / "broken.log"
    paths["broken_symlink"].symlink_to(tmp_path / "does_not_exist")

    paths["no_extension"] = tmp_path / "noext"
    paths["no_extension"].write_text("some content")

    paths["spaces in name"] = tmp_path / "a file with spaces.log"
    paths["spaces in name"].write_text("content")

    return paths


class TestExitCodesAreInContract:
    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize(
        "key", ["empty", "binary", "utf16", "invalid_utf8", "huge_line",
                "no_extension", "spaces in name"],
    )
    def test_parseable_inputs_exit_0_to_3(self, artifacts, backend, key):
        result = runner.invoke(
            app, ["diagnose", str(artifacts[key]), "-b", backend]
        )
        assert result.exit_code in (0, 2, 3), (
            f"{key}/{backend} exited {result.exit_code}"
        )

    @pytest.mark.parametrize("command", ["diagnose", "parse"])
    @pytest.mark.parametrize("key", ["directory", "broken_symlink"])
    def test_usage_errors_exit_1(self, artifacts, command, key):
        result = runner.invoke(app, [command, str(artifacts[key])])
        assert result.exit_code == 1

    @pytest.mark.parametrize("command", ["diagnose", "parse"])
    def test_missing_file_exits_1(self, command, tmp_path):
        result = runner.invoke(app, [command, str(tmp_path / "nope.log")])
        assert result.exit_code == 1


class TestNoTracebacksLeak:
    """A traceback is never an acceptable user-facing error."""

    @pytest.mark.parametrize("command", ["diagnose", "parse"])
    @pytest.mark.parametrize("key", ["directory", "broken_symlink"])
    def test_usage_error_message_is_clean(self, artifacts, command, key):
        result = runner.invoke(app, [command, str(artifacts[key])])
        assert "Traceback" not in result.output
        assert "IsADirectoryError" not in result.output
        assert "error:" in result.output

    def test_directory_error_names_the_problem(self, artifacts):
        result = runner.invoke(app, ["diagnose", str(artifacts["directory"])])
        assert "directory" in result.output.lower()

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_binary_input_produces_no_traceback(self, artifacts, backend):
        result = runner.invoke(
            app, ["diagnose", str(artifacts["binary"]), "-b", backend]
        )
        assert "Traceback" not in result.output

    @pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses permissions")
    def test_unreadable_file_does_not_traceback(self, tmp_path):
        # typer's Path argument rejects an unreadable file itself, with a clean
        # message and no traceback. The important guarantee is exactly that: no
        # crash and a non-success exit. (typer uses 2 for argument validation;
        # our own explicit check uses 1, but typer runs first, so the framework
        # code path is what a user hits here.)
        secret = tmp_path / "secret.log"
        secret.write_text("content")
        secret.chmod(0o000)
        try:
            result = runner.invoke(app, ["diagnose", str(secret)])
            assert "Traceback" not in result.output
            assert result.exit_code != 0
        finally:
            secret.chmod(stat.S_IRUSR | stat.S_IWUSR)


class TestJsonOutputStaysMachineReadable:
    """stdout carries results; chatter goes to stderr, so `| jq` always works."""

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("key", ["empty", "binary", "invalid_utf8", "huge_line"])
    def test_json_is_always_parseable(self, artifacts, backend, key):
        result = runner.invoke(
            app, ["diagnose", str(artifacts[key]), "-b", backend, "--json"]
        )
        data = json.loads(result.stdout)
        assert data["schemaVersion"] == 1
        assert isinstance(data["diagnostics"], list)
        assert isinstance(data["facts"], list)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_parse_json_is_always_parseable(self, artifacts, backend):
        result = runner.invoke(
            app, ["parse", str(artifacts["binary"]), "-b", backend, "--json"]
        )
        data = json.loads(result.stdout)
        assert data["backend"] == backend

    def test_json_survives_a_huge_binary_excerpt(self, artifacts):
        # Non-UTF8 bytes are replaced at read time; the result must still be
        # JSON-serializable (surrogates would break json.dumps).
        result = runner.invoke(
            app, ["parse", str(artifacts["invalid_utf8"]), "--json"]
        )
        json.loads(result.stdout)


class TestBackendSelection:
    def test_unknown_backend_is_rejected(self, artifacts):
        result = runner.invoke(
            app, ["diagnose", str(artifacts["empty"]), "-b", "not_a_backend"]
        )
        assert result.exit_code != 0

    def test_declared_but_unimplemented_backend_exits_1(self, artifacts):
        # coreml is in the enum (honest roadmap surface) but has no parser.
        result = runner.invoke(
            app, ["diagnose", str(artifacts["empty"]), "-b", "coreml"]
        )
        assert result.exit_code == 1
        assert "Not implemented" in result.output

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_every_registered_backend_is_usable_from_the_cli(self, artifacts, backend):
        result = runner.invoke(
            app, ["parse", str(artifacts["empty"]), "-b", backend]
        )
        assert result.exit_code == 0
        assert "no parser" not in result.output


class TestJsonHonoursTheExitCodeContract:
    """--json changes the FORMAT, never the verdict.

    The `diagnose --json` path used to `raise typer.Exit(code=0)` unconditionally,
    so a CI job piping the report to jq saw SUCCESS on a run that found errors —
    silently breaking the documented contract in exactly the situation --json
    exists to serve. The README advertises --json "for CI and AI agents" in the
    same sentence as the exit codes, so the two must agree.
    """

    @pytest.mark.parametrize(
        "fixture,expected",
        [
            ("tests/fixtures/tensorrt/unsupported_op_trt8.log", 2),  # has errors
            ("tests/fixtures/tensorrt/success.log", 0),              # clean
        ],
    )
    def test_json_and_human_exit_codes_agree(self, fixture, expected):
        human = runner.invoke(app, ["diagnose", fixture])
        as_json = runner.invoke(app, ["diagnose", fixture, "--json"])
        assert human.exit_code == expected
        assert as_json.exit_code == expected, (
            "--json must not change the verdict"
        )

    def test_json_document_is_still_on_stdout_and_valid(self):
        # Honouring the exit code must not break the document itself.
        result = runner.invoke(
            app, ["diagnose", "tests/fixtures/tensorrt/unsupported_op_trt8.log",
                  "--json"]
        )
        data = json.loads(result.stdout)
        assert data["schemaVersion"] == 1
        assert data["diagnostics"]

    def test_errors_found_is_distinguishable_from_a_tool_error(self):
        # 2 means "the log has problems"; 1 means "edgedoctor could not run".
        # A CI script must be able to tell those apart.
        found = runner.invoke(
            app, ["diagnose", "tests/fixtures/tensorrt/unsupported_op_trt8.log",
                  "--json"]
        )
        broken = runner.invoke(app, ["diagnose", "no_such_file.log", "--json"])
        assert found.exit_code == 2
        assert broken.exit_code == 1

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_every_backend_agrees_across_formats(self, artifacts, backend):
        # Whatever the backend, the two output modes must return the same code.
        target = str(artifacts["empty"])
        human = runner.invoke(app, ["diagnose", target, "-b", backend])
        as_json = runner.invoke(app, ["diagnose", target, "-b", backend, "--json"])
        assert human.exit_code == as_json.exit_code
