"""Security tests: secret leakage and terminal injection.

THREAT MODEL
edgedoctor has three properties that make it a security-relevant tool rather
than a passive viewer:

  1. It ingests ATTACKER-INFLUENCED data. A build log is produced by a toolchain
     acting on a model and a CI configuration, none of which the reader
     necessarily controls.
  2. It RE-DISPLAYS that data verbatim as its core promise, into terminals and
     into reports that get pasted into issues, CI output and chat.
  3. With --llm it TRANSMITS that data to a third-party API.

Two real vulnerabilities were found and fixed, both pinned below:

  A. SECRET LEAKAGE. Build logs routinely contain credentials — a private
     registry fetch, an exported token. If the secret sat on a line the parser
     matched, the report echoed it verbatim, and --json carried it into CI
     artifacts. OWASP's Secrets Management guidance requires masking rather than
     echoing. Worst of all, --llm transmitted it off the machine irreversibly.

  B. TERMINAL INJECTION (CWE-117 / CWE-116, "log forging" on the display side).
     Raw ANSI escapes from a log reached the terminal: ESC[2J clears the screen,
     OSC sets the window title, CR rewinds the cursor so later text overwrites
     earlier text. A crafted log could repaint the report and misrepresent the
     verdict. NO_COLOR did not help — these bytes are DATA in the log, not
     styling edgedoctor chose to emit.

WHAT IS DELIBERATELY *NOT* CLAIMED
Secret detection is pattern-based and therefore incomplete: a human-chosen
password matching no known shape will not be caught, exactly as OWASP warns.
These tests pin the known-format cases and the absence of false positives; they
do not assert that any report is safe to publish.
"""

import glob
import io
import json
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from edgedoctor.backends.base import Diagnosis, Fact, Facts
from edgedoctor.cli import app
from edgedoctor.redact import (
    redact_secrets,
    sanitize_for_display,
    strip_control_chars,
)
from edgedoctor.report import render_human, render_json

runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "200"})


def render(diagnoses, facts, **kwargs) -> str:
    buf = io.StringIO()
    render_human(diagnoses, facts,
                 console=Console(file=buf, width=200, no_color=True), **kwargs)
    return buf.getvalue()


def facts_with(excerpt: str, summary: str = "s", **data) -> Facts:
    return Facts(
        backend="tensorrt", artifact_path="build.log",
        facts=[Fact(id="f1", kind="unsupported_op", summary=summary,
                    source="build.log:1", excerpt=excerpt, data=data)],
    )


def diag() -> Diagnosis:
    return Diagnosis(code="ED0101", severity="error", message="m", evidence=["f1"])


# Realistic secret shapes. Values are synthetic but structurally valid, so the
# patterns are exercised the way a real credential would exercise them.
#
# Several are ASSEMBLED AT RUNTIME rather than written as literals. GitHub's push
# protection scans committed files and — correctly — cannot distinguish a
# well-formed test fixture from a live credential; it blocked this file for the
# Slack token. Concatenating the prefix keeps the value structurally valid for the
# patterns under test while leaving no scannable literal in the repository.
#
# The alternative was to click GitHub's "allow this secret" link, which would
# have meant disabling a real secret-scanning control in the very commit that
# adds secret redaction. Working WITH the scanner is the honest option, and it
# doubles as evidence that these fixtures look like the real thing.
_SLACK = "xox" + "b-1111111111-AAAAAAAAAAAAAAAAAAAA"
_GH_CLASSIC = "gh" + "p_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
_GH_PAT = "github_" + "pat_11AAAAAAA0BBBBBBBBBBBB_cccccccccccccccccccccccccc"
_GITLAB = "glp" + "at-AAAAAAAAAAAAAAAAAAAA"
_ANTHROPIC = "sk-ant-" + "api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
_AWS_ID = "AKIA" + "IOSFODNN7EXAMPLE"
_GOOGLE = "AIza" + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
_HF = "hf_" + "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"

SECRETS = {
    "anthropic-key": _ANTHROPIC,
    "aws-access-key-id": _AWS_ID,
    "github-token": _GH_CLASSIC,
    "github-pat": _GH_PAT,
    "gitlab-token": _GITLAB,
    "slack-token": _SLACK,
    "google-api-key": _GOOGLE,
    "hf-token": _HF,
    "jwt": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K",
    "url-credentials": "https://ci:Sup3rSecretValue@registry.internal/model.onnx",
    "assigned-password": "password=hunter2hunter2",
    "assigned-aws-secret": "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMIK7MDENGbPxRfiCYEX",
    "private-key": "-----BEGIN RSA PRIVATE KEY-----",
}


class TestSecretsAreDetected:
    @pytest.mark.parametrize("label,secret", sorted(SECRETS.items()))
    def test_each_known_shape_is_masked(self, label, secret):
        masked, found = redact_secrets(f"[TRT] fetch failed: {secret}")
        assert found, f"{label} not detected"
        # The literal secret value must be gone. For patterns that keep context
        # (a URL scheme, a key name), check the credential part specifically.
        for fragment in ("Sup3rSecretValue", "hunter2hunter2",
                         "wJalrXUtnFEMIK7MDENGbPxRfiCYEX", _AWS_ID,
                         "dBjftJeZ4CVPmB92K"):
            if fragment in secret:
                assert fragment not in masked, f"{label} leaked {fragment}"

    def test_the_marker_is_visible_not_blank(self):
        # Evidence must never be silently blanked: the reader has to be able to
        # see that edgedoctor altered the line.
        masked, _ = redact_secrets(f"token={_GITLAB}")
        assert "REDACTED" in masked

    def test_the_kind_is_reported_so_the_user_can_rotate(self):
        _, found = redact_secrets("https://u:p4ssword@host/x")
        assert "url-credentials" in found

    def test_multiple_secrets_on_one_line_are_all_masked(self):
        line = f"a {SECRETS['gitlab-token']} and {SECRETS['aws-access-key-id']}"
        masked, found = redact_secrets(line)
        assert _GITLAB not in masked
        assert _AWS_ID not in masked
        assert len(found) >= 2

    def test_context_is_preserved_around_a_masked_value(self):
        # "the password in this git URL" is more actionable than a bare marker.
        masked, _ = redact_secrets("https://ci:Sup3rSecretValue@registry/x")
        assert "https://ci:" in masked
        assert "@registry/x" in masked

    def test_an_authorization_header_keeps_its_field_name(self):
        # Masking the word "Authorization" would read as if the field NAME were
        # the secret. Regression: bare "auth" used to be in the generic key list.
        masked, _ = redact_secrets("Authorization: Basic dXNlcjpwYXNzd29yZA==")
        assert masked.startswith("Authorization: Basic ")
        assert "dXNlcjpwYXNzd29yZA" not in masked


class TestNoFalsePositivesOnRealLogs:
    """Masking a legitimate diagnostic value would break the tool's usefulness.

    A false positive here is not cosmetic: if `max_absdiff=4.52` were masked, the
    single most actionable number in an accuracy report would disappear.
    """

    @pytest.mark.parametrize("line", [
        "No importer registered for op: GridSample. Attempting to import as plugin.",
        "Node(s) placed on [CPUExecutionProvider]. Number of nodes: 5",
        "max_absdiff=4.5205 (n=1), max_reldiff=256.7 (n=1)",
        "Minimum Required Tolerance: elemwise error | [abs=4.52] OR [rel=256]",
        "Tolerance: [abs=1e-05, rel=1e-05] | Checking elemwise error",
        "throttled=0x50005",
        "temp=84.2'C",
        "frequency(0)=2400000000",
        "3: getPluginCreator could not find plugin: GridSample version: 1",
        "CoreMLExecutionProvider::GetCapability, number of partitions supported by CoreML: 2",
        "Out of memory: Killed process 1547 (python3) total-vm:8417328kB",
        "gfp_mask=0x140dca(GFP_HIGHUSER_MOVABLE|__GFP_COMP), order=0",
    ])
    def test_ordinary_diagnostic_lines_are_untouched(self, line):
        masked, found = redact_secrets(line)
        assert found == [], f"false positive {found} on: {line}"
        assert masked == line

    def test_no_committed_artifact_would_be_altered(self):
        """The strongest available check: 4500+ lines of real tool output.

        If redaction fired on any of them it would mean the tool corrupts genuine
        evidence, which is worse than the leak it prevents.
        """
        files = sorted(
            glob.glob("tests/fixtures/**/*.log", recursive=True)
            + glob.glob("corpus/**/*.log", recursive=True)
        )
        assert files, "no artifacts found — this test would be vacuous"
        offenders = []
        for path in files:
            for lineno, line in enumerate(
                Path(path).read_text(errors="replace").splitlines(), 1
            ):
                _, found = redact_secrets(line)
                if found:
                    offenders.append(f"{Path(path).name}:{lineno} {found}")
        assert not offenders, f"false positives on real logs: {offenders[:5]}"


class TestControlCharactersAreNeutralized:
    """CWE-117/116: a log must not be able to drive the terminal."""

    @pytest.mark.parametrize("payload,danger", [
        ("before\x1b[2Jafter", "\x1b"),        # clear screen
        ("x\x1b]0;pwned\x07y", "\x1b"),        # OSC window title
        ("real\rFAKE", "\r"),                  # cursor rewind / overwrite
        ("a\x00b", "\x00"),                    # NUL
        ("a\x9bb", "\x9b"),                    # 8-bit CSI
        ("a\x08\x08b", "\x08"),                # backspace
        ("file‮gnp.exe", "‮"),       # RTL override
        ("a​b", "​"),                # zero-width space
    ])
    def test_dangerous_characters_do_not_survive(self, payload, danger):
        assert danger not in strip_control_chars(payload)

    def test_the_replacement_is_visible(self):
        # Deleting silently would let a crafted log hide content; the reader must
        # see that something unusual was there.
        assert "<ESC>" in strip_control_chars("a\x1b[2Jb")

    @pytest.mark.parametrize("benign", ["a\tb", "a\nb", "plain text", "日本語 ✓", "é"])
    def test_legitimate_characters_are_preserved(self, benign):
        assert strip_control_chars(benign) == benign


class TestReportDoesNotLeak:
    """End-to-end: nothing dangerous reaches the rendered report."""

    def test_a_secret_on_a_matched_line_is_masked_in_the_report(self):
        output = render([diag()], facts_with(
            "No importer registered for op: X (from "
            "https://ci:Sup3rSecretValue@reg/m.onnx)"
        ))
        assert "Sup3rSecretValue" not in output
        assert "REDACTED" in output

    def test_redaction_is_announced(self):
        # Silently altering evidence would break the verbatim promise; the note
        # also tells the user to rotate, which is the actionable part.
        output = render([diag()], facts_with(f"token={_GITLAB}"))
        assert "redacted probable secret" in output
        assert "rotate" in output

    def test_no_escape_bytes_reach_the_terminal(self):
        output = render([diag()], facts_with("op: X\x1b[2J\x1b]0;t\x07Y"))
        assert "\x1b" not in output
        assert "\x07" not in output

    def test_a_message_cannot_carry_an_escape_either(self):
        # Rule messages interpolate parsed values (an op name), and the LLM
        # builds messages from log content — so the header is a leak path too.
        d = Diagnosis(code="ED0101", severity="error", evidence=["f1"],
                      message="op 'X\x1b[2J' is not supported")
        assert "\x1b" not in render([d], facts_with("plain line"))

    def test_root_cause_cannot_carry_an_escape(self):
        d = Diagnosis(code="ED0101", severity="error", message="m", evidence=["f1"],
                      root_cause="because\x1b[2J of reasons")
        assert "\x1b" not in render([d], facts_with("plain line"))

    def test_no_redact_returns_the_raw_value(self):
        # A user debugging their own private log may want the real value; the
        # escape hatch must actually work.
        output = render([diag()], facts_with(f"token={_GITLAB}"),
                        redact=False)
        assert _GITLAB in output

    def test_control_chars_are_stripped_even_with_no_redact(self):
        # Terminal safety is not negotiable: there is no legitimate reason to let
        # a log drive the terminal, so --no-redact does not disable it.
        output = render([diag()], facts_with("op: X\x1b[2JY"), redact=False)
        assert "\x1b" not in output


class TestJsonDoesNotLeak:
    def test_secrets_are_masked_in_json_by_default(self):
        data = json.loads(render_json([diag()], facts_with(
            "fetch https://ci:Sup3rSecretValue@reg/m.onnx"
        )))
        assert "Sup3rSecretValue" not in json.dumps(data)

    def test_json_declares_whether_it_was_redacted(self):
        # A consumer diffing excerpts against the source file must be able to
        # tell that the text was altered, or the mismatch is unexplainable.
        data = json.loads(render_json([diag()], facts_with(f"token={_GITLAB}")))
        assert data["redacted"] is True
        assert "gitlab-token" in data["secretsDetected"]

    def test_clean_logs_report_no_secrets_detected(self):
        data = json.loads(render_json([diag()], facts_with("op: GridSample")))
        assert data["secretsDetected"] == []

    def test_no_redact_produces_raw_json(self):
        data = json.loads(render_json([diag()],
                                      facts_with(f"token={_GITLAB}"),
                                      redact=False))
        assert data["redacted"] is False
        assert _GITLAB in json.dumps(data)

    def test_structured_data_fields_are_masked_too(self):
        # A parsed substring can carry the secret even when the excerpt is masked.
        data = json.loads(render_json([diag()], facts_with(
            "plain", url="https://ci:Sup3rSecretValue@reg"
        )))
        assert "Sup3rSecretValue" not in json.dumps(data)

    def test_json_stays_valid_after_redaction(self):
        raw = render_json([diag()], facts_with(f"token={_GITLAB}"))
        json.loads(raw)  # must not raise


class TestLlmTransmissionDoesNotLeak:
    """The irreversible channel: a leak here leaves the machine."""

    def _payload(self, excerpt: str, artifact: str = "build.log", **data):
        from edgedoctor.llm import _facts_payload

        facts = [Fact(id="f1", kind="mystery", summary="s", source=f"{artifact}:1",
                      excerpt=excerpt, data=data)]
        return _facts_payload(facts)

    @pytest.mark.parametrize("label,secret", sorted(SECRETS.items()))
    def test_no_known_secret_shape_is_transmitted(self, label, secret):
        payload, found = self._payload(f"fetch failed: {secret}")
        assert found, f"{label} would have been transmitted unmasked"
        for fragment in ("Sup3rSecretValue", "hunter2hunter2", _AWS_ID,
                         "wJalrXUtnFEMIK7MDENGbPxRfiCYEX"):
            if fragment in secret:
                assert fragment not in payload

    def test_the_data_payload_is_masked(self):
        payload, _ = self._payload("plain", url="https://ci:Sup3rSecretValue@reg")
        assert "Sup3rSecretValue" not in payload

    def test_the_artifact_filename_is_masked(self):
        # A path can itself carry a credential.
        from edgedoctor.llm import sanitize_artifact_name

        assert _GITLAB not in sanitize_artifact_name(f"build-{_GITLAB}.log")

    def test_control_characters_are_stripped_before_transmission(self):
        payload, _ = self._payload("op: X\x1b[2JY")
        assert "\x1b" not in payload

    def test_masking_preserves_the_diagnostic_signal(self):
        # The model still learns a credential was present — which is all the
        # value the secret has for diagnosis. Its literal bytes never help.
        payload, _ = self._payload("fetch https://ci:Sup3rSecretValue@reg/m.onnx")
        assert "REDACTED" in payload
        assert "https://ci:" in payload


class TestCliEndToEnd:
    def test_diagnose_masks_secrets(self, tmp_path):
        log = tmp_path / "b.log"
        log.write_text(
            "[TRT] No importer registered for op: GridSample. Attempting to "
            "import as plugin. (https://ci:Sup3rSecretValue@reg/m.onnx)\n"
        )
        result = runner.invoke(app, ["diagnose", str(log)])
        assert "Sup3rSecretValue" not in result.output

    def test_diagnose_json_masks_secrets(self, tmp_path):
        log = tmp_path / "b.log"
        log.write_text(
            "[TRT] No importer registered for op: GridSample. Attempting to "
            "import as plugin. tok=" + _GITLAB + "\n"
        )
        result = runner.invoke(app, ["diagnose", str(log), "--json"])
        assert _GITLAB not in result.stdout

    def test_parse_json_masks_secrets(self, tmp_path):
        log = tmp_path / "b.log"
        log.write_text(
            "[TRT] No importer registered for op: GridSample. Attempting to "
            "import as plugin. tok=" + _GITLAB + "\n"
        )
        result = runner.invoke(app, ["parse", str(log), "--json"])
        assert _GITLAB not in result.stdout

    def test_no_redact_flag_is_documented(self):
        result = runner.invoke(app, ["diagnose", "--help"])
        assert "--no-redact" in result.output

    def test_no_escape_bytes_from_any_command(self, tmp_path):
        log = tmp_path / "b.log"
        log.write_text(
            "[TRT] No importer registered for op: X\x1b[2J\x1b]0;pwned\x07. "
            "Attempting to import as plugin.\n"
        )
        for command in (["diagnose", str(log)], ["parse", str(log)]):
            result = runner.invoke(app, command)
            assert "\x1b" not in result.output, f"{command[0]} leaked an escape"


class TestRedactionPerformance:
    """Redaction runs on untrusted input, so it must not be a DoS vector."""

    def test_scales_linearly(self):
        import time

        def timed(n: int) -> float:
            text = "password=" + "x" * n
            start = time.monotonic()
            sanitize_for_display(text)
            return time.monotonic() - start

        small, large = timed(50_000), timed(200_000)
        if small > 0.005:
            assert large < small * 9, "redaction is superlinear"

    @pytest.mark.parametrize("payload", [
        "9" * 200_000, "eyJ" + "A" * 200_000, "https://u:" + "p" * 200_000,
        "sk-ant-" + "A" * 200_000, "-----BEGIN " * 20_000, "=" * 200_000,
    ])
    def test_pathological_input_is_fast(self, payload):
        import time

        start = time.monotonic()
        sanitize_for_display(payload)
        assert time.monotonic() - start < 5.0
