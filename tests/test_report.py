"""Tests for the terminal report renderer.

The headline test here is TestEvidenceIsVerbatim. report.py contains a comment
promising "Covered by tests/test_report.py::TestEvidenceIsVerbatim." — this file
makes that promise true. The core product claim is that the evidence lines shown
to a user are their OWN log lines, unaltered. rich's markup engine would happily
eat any "[...]" in a log line (and real logs are full of them), so the renderer
disables markup on excerpts; these tests pin that it stays disabled.
"""

import io

from rich.console import Console

from edgedoctor.backends.base import Diagnosis, Fact, Facts, Suggestion
from edgedoctor.report import MAX_EVIDENCE_SHOWN, _clip, render_human


def render(diagnoses, facts) -> str:
    """Render to a plain string, forcing deterministic non-interactive output.

    force_terminal=False + no_color keeps ANSI escapes out so assertions can
    match on the literal text a user would read.
    """
    buf = io.StringIO()
    con = Console(file=buf, width=100, no_color=True, force_terminal=False, highlight=False)
    render_human(diagnoses, facts, console=con)
    return buf.getvalue()


def make_facts(*facts: Fact) -> Facts:
    return Facts(backend="polygraphy", artifact_path="test.log", facts=list(facts))


class TestEvidenceIsVerbatim:
    """A user's log line must appear in the report exactly as written."""

    def test_square_brackets_survive_rendering(self):
        # This is the exact line Polygraphy emits. If rich markup were left on,
        # "[abs=1e-05, rel=1e-05]" would be parsed as a style tag and DELETED.
        raw = "Tolerance: [abs=1e-05, rel=1e-05] | Checking elemwise error"
        fact = Fact(
            id="f1",
            kind="tolerance_setting",
            summary="tolerance used",
            source="test.log:16",
            excerpt=raw,
        )
        diag = Diagnosis(
            code="ED0201",
            severity="error",
            message="output diverged",
            evidence=["f1"],
        )
        output = render([diag], make_facts(fact))
        assert "[abs=1e-05, rel=1e-05]" in output

    def test_excerpt_is_not_paraphrased(self):
        raw = "FAILED | Output: 'output' | Difference exceeds tolerance (rel=256.7, abs=4.52)"
        fact = Fact(
            id="f1",
            kind="output_mismatch",
            summary="a paraphrase that must NOT appear verbatim in place of the line",
            source="test.log:44",
            excerpt=raw,
        )
        diag = Diagnosis(code="ED0201", severity="error", message="x", evidence=["f1"])
        output = render([diag], make_facts(fact))
        assert raw in output


class TestEvidenceCap:
    """A layer-wise run can attach ~200 facts to one diagnosis. The report caps
    what it prints, but the cap must be announced, never silent."""

    def _many_facts(self, n: int) -> Facts:
        facts = [
            Fact(id=f"f{i}", kind="required_tolerance", summary=f"tol {i}",
                 source=f"test.log:{i}", excerpt=f"Minimum Required Tolerance {i}")
            for i in range(1, n + 1)
        ]
        return make_facts(*facts)

    def test_caps_number_of_evidence_blocks(self):
        n = 50
        facts = self._many_facts(n)
        diag = Diagnosis(code="ED0202", severity="error", message="many diverged",
                         evidence=[f.id for f in facts.facts])
        output = render([diag], facts)
        # The last excerpt must not be printed once the cap kicks in.
        assert "Minimum Required Tolerance 50" not in output
        # The first few must be.
        assert "Minimum Required Tolerance 1" in output

    def test_announces_how_many_were_hidden(self):
        n = 50
        facts = self._many_facts(n)
        diag = Diagnosis(code="ED0202", severity="error", message="many diverged",
                         evidence=[f.id for f in facts.facts])
        output = render([diag], facts)
        hidden = n - MAX_EVIDENCE_SHOWN
        assert f"and {hidden} more" in output
        assert "use --json" in output

    def test_no_cap_notice_when_under_limit(self):
        facts = self._many_facts(2)
        diag = Diagnosis(code="ED0201", severity="error", message="one diverged",
                         evidence=[f.id for f in facts.facts])
        output = render([diag], facts)
        assert "more supporting fact" not in output


class TestClip:
    def test_short_text_is_untouched(self):
        assert _clip("short line") == "short line"

    def test_long_text_is_marked(self):
        long = "x" * 5000
        clipped = _clip(long)
        assert len(clipped) < len(long)
        assert "use --json for the full line" in clipped
        assert "chars" in clipped

    def test_reports_the_number_of_omitted_chars(self):
        long = "y" * 400
        clipped = _clip(long, limit=100)
        assert "+300 chars" in clipped


class TestSummaryLine:
    def test_error_run_counts_errors(self):
        fact = Fact(id="f1", kind="output_mismatch", summary="s", source="test.log:1")
        diag = Diagnosis(code="ED0201", severity="error", message="m", evidence=["f1"])
        output = render([diag], make_facts(fact))
        assert "1 error" in output

    def test_info_only_run_is_stated_not_left_blank(self):
        # Regression: an info-only run used to render "summary:  · parsed N".
        fact = Fact(id="f1", kind="all_outputs_matched", summary="s", source="test.log:1")
        diag = Diagnosis(code="ED0205", severity="info", message="all matched",
                         evidence=["f1"])
        output = render([diag], make_facts(fact))
        assert "1 note" in output
        assert "summary:  ·" not in output  # no empty clause

    def test_no_diagnoses_prints_honest_nonanswer(self):
        facts = make_facts(Fact(id="f1", kind="run_verdict", summary="s",
                                source="test.log:1"))
        output = render([], facts)
        assert "No known failure patterns matched" in output


class TestReportStructure:
    def test_shows_code_severity_and_help(self):
        fact = Fact(id="f1", kind="output_mismatch", summary="s",
                    source="test.log:44", excerpt="FAILED | Output: 'output'")
        diag = Diagnosis(
            code="ED0201",
            severity="error",
            message="output diverged beyond tolerance",
            root_cause="reduced precision exceeds the FP32 default tolerance",
            suggestions=[
                Suggestion(summary="judge on a task metric", applicability="maybe-incorrect"),
                Suggestion(summary="find the first layer",
                           command="polygraphy run ...",
                           applicability="machine-applicable"),
            ],
            evidence=["f1"],
            confidence="high",
        )
        output = render([diag], make_facts(fact))
        assert "ED0201" in output
        assert "error" in output
        assert "note:" in output
        assert "help:" in output
        assert "help (safe to apply):" in output  # machine-applicable label
        assert "confidence: high" in output
