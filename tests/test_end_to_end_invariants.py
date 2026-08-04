"""Whole-pipeline invariants, checked against EVERY artifact in the repo.

WHY THIS FILE EXISTS SEPARATELY
The per-backend test files each verify their own layer deeply. This one verifies
properties that span layers — parser output feeding the diagnoser feeding the
renderer — for every artifact at once. Those cross-layer failures are the ones
single-layer tests structurally cannot catch: a parser can emit a perfectly valid
fact, and a rule can be perfectly well-formed, and the combination can still
produce a report citing evidence that isn't there.

It is driven by a filesystem sweep, so a NEW ARTIFACT IS COVERED THE MOMENT IT
LANDS. No test needs editing when a corpus log or fixture is added, which is what
keeps this honest as the project grows.

The seven invariants below are the tool's actual promises, restated as
assertions. If any breaks, edgedoctor is lying to its user somewhere.
"""

import io
import json
from pathlib import Path

import pytest
from rich.console import Console

from edgedoctor.backends import get_parser
from edgedoctor.diagnoser import diagnose
from edgedoctor.report import render_human, render_json

REPO = Path(__file__).parent.parent
CORPUS = REPO / "corpus" / "onnxruntime"
FIXTURES = REPO / "tests" / "fixtures"

# Artifact -> the backend that owns it. Built by convention, because the corpus
# directory holds two families (Polygraphy comparison logs and ORT session logs)
# that a single glob cannot distinguish.
def _artifacts() -> list[tuple[str, Path]]:
    pairs: list[tuple[str, Path]] = []
    for p in sorted((FIXTURES / "tensorrt").glob("*.log")):
        pairs.append(("tensorrt", p))
    for p in sorted((FIXTURES / "raspberrypi").glob("*.log")):
        pairs.append(("raspberrypi", p))
    for p in sorted(CORPUS.glob("*.log")):
        # ort_* are ONNX Runtime session logs; the rest are Polygraphy runs.
        pairs.append(("onnxruntime" if p.name.startswith("ort_") else "polygraphy", p))
    for p in sorted(CORPUS.glob("*.json")):
        pairs.append(("ort_profile", p))
    return pairs


ARTIFACTS = _artifacts()
IDS = [f"{b}:{p.name}" for b, p in ARTIFACTS]


def pipeline(backend: str, path: Path):
    facts = get_parser(backend).parse(path)
    return facts, diagnose(facts)


def render(diagnoses, facts) -> str:
    buf = io.StringIO()
    render_human(diagnoses, facts,
                 console=Console(file=buf, width=100, no_color=True))
    return buf.getvalue()


def test_artifacts_were_discovered():
    """Guard on the guards: an empty sweep makes every test below vacuous."""
    assert len(ARTIFACTS) >= 20, f"only found {len(ARTIFACTS)} artifacts"


@pytest.mark.parametrize("backend,path", ARTIFACTS, ids=IDS)
class TestPipelineInvariants:
    """Each of these is a promise edgedoctor makes to its user."""

    def test_all_evidence_ids_resolve(self, backend, path):
        # "Every claim is traceable" — a dangling id makes a claim unverifiable.
        facts, diags = pipeline(backend, path)
        ids = {f.id for f in facts.facts}
        for d in diags:
            for eid in d.evidence:
                assert eid in ids, f"{d.code} cites missing fact {eid}"

    def test_no_diagnosis_lacks_evidence(self, backend, path):
        # A grounded diagnosis with nothing behind it is a contradiction.
        _, diags = pipeline(backend, path)
        for d in diags:
            assert d.evidence, f"{d.code} has no evidence"

    def test_no_unresolved_placeholders_reach_the_user(self, backend, path):
        # A literal "{count}" in the output means a rule references a field its
        # own evidence doesn't provide.
        _, diags = pipeline(backend, path)
        for d in diags:
            assert "{" not in d.message, f"{d.code}: {d.message!r}"

    def test_no_python_reprs_leak_into_messages(self, backend, path):
        # "['a', 'b']" in a sentence means a raw list reached the user.
        _, diags = pipeline(backend, path)
        for d in diags:
            assert "['" not in d.message and "{'" not in d.message, d.message

    def test_evidence_is_never_cited_twice(self, backend, path):
        _, diags = pipeline(backend, path)
        for d in diags:
            assert len(d.evidence) == len(set(d.evidence)), f"{d.code} duplicates"

    def test_json_output_is_strictly_valid(self, backend, path):
        # `--json` is advertised for piping into jq/JS/Go, which reject the
        # NaN/Infinity tokens Python's json emits by default.
        facts, diags = pipeline(backend, path)

        def reject(token):
            raise ValueError(f"non-spec JSON token: {token}")

        json.loads(render_json(diags, facts), parse_constant=reject)

    def test_json_round_trips_into_the_contracts(self, backend, path):
        # The JSON report must deserialize back into the pydantic models, or
        # downstream consumers of the documented schema would break.
        from edgedoctor.backends.base import Diagnosis, Fact

        facts, diags = pipeline(backend, path)
        data = json.loads(render_json(diags, facts))
        for raw in data["facts"]:
            Fact.model_validate(raw)
        for raw in data["diagnostics"]:
            Diagnosis.model_validate(raw)

    def test_human_report_renders_and_stays_bounded(self, backend, path):
        # A report nobody can read is a failure even if every fact is correct;
        # the layer-wise Polygraphy log once produced 1022 lines.
        facts, diags = pipeline(backend, path)
        output = render(diags, facts)
        assert len(output.splitlines()) < 400, "report too long to be usable"

    def test_report_never_emits_a_carriage_return(self, backend, path):
        # \r rewinds the terminal cursor, letting later text overwrite earlier
        # text — a way to hide part of a report.
        facts, diags = pipeline(backend, path)
        assert "\r" not in render(diags, facts)

    def test_no_forged_structural_lines(self, backend, path):
        # Exactly the diagnoses reported may begin a header line. More would
        # mean content forged one.
        facts, diags = pipeline(backend, path)
        output = render(diags, facts)
        headers = [
            ln for ln in output.splitlines()
            if ln.startswith(("error[", "warning[", "info["))
        ]
        assert len(headers) == len(diags), (
            f"{len(headers)} header lines for {len(diags)} diagnoses"
        )

    def test_parsing_is_deterministic(self, backend, path):
        first, _ = pipeline(backend, path)
        second, _ = pipeline(backend, path)
        assert first.model_dump() == second.model_dump()

    def test_diagnosis_is_deterministic(self, backend, path):
        _, first = pipeline(backend, path)
        _, second = pipeline(backend, path)
        assert [d.model_dump() for d in first] == [d.model_dump() for d in second]

    def test_severity_is_always_a_known_value(self, backend, path):
        # The CLI maps severity to exit codes, so an unknown value would make
        # the exit code meaningless.
        _, diags = pipeline(backend, path)
        for d in diags:
            assert d.severity in ("error", "warning", "info"), d.severity

    def test_rule_diagnoses_are_marked_as_rule_origin(self, backend, path):
        # Nothing on the offline path may claim to be synthesized, or the
        # "(synthesized)" marker would lose its meaning.
        _, diags = pipeline(backend, path)
        for d in diags:
            assert d.origin == "rules", f"{d.code} claims origin={d.origin}"

    def test_severity_ordering_puts_errors_first(self, backend, path):
        # A reader must hit the most serious finding first.
        _, diags = pipeline(backend, path)
        rank = {"error": 0, "warning": 1, "info": 2}
        seen = [rank[d.severity] for d in diags]
        assert seen == sorted(seen), f"unsorted severities: {[d.severity for d in diags]}"


@pytest.mark.parametrize("backend,path", ARTIFACTS, ids=IDS)
def test_every_fact_is_traceable(backend, path):
    """Facts must cite a real, checkable location in their own artifact."""
    facts = get_parser(backend).parse(path)
    for f in facts.facts:
        assert f.source.startswith(f"{path.name}:"), f.source
        assert f.kind and f.summary, "a fact needs a kind and a summary"


@pytest.mark.parametrize(
    "backend,path",
    [(b, p) for b, p in ARTIFACTS if p.suffix == ".log"],
    ids=[f"{b}:{p.name}" for b, p in ARTIFACTS if p.suffix == ".log"],
)
def test_excerpts_are_verbatim_source_lines(backend, path):
    """The core product claim, checked for every text artifact at once.

    An excerpt that differs from its cited line means the tool is showing the
    user something other than their own log.
    """
    lines = path.read_text(errors="replace").splitlines()
    for f in get_parser(backend).parse(path).facts:
        ref = f.source.rsplit(":", 1)[1]
        if not ref.isdigit():
            continue
        lineno = int(ref)
        assert 1 <= lineno <= len(lines), f"cites line {lineno} of {len(lines)}"
        assert f.excerpt == lines[lineno - 1].strip(), (
            f"{path.name}:{lineno} excerpt differs from the source line"
        )


@pytest.mark.parametrize("backend,path", ARTIFACTS, ids=IDS)
def test_exit_code_matches_the_reported_severities(backend, path):
    """The documented exit-code contract: 2 = errors, 3 = warnings only, 0 = clean.

    CI systems branch on this, so a mismatch between what the report SAYS and
    what the process RETURNS is a silent integration bug.
    """
    _, diags = pipeline(backend, path)
    has_error = any(d.severity == "error" for d in diags)
    has_warning = any(d.severity == "warning" for d in diags)
    expected = 2 if has_error else 3 if has_warning else 0

    from typer.testing import CliRunner

    from edgedoctor.cli import app

    runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "100"})
    result = runner.invoke(app, ["diagnose", str(path), "-b", backend])
    assert result.exit_code == expected, (
        f"{path.name}: report has "
        f"error={has_error} warning={has_warning} but exit={result.exit_code}"
    )
