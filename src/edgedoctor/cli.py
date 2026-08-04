"""edgedoctor command-line interface.

A *thin* surface over the library core: argument parsing and presentation live
here, logic lives in the library, so the future MCP server wraps the same core.

Built with typer: CLI args/options are declared as typed function parameters,
and typer derives parsing, validation, help text, and shell completion from
them. It also brings rich, which we use for readable terminal output.

Output discipline (per clig.dev):
  - primary results   -> stdout   (so `edgedoctor ... | jq` works)
  - progress/messages -> stderr
  - exit codes: 0 = healthy, 1 = tool error, 2 = errors found, 3 = warnings only
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

import typer
from rich.console import Console

from . import __version__
from .backends import PARSER_REGISTRY, get_parser
from .redact import sanitize_for_display, strip_control_chars

app = typer.Typer(
    name="edgedoctor",
    help="The universal, cross-vendor edge-AI deployment diagnostician.",
    add_completion=True,
    no_args_is_help=True,
)

# stdout for reports, stderr for chatter — keeps piped output clean.
out = Console()
err = Console(stderr=True)


class BackendName(str, Enum):
    """Backends edgedoctor knows the *name* of.

    Only some have parsers; the rest are declared so --help honestly shows the
    planned cross-vendor surface without claiming they work. (An Enum gives
    typer choice-validation + tab completion.)

    Two entries aren't silicon at all, because what this flag really selects is
    "which parser + rule family":
      - `polygraphy` is NVIDIA's model-comparison tool, whose logs describe a
        comparison BETWEEN backends, so it belongs to neither.
      - `raspberrypi` describes the HOST rather than a model — throttling and
        OOM state, which say whether a measurement is trustworthy at all.
      - `ort_profile` is a second ONNX Runtime lane: the placement log says
        WHERE ops ran, a profiling trace says what they COST.
    """

    tensorrt = "tensorrt"
    polygraphy = "polygraphy"
    onnxruntime = "onnxruntime"
    raspberrypi = "raspberrypi"
    ort_profile = "ort_profile"
    coreml = "coreml"
    tflite = "tflite"
    executorch = "executorch"


# Backends whose full diagnose pipeline (parse → rules → report) works.
# Derived from the registry so this can never drift out of sync with reality.
IMPLEMENTED_BACKENDS: set[BackendName] = {
    b for b in BackendName if b.value in PARSER_REGISTRY
}


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"edgedoctor {__version__}")
        raise typer.Exit()


@app.callback()
def _app_callback(
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """The universal, cross-vendor edge-AI deployment diagnostician."""


def _require_readable_file(artifact: Path) -> None:
    """Exit 1 with a clean message unless `artifact` is a readable file.

    Checked explicitly rather than relying on the later read_text(): passing a
    directory used to surface a raw IsADirectoryError traceback, which violates
    the tool's own output discipline (errors belong on stderr as one line, and a
    traceback tells a user nothing actionable). A path that exists but is not a
    regular file — a directory, a device node, a broken symlink — is a usage
    error, not a tool crash.
    """
    if artifact.is_dir():
        err.print(f"[red]error:[/red] {artifact} is a directory, not a log file")
        raise typer.Exit(code=1)
    if not artifact.exists():
        # Covers both a missing path and a dangling symlink, since exists()
        # follows links.
        err.print(f"[red]error:[/red] file not found: {artifact}")
        raise typer.Exit(code=1)
    if not artifact.is_file():
        err.print(f"[red]error:[/red] {artifact} is not a regular file")
        raise typer.Exit(code=1)
    if not os.access(artifact, os.R_OK):
        # A readable check here keeps the "1 = usage error" contract: without
        # it, read_text() raises PermissionError, which surfaces as exit 2 —
        # the code reserved for "errors found in the log". A file we can't open
        # is a usage problem, not a diagnosis result.
        err.print(f"[red]error:[/red] cannot read {artifact} (permission denied)")
        raise typer.Exit(code=1)


@app.command()
def diagnose(
    artifact: Path = typer.Argument(
        ...,
        help="Build log or artifact to diagnose (e.g. build.log).",
        # readable=False disables typer/click's own permission check, which
        # would otherwise raise a UsageError and exit 2 — the code this tool
        # reserves for "errors were found in the log". A file we cannot open is
        # a usage problem (exit 1), and conflating the two would make a CI job
        # treat a permissions mistake as a failed model. _require_readable_file
        # below does the check and reports it correctly.
        readable=False,
    ),
    backend: BackendName = typer.Option(
        BackendName.tensorrt, "--backend", "-b", help="Target edge backend."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON on stdout."
    ),
    llm: bool = typer.Option(
        False, "--llm",
        help="Additionally try to explain facts no rule matched, using an LLM "
             "(needs ANTHROPIC_API_KEY and the [llm] extra). Off by default: "
             "the rules-only path is deterministic and free.",
    ),
    no_redact: bool = typer.Option(
        False, "--no-redact",
        help="Show evidence unmasked. By default probable secrets (tokens, "
             "URL credentials) are replaced with a visible marker, because "
             "build logs contain credentials and reports get shared. Control "
             "characters are always neutralized regardless.",
    ),
) -> None:
    """Diagnose why a model fails or underperforms on an edge backend.

    Parses the artifact, matches facts against the rule knowledge base, and
    renders a grounded report with evidence, root cause, and fix suggestions.
    Works fully offline by default — no LLM, no API key.

    Pass --llm to opt into synthesis for facts the rules don't cover. It is
    strictly additive: it only sees unmatched facts, every claim it makes must
    cite a real parsed fact, and any failure leaves the rules-based report
    exactly as it would have been.
    """
    err.print(f"[dim]edgedoctor v{__version__} · artifact={artifact} · "
              f"backend={backend.value}[/dim]")

    if backend not in IMPLEMENTED_BACKENDS:
        err.print()
        err.print("[yellow]⚠  Not implemented yet.[/yellow]")
        err.print(
            f"   The '{backend.value}' diagnosis pipeline is still being built."
        )
        err.print("   See ROADMAP.md for what's now / next / later.")
        raise typer.Exit(code=1)

    _require_readable_file(artifact)

    # 1. Parse
    facts = get_parser(backend.value).parse(artifact)

    # 2. Diagnose (rule-based, deterministic)
    from .diagnoser import diagnose as run_diagnosis
    diagnoses = run_diagnosis(facts)

    # 2b. Optional synthesis over whatever the rules couldn't explain.
    # Reported on stderr so it never pollutes --json on stdout.
    if llm:
        from .llm import availability, synthesize

        ok, reason = availability()
        if not ok:
            err.print(f"[yellow]note:[/yellow] --llm requested but {reason}.")
            err.print("[dim]      Continuing with rules-only diagnosis.[/dim]")
        else:
            err.print("[dim]synthesizing unmatched facts...[/dim]")
            synthesized = synthesize(facts, diagnoses)
            if synthesized:
                diagnoses = diagnoses + synthesized
            else:
                err.print("[dim]no additional grounded diagnoses.[/dim]")

    # 3. Render
    from .report import render_human, render_json

    if json_output:
        print(render_json(diagnoses, facts, redact=not no_redact))
        # Falls through to the SAME exit-code logic as the human path. It used to
        # exit 0 unconditionally, which broke the documented contract in exactly
        # the situation --json exists for: a CI job piping the report to jq saw
        # success on a run that found errors. The report format must not change
        # the verdict.
    else:
        render_human(diagnoses, facts, console=out, redact=not no_redact)

    # Exit code per contract: 2 = errors found, 3 = warnings only, 0 = clean.
    #
    # A SYNTHESIZED finding can never produce exit 2, only 3. The exit code is
    # an API that CI gates branch on, and an LLM diagnosis is unreviewed and
    # capped at medium confidence — letting one fail a build would contradict
    # this layer's whole premise, documented in docs/adr/0001, that `--llm` can
    # only ever add and never degrade a run. Capping at 3 keeps a genuine
    # LLM-only finding visible to automation (it is not silently swallowed)
    # while reserving the hard-failure code for curated, human-reviewed rules.
    rule_diagnoses = [d for d in diagnoses if d.origin != "llm"]
    has_rule_errors = any(d.severity == "error" for d in rule_diagnoses)
    has_rule_warnings = any(d.severity == "warning" for d in rule_diagnoses)
    # Any synthesized finding at all is worth a non-zero code, at warning level.
    has_synthesized = len(diagnoses) != len(rule_diagnoses)

    if has_rule_errors:
        raise typer.Exit(code=2)
    elif has_rule_warnings or has_synthesized:
        raise typer.Exit(code=3)
    raise typer.Exit(code=0)


@app.command()
def parse(
    artifact: Path = typer.Argument(
        ...,
        help="Raw artifact to parse (e.g. a trtexec build log).",
        readable=False,  # see the note on diagnose(); we report this ourselves
    ),
    backend: BackendName = typer.Option(
        BackendName.tensorrt, "--backend", "-b", help="Backend that produced the artifact."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit the Facts as JSON on stdout."
    ),
    no_redact: bool = typer.Option(
        False, "--no-redact",
        help="Show evidence unmasked. By default probable secrets (tokens, "
             "URL credentials) are replaced with a visible marker, because "
             "build logs contain credentials and reports get shared. Control "
             "characters are always neutralized regardless.",
    ),
) -> None:
    """Extract structured Facts from a raw vendor artifact (no LLM, no network).

    This is the deterministic first stage of diagnosis, exposed directly so
    you can inspect exactly what the parser saw — every fact cites file:line.
    """
    _require_readable_file(artifact)

    if backend.value not in PARSER_REGISTRY:
        err.print(f"[red]error:[/red] no parser for '{backend.value}' yet — see ROADMAP.md")
        raise typer.Exit(code=1)

    facts = get_parser(backend.value).parse(artifact)

    if json_output:
        # Machine consumers get the exact pydantic contract on stdout, with
        # secrets masked by default — this is the path that feeds CI artifacts
        # and agents, where a credential is most likely to be persisted or
        # forwarded. --no-redact opts out.
        if not no_redact:
            from .report import redacted_facts

            facts, kinds = redacted_facts(facts)
            if kinds:
                err.print(f"[yellow]note:[/yellow] redacted probable secret(s) "
                          f"({', '.join(kinds)}) — rotate and scrub the log.")
        print(facts.model_dump_json(indent=2))
        return

    if not facts.facts:
        out.print(f"No known signatures matched in [bold]{artifact.name}[/bold].")
        out.print("[dim]This may be a clean log, or a failure mode edgedoctor "
                  "doesn't know yet — if the run DID fail, please open an issue "
                  "with the log attached.[/dim]")
        return

    from rich.table import Table

    redacted: set[str] = set()
    table = Table(title=f"Facts extracted from {artifact.name}")
    table.add_column("source", style="dim", no_wrap=True)
    table.add_column("kind", style="cyan")
    table.add_column("observation")
    for f in facts.facts:
        # The summary is built from log-derived data (op names, tensor names), so
        # it is untrusted text: neutralize control characters and mask probable
        # secrets before printing. Same reasoning as the diagnose report — see
        # edgedoctor.redact.
        safe_summary, kinds = sanitize_for_display(f.summary, redact=not no_redact)
        redacted.update(kinds)
        table.add_row(strip_control_chars(f.source), f.kind, safe_summary)
    out.print(table)
    if redacted:
        err.print(f"[yellow]note:[/yellow] redacted probable secret(s) "
                  f"({', '.join(sorted(redacted))}) — rotate and scrub the log.")
    out.print(f"[dim]{len(facts.facts)} fact(s). Run with --json for the full "
              f"structured output including verbatim excerpts.[/dim]")


def main() -> None:
    """Entry point for `[project.scripts]` in pyproject.toml."""
    app()


if __name__ == "__main__":
    main()
