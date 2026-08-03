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

from enum import Enum
from pathlib import Path

import typer
from rich.console import Console

from . import __version__
from .backends import PARSER_REGISTRY, get_parser

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

    `polygraphy` is the odd one out: it isn't a piece of silicon but NVIDIA's
    model-comparison tool, whose logs describe a comparison BETWEEN backends.
    It's listed here because "which parser + rule family" is exactly what this
    flag selects.
    """

    tensorrt = "tensorrt"
    polygraphy = "polygraphy"
    onnxruntime = "onnxruntime"
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


@app.command()
def diagnose(
    artifact: Path = typer.Argument(
        ..., help="Build log or artifact to diagnose (e.g. build.log)."
    ),
    backend: BackendName = typer.Option(
        BackendName.tensorrt, "--backend", "-b", help="Target edge backend."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON on stdout."
    ),
) -> None:
    """Diagnose why a model fails or underperforms on an edge backend.

    Parses the artifact, matches facts against the rule knowledge base, and
    renders a grounded report with evidence, root cause, and fix suggestions.
    Works fully offline — no LLM, no API key.
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

    if not artifact.exists():
        err.print(f"[red]error:[/red] file not found: {artifact}")
        raise typer.Exit(code=1)

    # 1. Parse
    facts = get_parser(backend.value).parse(artifact)

    # 2. Diagnose (rule-based, deterministic)
    from .diagnoser import diagnose as run_diagnosis
    diagnoses = run_diagnosis(facts)

    # 3. Render
    from .report import render_human, render_json

    if json_output:
        print(render_json(diagnoses, facts))
        raise typer.Exit(code=0)

    render_human(diagnoses, facts, console=out)

    # Exit code per contract: 2 = errors found, 3 = warnings only, 0 = clean.
    has_errors = any(d.severity == "error" for d in diagnoses)
    has_warnings = any(d.severity == "warning" for d in diagnoses)
    if has_errors:
        raise typer.Exit(code=2)
    elif has_warnings:
        raise typer.Exit(code=3)
    raise typer.Exit(code=0)


@app.command()
def parse(
    artifact: Path = typer.Argument(
        ..., help="Raw artifact to parse (e.g. a trtexec build log)."
    ),
    backend: BackendName = typer.Option(
        BackendName.tensorrt, "--backend", "-b", help="Backend that produced the artifact."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit the Facts as JSON on stdout."
    ),
) -> None:
    """Extract structured Facts from a raw vendor artifact (no LLM, no network).

    This is the deterministic first stage of diagnosis, exposed directly so
    you can inspect exactly what the parser saw — every fact cites file:line.
    """
    if not artifact.exists():
        err.print(f"[red]error:[/red] file not found: {artifact}")
        raise typer.Exit(code=1)

    if backend.value not in PARSER_REGISTRY:
        err.print(f"[red]error:[/red] no parser for '{backend.value}' yet — see ROADMAP.md")
        raise typer.Exit(code=1)

    facts = get_parser(backend.value).parse(artifact)

    if json_output:
        # Machine consumers get the exact pydantic contract on stdout.
        print(facts.model_dump_json(indent=2))
        return

    if not facts.facts:
        out.print(f"No known signatures matched in [bold]{artifact.name}[/bold].")
        out.print("[dim]This may be a clean log, or a failure mode edgedoctor "
                  "doesn't know yet — if the run DID fail, please open an issue "
                  "with the log attached.[/dim]")
        return

    from rich.table import Table

    table = Table(title=f"Facts extracted from {artifact.name}")
    table.add_column("source", style="dim", no_wrap=True)
    table.add_column("kind", style="cyan")
    table.add_column("observation")
    for f in facts.facts:
        table.add_row(f.source, f.kind, f.summary)
    out.print(table)
    out.print(f"[dim]{len(facts.facts)} fact(s). Run with --json for the full "
              f"structured output including verbatim excerpts.[/dim]")


def main() -> None:
    """Entry point for `[project.scripts]` in pyproject.toml."""
    app()


if __name__ == "__main__":
    main()
