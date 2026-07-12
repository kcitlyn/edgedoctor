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

    Only tensorrt has even a stub implementation; the rest are declared so
    --help honestly shows the planned cross-vendor surface without claiming
    they work. (An Enum gives typer choice-validation + tab completion.)
    """

    tensorrt = "tensorrt"
    onnxruntime = "onnxruntime"
    coreml = "coreml"
    tflite = "tflite"
    executorch = "executorch"


IMPLEMENTED_BACKENDS: set[BackendName] = set()  # none fully built yet — honest


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
    model: Path = typer.Argument(..., help="Path to the model (e.g. model.onnx)."),
    backend: BackendName = typer.Option(
        BackendName.tensorrt, "--backend", "-b", help="Target edge backend."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON on stdout."
    ),
) -> None:
    """Diagnose why a model fails or underperforms on an edge backend."""
    err.print(f"[dim]edgedoctor v{__version__} · model={model} · backend={backend.value}[/dim]")

    if backend not in IMPLEMENTED_BACKENDS:
        # Honest stub: report status instead of faking a diagnosis.
        err.print()
        err.print("[yellow]⚠  Not implemented yet.[/yellow]")
        err.print(
            f"   The '{backend.value}' diagnosis pipeline is still being built (Phases 1–2)."
        )
        err.print("   See ROADMAP.md for what's now / next / later.")
        raise typer.Exit(code=1)


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

    if backend is not BackendName.tensorrt:
        err.print(f"[red]error:[/red] no parser for '{backend.value}' yet — see ROADMAP.md")
        raise typer.Exit(code=1)

    from .backends.tensorrt import TensorRTBackend

    facts = TensorRTBackend().parse(artifact)

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
