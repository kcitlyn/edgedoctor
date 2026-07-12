"""Report renderer — turns Diagnosis objects into terminal output.

Implements the rustc-style report anatomy:
  error[ED0101]: op 'GridSample' is not supported by this TensorRT ONNX parser
    --> trtexec.log:412
     |
  412 | No importer registered for op: GridSample. Attempting to import as plugin.
     |
     = note: The TensorRT ONNX parser has no importer for this operator...
     = help: Re-export with a newer ONNX opset
     = confidence: high

Also handles JSON output (just serializes Diagnosis models) and the end-of-run
summary line.
"""

from __future__ import annotations

import json

from rich.console import Console

from .backends.base import Diagnosis, Facts

# Severity → color/style for the header.
_SEVERITY_STYLE = {
    "error": "bold red",
    "warning": "bold yellow",
    "info": "bold blue",
}


def render_human(
    diagnoses: list[Diagnosis],
    facts: Facts,
    console: Console | None = None,
) -> None:
    """Print the rustc-style report to the console."""
    con = console or Console()

    if not diagnoses:
        con.print("[dim]No known failure patterns matched.[/dim]")
        con.print("[dim]The log may be clean, or the failure mode is one "
                  "edgedoctor doesn't know yet.[/dim]")
        con.print("[dim]Run with `edgedoctor parse <log>` to see raw extracted facts.[/dim]")
        return

    # Build a lookup from fact id → Fact for rendering evidence.
    fact_map = {f.id: f for f in facts.facts}

    for diag in diagnoses:
        style = _SEVERITY_STYLE.get(diag.severity, "bold")
        # Header line: error[ED0101]: message
        con.print(
            f"[{style}]{diag.severity}[{diag.code}][/{style}]: {diag.message}"
        )

        # Evidence block — show the user's own log lines.
        for fid in diag.evidence:
            fact = fact_map.get(fid)
            if not fact:
                continue
            con.print(f"  [dim]--> {fact.source}[/dim]")
            if fact.excerpt:
                # Indent the verbatim excerpt like rustc does.
                lineno = fact.source.rsplit(":", 1)[-1] if ":" in fact.source else ""
                con.print("   [dim]|[/dim]")
                con.print(f"[dim]{lineno:>4}[/dim] | {fact.excerpt}")
                con.print("   [dim]|[/dim]")

        # Note (the "why").
        if diag.root_cause:
            # Wrap long text to terminal width, indented under " = note:"
            lines = diag.root_cause.splitlines()
            con.print(f"   = [bold]note:[/bold] {lines[0]}")
            for line in lines[1:]:
                con.print(f"          {line}")

        # Help (the "fix").
        for sug in diag.suggestions:
            label = "help"
            if sug.applicability == "machine-applicable":
                label = "help (safe to apply)"
            con.print(f"   = [bold green]{label}:[/bold green] {sug.summary}")
            if sug.command:
                con.print(f"             [cyan]{sug.command}[/cyan]")

        # Confidence.
        con.print(f"   = [dim]confidence: {diag.confidence}[/dim]")
        con.print()  # blank line between diagnoses

    # Summary line at the end.
    errors = sum(1 for d in diagnoses if d.severity == "error")
    warnings = sum(1 for d in diagnoses if d.severity == "warning")
    parts = []
    if errors:
        parts.append(f"[red]{errors} error{'s' if errors != 1 else ''}[/red]")
    if warnings:
        parts.append(f"[yellow]{warnings} warning{'s' if warnings != 1 else ''}[/yellow]")
    con.print(f"[bold]summary:[/bold] {', '.join(parts)} · "
              f"parsed {len(facts.facts)} fact(s) from {facts.artifact_path}")


def render_json(diagnoses: list[Diagnosis], facts: Facts) -> str:
    """Serialize to the machine-readable JSON report format."""
    report = {
        "schemaVersion": 1,
        "tool": {"name": "edgedoctor", "version": "0.1.0"},
        "backend": facts.backend,
        "artifact": facts.artifact_path,
        "diagnostics": [d.model_dump() for d in diagnoses],
        "facts": [f.model_dump() for f in facts.facts],
    }
    return json.dumps(report, indent=2)
