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

from . import __version__
from .backends.base import Diagnosis, Facts

# Max evidence blocks printed per diagnosis before the rest are summarized.
# 4 keeps a diagnosis readable on one screen while still showing the primary
# proof plus a couple of supporting measurements.
MAX_EVIDENCE_SHOWN = 4

# Max characters of a single excerpt line shown in the terminal. Polygraphy
# prints its full mismatched-output list on one line, which for a layer-wise
# run is ~3000 characters. Clipping is marked with a visible "… (+N chars)" so
# the reader knows the line continued; --json keeps the untouched excerpt.
MAX_EXCERPT_CHARS = 300


def _clip(text: str, limit: int = MAX_EXCERPT_CHARS) -> str:
    """Shorten an over-long excerpt, marking the omission explicitly."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… (+{len(text) - limit} chars, use --json for the full line)"

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
        # LLM-synthesized diagnoses are marked inline. A reader must never have
        # to guess whether an explanation was human-reviewed (a curated rule) or
        # generated on the spot — they carry very different weight, and an
        # unmarked synthesis would borrow trust the rules earned.
        marker = " [magenta](synthesized)[/magenta]" if diag.origin == "llm" else ""
        con.print(
            f"[{style}]{diag.severity}[{diag.code}][/{style}]{marker}: {diag.message}"
        )

        # Evidence block — show the user's own log lines.
        # Capped: a layer-wise Polygraphy run can cite ~200 facts for one
        # diagnosis, which buries the explanation in 1000 lines of log. rustc
        # does the same thing with repeated errors. The cap is ANNOUNCED, never
        # silent — an omission the reader can't see would misrepresent how much
        # evidence exists. `--json` always contains every fact id.
        shown = 0
        for fid in diag.evidence:
            fact = fact_map.get(fid)
            if not fact:
                continue
            if shown >= MAX_EVIDENCE_SHOWN:
                remaining = len(diag.evidence) - shown
                con.print(
                    f"   [dim]... and {remaining} more supporting fact(s) "
                    f"(use --json to see all)[/dim]"
                )
                break
            shown += 1
            con.print(f"  [dim]--> {fact.source}[/dim]")
            if fact.excerpt:
                # Indent the verbatim excerpt like rustc does.
                lineno = fact.source.rsplit(":", 1)[-1] if ":" in fact.source else ""
                con.print("   [dim]|[/dim]")
                # markup=False and highlight=False are REQUIRED, not stylistic.
                # Real logs contain square brackets — Polygraphy prints
                # "Tolerance: [abs=1e-05, rel=1e-05]" — and rich would parse
                # "[abs=1e-05, rel=1e-05]" as a style tag and DELETE it from the
                # output. Silently altering the user's own log line breaks the
                # core promise that evidence is verbatim. Covered by
                # tests/test_report.py::TestEvidenceIsVerbatim.
                con.print(f"[dim]{lineno:>4}[/dim] | ", end="")
                con.print(
                    _clip(fact.excerpt), markup=False, highlight=False, soft_wrap=True
                )
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
    infos = sum(1 for d in diagnoses if d.severity == "info")
    parts = []
    if errors:
        parts.append(f"[red]{errors} error{'s' if errors != 1 else ''}[/red]")
    if warnings:
        parts.append(f"[yellow]{warnings} warning{'s' if warnings != 1 else ''}[/yellow]")
    if infos:
        parts.append(f"[blue]{infos} note{'s' if infos != 1 else ''}[/blue]")
    # An info-only run used to render "summary:  · parsed N fact(s)" — counting
    # only errors/warnings left the clause empty. A clean result deserves to be
    # stated, not implied by absence.
    if not parts:
        parts.append("no issues found")
    con.print(f"[bold]summary:[/bold] {', '.join(parts)} · "
              f"parsed {len(facts.facts)} fact(s) from {facts.artifact_path}")


def render_json(diagnoses: list[Diagnosis], facts: Facts) -> str:
    """Serialize to the machine-readable JSON report format."""
    report = {
        "schemaVersion": 1,
        "tool": {"name": "edgedoctor", "version": __version__},
        "backend": facts.backend,
        "artifact": facts.artifact_path,
        "diagnostics": [d.model_dump() for d in diagnoses],
        "facts": [f.model_dump() for f in facts.facts],
    }
    return json.dumps(report, indent=2)
