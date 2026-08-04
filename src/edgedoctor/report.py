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
from typing import Any

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


# Max characters of a rendered message. Long enough for any real headline,
# short enough that one diagnosis can't scroll the others off screen.
MAX_MESSAGE_CHARS = 400

# A root cause is a paragraph, so it gets more room than a headline — but still
# bounded, so one diagnosis cannot bury the rest of the report.
MAX_CAUSE_LINE_CHARS = 500
MAX_CAUSE_LINES = 12

# A suggestion is a single actionable sentence, plus at most one command.
MAX_SUGGESTION_CHARS = 300


def _clip(text: str, limit: int = MAX_EXCERPT_CHARS) -> str:
    """Shorten an over-long excerpt, marking the omission explicitly."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}… (+{len(text) - limit} chars, use --json for the full line)"


def _oneline(text: str, limit: int = MAX_MESSAGE_CHARS) -> str:
    """Flatten a single-line field so it cannot forge report structure.

    THIS IS A SECURITY BOUNDARY, not cosmetics. The report's structure IS its
    meaning: `error[ED0101]: ...` at the start of a line means "edgedoctor
    asserts this", and `= help:` means "edgedoctor recommends this". A newline
    inside a one-line field lets its content start a new line that mimics either,
    producing a fabricated diagnosis or a fabricated recommendation that a reader
    cannot distinguish from a real one.

    That is reachable in practice: the LLM synthesis layer builds `message` and
    `root_cause` from parsed log content, so a crafted log could carry
    "ok\\nerror[ED0101]: run rm -rf /" into a header position. Rule-authored text
    is trusted, but the renderer must not depend on its input being trusted.

    Carriage returns are stripped too — on a terminal, `\\r` rewinds to the line
    start, so trailing text can visually overwrite what preceded it.
    """
    flattened = " ".join(str(text).split())
    if len(flattened) <= limit:
        return flattened
    return f"{flattened[:limit]}… (truncated, use --json for the full text)"

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
        # The message is printed with markup=False so a message containing
        # "[bold red]" can't restyle the report, and flattened by _oneline so it
        # can't start a line that mimics a header. The prefix is printed
        # separately because it is OUR text and does need its styling.
        con.print(
            f"[{style}]{diag.severity}[{diag.code}][/{style}]{marker}: ", end=""
        )
        con.print(_oneline(diag.message), markup=False, highlight=False,
                  soft_wrap=True)

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
            # Multi-line causes are legitimate (rule authors write paragraphs),
            # but every continuation line is INDENTED so it cannot begin at
            # column 0 where a forged "error[...]" header would live, and each
            # is printed with markup disabled.
            lines = str(diag.root_cause).splitlines() or [""]
            con.print("   = [bold]note:[/bold] ", end="")
            con.print(_oneline(lines[0], limit=MAX_CAUSE_LINE_CHARS),
                      markup=False, highlight=False, soft_wrap=True)
            for line in lines[1:MAX_CAUSE_LINES]:
                con.print("          ", end="")
                con.print(_oneline(line, limit=MAX_CAUSE_LINE_CHARS),
                          markup=False, highlight=False, soft_wrap=True)
            if len(lines) > MAX_CAUSE_LINES:
                con.print(f"          [dim]... ({len(lines) - MAX_CAUSE_LINES} "
                          f"more line(s), use --json)[/dim]")

        # Help (the "fix").
        for sug in diag.suggestions:
            label = "help"
            if sug.applicability == "machine-applicable":
                label = "help (safe to apply)"
            # Flattened for the same reason as the message: "help (safe to
            # apply)" is a claim edgedoctor makes about a command's safety, so a
            # suggestion that could forge that line could get an unreviewed
            # command run unattended by an agent.
            con.print(f"   = [bold green]{label}:[/bold green] ", end="")
            con.print(_oneline(sug.summary, limit=MAX_SUGGESTION_CHARS),
                      markup=False, highlight=False, soft_wrap=True)
            if sug.command:
                con.print("             ", end="")
                con.print(_oneline(sug.command, limit=MAX_SUGGESTION_CHARS),
                          markup=False, highlight=False, soft_wrap=True,
                          style="cyan")

        # Confidence.
        con.print(f"   = [dim]confidence: {diag.confidence}[/dim]")
        con.print()  # blank line between diagnoses

    # Summary line at the end.
    errors = sum(1 for d in diagnoses if d.severity == "error")
    warnings = sum(1 for d in diagnoses if d.severity == "warning")
    infos = sum(1 for d in diagnoses if d.severity == "info")
    # Anything with a severity we don't recognize still has to be counted, or
    # the summary contradicts the report it's summarizing.
    other = len(diagnoses) - errors - warnings - infos
    parts = []
    if errors:
        parts.append(f"[red]{errors} error{'s' if errors != 1 else ''}[/red]")
    if warnings:
        parts.append(f"[yellow]{warnings} warning{'s' if warnings != 1 else ''}[/yellow]")
    if infos:
        parts.append(f"[blue]{infos} note{'s' if infos != 1 else ''}[/blue]")
    if other:
        # An unknown severity is itself worth flagging: it means a rule declared
        # something outside the documented set, and the CLI's exit code (which
        # keys off error/warning) will not reflect it.
        parts.append(f"{other} of unknown severity")
    # An info-only run used to render "summary:  · parsed N fact(s)" — counting
    # only errors/warnings left the clause empty. A clean result deserves to be
    # stated, not implied by absence.
    #
    # Guarded on `diagnoses` being empty, NOT on `parts`: a diagnosis with an
    # unrecognized severity previously produced "no issues found" while a
    # diagnosis was printed directly above it — the summary flatly contradicting
    # the report.
    if not diagnoses:
        parts.append("no issues found")
    con.print(f"[bold]summary:[/bold] {', '.join(parts)} · "
              f"parsed {len(facts.facts)} fact(s) from {facts.artifact_path}")


def render_json(diagnoses: list[Diagnosis], facts: Facts) -> str:
    """Serialize to the machine-readable JSON report format.

    Emits SPEC-COMPLIANT JSON. Two of Python's json defaults would otherwise
    produce output that strict consumers reject, and `--json` exists precisely
    to be piped into those (`| jq`, JS, Go):

      - allow_nan=False: Python defaults to writing bare `NaN`/`Infinity`, which
        are not valid JSON. A fact could carry a non-finite float (a malformed
        `temp=inf` reading, say), so we serialize such values as their string
        form via `default` rather than emitting a token jq would choke on.
      - default=str: a Fact.data value that isn't JSON-native (a set, a Path)
        would raise TypeError mid-serialization and abort the whole report.
        Coercing to str keeps the report producible; parsers should emit native
        types, but the output path must not depend on their doing so.
    """
    report = {
        "schemaVersion": 1,
        "tool": {"name": "edgedoctor", "version": __version__},
        "backend": facts.backend,
        "artifact": facts.artifact_path,
        "diagnostics": [d.model_dump() for d in diagnoses],
        "facts": [f.model_dump() for f in facts.facts],
    }

    def _fallback(obj: Any) -> str:
        # Reached for non-serializable values (sets, Paths, ...). Non-finite
        # floats do NOT reach here — allow_nan=False raises on them first — so
        # they are handled by the pre-pass below.
        return str(obj)

    # allow_nan=False makes json.dumps raise ValueError on inf/nan instead of
    # emitting an invalid token. We catch that and retry with the floats
    # sanitized, so the common (finite) case pays no cost.
    #
    # RecursionError is caught too: `indent=2` forces json's pure-PYTHON encoder
    # (the C fast path only handles the compact form), so pathologically nested
    # data blows the stack during encoding. Real parsers nest Fact.data at most
    # two deep, but a library caller can hand us anything, and a diagnostic tool
    # must not die while trying to report. The fallback drops indentation to take
    # the C encoder, which handles far deeper nesting; if even that fails we emit
    # a valid, honest error document rather than nothing.
    try:
        return json.dumps(report, indent=2, allow_nan=False, default=_fallback)
    except ValueError:
        return json.dumps(
            _finite_only(report), indent=2, allow_nan=False, default=_fallback
        )
    except RecursionError:
        try:
            return json.dumps(report, allow_nan=False, default=_fallback)
        except (ValueError, RecursionError):
            return json.dumps({
                "schemaVersion": 1,
                "tool": {"name": "edgedoctor", "version": __version__},
                "backend": facts.backend,
                "artifact": facts.artifact_path,
                "error": "report data was too deeply nested to serialize",
                "diagnosticCount": len(diagnoses),
                "factCount": len(facts.facts),
            }, indent=2)


#: Depth past which _finite_only stops descending. Real parsers emit Fact.data
#: nested at most 2 deep, so this is a runaway guard, not a limit anyone should
#: hit. It exists because this helper recurses in Python while json.dumps
#: recurses in C: json can handle nesting that would blow our stack, so without
#: a bound a hand-constructed pathological Facts object could turn a valid
#: report into a RecursionError.
_MAX_SANITIZE_DEPTH = 200


def _finite_only(obj: Any, _depth: int = 0) -> Any:
    """Recursively replace non-finite floats with their string form.

    NaN/Infinity aren't valid JSON, but they ARE real values a parser might
    carry (a divergence metric, a malformed sensor reading). Dropping them would
    lose information; stringifying them keeps the value visible and the document
    valid.
    """
    import math

    if _depth > _MAX_SANITIZE_DEPTH:
        # Too deep to descend safely. Stringify wholesale rather than risk a
        # stack overflow — the alternative is no report at all.
        return str(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)  # "inf", "-inf", "nan"
    if isinstance(obj, dict):
        return {k: _finite_only(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_finite_only(v, _depth + 1) for v in obj]
    return obj
