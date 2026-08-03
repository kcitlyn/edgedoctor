"""Shared path normalization for corpus generators.

WHY THIS EXISTS
Vendor tools echo the absolute paths they were invoked with, so a captured log
contains the generating machine's home directory and virtualenv layout —
`/Users/alice/personal/edgedoctor/.venv/bin/polygraphy`. The corpus is committed
to a public repo, so that leaks a username, and it also makes logs
machine-specific noise rather than portable fixtures: regenerating on another
machine would produce a diff on every such line even when nothing changed.

WHY REWRITING IS STILL "REAL ARTIFACTS ONLY"
corpus/README.md rule 1 requires every artifact to be genuine tool output. This
substitution replaces an absolute prefix with a placeholder and nothing else:
no line is added, removed, reordered, or reworded, so LINE NUMBERS ARE
PRESERVED. That matters more than it sounds — edgedoctor cites `file:line`, and
the snapshot tests assert each fact's excerpt equals the source line. Any
transformation that shifted lines would silently invalidate every citation.

It is applied by the generator at capture time, so committed logs are normalized
by construction rather than by a cleanup pass someone has to remember to run.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

#: Stand-in for the repo root in committed logs. Chosen to look obviously like a
#: placeholder, so nobody mistakes it for a path that ever existed.
REPO_PLACEHOLDER = "<REPO>"

#: Stand-in for the Python environment root, which sits inside the repo for uv
#: but may be anywhere for a global install.
VENV_PLACEHOLDER = "<VENV>"


def _prefixes() -> list[tuple[str, str]]:
    """Absolute prefixes to mask, longest first.

    Longest-first matters: the venv lives *inside* the repo, so masking the repo
    prefix first would leave a half-rewritten venv path behind.
    """
    pairs: list[tuple[str, str]] = []

    venv = os.environ.get("VIRTUAL_ENV") or str(Path(sys.prefix))
    if venv and venv != "/":
        pairs.append((venv, VENV_PLACEHOLDER))

    repo = str(Path(__file__).resolve().parent.parent)
    if repo and repo != "/":
        pairs.append((repo, REPO_PLACEHOLDER))

    # The home directory is masked last as a backstop, in case a tool reports a
    # path outside both the repo and the venv (a cache dir, say).
    home = str(Path.home())
    if home and home != "/":
        pairs.append((home, "<HOME>"))

    return sorted(pairs, key=lambda p: len(p[0]), reverse=True)


def scrub_text(text: str) -> str:
    """Replace machine-specific absolute paths with stable placeholders.

    Line count and line order are unchanged by construction — this only ever
    substitutes within a line.
    """
    for prefix, placeholder in _prefixes():
        text = text.replace(prefix, placeholder)
        # Tools sometimes print a resolved path (/private/var/... on macOS)
        # where the caller passed a symlinked one (/var/...).
        resolved = str(Path(prefix).resolve())
        if resolved != prefix:
            text = text.replace(resolved, placeholder)
    return text


def scrub_log(path: Path) -> int:
    """Normalize a captured log in place. Returns the number of lines changed.

    Asserts the line count is preserved: a corpus log whose lines shifted would
    invalidate every `file:line` citation edgedoctor makes about it, so this
    fails loudly rather than corrupting the fixtures quietly.
    """
    original = path.read_text(errors="replace")
    scrubbed = scrub_text(original)
    if scrubbed == original:
        return 0

    before, after = original.splitlines(), scrubbed.splitlines()
    if len(before) != len(after):  # pragma: no cover - defensive
        raise RuntimeError(
            f"scrubbing changed the line count of {path} "
            f"({len(before)} -> {len(after)}); citations would break"
        )
    path.write_text(scrubbed)
    return sum(1 for b, a in zip(before, after) if b != a)


def find_machine_paths(text: str) -> list[str]:
    """Absolute home-directory paths still present in `text`.

    Used by tests to assert the committed corpus stays portable.
    """
    return re.findall(r"(?:/Users/|/home/)[^\s,)'\"\]]+", text)
