"""Smoke tests for the scripts/ directory.

WHY THESE EXIST
The generators are how the corpus is reproduced. A syntax error or a broken
import in one of them is a SILENT failure: nothing notices until someone tries
to regenerate months later, by which point the artifacts may not be
reproducible at all — which would quietly break the project's "real artifacts,
reproducibly generated" guarantee.

These tests deliberately do NOT run the generators (that needs torch, polygraphy
and several minutes). They check the cheap properties that catch the realistic
breakages: the module parses, its argparse surface is intact, and its pure
helper functions behave. Anything requiring heavy deps is skipped rather than
faked.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
SCRIPTS = sorted(p for p in SCRIPTS_DIR.glob("*.py") if not p.name.startswith("__"))

# Scripts that expose a CLI (i.e. have a main() guarded by __main__).
# _corpus_paths is a library module, so it has no CLI.
CLI_SCRIPTS = [p for p in SCRIPTS if p.name != "_corpus_paths.py"]


def test_scripts_directory_is_not_empty():
    # Guard on the guards: an empty glob would make everything below vacuous.
    assert SCRIPTS


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_parses(script):
    """A syntax error here breaks corpus reproduction, silently."""
    ast.parse(script.read_text())


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_has_a_module_docstring(script):
    """Every script must explain WHY it exists, not just what it does.

    These are the project's reproducibility record; an unexplained generator is
    an artifact nobody can trust or re-run confidently.
    """
    tree = ast.parse(script.read_text())
    doc = ast.get_docstring(tree)
    assert doc and len(doc) > 100, f"{script.name} needs a real module docstring"


@pytest.mark.parametrize("script", CLI_SCRIPTS, ids=lambda p: p.name)
def test_script_defines_a_main_entry_point(script):
    tree = ast.parse(script.read_text())
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "main" in names, f"{script.name} has no main()"


@pytest.mark.parametrize("script", CLI_SCRIPTS, ids=lambda p: p.name)
def test_script_help_does_not_crash(script):
    """--help must work without the heavy optional dependencies installed.

    argparse is built before any model library is touched, so a generator should
    be able to explain itself on a machine that can't run it. If a script
    imports torch at module scope, this catches it.
    """
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        # A missing heavy dep at import time is the thing we're detecting, but
        # it's only a failure if it's OUR import that's misplaced. Report it
        # with the actual error so the cause is visible.
        combined = result.stdout + result.stderr
        pytest.fail(f"{script.name} --help failed:\n{combined[-800:]}")
    assert "usage:" in result.stdout.lower()


class TestCorpusPathsHelpers:
    """_corpus_paths rewrites committed fixtures, so its guarantees matter.

    Covered more thoroughly in test_corpus_hygiene.py against the real corpus;
    these are the unit-level properties.
    """

    def test_scrub_text_is_a_noop_without_paths(self):
        from _corpus_paths import scrub_text

        assert scrub_text("no paths here\nsecond line\n") == "no paths here\nsecond line\n"

    def test_scrub_text_preserves_line_count(self):
        from _corpus_paths import scrub_text

        text = f"{Path.home()}/a\n{Path.home()}/b\nplain\n"
        assert len(scrub_text(text).splitlines()) == len(text.splitlines())

    def test_scrub_text_masks_the_home_directory(self):
        from _corpus_paths import find_machine_paths, scrub_text

        assert not find_machine_paths(scrub_text(f"{Path.home()}/x/y.log"))

    def test_scrub_text_handles_empty_input(self):
        from _corpus_paths import scrub_text

        assert scrub_text("") == ""

    def test_find_machine_paths_detects_both_platforms(self):
        from _corpus_paths import find_machine_paths

        # A log generated on Linux (the Pi, the ThinkPad) must be detectable
        # from macOS, or the hygiene test would miss a foreign leak.
        assert find_machine_paths("/home/pi/proj/x.py")
        assert find_machine_paths("/Users/someone/proj/x.py")

    def test_find_machine_paths_ignores_ordinary_text(self):
        from _corpus_paths import find_machine_paths

        assert find_machine_paths("no paths, just prose") == []


@pytest.fixture(autouse=True, scope="module")
def _scripts_importable():
    """Make scripts/ importable for the helper tests above."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    yield
    sys.path.remove(str(SCRIPTS_DIR))
