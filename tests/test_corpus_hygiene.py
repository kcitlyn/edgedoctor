"""Guards on the committed corpus itself, not on any parser.

The corpus is committed to a public repo and is the input to every parser test,
so two properties have to hold for all of it at once. Both were violated at some
point, which is why they're pinned here rather than trusted:

  1. No machine-specific absolute paths. Vendor tools echo the paths they were
     invoked with, which leaks the generating machine's username and makes logs
     non-portable. Generators mask these at capture time.
  2. Every artifact has a .meta.md sidecar. corpus/README.md makes the sidecar
     the ground-truth LABEL for the sample; an unlabelled log is unusable as
     test data because nobody can say what it's supposed to prove.

These run over whatever is in corpus/, so a newly added log is covered the
moment it lands — no test needs updating.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
CORPUS = REPO / "corpus"

sys.path.insert(0, str(REPO / "scripts"))
from _corpus_paths import find_machine_paths, scrub_text  # noqa: E402

ALL_LOGS = sorted(CORPUS.glob("*/*.log"))


def test_corpus_is_not_empty():
    # A guard on the guards: if the glob silently matched nothing, every test
    # below would vacuously pass and we'd believe the corpus was clean.
    assert ALL_LOGS, "no corpus logs found — the other tests here would be vacuous"


@pytest.mark.parametrize("log", ALL_LOGS, ids=lambda p: p.name)
def test_no_machine_specific_paths(log):
    """No committed log may contain a real home-directory path."""
    found = find_machine_paths(log.read_text(errors="replace"))
    assert not found, (
        f"{log.relative_to(REPO)} contains machine paths: {sorted(set(found))[:3]}. "
        "Regenerate it with its scripts/make_*_corpus.py, which masks these at "
        "capture time."
    )


@pytest.mark.parametrize("log", ALL_LOGS, ids=lambda p: p.name)
def test_every_log_has_a_sidecar(log):
    """corpus/README.md requires a .meta.md label per artifact."""
    sidecar = log.with_suffix(".meta.md")
    assert sidecar.exists(), (
        f"{log.relative_to(REPO)} has no .meta.md sidecar recording what it is "
        "and what actually went wrong"
    )


@pytest.mark.parametrize("log", ALL_LOGS, ids=lambda p: p.name)
def test_sidecar_records_ground_truth(log):
    """A sidecar that omits the outcome or cause isn't a label, just a filename."""
    text = log.with_suffix(".meta.md").read_text()
    for field in ("- command:", "- outcome:", "- root cause"):
        assert field in text, f"{log.name}'s sidecar is missing '{field}'"


class TestScrubberSafety:
    """The scrubber rewrites committed fixtures, so its safety property matters.

    edgedoctor cites file:line and the parser tests assert each fact's excerpt
    equals its source line. A scrub that added, removed, or reordered a line
    would silently invalidate every citation in the corpus.
    """

    def test_preserves_line_count(self):
        for log in ALL_LOGS:
            original = log.read_text(errors="replace")
            assert len(scrub_text(original).splitlines()) == len(original.splitlines())

    def test_is_idempotent(self):
        # Re-running a generator must not progressively mangle a log.
        for log in ALL_LOGS:
            once = scrub_text(log.read_text(errors="replace"))
            assert scrub_text(once) == once

    def test_committed_corpus_is_already_scrubbed(self):
        # If this fails, someone committed a log that bypassed the generator.
        for log in ALL_LOGS:
            text = log.read_text(errors="replace")
            assert scrub_text(text) == text, (
                f"{log.name} is not normalized — regenerate it via its script"
            )

    def test_replaces_a_known_prefix(self):
        # Proves the scrubber actually does something, so the assertions above
        # aren't passing because it's a no-op.
        assert "/Users/" not in scrub_text(str(Path.home() / "x/y.log"))
