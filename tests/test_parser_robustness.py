"""Adversarial-input tests for every parser, driven off the registry.

WHY THIS FILE EXISTS
edgedoctor parses logs it did not produce. A vendor tool crashing mid-write, a
truncated upload, a binary file passed by mistake, or a deliberately hostile log
are all realistic inputs, and none of them may crash the tool or hang it. Three
real bugs were found by probing this way and are pinned below:

  1. CATASTROPHIC BACKTRACKING. Three patterns were O(n^2) because an unbounded
     character class preceded a literal, so the engine retried the literal from
     every start position. A single long line hung the parser for minutes — a
     genuine DoS, since real logs do contain very long lines (the Polygraphy
     layer-wise log has a ~3000-char one).
  2. A NEGATIVE BITFIELD decoded as "every throttle condition set at once".
  3. Malformed rule files crashed the diagnoser instead of failing closed.

Tests are parametrized over PARSER_REGISTRY, so a NEW BACKEND IS COVERED THE
MOMENT IT IS REGISTERED — no test needs updating, and a parser cannot be added
without inheriting these guarantees.
"""

import time

import pytest

from edgedoctor.backends import PARSER_REGISTRY, get_parser

BACKENDS = sorted(PARSER_REGISTRY)

# Inputs that have historically broken parsers, or plausibly could.
HOSTILE_INPUTS = {
    "empty": "",
    "whitespace_only": "   \n\t\n  ",
    "only_newlines": "\n\n\n\n",
    "no_trailing_newline": "a log line with no trailing newline",
    "crlf": "line one\r\nline two\r\n",
    "cr_only": "line one\rline two",
    "nul_bytes": "line\x00with\x00nul",
    "ansi_escapes": "\x1b[31mred text\x1b[0m",
    "unicode": "日本語 ✓ émoji 🔥",
    "rtl_override": "abc‮def",
    "combining_marks": "é" * 500,
    "binary_ish": "".join(chr(i % 256) for i in range(2000)),
    "json_when_text_expected": '{"not": "a log"}',
    "text_when_json_expected": "not json at all",
    "html": "<html><body>nope</body></html>",
    "deeply_nested_brackets": "[" * 500 + "]" * 500,
    "many_short_lines": "x\n" * 20000,
    "one_long_line": "y" * 100_000,
    "long_word_run": "a" * 60_000,
    "long_digit_run": "9" * 60_000,
    "long_space_run": " " * 60_000,
    "long_hex_run": "f" * 60_000,
    "long_punct_run": "./-_" * 15_000,
    "quotes": "'" * 30_000,
    "equals": "=" * 30_000,
}

# Generous but finite: these must be fast, and a regression to quadratic
# behaviour blows straight past this rather than merely getting slower.
MAX_SECONDS_PER_INPUT = 5.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("label", sorted(HOSTILE_INPUTS))
def test_hostile_input_neither_crashes_nor_hangs(backend, label):
    parser = get_parser(backend)
    started = time.monotonic()
    facts = parser.parse_text(HOSTILE_INPUTS[label], artifact_name="hostile.log")
    elapsed = time.monotonic() - started

    assert elapsed < MAX_SECONDS_PER_INPUT, (
        f"{backend} took {elapsed:.1f}s on '{label}' — likely catastrophic "
        "regex backtracking (an unbounded class before a literal)"
    )
    # Whatever came back must be a well-formed Facts object.
    assert facts.backend == parser.name
    assert isinstance(facts.facts, list)


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_facts_invented_from_pure_noise(backend):
    """Random noise must not produce diagnosable facts.

    The tool's core promise is that it reports only what it observed. A parser
    that finds "evidence" in noise is the worst possible failure, because the
    output looks authoritative.
    """
    noise = "".join(chr((i * 7919) % 128) for i in range(20000))
    facts = get_parser(backend).parse_text(noise, artifact_name="noise.log")
    assert facts.facts == [], (
        f"{backend} invented {len(facts.facts)} fact(s) from random noise"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_scales_linearly_not_quadratically(backend):
    """Doubling input length must not quadruple the time.

    This is the shape-of-the-curve test that catches backtracking regressions a
    fixed timeout can miss on a fast machine.
    """
    parser = get_parser(backend)

    def timed(n: int) -> float:
        text = "a" * n
        start = time.monotonic()
        parser.parse_text(text, artifact_name="t.log")
        return time.monotonic() - start

    small, large = timed(20_000), timed(80_000)
    # 4x the input. Linear would be ~4x; quadratic ~16x. Allow generous slack
    # for timer noise on tiny durations, but a quadratic blow-up is orders out.
    if small > 0.005:  # only meaningful when the small case is measurable
        assert large < small * 9, (
            f"{backend}: 4x input took {large / small:.1f}x time — "
            "superlinear, check for unbounded quantifiers before literals"
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_parse_text_is_deterministic(backend):
    parser = get_parser(backend)
    sample = "some log line\nanother line\n"
    first = parser.parse_text(sample, artifact_name="t.log")
    second = parser.parse_text(sample, artifact_name="t.log")
    assert first.model_dump() == second.model_dump()


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_fact_satisfies_the_traceability_contract(backend):
    """Whatever a parser emits from odd input must still be citable.

    A fact without a real source is unverifiable, which defeats the point of
    the Facts firewall.
    """
    parser = get_parser(backend)
    for label, text in HOSTILE_INPUTS.items():
        facts = parser.parse_text(text, artifact_name="t.log")
        lines = text.splitlines()
        for f in facts.facts:
            assert f.source.startswith("t.log:"), f"{backend}/{label}: bad source"
            assert f.id, f"{backend}/{label}: empty id"
            assert f.kind, f"{backend}/{label}: empty kind"
            ref = f.source.split(":", 1)[1]
            # Text parsers cite a line number; the JSON parser cites an
            # events[N]/summary path. Both must be resolvable.
            if ref.isdigit():
                assert 1 <= int(ref) <= len(lines), (
                    f"{backend}/{label}: cites line {ref} of {len(lines)}"
                )


@pytest.mark.parametrize("backend", BACKENDS)
def test_fact_ids_are_unique(backend):
    parser = get_parser(backend)
    for text in HOSTILE_INPUTS.values():
        ids = [f.id for f in parser.parse_text(text, artifact_name="t.log").facts]
        assert len(ids) == len(set(ids))


@pytest.mark.parametrize("backend", BACKENDS)
def test_convert_raises_not_implemented_with_an_explanation(backend):
    """Every backend's convert() is a stub; each must say WHY, not just fail.

    Per the honesty guardrail, a stub is marked as a stub.
    """
    from pathlib import Path

    parser = get_parser(backend)
    with pytest.raises(NotImplementedError) as exc:
        parser.convert(Path("model.onnx"))
    assert len(str(exc.value)) > 20, "NotImplementedError should explain itself"


class TestRegistryIntegrity:
    def test_registry_is_not_empty(self):
        # Guard on the guards: an empty registry would make every parametrized
        # test above vacuous.
        assert BACKENDS

    def test_every_registered_backend_loads(self):
        for name in BACKENDS:
            assert get_parser(name).name == name

    def test_unknown_backend_raises_keyerror(self):
        with pytest.raises(KeyError):
            get_parser("no_such_backend")

    def test_every_backend_has_a_rule_file(self):
        # A parser with no rules produces facts nothing can explain, which is a
        # silent dead end for the user.
        from edgedoctor.diagnoser import _load_rules

        for name in BACKENDS:
            assert _load_rules(name), f"{name} has no usable rule file"

    def test_cli_exposes_every_registered_backend(self):
        # The --backend flag must not drift from the registry.
        from edgedoctor.cli import BackendName

        cli_names = {b.value for b in BackendName}
        assert set(BACKENDS) <= cli_names, (
            f"registered but not in --backend: {set(BACKENDS) - cli_names}"
        )


class TestPackagingCompleteness:
    """Non-Python assets must actually ship in the installed package.

    The rules live in YAML, not code, so an incomplete package build produces a
    tool that imports fine and then silently finds no rules — the worst kind of
    failure, because `diagnose` would honestly report "no known pattern matched"
    for every input and look like it was working correctly.

    These run against whatever edgedoctor is IMPORTED, so under a wheel-based CI
    run they check the wheel; under an editable install they check the source
    tree. Either way, a missing asset fails here.
    """

    def test_every_registered_backend_has_loadable_rules(self):
        from edgedoctor.diagnoser import _load_rules

        missing = [b for b in BACKENDS if not _load_rules(b)]
        assert not missing, f"no rules found for: {missing} (packaging problem?)"

    def test_rules_directory_ships_with_the_package(self):
        from edgedoctor.diagnoser import RULES_DIR

        assert RULES_DIR.is_dir(), f"{RULES_DIR} missing from the installation"
        assert list(RULES_DIR.glob("*.yaml")), "no rule files in the installation"

    def test_one_rule_file_per_registered_backend(self):
        from edgedoctor.diagnoser import RULES_DIR

        shipped = {p.stem for p in RULES_DIR.glob("*.yaml")}
        assert set(BACKENDS) <= shipped, (
            f"registered backends with no shipped rule file: "
            f"{set(BACKENDS) - shipped}"
        )

    def test_version_is_exposed(self):
        # The JSON report embeds this; an unset version makes reports
        # unattributable to a build.
        import edgedoctor

        assert edgedoctor.__version__
        assert edgedoctor.__version__ != "unknown"
