"""Tests for the Raspberry Pi host-health parser (throttling + OOM).

NOTE ON FIXTURES vs CORPUS: these inputs live in tests/fixtures/, not corpus/,
because the Pi hasn't arrived — they are hand-written to the documented formats,
so per corpus/README.md rule 1 they are explicitly NOT corpus material. When the
real hardware produces real output, that goes in corpus/raspberrypi/ and these
tests gain a second layer against it. The bit MEANINGS are from Raspberry Pi's
official documentation, so the decoder is grounded even though the fixtures
aren't.

The most important test here is TestDoesNotInventThrottling: a permissive hex
pattern originally matched `gfp_mask=0x140dca` in a kernel OOM log and reported
undervoltage that never happened. That's the failure mode this whole project
exists to prevent, so it's pinned.
"""

from pathlib import Path

import pytest

from edgedoctor.backends.raspberrypi import (
    LIVE_BITS,
    STICKY_BITS,
    RaspberryPiBackend,
    decode_throttled,
)

FIXTURES = Path(__file__).parent / "fixtures" / "raspberrypi"

parser = RaspberryPiBackend()


def parse_fixture(name: str):
    return parser.parse(FIXTURES / name)


def kinds_in(facts) -> set[str]:
    return {f.kind for f in facts.facts}


def fact_of(facts, kind):
    return next(f for f in facts.facts if f.kind == kind)


class TestBitfieldDecoding:
    """Decoded against Raspberry Pi's official bit table."""

    def test_zero_is_healthy(self):
        d = decode_throttled(0x0)
        assert d["healthy"] is True
        assert d["live"] == [] and d["sticky"] == []

    @pytest.mark.parametrize(
        "value,expected",
        [
            (0x1, "undervoltage"),
            (0x2, "freq_capped"),
            (0x4, "throttled"),
            (0x8, "soft_temp_limit"),
        ],
    )
    def test_each_live_bit(self, value, expected):
        assert decode_throttled(value)["live"] == [expected]
        assert decode_throttled(value)["sticky"] == []

    @pytest.mark.parametrize(
        "value,expected",
        [
            (0x10000, "undervoltage_occurred"),
            (0x20000, "freq_capped_occurred"),
            (0x40000, "throttled_occurred"),
            (0x80000, "soft_temp_limit_occurred"),
        ],
    )
    def test_each_sticky_bit(self, value, expected):
        assert decode_throttled(value)["sticky"] == [expected]
        assert decode_throttled(value)["live"] == []

    def test_live_and_sticky_are_reported_separately(self):
        # The central distinction: 0x50005 means "happening now AND happened
        # before". Collapsing the halves would lose which is which.
        d = decode_throttled(0x50005)
        assert d["live"] == ["undervoltage", "throttled"]
        assert d["sticky"] == ["undervoltage_occurred", "throttled_occurred"]

    def test_sticky_only_is_not_reported_as_live(self):
        # Would cry wolf on a currently-healthy machine.
        d = decode_throttled(0x50000)
        assert d["live"] == []
        assert d["sticky"] == ["undervoltage_occurred", "throttled_occurred"]

    def test_all_documented_bits(self):
        d = decode_throttled(0xF000F)
        assert len(d["live"]) == 4
        assert len(d["sticky"]) == 4
        assert d["unknown_bits"] == []

    def test_undocumented_bit_is_reported_not_guessed(self):
        # Bits 4-15 have no published meaning. Inventing one for a hardware
        # status flag would be worse than admitting the gap.
        d = decode_throttled(0x100)
        assert d["unknown_bits"] == [8]
        assert d["live"] == [] and d["sticky"] == []
        assert d["healthy"] is False  # an unexplained bit is not "healthy"

    def test_bit_tables_match_the_official_documentation(self):
        assert set(LIVE_BITS) == {0, 1, 2, 3}
        assert set(STICKY_BITS) == {16, 17, 18, 19}


class TestDoesNotInventThrottling:
    """The regression that matters most in this file."""

    def test_kernel_oom_log_reports_no_throttling(self):
        # This log contains gfp_mask=0x140dca. An earlier, more permissive
        # pattern decoded it as a throttle bitfield and reported undervoltage
        # and throttling that never happened.
        facts = parse_fixture("oom_kill.log")
        throttle_kinds = {
            "throttled_bitfield", "throttle_active", "throttle_occurred",
            "throttle_unknown_bits", "throttle_clear",
        }
        assert not (kinds_in(facts) & throttle_kinds)

    def test_bare_hex_is_not_a_throttle_reading(self):
        facts = parser.parse_text(
            "some_mask=0x50005\nother=0xdeadbeef\n", artifact_name="x.log"
        )
        assert "throttled_bitfield" not in kinds_in(facts)

    def test_the_throttled_key_is_required(self):
        facts = parser.parse_text("throttled=0x4", artifact_name="x.log")
        assert "throttle_active" in kinds_in(facts)


class TestThrottleActive:
    LOG = "throttle_active.log"

    def test_detects_live_throttling(self):
        facts = parse_fixture(self.LOG)
        live = fact_of(facts, "throttle_active")
        assert live.data["condition_keys"] == ["undervoltage", "throttled"]

    def test_conditions_render_as_prose_not_a_python_list(self):
        # This string is interpolated into a user-facing message; a repr like
        # "['undervoltage']" reads as a leaked internal.
        facts = parse_fixture(self.LOG)
        conditions = fact_of(facts, "throttle_active").data["conditions"]
        assert conditions == "undervoltage detected, currently throttled"
        assert "[" not in conditions

    def test_captures_supporting_temperature_and_clock(self):
        facts = parse_fixture(self.LOG)
        assert fact_of(facts, "soc_temperature").data["celsius"] == 84.2
        # Clock is reported at 1000 MHz — well below the Pi 5's 2400 MHz, which
        # is itself corroborating evidence of throttling.
        assert fact_of(facts, "arm_clock").data["mhz"] == 1000


class TestThrottleOccurred:
    def test_sticky_only_log_has_no_live_fact(self):
        facts = parse_fixture("throttle_occurred.log")
        assert "throttle_occurred" in kinds_in(facts)
        assert "throttle_active" not in kinds_in(facts)

    def test_clock_is_back_to_full_speed(self):
        facts = parse_fixture("throttle_occurred.log")
        assert fact_of(facts, "arm_clock").data["mhz"] == 2400


class TestThrottleClear:
    def test_healthy_host_produces_a_clear_fact(self):
        facts = parse_fixture("throttle_clear.log")
        assert "throttle_clear" in kinds_in(facts)

    def test_no_failure_facts_on_a_healthy_host(self):
        facts = parse_fixture("throttle_clear.log")
        bad = {"throttle_active", "throttle_occurred", "throttle_unknown_bits",
               "oom_kill", "allocation_failed"}
        assert not (kinds_in(facts) & bad)


class TestOom:
    def test_detects_the_killed_process(self):
        facts = parse_fixture("oom_kill.log")
        kill = fact_of(facts, "oom_kill")
        assert kill.data["process"] == "python3"
        assert kill.data["pid"] == "1547"

    def test_detects_the_invoking_process(self):
        facts = parse_fixture("oom_kill.log")
        assert fact_of(facts, "oom_invoked").data["invoker"] == "python3"

    def test_allocation_failure_is_a_distinct_kind(self):
        # A survivable allocator refusal, not a kernel kill — different fix.
        facts = parse_fixture("allocation_failed.log")
        alloc = fact_of(facts, "allocation_failed")
        assert alloc.data["bytes"] == 1073741824
        assert alloc.data["mib"] == 1024.0
        assert "oom_kill" not in kinds_in(facts)


class TestGroundingContract:
    LOGS = ["throttle_active.log", "throttle_occurred.log", "throttle_clear.log",
            "throttle_unknown_bit.log", "oom_kill.log", "allocation_failed.log"]

    def test_every_fact_is_traceable(self):
        for name in self.LOGS:
            facts = parse_fixture(name)
            assert facts.facts, f"{name} produced no facts"
            for f in facts.facts:
                assert f.source.startswith(f"{name}:")
                assert f.excerpt != ""

    def test_excerpt_matches_the_cited_line(self):
        for name in self.LOGS:
            lines = (FIXTURES / name).read_text().splitlines()
            for f in parse_fixture(name).facts:
                lineno = int(f.source.rsplit(":", 1)[1])
                assert f.excerpt == lines[lineno - 1].strip()

    def test_fact_ids_are_unique(self):
        for name in self.LOGS:
            ids = [f.id for f in parse_fixture(name).facts]
            assert len(ids) == len(set(ids))


class TestParserProperties:
    def test_deterministic(self):
        assert (parse_fixture("throttle_active.log").model_dump()
                == parse_fixture("throttle_active.log").model_dump())

    def test_empty_input_yields_no_facts(self):
        assert parser.parse_text("", artifact_name="e.log").facts == []

    def test_garbage_input_yields_no_facts(self):
        facts = parser.parse_text("hello\nworld\n42\n", artifact_name="j.log")
        assert facts.facts == []

    def test_backend_name(self):
        assert parse_fixture("throttle_clear.log").backend == "raspberrypi"

    def test_convert_is_honestly_unimplemented(self):
        with pytest.raises(NotImplementedError, match="host-health"):
            parser.convert(Path("model.onnx"))

    def test_registered(self):
        from edgedoctor.backends import PARSER_REGISTRY, get_parser

        assert "raspberrypi" in PARSER_REGISTRY
        assert get_parser("raspberrypi").name == "raspberrypi"


@pytest.mark.parametrize(
    "log",
    ["throttle_active.log", "throttle_occurred.log", "throttle_clear.log",
     "throttle_unknown_bit.log", "oom_kill.log", "allocation_failed.log"],
)
def test_snapshot(log, snapshot):
    assert parse_fixture(log).model_dump() == snapshot
