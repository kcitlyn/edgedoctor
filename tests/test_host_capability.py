"""Tests for the x86/ARM saturation-capability classifier.

This logic decides whether a divergence measurement from a given host means
anything at all, so getting it wrong doesn't produce a crash — it produces a
confident null result that looks like a successful experiment. Hence the
emphasis on the "unknown" and "wrong host" paths rather than just the happy one.

The rule being encoded, from ONNX Runtime's documentation:
    "There is no such issue on other CPU architectures (x64 with VNNI and Arm)."
So saturation needs x86 AND (AVX2 or AVX512) AND NOT VNNI.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import host_capability as hc  # noqa: E402


def classify(monkeypatch, machine: str, flags: set[str]):
    monkeypatch.setattr(hc.platform, "machine", lambda: machine)
    monkeypatch.setattr(hc, "_cpu_flags", lambda: flags)
    return hc.describe_host()


class TestSaturationRequiresAllConditions:
    """x86 AND (AVX2 or AVX512) AND NOT VNNI — all three, or no saturation."""

    def test_x86_avx2_no_vnni(self, monkeypatch):
        info = classify(monkeypatch, "x86_64", {"avx2", "sse4_2"})
        assert info["u8s8_saturation_possible"] is True
        assert info["suitable_as_x86_baseline"] is True

    def test_x86_avx512_no_vnni(self, monkeypatch):
        info = classify(monkeypatch, "x86_64", {"avx512f"})
        assert info["u8s8_saturation_possible"] is True

    def test_x86_with_vnni_is_not_affected(self, monkeypatch):
        # VNNI accumulates into 32-bit lanes, so there is nothing to saturate.
        info = classify(monkeypatch, "x86_64", {"avx512f", "avx512_vnni"})
        assert info["u8s8_saturation_possible"] is False
        assert "VNNI" in info["reason"]

    def test_vnni_spelling_variants_are_all_detected(self, monkeypatch):
        # Kernels and vendors spell this differently; missing a spelling would
        # misclassify a VNNI machine as an affected one.
        for spelling in ("avx512_vnni", "avx512vnni", "avx_vnni"):
            info = classify(monkeypatch, "x86_64", {"avx512f", spelling})
            assert info["has_vnni"] is True, spelling
            assert info["u8s8_saturation_possible"] is False, spelling

    def test_x86_without_avx2_or_avx512(self, monkeypatch):
        info = classify(monkeypatch, "x86_64", {"sse4_2"})
        assert info["u8s8_saturation_possible"] is False


class TestArmIsNeverAnX86Baseline:
    """The trap this module exists to prevent."""

    @pytest.mark.parametrize("machine", ["arm64", "aarch64", "armv7l"])
    def test_arm_hosts_cannot_exhibit_saturation(self, monkeypatch, machine):
        info = classify(monkeypatch, machine, set())
        assert info["is_arm"] is True
        assert info["u8s8_saturation_possible"] is False
        assert info["suitable_as_x86_baseline"] is False

    def test_apple_silicon_is_rejected_as_a_baseline(self, monkeypatch):
        # An M-series Mac is the SAME side of the issue as the Pi's Cortex-A76,
        # so a Mac-vs-Pi comparison is ARM vs ARM: guaranteed to show no
        # divergence, and proving nothing.
        info = classify(monkeypatch, "arm64", set())
        assert info["suitable_as_x86_baseline"] is False
        assert "Arm" in info["reason"]

    def test_this_actual_host_is_classified(self):
        # Not asserting a specific answer — this must work wherever it runs.
        info = hc.describe_host()
        assert info["u8s8_saturation_possible"] in (True, False, None)
        assert info["reason"]


class TestUnknownIsNotFalse:
    def test_unreadable_flags_on_x86_yields_none_not_false(self, monkeypatch):
        # The important distinction: "we couldn't check" must never be reported
        # as "not affected", or an unverified host looks like a cleared one.
        info = classify(monkeypatch, "x86_64", set())
        assert info["u8s8_saturation_possible"] is None
        assert info["suitable_as_x86_baseline"] is False  # unknown != suitable
        assert "UNVERIFIED" in info["reason"]

    def test_unknown_flags_are_reported_as_none(self, monkeypatch):
        info = classify(monkeypatch, "x86_64", set())
        assert info["has_avx2"] is None
        assert info["has_vnni"] is None

    def test_exotic_architecture_is_not_applicable(self, monkeypatch):
        info = classify(monkeypatch, "riscv64", set())
        assert info["u8s8_saturation_possible"] is False
        assert "neither x86 nor ARM" in info["reason"]


class TestOutputContract:
    def test_reports_every_field_the_scripts_depend_on(self):
        info = hc.describe_host()
        for key in ("machine", "is_x86", "is_arm", "u8s8_saturation_possible",
                    "reason", "suitable_as_x86_baseline", "cpu_flags_readable"):
            assert key in info

    def test_json_serializable(self):
        # make_cross_host_baseline.py writes this to host_<tag>.json.
        import json

        json.dumps(hc.describe_host())

    def test_reason_is_always_populated(self, monkeypatch):
        for machine in ("x86_64", "arm64", "riscv64"):
            for flags in (set(), {"avx2"}, {"avx512f", "avx512_vnni"}):
                assert classify(monkeypatch, machine, flags)["reason"]
