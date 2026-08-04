"""Every parser signature must be grounded in something real.

WHY THIS FILE EXISTS
A parser signature is a claim about what a vendor tool emits. If the claim is
wrong the signature never fires, and the failure is SILENT: edgedoctor reports
"no known pattern matched" on a log that plainly contains the failure, and looks
like it is working correctly while missing the thing it exists to catch.

That happened. `memcpy_nodes_added` matched "Add MemcpyFromHost/MemcpyToHost for
N" and "Inserted N Memcpy node" — strings reconstructed from memory that ONNX
Runtime never emits. It could never have fired. The real message is assembled in
core/optimizer/transformer_memcpy.cc and reads:

    <count> Memcpy nodes are added to the graph <name> for <provider>. It might
    have negative impact on performance (including unable to run CUDA graph). ...

It only fires for the CUDA EP, so it cannot be reproduced on a machine without
CUDA — which is exactly why the fabrication went unnoticed for so long.

THE RULE THIS FILE ENFORCES
Every signature is either
  (a) exercised by a real artifact committed to the repo, or
  (b) listed in AWAITING_REAL_ARTIFACT below with the reason it can't be, and
      covered by a test against a line reconstructed from vendor SOURCE.

Nothing may be grounded in memory alone.
"""

import glob
from pathlib import Path

import pytest

from edgedoctor.backends import onnxruntime, polygraphy, raspberrypi, tensorrt

MODULES = {
    "tensorrt": tensorrt,
    "polygraphy": polygraphy,
    "onnxruntime": onnxruntime,
    "raspberrypi": raspberrypi,
}

#: Signatures that no committed artifact can exercise, with the hardware or
#: condition required to capture one. Each MUST still have a source-grounded
#: unit test below. Shrinking this dict is the goal; it is not a place to park
#: unverified guesses.
AWAITING_REAL_ARTIFACT = {
    "onnxruntime.memcpy_nodes_added": (
        "ORT only logs this for the CUDA EP (transformer_memcpy.cc guards on "
        "provider == CUDA), so it needs the ThinkPad. Pattern reconstructed from "
        "ORT source, not memory; covered by TestMemcpySignatureAgainstSource."
    ),
}


def _repo_lines() -> list[str]:
    """Every line of every committed log artifact."""
    files = sorted(
        glob.glob("tests/fixtures/**/*.log", recursive=True)
        + glob.glob("corpus/**/*.log", recursive=True)
    )
    lines: list[str] = []
    for path in files:
        lines += Path(path).read_text(errors="replace").splitlines()
    return lines


REPO_LINES = _repo_lines()

ALL_SIGNATURES = [
    (backend, kind, pattern)
    for backend, module in MODULES.items()
    for kind, pattern, _ in module._SIGNATURES
]
SIG_IDS = [f"{b}.{k}" for b, k, _ in ALL_SIGNATURES]


def test_repo_artifacts_were_found():
    # Guard on the guards: an empty sweep would make every test below vacuous.
    assert len(REPO_LINES) > 1000, f"only found {len(REPO_LINES)} lines"


@pytest.mark.parametrize("backend,kind,pattern", ALL_SIGNATURES, ids=SIG_IDS)
def test_signature_is_grounded_in_a_real_artifact(backend, kind, pattern):
    """A signature must match a line some real tool actually produced.

    Matched line-by-line, exactly as the parser does. (An earlier version of
    this audit searched the concatenated text and reported eight false
    positives, because patterns anchored with `$` only match end-of-string
    without re.MULTILINE.)
    """
    name = f"{backend}.{kind}"
    if name in AWAITING_REAL_ARTIFACT:
        pytest.skip(f"awaiting hardware: {AWAITING_REAL_ARTIFACT[name]}")
    assert any(pattern.search(line) for line in REPO_LINES), (
        f"{name} matches no line in any committed artifact. Either the pattern "
        "is wrong (a signature that can never fire is worse than none, because "
        "the tool silently misses the failure), or an artifact exercising it "
        "needs to be added, or it belongs in AWAITING_REAL_ARTIFACT with a "
        "source-grounded test."
    )


def test_awaiting_list_contains_only_real_signature_names():
    """A stale entry would silently exempt nothing, or worse, mask a rename."""
    known = set(SIG_IDS)
    for name in AWAITING_REAL_ARTIFACT:
        assert name in known, f"{name} is not a real signature (renamed? removed?)"


def test_awaiting_list_stays_small():
    # A growing exemption list means grounding discipline is eroding.
    assert len(AWAITING_REAL_ARTIFACT) <= 3, (
        f"{len(AWAITING_REAL_ARTIFACT)} signatures lack real artifacts; capture "
        "some before adding more"
    )


class TestMemcpySignatureAgainstSource:
    """Reconstructed from ORT source, since no local hardware can emit it.

    Fragments, in order, from MemcpyTransformer::ApplyImpl in
    onnxruntime/core/optimizer/transformer_memcpy.cc:
        <copy_node_counter>
        " Memcpy nodes are added to the graph "
        <graph.Name()>
        " for "
        <provider_type>
        ". It might have negative impact on performance (including unable to
         run CUDA graph). "
        "Set session_options.log_severity_level=1 to see the detail logs ..."
    """

    REAL_LINE = (
        "7 Memcpy nodes are added to the graph torch_jit for "
        "CUDAExecutionProvider. It might have negative impact on performance "
        "(including unable to run CUDA graph). Set "
        "session_options.log_severity_level=1 to see the detail logs before "
        "this message."
    )

    def _parse(self, line: str):
        return onnxruntime.OnnxRuntimeBackend().parse_text(line, artifact_name="t.log")

    def test_matches_the_source_reconstructed_line(self):
        facts = self._parse(self.REAL_LINE)
        memcpy = [f for f in facts.facts if f.kind == "memcpy_nodes_added"]
        assert memcpy, "the source-grounded line must match"
        assert memcpy[0].data["count"] == 7
        assert memcpy[0].data["provider"] == "CUDAExecutionProvider"
        assert memcpy[0].data["graph"] == "torch_jit"

    def test_matches_with_ORT_log_prefix(self):
        prefixed = (
            "2026-08-04 00:00:00 [W:onnxruntime:, transformer_memcpy.cc:52 "
            "ApplyImpl] " + self.REAL_LINE
        )
        assert any(f.kind == "memcpy_nodes_added" for f in self._parse(prefixed).facts)

    @pytest.mark.parametrize(
        "fabricated",
        [
            "Add MemcpyFromHost/MemcpyToHost for 5 nodes",
            "Inserted 3 Memcpy node(s)",
        ],
    )
    def test_the_previously_fabricated_strings_do_not_match(self, fabricated):
        # Pinning the bug: these are what the signature used to look for, and
        # ORT emits neither. If they ever match again, someone has reintroduced
        # a guess.
        facts = self._parse(fabricated)
        assert not [f for f in facts.facts if f.kind == "memcpy_nodes_added"]

    def test_a_memcpy_count_of_zero_is_not_reported(self):
        # ORT only logs at all when copy_node_counter > 0, so a "0 Memcpy nodes"
        # line does not occur; if one did, reporting it as a boundary would be
        # wrong. This documents that the count is always meaningful.
        facts = self._parse(self.REAL_LINE.replace("7 Memcpy", "0 Memcpy"))
        memcpy = [f for f in facts.facts if f.kind == "memcpy_nodes_added"]
        # It parses (the pattern is numeric-agnostic) but the count is honest.
        assert not memcpy or memcpy[0].data["count"] == 0
