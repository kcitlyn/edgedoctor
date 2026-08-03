"""ONNX Runtime backend — failure class (A), CPU fallback and EP placement.

This is edgedoctor's SECOND vendor, and it exists to prove the cross-vendor
claim: the same Facts/Diagnosis contracts, the same report format, a different
tool's logs. It is also the Raspberry Pi 5 milestone — TensorRT cannot run on a
Pi (no CUDA), but ORT installs via pip on Pi OS aarch64 and this exact parser
handles its output, so Pi day is a host swap rather than a build day.

WHY CPU FALLBACK LIVES HERE AND NOT IN tensorrt.py
TensorRT refuses to build when it meets an op it can't handle — that's ED0101,
a loud failure. ORT does the opposite: it silently *succeeds*, quietly running
the unsupported parts of your graph on CPU. Nothing alerts you. You just get
a model that's inexplicably slower than the accelerator you bought. That
silence is the failure mode, which is why it belongs to this backend.

WHAT MAKES THIS PARSER DIFFERENT FROM THE OTHER TWO
  - tensorrt.py is stateless: one failure, one line.
  - polygraphy.py is block-structured: a per-tensor block whose later lines
    don't repeat the tensor name.
  - This one is block-structured AND aggregating: the "Node placements" section
    lists one line per execution provider, each followed by that EP's nodes.
    The interesting facts are the RELATIONSHIPS between those groups (is more
    than one EP present? did the EP you asked for get any nodes at all?), so
    parse_text does a second pass to derive them once the whole section is read.

THE HONESTY TRAP THIS PARSER IS BUILT TO AVOID
"All nodes placed on [CPUExecutionProvider]" is either completely fine or the
whole problem, and the line itself cannot tell you which. A deliberate CPU-only
session looks identical to a failed accelerator. So this parser records what was
REQUESTED (`session_providers`, `provider_unavailable`) separately from what
HAPPENED (`node_placement`), and leaves the judgement to rules that can see
both. See ort_cpu_only.log vs ort_missing_provider.log in corpus/onnxruntime/ —
their placement lines are identical and their diagnoses must differ.

Signatures are verified against real logs from scripts/make_ort_corpus.py and,
where the string comes from Python, against ORT's own source (cited per
signature). The C++ strings come from onnxruntime/core/framework/session_state.cc
(`VerifyEachNodeIsAssignedToAnEp`).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from .base import Backend, Fact, Facts

_Sig = tuple[str, re.Pattern[str], Callable[[re.Match[str]], str]]

# ORT prefixes every C++ log line with a timestamp, pid, severity and source
# location. We anchor on the message text rather than the prefix, since the
# prefix format varies between platforms and log sinks.

_SIGNATURES: list[_Sig] = [
    # ── Placement: what actually happened ─────────────────────────────────
    # Every node on a single EP. Source: session_state.cc VerifyEachNode...,
    # the "All nodes placed on [X]" branch.
    (
        "all_nodes_one_provider",
        re.compile(
            r"All nodes placed on \[(?P<provider>\w+)\]\. "
            r"Number of nodes: (?P<count>\d+)"
        ),
        lambda m: (
            f"All {m['count']} node group(s) placed on {m['provider']} "
            "(single partition)"
        ),
    ),
    # The graph was split: one such line PER execution provider involved.
    # Source: session_state.cc, the per-EP branch.
    (
        "node_placement",
        re.compile(
            r"Node\(s\) placed on \[(?P<provider>\w+)\]\. "
            r"Number of nodes: (?P<count>\d+)"
        ),
        lambda m: f"{m['count']} node group(s) placed on {m['provider']}",
    ),
    # How many partitions an EP could claim, and how much of the graph.
    # This is the performance-relevant line: partition COUNT matters more than
    # node count, because each boundary is a synchronization point.
    (
        "provider_capability",
        # PERFORMANCE, not style: the provider group is bounded and the pattern
        # leads with a LITERAL. An earlier version began with `(?P<provider>\w+)`
        # followed by "::GetCapability", which made matching O(n^2) — on a long
        # run of word characters the engine retries the literal from every start
        # position. A single 200 KB line hung the parser for minutes. Real logs
        # do contain very long lines, so this is a genuine DoS on untrusted
        # input, not a hypothetical. Covered by tests/test_parser_robustness.py.
        # The provider group is BOUNDED ({1,64}); an unbounded `\w+` here is what
        # caused the blow-up, because it could start anywhere in a long word run.
        # A bounded quantifier caps the backtracking work per start position.
        re.compile(
            r"(?P<provider>\w{1,64})::GetCapability, "
            r"number of partitions supported by "
            r"(?P<short_name>\w{1,64}): (?P<partitions>\d+) "
            r"number of nodes in the graph: (?P<graph_nodes>\d+) "
            r"number of nodes supported by (?P<short_name2>\w{1,64}): "
            r"(?P<supported>\d+)"
        ),
        lambda m: (
            f"{m['provider']} claimed {m['supported']}/{m['graph_nodes']} nodes "
            f"across {m['partitions']} partition(s)"
        ),
    ),
    # ── Requested vs available: what the user ASKED for ───────────────────
    # A requested EP that isn't in this build. ORT warns and DROPS it rather
    # than failing, so the session succeeds on CPU and nothing looks wrong.
    # Source: onnxruntime/capi/onnxruntime_inference_collection.py:148.
    (
        "provider_unavailable",
        re.compile(
            r"Specified provider '(?P<requested>\w+)' is not in available "
            r"provider names\.\s*Available providers: '(?P<available>[^']*)'"
        ),
        lambda m: (
            f"Requested provider {m['requested']} is NOT available in this "
            f"onnxruntime build"
        ),
    ),
    # The providers the session ended up with. Emitted by our corpus driver via
    # sess.get_providers(); it is the ground truth for "what did I actually
    # get", and is the fact that makes an intentional CPU run distinguishable
    # from a silently degraded one.
    (
        "session_providers",
        re.compile(r"SESSION_PROVIDERS: (?P<providers>.+)$"),
        lambda m: f"Session initialized with: {m['providers']}",
    ),
    (
        "session_failed",
        re.compile(r"SESSION_FAILED: (?P<error>.+)$"),
        lambda m: f"Session construction failed: {m['error']}",
    ),
    # ── Data movement inserted by the framework ───────────────────────────
    # Memcpy nodes are ORT's stitching between EPs that don't share memory —
    # direct evidence of a host/device boundary inside the graph.
    (
        "memcpy_nodes_added",
        re.compile(
            r"Add MemcpyFromHost/MemcpyToHost for (?P<count>\d+)|"
            r"Inserted (?P<count2>\d+) Memcpy node"
        ),
        lambda m: "Memcpy node(s) inserted to bridge execution providers",
    ),
    (
        "memcpy_transformer",
        re.compile(r"GraphTransformer MemcpyTransformer modified: (?P<modified>\d+)"),
        lambda m: (
            "MemcpyTransformer inserted cross-provider copies"
            if m["modified"] != "0"
            else "MemcpyTransformer made no changes (no cross-provider copies needed)"
        ),
    ),
    # ── Errors ────────────────────────────────────────────────────────────
    (
        "load_failed",
        re.compile(r"Load model .*? failed|Failed to load model"),
        lambda m: "Model failed to load",
    ),
    (
        "protobuf_error",
        re.compile(r"Protobuf parsing failed|Error parsing message"),
        lambda m: "Model file is not a valid ONNX protobuf",
    ),
]

# Node-detail lines inside a placement group, e.g. "  Erf (Erf_0)". Handled
# separately from the signature table because they only mean anything in the
# context of the group heading above them, and there are many per group.
_NODE_DETAIL = re.compile(
    r"VerifyEachNodeIsAssignedToAnEp\]\s+(?P<op>[\w./-]+) \((?P<name>[^)]+)\)\s*$"
)

# EP-compiled subgraphs get synthetic names like
# "7615378459790495232_CoreML_7615378459790495232_0". Those are hashes, not ops,
# and reporting them to a user as "the op that fell back" would be noise.
_SYNTHETIC_NODE = re.compile(r"^\d{6,}_|_\d{6,}_")

# Providers that are not real accelerators, for deciding whether a placement
# constitutes "fallback". CPU is ORT's guaranteed last resort.
CPU_PROVIDER = "CPUExecutionProvider"


class OnnxRuntimeBackend(Backend):
    """Parses ONNX Runtime verbose session logs into placement/fallback Facts."""

    name = "onnxruntime"

    def convert(self, model_path: Path, **options: Any) -> list[Path]:
        """Not applicable: ORT consumes ONNX directly, there is nothing to convert.

        Producing these artifacts means *running a session with verbose logging*,
        which is scripted in scripts/make_ort_corpus.py because it needs a
        subprocess (ORT's placement log is written from C++ to the process's
        stderr and can't be captured in-process).
        """
        raise NotImplementedError(
            "ONNX Runtime runs ONNX models directly — there is no conversion "
            "step. Generate placement logs with scripts/make_ort_corpus.py."
        )

    def parse(self, artifact_path: Path) -> Facts:
        text = artifact_path.read_text(errors="replace")
        return self.parse_text(text, artifact_name=artifact_path.name)

    def parse_text(self, text: str, artifact_name: str = "<string>") -> Facts:
        facts: list[Fact] = []
        # Which EP's node list we are currently inside. ORT prints the group
        # heading, then that group's nodes, with nothing to close the group.
        current_provider: str | None = None

        def add(kind: str, summary: str, lineno: int, line: str,
                data: dict[str, Any]) -> None:
            facts.append(
                Fact(
                    id=f"f{len(facts) + 1}",
                    kind=kind,
                    summary=summary,
                    source=f"{artifact_name}:{lineno}",
                    excerpt=line.strip(),
                    data=data,
                )
            )

        for lineno, line in enumerate(text.splitlines(), start=1):
            matched = False
            for kind, pattern, summarize in _SIGNATURES:
                m = pattern.search(line)
                if m is None:
                    continue
                groups = {k: v for k, v in m.groupdict().items() if v is not None}

                if kind in ("node_placement", "all_nodes_one_provider"):
                    current_provider = groups.get("provider")
                    groups["count"] = int(groups.get("count", 0))
                if kind == "provider_capability":
                    # Normalize the duplicate short-name group out of the payload
                    # and expose the numbers as ints for rule conditions.
                    groups.pop("short_name2", None)
                    for key in ("partitions", "graph_nodes", "supported"):
                        if key in groups:
                            groups[key] = int(groups[key])
                    groups["unsupported"] = (
                        groups.get("graph_nodes", 0) - groups.get("supported", 0)
                    )
                if kind == "session_providers":
                    groups["providers"] = [
                        p.strip() for p in groups["providers"].split(",") if p.strip()
                    ]
                if kind == "provider_unavailable":
                    groups["available"] = [
                        p.strip().strip("'")
                        for p in groups.get("available", "").split(",")
                        if p.strip()
                    ]
                if kind == "memcpy_nodes_added":
                    groups["count"] = int(groups.pop("count", None)
                                          or groups.pop("count2", None) or 0)

                add(kind, summarize(m), lineno, line, groups)
                matched = True
                break

            if matched:
                continue

            # Node-detail line inside the current placement group.
            nd = _NODE_DETAIL.search(line)
            if nd and current_provider:
                op = nd.group("op")
                if _SYNTHETIC_NODE.search(op):
                    # An EP-compiled subgraph, not a real op. Skipping keeps the
                    # op list meaningful instead of full of hashes.
                    continue
                add(
                    "node_on_provider",
                    f"Node '{nd.group('name')}' (op {op}) runs on {current_provider}",
                    lineno,
                    line,
                    {"op": op, "node": nd.group("name"),
                     "provider": current_provider},
                )

        facts.extend(self._derive(facts, artifact_name))
        return Facts(backend=self.name, artifact_path=artifact_name, facts=facts)

    def _derive(self, facts: list[Fact], artifact_name: str) -> list[Fact]:
        """Second pass: facts about the RELATIONSHIP between placement groups.

        These are still observations, not interpretations — "two providers appear
        in this log" is as literal as "one provider appears on line 42". But they
        can only be seen once the whole placement section has been read, which a
        single-pass line scanner cannot do.

        Each derived fact cites the line of the evidence it was derived from, so
        traceability holds: the user can still go look at the log.
        """
        derived: list[Fact] = []
        placements = [f for f in facts if f.kind == "node_placement"]
        providers = [f.data.get("provider") for f in placements]

        # A graph split across more than one EP. This is the fact that
        # distinguishes real partial fallback from any single-EP run.
        if len(set(providers)) > 1 and CPU_PROVIDER in providers:
            accel = [p for p in providers if p != CPU_PROVIDER]
            anchor = placements[0]
            cpu_groups = sum(
                f.data.get("count", 0) for f in placements
                if f.data.get("provider") == CPU_PROVIDER
            )
            derived.append(
                Fact(
                    id=f"f{len(facts) + len(derived) + 1}",
                    kind="split_execution",
                    summary=(
                        f"Graph split across {len(set(providers))} providers: "
                        f"{', '.join(sorted(set(providers)))} — "
                        f"{cpu_groups} group(s) fell back to CPU"
                    ),
                    source=anchor.source,
                    excerpt=anchor.excerpt,
                    data={
                        "providers": sorted(set(providers)),
                        "accelerators": sorted(set(accel)),
                        "cpu_node_groups": cpu_groups,
                    },
                )
            )

        # The ops that landed on CPU while an accelerator was present — the
        # actionable list, since these are what to replace or remove.
        cpu_ops = [
            f.data["op"] for f in facts
            if f.kind == "node_on_provider" and f.data.get("provider") == CPU_PROVIDER
        ]
        if cpu_ops and len(set(providers)) > 1:
            anchor = next(
                f for f in facts
                if f.kind == "node_on_provider"
                and f.data.get("provider") == CPU_PROVIDER
            )
            derived.append(
                Fact(
                    id=f"f{len(facts) + len(derived) + 1}",
                    kind="cpu_fallback_ops",
                    summary=(
                        f"{len(cpu_ops)} op(s) ran on CPU despite an accelerator "
                        f"being available: {', '.join(sorted(set(cpu_ops)))}"
                    ),
                    source=anchor.source,
                    excerpt=anchor.excerpt,
                    data={"ops": sorted(set(cpu_ops)), "count": len(cpu_ops),
                          "first_op": cpu_ops[0]},
                )
            )

        # An accelerator was requested but got nothing. Only derivable by
        # comparing the request against the placement — the single most
        # misleading situation in ORT, because the run reports success.
        unavailable = [f for f in facts if f.kind == "provider_unavailable"]
        got = next((f for f in facts if f.kind == "session_providers"), None)
        if unavailable and got:
            active = got.data.get("providers", [])
            if active == [CPU_PROVIDER]:
                derived.append(
                    Fact(
                        id=f"f{len(facts) + len(derived) + 1}",
                        kind="silent_cpu_only",
                        summary=(
                            f"Requested {unavailable[0].data.get('requested')} was "
                            "unavailable, so the session ran entirely on CPU"
                        ),
                        source=unavailable[0].source,
                        excerpt=unavailable[0].excerpt,
                        data={
                            "requested": unavailable[0].data.get("requested"),
                            "actual": active,
                        },
                    )
                )
        return derived
