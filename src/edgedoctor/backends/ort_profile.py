"""ONNX Runtime profiling-JSON parser — where the time actually goes.

WHY THIS IS SEPARATE FROM onnxruntime.py
That module answers "WHERE did each op run?" from a text log. This one answers
"what did it COST?" from a JSON trace. They are complementary halves of the same
question and are usually read together: ED0301 tells you the graph was split
across providers, and this tells you whether that split actually cost anything.
A two-partition split around a 0.1% operator is a curiosity; the same split
around a 40% operator is the bug.

They are separate modules because the artifacts are separate — you get the text
log from stderr and the JSON from `end_profiling()` — and because the parsing has
nothing in common.

A THIRD PARSER SHAPE
  - tensorrt.py    stateless, line-based
  - polygraphy.py  block-structured text
  - onnxruntime.py block-structured AND aggregating
  - this one       structured JSON, aggregating across thousands of events

Because it aggregates, it must be careful about a specific dishonesty: a
percentage is meaningless without its denominator, and a "slowest op" claim is
meaningless if the run was one warm-up iteration. So every derived fact here
carries the totals it was computed from, and the parser records the iteration
count so a rule can decline to draw conclusions from a single noisy sample.

THE HONESTY TRAP HERE
Profiling data invites invented causation. "Conv is 60% of runtime" is a
measurement; "Conv is your bottleneck" is a conclusion that may be wrong (60% of
a 2ms model may be irrelevant, and the first iteration includes one-time
allocation). This parser records measurements and shares of measured total,
never verdicts. Interpretation lives in the rules, which can see the iteration
count and the absolute times.

GROUNDING NOTE ON `source`
Every other parser cites `file:line`. A JSON trace has no meaningful line
numbers, so facts here cite `file:<json-pointer-ish path>` (e.g.
`prof.json:events[17]`) — still a precise, checkable location in the artifact,
which is what the traceability contract actually requires.

Format verified against real output from onnxruntime 1.27 (`enable_profiling`).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .base import Backend, Fact, Facts

#: Ops contributing at least this share of measured node time are recorded
#: individually. Below this they're noise for diagnosis purposes and would bury
#: the signal.
SIGNIFICANT_SHARE = 0.05

#: A run with fewer than this many iterations can't support timing claims —
#: the first iteration includes one-time allocation and cache warming.
MIN_RELIABLE_ITERATIONS = 3

#: EP-compiled subgraphs are named with a hash, e.g.
#: "7615378459790495232_CoreML_7615378459790495232_0". Showing that to a user as
#: "the expensive op" is noise — it names nothing they can act on. We keep the
#: raw name in `data` for traceability but present a readable label.
_HASH_PREFIX_LEN = 6


#: The two shapes ORT uses for an EP-compiled subgraph. Matched STRICTLY, in
#: full, rather than by "contains a long digit run" — an earlier heuristic
#: rewrote the ordinary op name `a_1234567890_b` into
#: "a compiled subgraph #?", inventing a fused subgraph that doesn't exist.
#: Renaming a real operator is worse than leaving a hash visible: the user goes
#: looking for something that isn't in their graph.
#:
#:   <hash>_<EP>_<hash>_<index>                          (op_name form)
#:   <EP>ExecutionProvider_<hash>_<EP>_<hash>_<i>_<j>    (node name form)
_SUBGRAPH_OP = re.compile(
    r"^\d{%d,}_(?P<ep>\w+?)_\d{%d,}_(?P<index>\d+)$" % (_HASH_PREFIX_LEN, _HASH_PREFIX_LEN)
)
_SUBGRAPH_NODE = re.compile(
    r"^(?P<ep>\w+?)_\d{%d,}_\w+?_\d{%d,}_(?P<index>\d+)_\d+$"
    % (_HASH_PREFIX_LEN, _HASH_PREFIX_LEN)
)


def _readable_op(op: str) -> str:
    """Turn an EP-compiled subgraph's hash name into something legible.

    ORT names a fused node and its op type differently, so both forms are
    handled (see _SUBGRAPH_OP / _SUBGRAPH_NODE). Neither names anything a user
    can act on, so both collapse to the same readable label.

    Anything that does NOT match one of those exact shapes is returned unchanged.
    That conservatism is deliberate: a real operator name may legitimately
    contain digits and underscores, and mislabelling one as a compiled subgraph
    would send the reader hunting for a node their model doesn't contain.
    """
    for pattern in (_SUBGRAPH_OP, _SUBGRAPH_NODE):
        m = pattern.match(op)
        if m:
            ep = m.group("ep").removesuffix("ExecutionProvider")
            return f"{ep} compiled subgraph #{m.group('index')}"
    return op


class OrtProfileBackend(Backend):
    """Parses ONNX Runtime profiling JSON into per-op cost Facts."""

    name = "ort_profile"

    def convert(self, model_path: Path, **options: Any) -> list[Path]:
        """Not applicable: profiling is produced by running a session.

        Enable it with SessionOptions.enable_profiling = True, then call
        sess.end_profiling() to get the JSON path.
        """
        raise NotImplementedError(
            "ort_profile reads a profiling trace, it doesn't create models. "
            "Produce one with SessionOptions.enable_profiling = True."
        )

    def parse(self, artifact_path: Path) -> Facts:
        text = artifact_path.read_text(errors="replace")
        return self.parse_text(text, artifact_name=artifact_path.name)

    def parse_text(self, text: str, artifact_name: str = "<string>") -> Facts:
        facts: list[Fact] = []

        try:
            events = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            # Not JSON at all. Returning no facts (rather than raising) matches
            # every other parser: a wrong-format artifact is a clean "nothing
            # matched", not a crash.
            return Facts(backend=self.name, artifact_path=artifact_name, facts=[])

        # ORT writes a bare list; tolerate a {"traceEvents": [...]} wrapper too,
        # since that's the standard Chrome-trace envelope.
        if isinstance(events, dict):
            events = events.get("traceEvents", [])
        if not isinstance(events, list):
            return Facts(backend=self.name, artifact_path=artifact_name, facts=[])

        def add(kind: str, summary: str, index: int | str,
                data: dict[str, Any], excerpt: str) -> None:
            facts.append(
                Fact(
                    id=f"f{len(facts) + 1}",
                    kind=kind,
                    summary=summary,
                    source=f"{artifact_name}:{index}",
                    excerpt=excerpt,
                    data=data,
                )
            )

        # ── Session-level phases ──────────────────────────────────────────
        # Model loading and session init are one-time costs. Worth separating:
        # a user timing "inference" who actually measured session creation is a
        # common and very misleading mistake.
        for i, ev in enumerate(events):
            if not isinstance(ev, dict) or ev.get("cat") != "Session":
                continue
            name, dur = ev.get("name", ""), ev.get("dur")
            # A negative or non-finite duration is not a measurement; recording
            # it would put nonsense in the report and skew every share computed
            # from the total.
            if not isinstance(dur, int | float) or isinstance(dur, bool) or dur < 0:
                continue
            if name in ("model_loading_uri", "model_loading_array",
                        "session_initialization"):
                add(
                    "session_phase",
                    f"{name} took {dur / 1000:.1f} ms",
                    f"events[{i}]",
                    {"phase": name, "us": int(dur), "ms": round(dur / 1000, 3)},
                    f'{{"name": "{name}", "dur": {dur}}}',
                )

        # ── Per-node timings ──────────────────────────────────────────────
        node_events = [
            (i, ev) for i, ev in enumerate(events)
            if isinstance(ev, dict) and ev.get("cat") == "Node"
            and isinstance(ev.get("dur"), int | float)
            and not isinstance(ev.get("dur"), bool)
            # Negative durations are rejected for the same reason as above: a
            # share of a total polluted by negative values is meaningless.
            and ev["dur"] >= 0
            and str(ev.get("name", "")).endswith("_kernel_time")
        ]

        if not node_events:
            return Facts(backend=self.name, artifact_path=artifact_name, facts=facts)

        # Aggregate by op type and by provider.
        by_op: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"us": 0, "calls": 0, "nodes": set()}
        )
        by_provider: dict[str, int] = defaultdict(int)
        # Iterations are inferred from how many times the same node is timed —
        # ORT emits one event per node per run.
        node_call_counts: dict[str, int] = defaultdict(int)
        total_us = 0
        slowest: tuple[int, dict[str, Any]] | None = None

        for i, ev in node_events:
            args = ev.get("args") or {}
            dur = int(ev["dur"])
            op = str(args.get("op_name") or "Unknown")
            provider = str(args.get("provider") or "Unknown")
            node = str(ev.get("name", "")).removesuffix("_kernel_time")

            total_us += dur
            by_op[op]["us"] += dur
            by_op[op]["calls"] += 1
            by_op[op]["nodes"].add(node)
            by_provider[provider] += dur
            node_call_counts[node] += 1
            if slowest is None or dur > int(slowest[1]["dur"]):
                slowest = (i, ev)

        iterations = max(node_call_counts.values()) if node_call_counts else 0

        add(
            "profile_summary",
            f"{len(node_call_counts)} node(s) timed over ~{iterations} iteration(s), "
            f"{total_us / 1000:.1f} ms total node time",
            "summary",
            {
                "total_node_us": total_us,
                "total_node_ms": round(total_us / 1000, 3),
                "unique_nodes": len(node_call_counts),
                "iterations": iterations,
                "op_types": len(by_op),
            },
            f'{len(node_events)} Node kernel_time events, total dur {total_us} us',
        )

        # Iteration count is its own fact so a rule can require it — a timing
        # claim from one warm-up iteration is not a finding.
        if iterations < MIN_RELIABLE_ITERATIONS:
            add(
                "few_iterations",
                f"only ~{iterations} iteration(s) profiled — timings include "
                "one-time warm-up costs",
                "summary",
                {"iterations": iterations, "minimum": MIN_RELIABLE_ITERATIONS},
                f"max per-node call count = {iterations}",
            )

        # ── Per-op-type costs, with their denominator attached ────────────
        for op, agg in sorted(by_op.items(), key=lambda kv: -kv[1]["us"]):
            share = agg["us"] / total_us if total_us else 0.0
            if share < SIGNIFICANT_SHARE:
                continue
            add(
                "op_cost",
                f"{_readable_op(op)}: {agg['us'] / 1000:.1f} ms "
                f"({share * 100:.1f}% of measured node time)",
                f"op:{op}",
                {
                    "op": _readable_op(op),
                    "raw_op": op,
                    "us": agg["us"],
                    "ms": round(agg["us"] / 1000, 3),
                    "share_pct": round(share * 100, 2),
                    "calls": agg["calls"],
                    "distinct_nodes": len(agg["nodes"]),
                    # The denominator travels with the percentage: a share is
                    # uninterpretable without knowing what it's a share OF.
                    "total_node_ms": round(total_us / 1000, 3),
                },
                f'op_name={op}, total dur={agg["us"]} us over {agg["calls"]} call(s)',
            )

        # ── Time split across providers ───────────────────────────────────
        # The cost half of ED0301: how much time actually ran on each provider.
        if len(by_provider) > 1:
            parts = ", ".join(
                f"{p} {us / 1000:.1f} ms ({us / total_us * 100:.0f}%)"
                for p, us in sorted(by_provider.items(), key=lambda kv: -kv[1])
            )
            cpu_us = sum(us for p, us in by_provider.items() if p == "CPUExecutionProvider")
            add(
                "provider_time_split",
                f"node time split across providers: {parts}",
                "summary",
                {
                    "providers": {p: us for p, us in by_provider.items()},
                    "provider_count": len(by_provider),
                    "cpu_us": cpu_us,
                    "cpu_share_pct": round(cpu_us / total_us * 100, 2) if total_us else 0,
                    "total_node_ms": round(total_us / 1000, 3),
                },
                f"{len(by_provider)} providers in Node events: "
                f"{', '.join(sorted(by_provider))}",
            )

        # ── The single slowest node ───────────────────────────────────────
        if slowest is not None and total_us:
            idx, ev = slowest
            args = ev.get("args") or {}
            dur = int(ev["dur"])
            node = str(ev.get("name", "")).removesuffix("_kernel_time")
            add(
                "slowest_node",
                f"slowest single node: '{_readable_op(node)}' "
                f"({_readable_op(str(args.get('op_name') or '?'))}) "
                f"at {dur / 1000:.2f} ms",
                f"events[{idx}]",
                {
                    "node": _readable_op(node),
                    "raw_node": node,
                    "op": _readable_op(str(args.get("op_name") or "Unknown")),
                    "us": dur,
                    "ms": round(dur / 1000, 3),
                    "provider": str(args.get("provider") or "Unknown"),
                    "share_pct": round(dur / total_us * 100, 2),
                    "total_node_ms": round(total_us / 1000, 3),
                },
                f'"{ev.get("name")}": dur={dur}, op_name={args.get("op_name")}',
            )

        return Facts(backend=self.name, artifact_path=artifact_name, facts=facts)
