# Vision

> Think big. This document describes the whole target — not where the build is
> today (see [ROADMAP.md](ROADMAP.md) for that). The ambition and the disciplined
> build order are not in tension: building one slice undeniably well is *how* the
> universal version becomes believable.

## The target

Build the **universal, cross-vendor edge-AI deployment diagnostician**: the open
tool that stands between *any* model and *any* edge target and explains, in plain
language, **why it will break, why it's slow, and how to fix it.**

Today, getting a model onto edge hardware is a minefield of cryptic,
vendor-specific failures. Every vendor ships tools that produce raw data (logs,
profiler dumps) but never explanations — and each only covers its own silicon.
No one owns the cross-vendor *explanation* layer, because no single vendor has
the incentive to build a tool that spans their competitors. **That gap is the
opportunity.**

## The full vision includes

- **Every major backend:** TensorRT, CoreML, ONNX Runtime, TFLite / LiteRT,
  ExecuTorch, and vendor NPUs.
- **The full failure taxonomy:** conversion / op-support failures, CPU fallback,
  accuracy divergence, memory / arena overflow, and (eventually) performance
  regressions.
- **An LLM-grounded explanation engine** that turns raw artifacts into a
  plain-English root cause + concrete fix — never hallucinated, always traceable
  to parsed facts.
- **A delivery surface for both humans and agents:** a CLI *and* an MCP server,
  so AI coding agents (Claude Code, Cursor) can call it directly during their own
  deploy loops.
- **A reference benchmark:** a curated corpus of models-that-fail-in-known-ways
  that becomes the dataset for evaluating edge-deployment tooling.
- **A community of contributors,** each adding backend modules and failure
  detectors.

This is genuinely ambitious and genuinely unfilled. We aim here.

## Why no incumbent closes the gap

- The vendor tools (`trtexec`, Polygraphy, ETDump, `coremltools`) produce raw
  data but don't explain it.
- The cross-vendor, LLM-grounded explanation layer is essentially empty. The
  closest competitor ("Cellulose") is abandoned; NVIDIA's own forum auto-replies
  with a context-blind LLM bot that gives wrong advice — proof the pain is real
  and unsolved.
- A big vendor won't build it: the useful version is cross-vendor (against any
  one vendor's interest), it's a product/UX problem (not their engineering
  culture), it's small-TAM / high-annoyance, and reliable LLM-grounding is new and
  liability-laden for a megacorp. → a durable opening for an independent builder.

---

## The expansion path (the vision, unlocked in order)

We ship a **beachhead** first, then expand. Each step below is a real milestone
and its own mini-release — not a "nice to have."

### Beachhead — prove the core thesis

- **Input:** a PyTorch model.
- **One backend first — TensorRT** (loudest pain, abandoned competitor, best
  NVIDIA/edge resume value, and we have an RTX 500 Ada). Fallback: **CoreML** on
  the Mac if CUDA/TensorRT setup blocks week 1.
- **Two cleanly-traceable failure classes:**
  - **(A) Conversion / op failures & CPU fallback** — logs name the op →
    traceable.
  - **(B) Accuracy divergence** (FP32 vs INT8) — rank diverging layers (e.g. by
    SQNR) → numerically traceable.
- **The grounded diagnoser + a CLI.**

### Expansion steps

1. **2nd & 3rd backends** (CoreML, ONNX-RT, TFLite/ExecuTorch) — proves
   "cross-vendor."
2. **Memory / arena-overflow failure class**, then more of the taxonomy.
3. **MCP server wrapper** — agents call edgedoctor directly (our unique
   differentiator).
4. **Synthetic failure-corpus + LLM-as-judge eval harness** — becomes the
   reference benchmark and an evaluation story.
5. **MCU / on-device divergence module** (when the Raspberry Pi 5 arrives).
6. **Pre-flight prediction** — diagnose *before* converting. The hardest,
   highest-value frontier.

### The one genuine off-limits item

⛔ **Fused-engine latency attribution** ("why is my engine 10 ms slower")
near-term. It depends on undocumented post-fusion node→source mapping and can
burn a month for nothing. Avoided because it's a *wall*, not because it's
ambitious — it can return later as a research-grade frontier once everything
else is solid.

> Everything else in the vision is in-bounds — just **sequenced**. When asked to
> jump ahead: confirm the current slice works, then proceed along this path.
> Bias toward "make the current slice real," not toward sprawling before
> anything runs.
