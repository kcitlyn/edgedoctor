# Design

This document captures the *rationale* and the *architecture*. The headline idea:
a **simple core, designed to extend** — we architect for the whole vision on day
one (the seams exist) but only fully implement the current slice.

---

## Why the gap is credible and durable

- The underlying vendor tools (`trtexec`, Polygraphy, ETDump, `coremltools`)
  produce raw data but don't explain it.
- The cross-vendor, LLM-grounded explanation layer is essentially empty. The
  closest competitor ("Cellulose") is abandoned; NVIDIA's own forum auto-replies
  with a context-blind LLM bot that gives wrong advice — proof the pain is real
  and unsolved.
- Why no big vendor will close it: the useful version is cross-vendor (against
  any one vendor's interest), it's a product/UX problem (not their engineering
  culture), it's small-TAM / high-annoyance, and reliable LLM-grounding is new and
  liability-laden for a megacorp. → durable opening for an independent builder.

---

## Build strategy: beachhead → expansion

**Principle:** the fastest way to make the big vision believable is to make one
slice *undeniably* work, then expand outward along a clear path. A working slice
proves the universal version is real; a half-built "everything tool" reads as
vaporware and hurts credibility. So we think big and build in order.

**The beachhead** (prove the core thesis first):

- **Input:** PyTorch model.
- **One backend first — TensorRT** (loudest pain, abandoned competitor, best
  NVIDIA/edge resume value, RTX 500 Ada on hand). Fallback: CoreML on the Mac if
  CUDA/TensorRT setup blocks week 1.
- **Two cleanly-traceable failure classes:**
  - (A) Conversion / op failures & CPU fallback — logs name the op → traceable.
  - (B) Accuracy divergence (FP32 vs INT8) — rank diverging layers (e.g. by SQNR)
    → numerically traceable.
- The grounded diagnoser + a CLI.

The full expansion path lives in [VISION.md](../VISION.md). The one genuine
off-limits item near-term is **fused-engine latency attribution** — a technical
dead-end (undocumented post-fusion node→source mapping), not a scope-shrink.

---

## Quality bar — the differentiator (non-negotiable)

- The diagnoser may **only assert facts extracted from the user's real, parsed
  artifacts.** No free-association, no invented causes/fixes, no guessing beyond
  the evidence. A wrong explanation is worse than none — it's exactly how the
  NVIDIA bot fails, and how edgedoctor wins by contrast.
- **Every claim must be traceable** to a parsed log line / measured value.
  *"I don't have enough info to say X"* is an acceptable — often correct — output.
- This grounding discipline is the project's moat. Apply it rigorously.

---

## Architecture

```
[ PyTorch model ]
      │  export + convert  (PyTorch→ONNX→TensorRT, or →CoreML, …)
      ▼
[ raw artifacts: conversion logs, profiler output, layer-wise numeric dumps ]
      │  deterministic, per-backend PARSERS  →  structured "facts" (JSON)
      ▼
[ GROUNDED LLM diagnoser ]  (facts in → plain-English cause + fix out)
      ▼
[ CLI report ]   (+ later: MCP server exposing the same core)
```

### Design rules

1. **Pluggable per-backend parser layer.** A clean `Backend` / parser interface
   exists from day one, even though only one backend is implemented first. This
   is how we think big in the architecture without building everything: the seams
   for expansion exist; only one is filled. See
   [`src/edgedoctor/backends/base.py`](../src/edgedoctor/backends/base.py).
2. **The LLM only ever sees parsed facts** — never raw blobs it could hallucinate
   over. The deterministic parser is the firewall between messy vendor output and
   the explanation engine.
3. **Library + thin CLI.** The MCP server is a *later surface over the same
   core* — not a separate product, and not a framework. Keep the core importable
   and side-effect-free so multiple surfaces can wrap it.

### The two data contracts

- **`Facts`** — what a parser emits: a structured, JSON-serializable record of
  *only what was observed* in the artifact (op names, log line numbers, measured
  SQNR per layer, …). No interpretation.
- **`Diagnosis`** — what the diagnoser emits: `root_cause`, `evidence`
  (pointers back into the Facts), `fix`, and a `confidence` / `insufficient_info`
  signal. Every field traces to a Fact.

Keeping these two contracts separate is what enforces the grounding discipline
structurally — the diagnoser physically cannot see anything the parser didn't
record.

---

## Hardware & environment

- **MacBook Pro (Apple Silicon):** PyTorch training (MPS), CoreML / Neural
  Engine, ONNX Runtime. Primary dev + training machine.
- **ThinkPad P15s — NVIDIA RTX 500 Ada (4 GB), CUDA 13.2:** the TensorRT lane.
  4 GB ⇒ small edge models (YOLO-nano/small, MobileNet-SSD). CUDA 13.2 is very
  new ⇒ TensorRT version matching is fiddly — treat version mismatch as a
  first-class failure the tool should diagnose (free dogfooding).
- **Raspberry Pi 5:** later (expansion step 5), not in hand yet — don't design
  around it now.
- **Division of labor:** train on the Mac; convert / quantize / diagnose on
  whichever backend matches the current target.
