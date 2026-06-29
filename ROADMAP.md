# Roadmap

Status legend: ✅ done · 🔜 **now / next** · 🗺️ later

> Phases ship something real at every step. We keep `main` runnable and honest:
> a small working slice beats a big broken one. The full ambition lives in
> [VISION.md](VISION.md); this file tracks *order* and *state*.

---

## Phase 0 — De-risk  ✅ (½ day)

- [x] Confirm no exact competitor launched recently
      (search: "TensorRT explain failure LLM", "ONNX conversion diagnose",
      "edge deployment diagnostics LLM").
- [x] Write the one-paragraph gap statement → README intro.
- **Kill criterion:** a maintained, popular tool already does exactly this →
  stop. *(Not triggered.)*

## Phase 1 — Foundation  🔜 **NOW** (2–3 weekends)

The first real ML/edge slice. Agent builds + explains each step; Kaitlyn follows
along and runs it.

- [ ] Train / fine-tune an object detector in PyTorch.
- [ ] Evaluate it (mAP / IoU / precision / recall).
- [ ] Convert to the beachhead backend (PyTorch → ONNX → TensorRT).
- [ ] Deploy & measure (latency / FPS).
- [ ] **Log every failure** — these become the tool's first real test corpus.

*Standalone resume value even on its own.*

## Phase 2 — Diagnoser MVP  🔜 **NEXT** (1–2 weekends)

- [ ] Deterministic parsers for the Phase-1 artifacts → structured "facts".
- [ ] Grounded LLM diagnosis for failure class **(A)** conversion / CPU fallback.
- [ ] Grounded LLM diagnosis for failure class **(B)** accuracy divergence.
- [ ] CLI report.
- **Done =** a 30-second demo: broken model in → cryptic log → 3 plain-English
  sentences + a fix.

## Phase 3+ — Expand toward the vision  🗺️ later (ongoing)

Each is its own mini-release (see [VISION.md](VISION.md) for full rationale):

- [ ] 2nd & 3rd backends — CoreML, ONNX-RT, TFLite/ExecuTorch → proves
      "cross-vendor."
- [ ] Memory / arena-overflow failure class, then more taxonomy.
- [ ] MCP server wrapper — agents call edgedoctor directly.
- [ ] Synthetic failure-corpus + LLM-as-judge eval harness → reference benchmark.
- [ ] MCU / on-device divergence module (when the Raspberry Pi 5 arrives).
- [ ] Pre-flight prediction — diagnose before converting (hardest frontier).

### Out of scope (near-term)

- ⛔ Fused-engine latency attribution — a technical dead-end near-term (see
  [VISION.md](VISION.md)).

---

## Kill / pivot criteria

- **Phase 0:** exact maintained competitor exists → stop.
- **End of Phase 1:** if the ML-on-hardware work isn't enjoyable → successful
  *direction experiment*; pivot to a pure-embedded project. Not a failure.
- **End of Phase 2:** if diagnoses can't be made reliably correct on real logs →
  downscope the diagnoser to "parser + KB lookup" (still useful), keep the vision.
