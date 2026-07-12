# Working Plan — July 12 → mid-August 2026 (Pi 5 milestone)

Constraint model: internship until ~Aug 7 → weekdays are light (≤1 hr),
weekends are the real build blocks. After Aug 7: near-full-time until the Pi
arrives (~Aug 15). Every week ends with something that runs.

Design principles this plan serves (see docs/DESIGN.md + research notes):
- **Rules-first, LLM-second.** A curated YAML knowledge base (error signature →
  cause → fix) gives a useful deterministic diagnosis with ZERO config/API key.
  The LLM is an optional layer for unseen variants + prose synthesis
  (k8sgpt-proven pattern). This also pre-builds the Phase-2 kill-criterion
  fallback, so the risk is retired early.
- **rustc-style report anatomy**: header `error[ED0xx]` → verbatim evidence
  (user's own log lines, never paraphrased) → `note:` (why) → `help:` (fix) →
  confidence → doc link. `--json` (LSP-shaped schema) for agents. Exit codes:
  0 healthy / 1 tool error / 2 error found / 3 warnings only.
- **The Pi milestone = the ONNX Runtime backend.** TensorRT cannot run on a Pi
  (no CUDA). ORT aarch64 installs via pip on Pi OS, and the SAME backend runs on
  the Mac — so it's built and tested at the desk first; Pi day is a host swap,
  not a build day.

## Week 1 — Jul 13–19 · Phase 1 starts: first real pipeline
- **Mon–Fri (evenings, ~30–60 min each):**
  - Mon: `uv` project setup; swap CLI scaffold to typer+pydantic stack.
  - Tue: `scripts/export_onnx.py` — torchvision MobileNetV3/ResNet18 → ONNX
    (dynamo exporter); inspect with `polygraphy inspect` / Netron.
  - Wed: ONNX Runtime parity check vs PyTorch outputs (the "golden" baseline).
  - Thu: ThinkPad: install TensorRT matching CUDA 13.2; record every
    version-mismatch error verbatim → `corpus/` (first real test data!).
  - Fri: buffer / catch-up.
- **Weekend (Sat 18–Sun 19):** first successful `trtexec` engine build (FP32,
  FP16) with `--verbose` logs saved; then deliberately BREAK it (unsupported op
  e.g. GridSample at old opset, wrong shapes) and archive every failure log in
  `corpus/tensorrt/`. Goal: ≥5 real failure artifacts.

## Week 2 — Jul 20–26 · Phase 1 finishes: quantization + divergence data
- **Weekdays:** small: calibration-set prep; read TRT quantization docs (explicit
  Q/DQ — implicit calibrator API is deprecated since TRT 10.1); start
  `corpus/README.md` cataloging each artifact + ground-truth cause.
- **Weekend (25–26):** INT8 build via explicit Q/DQ (ModelOpt); run
  `polygraphy run --trt --onnxrt` comparisons; capture accuracy-divergence
  artifacts; compute per-layer SQNR ranking on at least one model.
  **Phase 1 done = trained→exported→converted→measured, with a labeled corpus.**

## Week 3 — Jul 27–Aug 2 · Phase 2 starts: parsers + rule KB + report UX
- **Weekdays:** pydantic `Fact`/`Facts`/`Diagnosis` models finalized; first
  parser: TensorRT build-log → Facts (regex signatures from research:
  `No importer registered for op:`, `Could not find any implementation for
  node`, `getPluginCreator could not find plugin`, both TRT ≤8.x and 10.x node
  formats). Golden-file tests with syrupy against `corpus/`.
- **Weekend (Aug 1–2):** YAML rule KB v1 (~10 rules, `ED0xx` codes); the
  rustc-style terminal report (rich) + `--json`; wire into CLI. **Demo: broken
  model in → `error[ED0xx]` + evidence + fix out, no API key needed.**

## Week 4 — Aug 3–9 · Grounded LLM layer (internship ends Aug 7 🎉)
- **Weekdays (through Thu):** Polygraphy-comparison parser (Class B facts).
- **Fri Aug 7–Sun Aug 9:** LLM synthesis via `client.messages.parse()` +
  pydantic schema (Haiku default, ~half a cent per diagnosis); system prompt:
  facts-only, must cite Fact ids, `insufficient_info` allowed; degrade to
  rules-only on any API failure. pytest-recording cassettes for offline CI.
  Groundedness eval: every cited fact id must exist in input.
  **Phase 2 done = the 30-second demo, now with plain-English synthesis.**

## Week 5 — Aug 10–16 · Second backend (ONNX Runtime) + Pi arrival
- **Mon–Wed (on the Mac — no Pi needed):** `backends/onnxruntime.py`:
  parse ORT verbose "Node placements" log (CPU-fallback detection — this class
  lives HERE, not in TensorRT) + ORT profiling JSON. Rules for
  `CPUExecutionProvider` fallback, Memcpy node insertion.
- **Thu–Fri:** Pi-specific fact sources, testable in isolation:
  `vcgencmd get_throttled` bitfield parser (thermal/undervolt), OOM detection.
- **Pi arrives (~Aug 15): the milestone script**
  1. `pip install onnxruntime` on Pi OS 64-bit; run the same models.
  2. Reproduce + diagnose on-device: (a) x86-vs-ARM INT8 divergence (verified
     real: x86 U8S8 saturation doesn't occur on Cortex-A76), (b) thermal
     throttling skewing benchmarks (get_throttled bits 2/3/18/19),
     (c) OOM on large model.
  3. `edgedoctor diagnose` runs ON the Pi → cross-vendor proof: same tool, same
     report format, second backend. 📸 for README.

## Explicitly deferred (post-Pi)
- ExecuTorch backend (ETDump is the richest diagnostic surface — strong
  candidate for backend #3), LiteRT, MCP server, eval corpus at scale,
  Hailo NPU (needs x86 compile box + Developer Zone account — start
  registration early if wanted).
