# edgedoctor

**The universal, cross-vendor edge-AI deployment diagnostician.**

Getting a model onto edge hardware is a minefield of cryptic, vendor-specific
failures. Every vendor ships tools that produce *raw data* — conversion logs,
profiler dumps, layer-wise numeric traces — but never *explanations*, and each
only covers its own silicon. No one owns the cross-vendor explanation layer,
because no single vendor has the incentive to build a tool that spans their
competitors.

**edgedoctor stands between any model and any edge target and tells you, in
plain language, why it will break, why it's slow, and how to fix it** — grounded
in your real artifacts, never hallucinated.

---

## Why this gap is real (and durable)

- The underlying vendor tools (`trtexec`, Polygraphy, ETDump, `coremltools`)
  emit raw data but don't explain it.
- The cross-vendor, LLM-grounded *explanation* layer is essentially empty. The
  closest prior attempt ("Cellulose") is abandoned; NVIDIA's own forum
  auto-replies with a context-blind bot that gives wrong advice — proof the pain
  is real and unsolved.
- No big vendor will close it: the useful version is **cross-vendor** (against
  any one vendor's interest), it's a product/UX problem rather than an
  engineering-culture one, the TAM is small but the annoyance is high, and
  reliable LLM grounding is new and liability-laden for a megacorp. → a durable
  opening for an independent, open-source tool.

---

## 🚧 Status — v0.1, early development

This repo is **scaffold + vision**. Here is the honest state of the build:

| Piece                                   | State           |
| --------------------------------------- | --------------- |
| Vision, architecture, roadmap docs      | ✅ done          |
| Pluggable backend/parser interface      | ✅ scaffolded    |
| CLI skeleton (`edgedoctor diagnose`)    | ✅ honest stub   |
| PyTorch → TensorRT conversion slice     | 🔜 next (Phase 1) |
| Deterministic artifact parsers          | 🔜 next (Phase 2) |
| Grounded LLM diagnoser                  | 🔜 next (Phase 2) |
| CoreML / ONNX-RT / TFLite backends      | 🗺️ planned       |
| MCP server surface                      | 🗺️ planned       |

We are building **one slice end-to-end first** — PyTorch → TensorRT conversion &
accuracy diagnosis — then expanding outward. A working slice proves the universal
version is real; a half-built "everything tool" reads as vaporware. See
[ROADMAP.md](ROADMAP.md).

---

## Planned UX

```console
$ edgedoctor diagnose model.onnx --backend tensorrt

✗ Conversion failed.

  Root cause:  Op `GridSample` (node /decoder/grid_sample) is not supported by
               the TensorRT ONNX parser at opset 13.
  Evidence:    trtexec log line 412 — "No importer registered for op: GridSample"
  Fix:         Re-export from PyTorch with opset ≥ 16, or replace GridSample with
               an affine_grid + bilinear sampling fallback. See docs/ops.md.
```

> Every claim above is traceable to a parsed log line or measured value.
> If edgedoctor doesn't have the evidence, it says so — it does **not** guess.

---

## Failure taxonomy (what edgedoctor diagnoses)

| Class                                   | Backend(s)        | State       |
| --------------------------------------- | ----------------- | ----------- |
| (A) Conversion / op-support failures    | TensorRT (first)  | 🔜 building  |
| (A) CPU fallback (unsupported subgraph) | TensorRT (first)  | 🔜 building  |
| (B) Accuracy divergence (FP32 vs INT8)  | TensorRT (first)  | 🔜 building  |
| Memory / arena overflow                 | —                 | 🗺️ planned   |
| Performance regression (fused engines)  | —                 | ⛔ out of scope (near-term)¹ |

¹ Fused-engine latency attribution depends on undocumented post-fusion
node→source mapping — a technical dead-end near-term, not a scope choice. It may
return later as a research-grade frontier.

---

## Hardware lanes

- **MacBook Pro (Apple Silicon)** — PyTorch training (MPS), CoreML / Neural
  Engine, ONNX Runtime. Primary dev + training machine.
- **ThinkPad P15s, NVIDIA RTX 500 Ada (4 GB), CUDA 13.2** — the TensorRT lane.
  4 GB VRAM ⇒ small edge models (YOLO-nano/small, MobileNet-SSD). CUDA 13.2 is
  very new ⇒ TensorRT version matching is fiddly — which we treat as a
  first-class failure the tool should diagnose (free dogfooding).
- **Raspberry Pi 5** — later (MCU / on-device divergence), not in hand yet.

---

## Quickstart

> ⚠️ Placeholder — the conversion + diagnosis pipeline isn't built yet. For now
> the CLI runs and honestly reports what's implemented.

```console
$ pip install -e .
$ edgedoctor diagnose path/to/model.onnx --backend tensorrt
# → prints an honest "not implemented yet — see ROADMAP.md"
```

---

## Read next

- [VISION.md](VISION.md) — the full ambition + the expansion path.
- [ROADMAP.md](ROADMAP.md) — phases as checkboxes (now / next / later).
- [docs/DESIGN.md](docs/DESIGN.md) — architecture & rationale, including the
  pluggable-backend seam.

## License

[MIT](LICENSE).
