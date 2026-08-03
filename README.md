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
  closest prior attempt ("Cellulose", PyTorch→TensorRT diagnostics) has been
  inactive since 2023; an ISSTA'24 study of model-converter failures found ~75%
  of defects are node-conversion failures and roughly a third produce silently
  wrong models — the pain is real, studied, and unsolved.
- No big vendor will close it: the useful version is **cross-vendor** (against
  any one vendor's interest), it's a product/UX problem rather than an
  engineering-culture one, the TAM is small but the annoyance is high, and
  reliable LLM grounding is new and liability-laden for a megacorp. → a durable
  opening for an independent, open-source tool.

---

## 🚧 Status — v0.1, early development

The first slice works: **TensorRT build-log diagnosis, fully offline** (no LLM,
no API key). Honest state of the build:

| Piece                                        | State           |
| -------------------------------------------- | --------------- |
| TensorRT log parser (op failures, build errors) | ✅ works      |
| Polygraphy parser (accuracy divergence)      | ✅ works        |
| Rule knowledge base → root cause + fix       | ✅ works (10 rules, growing) |
| `edgedoctor diagnose` / `parse` CLI, `--json` | ✅ works       |
| PyTorch → ONNX export + verification scripts | ✅ works        |
| Accuracy divergence (FP32 vs INT8)           | ✅ works (real corpus) |
| Validation against real-hardware logs        | 🔜 in progress  |
| Optional grounded LLM synthesis layer (`--llm`) | ✅ works (untested vs live API) |
| ONNX Runtime backend (2nd vendor)            | 🗺️ planned (Aug) |
| CoreML / TFLite / ExecuTorch backends        | 🗺️ planned      |
| MCP server surface                           | 🗺️ planned      |

We are building **one slice end-to-end first** — PyTorch → TensorRT conversion &
accuracy diagnosis — then expanding outward. A working slice proves the universal
version is real; a half-built "everything tool" reads as vaporware. See
[ROADMAP.md](ROADMAP.md).

---

## What it looks like

Real output, current build:

```console
$ edgedoctor diagnose build.log

error[ED0101]: op 'GridSample' is not supported by this TensorRT ONNX parser
  --> build.log:12
   |
  12 | [TRT] No importer registered for op: GridSample. Attempting to import as plugin.
   |
   = note: The TensorRT ONNX parser has no importer for this operator, and no
           plugin with that name was found in the plugin registry. TensorRT
           cannot run unsupported ops on CPU — the build fails outright.
   = help: Re-export with a newer ONNX opset — many ops gained TensorRT support
           at higher opsets
             torch.onnx.export(..., opset_version=17)
   = confidence: high

summary: 3 errors · parsed 5 fact(s) from build.log
```

Every rule code (`ED0101`) maps to a curated cause + fix with reference URLs.
`--json` emits the same diagnosis as a structured document for CI and AI
agents. Exit codes: `0` clean · `2` errors found · `3` warnings only.

> Every claim is traceable to a parsed log line — the evidence block shows your
> own log, verbatim. If edgedoctor doesn't have the evidence, it says so — it
> does **not** guess.

---

## The optional LLM layer (`--llm`)

Everything above works with no API key. `--llm` opts into an extra pass that
tries to explain facts **no rule matched** — and it is built so that it cannot
degrade what the rules already give you:

| Guarantee | How it's enforced |
| --- | --- |
| Cannot contradict a curated rule | It only ever receives the *unmatched* facts, so it can't revisit covered ground |
| Cannot invent evidence | It sees parsed `Facts`, never the raw log. Any diagnosis citing a fact id that wasn't in its input is **dropped, not displayed** |
| Cannot pose as a reviewed rule | Marked `(synthesized)` in the report and `origin: "llm"` in `--json`; capped below `high` confidence; reserved `ED9001` code; its suggestions are never `machine-applicable` |
| Cannot break the tool | Missing SDK, missing key, timeout, malformed response → zero synthesized diagnoses, rules-based report untouched, exit code unchanged |
| Costs nothing when unneeded | If the rules explained every fact, no API call is made at all |

"I don't have enough information" is treated as a correct answer, not a failure.
Design rationale and rejected alternatives:
[docs/adr/0001-llm-synthesis-layer.md](docs/adr/0001-llm-synthesis-layer.md).

```console
$ pip install "edgedoctor[llm]" && export ANTHROPIC_API_KEY=...
$ edgedoctor diagnose build.log --llm
```

---

## Failure taxonomy (what edgedoctor diagnoses)

| Class                                   | Backend(s)        | State       |
| --------------------------------------- | ----------------- | ----------- |
| (A) Conversion / op-support failures    | TensorRT          | ✅ working   |
| (A) Build failures (no kernel, tactics) | TensorRT          | ✅ working   |
| (A) CPU fallback (unsupported subgraph) | ONNX Runtime²     | 🗺️ planned   |
| (B) Accuracy divergence (FP32 vs INT8)  | Polygraphy        | ✅ working   |
| (B) NaN / Inf in outputs                | Polygraphy        | ✅ working   |
| (B) Layer-wise first-diverging tensor   | Polygraphy        | ✅ working   |
| Memory / arena overflow                 | —                 | 🗺️ planned   |
| Performance regression (fused engines)  | —                 | ⛔ out of scope (near-term)¹ |

¹ Fused-engine latency attribution depends on undocumented post-fusion
node→source mapping — a technical dead-end near-term, not a scope choice. It may
return later as a research-grade frontier.

² "CPU fallback" is an ONNX Runtime concept (graph partitioning across
execution providers). A pure TensorRT engine has no per-op CPU fallback — an
unsupported op fails the build outright, which is failure class (A) above.

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

```console
$ git clone https://github.com/kcitlyn/edgedoctor && cd edgedoctor
$ uv sync

# Diagnose a TensorRT build log (works offline, no API key):
$ uv run edgedoctor diagnose path/to/trtexec_build.log

# Or inspect the raw extracted facts first:
$ uv run edgedoctor parse path/to/trtexec_build.log
$ uv run edgedoctor parse path/to/trtexec_build.log --json

# Try it on a bundled example:
$ uv run edgedoctor diagnose tests/fixtures/tensorrt/unsupported_op_trt8.log
```

To produce a log worth diagnosing (on an NVIDIA machine):

```console
$ trtexec --onnx=model.onnx --saveEngine=model.engine --verbose > build.log 2>&1
$ edgedoctor diagnose build.log
```

---

## Read next

- [VISION.md](VISION.md) — the full ambition + the expansion path.
- [ROADMAP.md](ROADMAP.md) — phases as checkboxes (now / next / later).
- [docs/DESIGN.md](docs/DESIGN.md) — architecture & rationale, including the
  pluggable-backend seam.

## License

[MIT](LICENSE).
