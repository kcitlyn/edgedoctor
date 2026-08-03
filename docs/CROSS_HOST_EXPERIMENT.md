# The x86-vs-ARM INT8 experiment

The Pi milestone's scientific claim: **INT8 quantization divergence appears on
x86 but not on ARM.** This is the setup for demonstrating it, and the reasons it
is easy to get wrong.

## The claim, precisely

ONNX Runtime's documentation states it directly:

> There is no such issue on other CPU architectures (x64 with VNNI and Arm).
> — [ORT quantization docs](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

**Mechanism.** With U8S8 quantization (uint8 activations, int8 weights) on
AVX2/AVX512 **without VNNI**, ORT uses `VPMADDUBSW`, which accumulates
`uint8 × int8` products into **16-bit lanes**. A large dot product overflows and
is *clamped*, so the result is not merely rounded — it is wrong. VNNI-capable x86
accumulates into 32 bits, and ARM's dot-product instructions behave the same way,
so neither can exhibit it.

**Saturation therefore requires all of:**

1. x86-64, **and**
2. AVX2 or AVX512, **and**
3. **not** VNNI, **and**
4. U8S8 quantization without `reduce_range`

## The trap

An Apple Silicon Mac is **ARM64 — the same side of this issue as the Pi's
Cortex-A76.** A Mac-vs-Pi comparison is ARM vs ARM: it will show no divergence,
which looks like a successful experiment but is a **null setup**. Nothing was
tested.

`scripts/make_cross_host_baseline.py --save` refuses to run on a host that cannot
exhibit the effect, for exactly this reason. Check any machine first:

```console
$ uv run python scripts/host_capability.py
```

It answers `YES` / `NO` / `UNKNOWN`, and treats "couldn't read CPU flags" as
**UNKNOWN rather than NO** — an unverified host must never look like a cleared
one.

## Running it

**Step 1 — on the x86 host (the ThinkPad P15s).** Must report `YES`; if the CPU
has VNNI, this experiment cannot be run on it.

```console
$ uv run python scripts/host_capability.py            # must say YES
$ uv run python scripts/make_cross_host_baseline.py --save
$ git add corpus/cross_host/ && git commit
```

This saves **inputs as well as outputs**. Both hosts must see byte-identical
inputs, or a difference could be the data rather than the architecture.

It captures two quantizations: plain U8S8, and U8S8 with `reduce_range=True`
(ORT's documented workaround, which quantizes weights to 7 bits so accumulations
stay in range). Capturing both is what **isolates the mechanism** — if
`reduce_range` removes the divergence, saturation is confirmed as the cause
rather than merely suspected.

**Step 2 — on the Raspberry Pi.**

```console
$ uv run python scripts/make_cross_host_baseline.py --compare
$ uv run edgedoctor diagnose corpus/cross_host/compare_*.log -b polygraphy
```

The comparison log is ordinary Polygraphy output, so the existing ED02xx rules
diagnose it with no new code.

## Reading the result honestly

- A difference is evidence about **arithmetic**, not about model quality.
- **Tolerance is a policy**, so the useful numbers are the measured divergence
  and the minimum tolerance that would have passed — not a pass/fail verdict.
  (Same discipline as the rest of the corpus; see `rules/polygraphy.yaml`.)
- If divergence **does not** appear, that is a real result worth recording, not a
  failure to hide. Check `host_*.json` first: the most likely explanation is that
  the x86 host had VNNI after all.
- A "no divergence" result from two ARM hosts means **nothing**. Verify both
  `host_*.json` files before drawing any conclusion.

## Status

- Tooling: **done**, tested on ARM (correctly refuses to produce a baseline).
- x86 capture: **not yet run** — needs the ThinkPad.
- ARM capture: **not yet run** — needs the Pi (~Aug 15).
