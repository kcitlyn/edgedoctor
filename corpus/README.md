# Failure corpus

Real artifacts from real conversion runs — the raw material the parsers are
built and tested against, and (later) the seed of the reference benchmark.

## Layout

```
corpus/
  tensorrt/          trtexec / Polygraphy logs from the ThinkPad (RTX 500 Ada)
  onnxruntime/       two families, both generated on the Mac:
                       int8_*/fp32_*/nan_*/shape_*  Polygraphy accuracy-
                         divergence logs (make_divergence_corpus.py)
                       ort_*                        ORT verbose session logs
                         showing EP placement / CPU fallback
                         (make_ort_corpus.py) — the same parser handles the
                         Pi 5's logs, which is why it's built here first
  <backend>/         one directory per backend as they come online
```

## Rules

1. **Real artifacts only.** Every file here was produced by an actual run of a
   vendor tool. Synthetic test inputs live in `tests/fixtures/`, never here.
   *Reproducibly generated* is still real: the `onnxruntime/` logs are genuine
   Polygraphy output from a real `polygraphy run`, driven by a committed script
   (`scripts/make_divergence_corpus.py`) so the exact bytes can be regenerated.
   The distinction that matters is the tool, not who typed the command — a
   hand-forged log that no tool ever emitted would belong in `tests/fixtures/`.
2. **One `.meta.md` sidecar per artifact** recording ground truth:
   what command produced it, on what hardware/versions, what actually went
   wrong (the human-verified root cause), and what fixed it. The sidecar is the
   label; the log is the sample.
3. **Failures are more valuable than successes.** Save every broken run —
   version mismatches, missing libs, unsupported ops, accuracy divergence.
   Keep a couple of clean success logs too (parsers must not hallucinate
   failures where there are none).
4. Scrub nothing except genuine secrets (there normally are none in these
   logs). Line numbers matter — parsers cite them.

## Sidecar template

```markdown
# <artifact filename>
- command:  <exact command line>
- machine:  <e.g. ThinkPad P15s, RTX 500 Ada, driver XXX, CUDA 13.2>
- versions: <TensorRT X.Y.Z, ONNX opset N, ...>
- outcome:  <failed | succeeded-with-warnings | succeeded>
- root cause (human-verified): <one paragraph>
- fix that worked: <what actually resolved it>
```
