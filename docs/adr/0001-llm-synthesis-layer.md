# ADR 0001 — The grounded LLM synthesis layer

Status: accepted · 2026-08-03

Phase 2's last piece. The rules engine already produces useful diagnoses with no
API key; this layer exists only for the facts a rule *didn't* cover. Six choices
had real alternatives, so they're recorded with their trade-offs.

## 1. Opt-in `--llm` flag, not auto-enable on a present API key

| | Pros | Cons |
|---|---|---|
| **Opt-in** (chosen) | No surprise network calls or spend; deterministic by default; the rules-only path stays the headline | Users must discover the flag |
| Auto-enable | Seamless | Silently makes output non-deterministic and costs money; and `diagnose`'s own help text says "Works fully offline — no LLM, no API key" |

Decisive: auto-enabling would make documented behaviour false, and a diagnostic
tool whose output silently varies between runs is worth less than one that
doesn't.

## 2. The LLM sees only UNMATCHED facts

| | Pros | Cons |
|---|---|---|
| **Unmatched only** (chosen) | Strictly additive — cannot contradict or override a curated rule; cheap (usually zero facts → zero calls) | Misses cross-cutting synthesis over the whole log |
| All facts | Richer prose | A rule and the LLM can disagree about the same fact, and there is no principled way to adjudicate. Curated knowledge would lose to generated text |

`diagnoser.py`'s docstring already scoped it this way. The rejected second half
of that note — "render matched diagnoses into richer prose" — is deliberately
NOT built: paraphrasing curated `cause`/`help` text risks corrupting reviewed
content for style.

## 3. Groundedness is a hard post-validation gate, not just a prompt instruction

| | Pros | Cons |
|---|---|---|
| **Prompt + reject invalid citations** (chosen) | A hallucinated fact id can never reach the user | Discards otherwise-plausible output |
| Prompt only | Keeps more output | Ungrounded claims get displayed, breaking the tool's central promise |

Any synthesized diagnosis citing a fact id absent from the input is **dropped
entirely**, and a diagnosis citing nothing at all is dropped too. Asking a model
to behave is not a guarantee; checking is.

## 4. Synthesized diagnoses are visibly labelled and capped at `medium` confidence

A user must be able to tell a curated, reviewed diagnosis from a generated one.
`Diagnosis` gains `origin: "rules" | "llm"` (defaulting to `"rules"`, so every
existing path and snapshot is unaffected), the report prints a `synthesized`
marker, and confidence is clamped — an unreviewed synthesis may never claim the
`high` confidence a curated rule earns.

## 5. Tests use an injected fake client, not recorded HTTP cassettes

The plan called for pytest-recording cassettes. Chosen differently:

| | Pros | Cons |
|---|---|---|
| **Injected fake client** (chosen) | Zero new deps; hermetic CI; needs no API key and no live spend to write; tests the logic that actually matters (grounding gate, degradation, prompt construction) | Doesn't verify real wire format |
| Cassettes | Pins the real response shape | Recording needs a live key and real spend; and an unrecorded `pytest-recording` dep is dead weight |

The failure modes worth testing are ours, not the SDK's. Verifying the live wire
format is a follow-up worth doing once, against a real key.

## 6. A separate `SynthesizedDiagnosis` wire schema

The model is handed a narrow schema containing only fields it should decide —
not `origin`, `code`, or anything else the tool controls. Structural, rather
than instructional, prevention of a model setting its own trust label.

## Non-negotiable: degradation

Any failure — missing SDK, missing key, timeout, malformed response, ungrounded
output — yields zero synthesized diagnoses and leaves the rules-based result
untouched. The LLM layer can only ever add; never remove or break.
