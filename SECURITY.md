# Security

## Reporting a vulnerability

Open a [GitHub security advisory](https://github.com/kcitlyn/edgedoctor/security/advisories/new)
rather than a public issue. This is a pre-alpha personal project, so there is no
SLA — but reports will be read and credited.

## Threat model

edgedoctor is not a passive log viewer, and three properties make it
security-relevant:

1. **It ingests attacker-influenced data.** A build log is produced by a
   toolchain acting on a model and a CI configuration. If you diagnose a log from
   a shared runner, a colleague's failed job, or a bug report, you are parsing
   input you did not author.
2. **It re-displays that data verbatim.** Showing your own log lines unaltered is
   the tool's core promise — and reports get pasted into issues, CI output and
   chat.
3. **With `--llm` it transmits that data off the machine** to a third-party API.

## What edgedoctor does about it

### Secret redaction (on by default)

Build logs routinely contain credentials: a private-registry fetch written as
`https://ci:TOKEN@registry/model.onnx`, an `export API_KEY=…` before invoking
`trtexec`, an `Authorization:` header in verbose curl output. Because the report
echoes matched lines, a secret on such a line would be re-displayed — and with
`--json`, stored in CI artifacts.

All output paths mask probable secrets before display or transmission:

| Path | Behaviour |
| --- | --- |
| Terminal report | Masked, and the redaction is **announced** with the secret kinds found |
| `--json` | Masked; `redacted` and `secretsDetected` are declared in the document |
| `--llm` prompt | Masked **before the request is built**, including the artifact filename |

Detected shapes include Anthropic/OpenAI/Google/HuggingFace keys, AWS access key
IDs, GitHub/GitLab/Slack tokens, JWTs, PEM private-key headers, credentials
embedded in URLs, `Authorization` headers, and secret-looking `key=value`
assignments.

A masked value is replaced with a **visible marker** (`[REDACTED:gitlab-token]`),
never blanked — silently altering evidence would break the verbatim promise more
badly than the leak it prevents. Surrounding context is preserved, so
`https://ci:[REDACTED:url-credentials]@registry/x` still tells you *which*
credential to rotate.

Pass `--no-redact` to see raw values when debugging your own private log.

**This is defence in depth, not a guarantee.** Detection is pattern-based, so a
human-chosen password matching no known shape will not be caught — exactly as
[OWASP's Secrets Management guidance](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
warns. Do not treat a redacted report as certified safe to publish. And note that
redaction protects the *report*, not the log: **if edgedoctor reports a secret was
found, rotate that credential and scrub the log.**

### Control-character neutralization (always on)

A log is untrusted data, and a terminal treats some bytes as commands: `ESC[2J`
clears the screen, `OSC` sets the window title, `CR` rewinds the cursor so later
text overwrites earlier text, and a Unicode RTL override can make a line render
differently from its bytes. Passing those through would let a crafted log repaint
the report and misrepresent the verdict — the display side of
[CWE-117](https://cwe.mitre.org/data/definitions/117.html) /
[CWE-116](https://cwe.mitre.org/data/definitions/116.html) ("log forging").

Control and invisible characters are escaped to a visible form (`<ESC>`,
`<U+202E>`) in every rendered field. This is **not** disabled by `--no-redact`:
there is no legitimate reason to let a log drive your terminal. `NO_COLOR` does
not help here either, because these bytes are data in the log rather than styling
edgedoctor chose to emit.

### Structural integrity of the report

The report's structure is its meaning: `error[ED0101]:` means "edgedoctor asserts
this", and `= help (safe to apply):` means "an agent may run this command
unattended". One-line fields are flattened and length-bounded so log content
cannot start a line that mimics either — a fabricated diagnosis or a fabricated
safe-to-run command.

### Other properties

- **No code execution from input.** No `eval`, `exec`, `pickle`, or
  `shell=True` anywhere in the package. Rule files load with `yaml.safe_load`.
- **Rule lookup is confined.** The backend name cannot traverse out of the rules
  directory.
- **Suggested commands are static.** No rule interpolates log-derived data into a
  suggested command, so a log cannot influence a command you might copy-paste.
- **Generated commands are never `machine-applicable`.** An LLM-suggested command
  is always marked for human review, so an agent will not run it unattended.
- **Prompt injection is structurally defeated.** Log content reaches the `--llm`
  prompt, so a crafted log can attempt to jailbreak the model — but every
  synthesized diagnosis must cite fact IDs that exist in the input, checked in
  code *after* the call. A fully-compliant jailbroken model still cannot make
  edgedoctor emit a claim about evidence that isn't there.
- **The API key is never printed.** It is read from `ANTHROPIC_API_KEY` and never
  logged, echoed, or included in an error message.
- **`--llm` cannot fail your CI gate.** A synthesized finding caps the exit code
  at `3`; only curated rules exit `2`.

## Known limitations

- Secret detection is incomplete by construction (see above).
- The live `--llm` API path has not been exercised against the real API; only its
  wire contract has, offline. See `tests/test_llm_live.py`.
- Corpus artifacts are scrubbed of machine-specific paths at capture time, but
  the corpus is not a substitute for reviewing any log you attach to a bug report.
