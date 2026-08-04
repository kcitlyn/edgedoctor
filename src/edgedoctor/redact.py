"""Secret redaction and control-character neutralization for untrusted log text.

WHY THIS MODULE EXISTS
edgedoctor's central promise is to show a user their OWN log lines, verbatim.
That promise is what makes the tool trustworthy — and it is also an
exfiltration channel, because the logs it ingests are not clean:

  1. BUILD LOGS CONTAIN SECRETS. A CI job that fetches a model from a private
     registry, or exports a token before invoking trtexec, puts that credential
     in the log. If the secret happens to sit on a line the parser matched, the
     report echoes it — and reports get pasted into GitHub issues, CI output and
     chat. OWASP's Secrets Management guidance is explicit that secrets must
     "never be logged... implement either an encryption or masking approach",
     and that a tool re-displaying such a log should mask rather than echo.
     With `--llm`, the same text is transmitted to a third-party API.

  2. LOGS CONTAIN CONTROL CHARACTERS. A log is attacker-influenced data. Raw
     ANSI escapes passed to a terminal can clear the screen (ESC[2J), reposition
     the cursor, set the window title (OSC), or hide text — letting a crafted
     log rewrite what the user appears to be reading. This is the display side
     of CWE-117 / CWE-116 ("log forging"): the report's structure is its
     meaning, so content that can repaint the terminal can lie about the
     verdict. Note NO_COLOR does not help: these bytes are DATA in the log, not
     styling edgedoctor chose to emit.

DESIGN CONSTRAINTS THIS MODULE RESPECTS

Redaction must not silently destroy evidence. An excerpt is the tool's proof, so
a masked value is replaced by a VISIBLE, labelled marker — never deleted, never
blanked. The reader can always see that something was removed and what kind of
thing it was, which keeps the report honest about its own alterations.

LENGTH AND LINE COUNT ARE NOT PRESERVED, deliberately, and that is safe here:
redaction happens at RENDER time, not at parse time. Facts keep the original
excerpt and the original `file:line` citation, so every number the tool reports
still points at the real line in the real file. Nothing about traceability
depends on the rendered string's length.

Detection is pattern-based and therefore INCOMPLETE by construction — a
human-chosen password matching no pattern will slip through, exactly as OWASP
warns. This is defence in depth, not a guarantee, and the docstrings say so
rather than implying safety it cannot deliver.
"""

from __future__ import annotations

import re

#: Marker written in place of a redacted value. Deliberately conspicuous: the
#: reader must be able to tell that edgedoctor altered the line, and that the
#: alteration was a redaction rather than a parsing artefact.
_MARK = "[REDACTED:{kind}]"

#: (kind, pattern) pairs. Ordered longest/most-specific first, because an
#: earlier match consumes the text a later, looser pattern would have caught.
#:
#: Each pattern is BOUNDED — `{n,m}` rather than `+` — for the same reason every
#: parser signature is: an unbounded quantifier before a literal turns matching
#: quadratic, and this runs over untrusted input of arbitrary length.
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # ── Provider-specific formats: highest confidence, near-zero false positives.
    ("anthropic-key", re.compile(r"sk-ant-(?:api\d\d-)?[A-Za-z0-9_\-]{16,120}")),
    ("openai-key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{20,80}\b")),
    ("aws-access-key-id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github-token", re.compile(
        r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,80}\b"
        r"|\bgithub_pat_[A-Za-z0-9_]{20,90}\b"
    )),
    ("gitlab-token", re.compile(r"\bglpat-[A-Za-z0-9_\-]{16,40}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,80}\b")),
    ("google-api-key", re.compile(r"\bAIza[A-Za-z0-9_\-]{35}\b")),
    ("hf-token", re.compile(r"\bhf_[A-Za-z0-9]{30,50}\b")),
    # A JWT: three base64url segments. Common in Authorization headers.
    ("jwt", re.compile(
        r"\beyJ[A-Za-z0-9_\-]{8,600}\.[A-Za-z0-9_\-]{4,600}\.[A-Za-z0-9_\-]{4,600}"
    )),
    # ── Private keys: the PEM header alone is enough to act on.
    ("private-key", re.compile(
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY(?: BLOCK)?-----"
    )),
    # ── Credentials embedded in a URL. Very common in build logs, because a
    # fetch of a private artifact is often written this way.
    ("url-credentials", re.compile(
        r"(?P<scheme>\b[a-zA-Z][a-zA-Z0-9+.\-]{1,20}://)"
        r"(?P<user>[^\s:/@]{1,64}):(?P<secret>[^\s/@]{1,200})@"
    )),
    # ── Authorization headers. Placed BEFORE the generic key=value rule so
    # the specific match wins: 'auth' appears in the generic key list, which
    # would otherwise mask the literal word "Authorization" and read as if
    # the header NAME were the secret.
    ("authorization-header", re.compile(
        r"(?i)\bauthorization\s*:\s*(?P<scheme>bearer|basic|token)\s+"
        r"(?P<secret>[A-Za-z0-9_\-.=+/]{8,600})"
    )),

    # ── Generic key=value assignments. Lower confidence, so the KEY NAME must
    # look secret-ish; the value alone is never enough to judge.
    ("assigned-secret", re.compile(
        r"(?i)\b(?P<key>"
        r"(?:[a-z0-9_\-]{0,24})"
        # NOTE: bare "auth" is deliberately NOT in this list. It matched the
        # literal word "Authorization" in a header, masking the header NAME and
        # reading as though the field name were the secret. The
        # authorization-header pattern above handles that case precisely.
        r"(?:secret|passwd|password|token|api[_\-]?key|access[_\-]?key"
        r"|credential|private[_\-]?key|session[_\-]?id)"
        r"(?:[a-z0-9_\-]{0,24})"
        r")\s*[=:]\s*(?P<quote>[\"']?)(?P<secret>[^\s\"',;]{4,200})(?P=quote)"
    )),
]

#: Control characters that must never reach a terminal from untrusted text.
#: TAB (\x09) is deliberately allowed — it is ordinary log formatting and cannot
#: reposition the cursor or repaint the screen.
#:
#: Covers C0 controls, DEL, and the C1 range (0x80-0x9F), which includes the
#: 8-bit forms of CSI/OSC that some terminals honour.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")

#: Unicode characters that are invisible or reorder text visually. A
#: right-to-left override can make a rendered line read differently from its
#: bytes, which is a way to lie about evidence without any ANSI escape at all.
_BIDI_AND_INVISIBLE = re.compile(
    "["
    "\u200b-\u200f"   # zero-width space/joiners, LRM/RLM
    "\u202a-\u202e"   # embedding / override (RLO is the dangerous one)
    "\u2060-\u2064"   # word joiner, invisible operators
    "\u2066-\u2069"   # isolates
    "\ufeff"          # BOM / zero-width no-break space
    "]"
)


def redact_secrets(text: str) -> tuple[str, list[str]]:
    """Mask probable secrets in `text`.

    Returns the masked text and the list of secret kinds found, so a caller can
    tell the user that redaction happened rather than silently altering their
    evidence.

    Detection is pattern-based and therefore INCOMPLETE: a password that matches
    no known shape will not be caught. Treat this as defence in depth against
    the common cases (provider tokens, URL credentials, obvious assignments),
    not as a guarantee that a report is safe to publish.
    """
    found: list[str] = []
    for kind, pattern in _SECRET_PATTERNS:
        def _replace(match: re.Match[str], _kind: str = kind) -> str:
            groups = match.groupdict()
            marker = _MARK.format(kind=_kind)
            # When a pattern captures surrounding context (a scheme, a key
            # name), keep that context and mask only the secret itself. The
            # context is often what makes the finding intelligible — "the
            # password in this git URL" is more useful than a bare marker.
            if "secret" in groups and groups["secret"] is not None:
                prefix = match.group(0)[: match.start("secret") - match.start(0)]
                suffix = match.group(0)[match.end("secret") - match.start(0):]
                return f"{prefix}{marker}{suffix}"
            return marker

        new_text, count = pattern.subn(_replace, text)
        if count:
            found.append(kind)
            text = new_text
    return text, found


def strip_control_chars(text: str) -> str:
    """Neutralize control and invisible characters from untrusted log text.

    A log is attacker-influenced data, and a terminal treats some of its bytes
    as commands: ESC[2J clears the screen, OSC sets the window title, CR
    rewinds the cursor so later text overwrites earlier text. Passing those
    through would let a crafted log repaint the report and misrepresent the
    verdict — the display-side form of log forging (CWE-117 / CWE-116).

    Replaced with a visible escape rather than deleted, so the reader can see
    that the line contained something unusual instead of silently reading a
    doctored version.
    """
    def _escape(match: re.Match[str]) -> str:
        char = match.group(0)
        # \x1b -> <ESC>, \x07 -> <0x07>: recognisable without being executable.
        return "<ESC>" if char == "\x1b" else f"<0x{ord(char):02x}>"

    text = _CONTROL_CHARS.sub(_escape, text)
    return _BIDI_AND_INVISIBLE.sub(lambda m: f"<U+{ord(m.group(0)):04X}>", text)


def sanitize_for_display(text: str, *, redact: bool = True) -> tuple[str, list[str]]:
    """Make untrusted log text safe to print, returning any secret kinds found.

    Control characters are ALWAYS neutralized — that is a terminal-safety
    property with no legitimate reason to opt out. Secret redaction is
    switchable because a user debugging their own local log may genuinely want
    the raw value, and forcing masking would break the verbatim-evidence promise
    for the common, private case.

    Order matters: redact first, then strip controls. A secret containing a
    control character would otherwise be broken up by escaping and could slip
    past the patterns.
    """
    found: list[str] = []
    if redact:
        text, found = redact_secrets(text)
    return strip_control_chars(text), found
