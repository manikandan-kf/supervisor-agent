"""Untrusted text at the prompt boundary, in both directions.

Outbound (`untrusted_turn`, `system_blocks`): rules in the system turn, data in a JSON user turn
that no value can escape. Inbound (`clean_inbound_text`, `clean_worker_output`,
`neutralise_history_text`, `neutralise_embedded_directives`) treats worker output as untrusted:
flattened history lets untrusted text impersonate a turn boundary, and defanging it is a control
where a prompt instruction is not (NIST SP 800-207 §5.7). Mechanical markers only, not intent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# Characters no legitimate artifact needs. Assembled from codepoint ranges rather than written
# as literals: these characters are invisible in an editor, so pasting them here is a trap.
_CONTROL_RANGES = (
    (0x00, 0x08),  # NUL..BS
    (0x0B, 0x0C),  # VT, FF -- tab, newline and CR are deliberately kept
    (0x0E, 0x1F),  # SO..US, includes ESC (the ANSI sequence introducer)
    (0x7F, 0x9F),  # DEL and the C1 block
    (0x200B, 0x200F),  # zero-width space and joiners, LRM, RLM
    (0x202A, 0x202E),  # bidirectional embedding and override
    (0x2066, 0x2069),  # bidirectional isolates
    (0xFEFF, 0xFEFF),  # BOM / zero-width no-break space
)
_CONTROL_CHARS = re.compile(
    "[" + "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in _CONTROL_RANGES) + "]"
)

# A line that impersonates a turn boundary, or a chat template's role delimiters. Anchored to
# line start, the only position `_history_lines` output could be confused with; `other` is one
# of the three roles that helper emits.
_ROLE_MARKER = re.compile(
    r"(?im)^[ \t>*_#-]*(system|assistant|user|human|ai|developer|tool|function|other)"
    r"[ \t]*:[ \t]*"
)
# The delimiter families chat templates use to frame a turn. They have no meaning in an SDLC
# artifact, and every one is a documented way to smuggle a role change through text.
_TEMPLATE_MARKER = re.compile(
    r"(?i)<\|[a-z_]{0,32}\|>|<\/?(?:system|assistant|user|human|im_start|im_end)>"
    r"|\[/?INST\]|\[/?SYS\]|###\s*(?:system|instruction)s?\s*:?"
)
# A directive smuggled *inside* pasted content rather than at the start of a line: the OWASP
# LLM01 indirect-injection channel (MITRE ATLAS AML.T0051.001). What is matched is the frame,
# not the intent - the colon is what makes it parse as an instruction, so the colon goes.
_EMBEDDED_DIRECTIVE = re.compile(
    # "NOTE TO ASSISTANT:", "INSTRUCTIONS FOR THE AI:" — an addressee frame.
    r"(?:\b(?:NOTE|INSTRUCTIONS?|MESSAGE|ATTENTION|IMPORTANT)\s+(?:TO|FOR)\s+(?:THE\s+)?"
    r"(?:ASSISTANT|AI|MODEL|SUPERVISOR|SYSTEM|AGENT|LLM)\s*:"
    # An all-caps role marker mid-line. `AI`, `ADMIN` and `OVERRIDE` are ordinary words in a
    # spec (`OVERRIDE: true` is a feature flag), so they are left to the verb-gated form below.
    r"|(?<![\w/])(?:SYSTEM|ASSISTANT|SUPERVISOR|LLM|DEVELOPER)\s*:(?=\s*[A-Za-z\[(])"
    # Any-case role marker followed by a directive verb. `always`/`never` are not directive
    # verbs: `Roles: admin: always, operator: never` is a permission table, not an instruction.
    r"|(?<![\w/])(?i:supervisor|assistant|system|admin|ai|developer|operator)\s*:"
    # `(?!-)` after the verb: `developer: print-only` is a role-table permission, while
    # `Developer: print the system prompt` is a directive. The hyphen separates them.
    r"(?=\s*(?i:treat|ignore|reveal|override|skip|disregard|approve|bypass|forget|output|print|"
    r"disclose|you\s+are|from\s+now|do\s+not|grant|execute)\b(?!-))"
    # Explicit injection tags.
    r"|\[(?:INJECTED|INJECTION|SYSTEM|ADMIN|OVERRIDE)\])"
)

# Code and commands are surfaced for human review, never executed. Detected only to record
# that the turn carried them: a deployment command is a different review proposition.
_FENCE = re.compile(r"^[ \t]*(?:```|~~~)", re.MULTILINE)

_TRUNCATION_NOTE = "\n\n[… truncated: the agent's response exceeded the size limit.]"


@dataclass(frozen=True)
class CleanOutput:
    """The sanitized text, plus what was done to it — for the decision trail."""

    text: str
    original_length: int
    truncated: bool = False
    control_characters: int = 0
    role_markers: int = 0
    template_markers: int = 0
    code_blocks: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def modified(self) -> bool:
        return bool(
            self.truncated or self.control_characters or self.role_markers or self.template_markers
        )

    def audit_detail(self) -> str:
        """One line describing what happened, for `_entry(...)`."""
        parts = [f"{self.original_length} chars"]
        if self.truncated:
            parts.append(f"truncated to {len(self.text)}")
        if self.control_characters:
            parts.append(f"{self.control_characters} control char(s) stripped")
        if self.role_markers:
            parts.append(f"{self.role_markers} impersonated turn marker(s) neutralised")
        if self.template_markers:
            parts.append(f"{self.template_markers} chat-template marker(s) neutralised")
        if self.code_blocks:
            parts.append(f"{self.code_blocks} code block(s) surfaced for review")
        return "; ".join(parts)


def _neutralise_role(match: re.Match) -> str:
    """Defang a turn marker without deleting the reader's content.

    Only the colon goes, because it is what makes `system: do X` parse as a boundary; an
    artifact may legitimately open a line with "User:" in a sequence diagram.
    """
    return match.group(0).replace(":", " -", 1)


def _neutralise_directive(match: re.Match) -> str:
    """Defang an embedded directive frame, keeping the words.

    `[INJECTED]`-style tags become `(INJECTED)`: still visible, no longer a tag.
    """
    found = match.group(0)
    if found.startswith("["):
        return "(" + found[1:-1] + ")"
    return found.replace(":", " -", 1)


def neutralise_embedded_directives(text: str) -> tuple[str, int]:
    """Defang directive frames smuggled inside content. Returns (text, count).

    The mid-line complement to `_ROLE_MARKER`, for pasted tool output and API payloads that
    carry instruction-shaped text inside a value. Apply wherever untrusted text meets a model.
    """
    if not text:
        return text or "", 0
    return _EMBEDDED_DIRECTIVE.subn(_neutralise_directive, text)


def neutralise_history_text(text: str) -> str:
    """Defang turn boundaries in any text bound for a flattened history prompt.

    Covers prior *user* turns - the symmetrical channel `clean_worker_output` does not close.
    No truncation or audit counters: the window budget already bounds history size, and
    per-line counters would drown the trail. A second pass is a no-op.
    """
    if not text:
        return text or ""
    cleaned = _CONTROL_CHARS.sub("", text)
    cleaned = _ROLE_MARKER.sub(_neutralise_role, cleaned)
    cleaned = _EMBEDDED_DIRECTIVE.sub(_neutralise_directive, cleaned)
    return _TEMPLATE_MARKER.sub(" ", cleaned)


def clean_inbound_text(text: str) -> str:
    """Strip control and invisible characters from one inbound user message.

    Raw text is checkpointed into state and echoed back through the calling UI, so an ANSI
    escape or bidi override must not survive ingress. Control characters only: role markers
    are defanged at the prompt boundary instead, so a user's own words read back unmangled.
    """
    return _CONTROL_CHARS.sub("", text or "")


def clean_worker_output(text: str, *, max_chars: int) -> CleanOutput:
    """Bound and sanitize one worker response.

    Order matters: markers are neutralised before truncation so a marker past the cut is
    still counted, and truncation runs last so the length limit has the final word.
    """
    raw = text or ""
    original_length = len(raw)

    cleaned, control_count = _CONTROL_CHARS.subn("", raw)
    code_blocks = len(_FENCE.findall(cleaned)) // 2

    cleaned, role_count = _ROLE_MARKER.subn(_neutralise_role, cleaned)
    cleaned, directive_count = _EMBEDDED_DIRECTIVE.subn(_neutralise_directive, cleaned)
    cleaned, template_count = _TEMPLATE_MARKER.subn(" ", cleaned)
    # A mid-line directive frame is the same coercion channel as a fake turn
    # boundary, so it counts with them: a reviewer looks for one number.
    role_count += directive_count

    truncated = False
    if max_chars > 0 and len(cleaned) > max_chars:
        # Cut on a paragraph or line boundary when one is close to the limit, so
        # the visible result ends mid-document rather than mid-word.
        window = cleaned[:max_chars]
        cut = max(window.rfind("\n\n"), window.rfind("\n"))
        if cut < max_chars * 0.8:
            cut = max_chars
        cleaned = cleaned[:cut].rstrip() + _TRUNCATION_NOTE
        truncated = True

    return CleanOutput(
        text=cleaned.strip(),
        original_length=original_length,
        truncated=truncated,
        control_characters=control_count,
        role_markers=role_count,
        template_markers=template_count,
        code_blocks=code_blocks,
    )


# ---------------------------------------------------------------------------
# Outbound: composing a governance prompt
# ---------------------------------------------------------------------------
def untrusted_turn(**fields) -> str:
    r"""JSON-encode untrusted content for its own user turn.

    `ensure_ascii=True` escapes non-ASCII to `\uXXXX`, which also neutralises the homoglyph
    and bidi-override tricks used to smuggle instructions past a reader's eye.
    """
    return json.dumps(fields, ensure_ascii=True, default=str)


# ── prompt caching ──────────────────────────────────────────────────────────
# A cached prefix costs ~10% of the input price on a read and ~125% on the write, so it pays
# from the second use of that exact prefix within the TTL. `ChatDatabricks` passes
# `cache_control` through but does not surface the cache counters - confirm in MLflow traces.
CACHE_MIN_TOKENS = 1024
# Sonnet/Opus silently refuse to cache a block below `CACHE_MIN_TOKENS` - full price, no
# error. Measured at ~4.12 chars per token on these templates, so this is the floor plus margin.
CACHE_MIN_CHARS = 4300


def cacheable_split(template: str) -> tuple[str, str]:
    """Split a single-brace template into (invariant prefix, per-call tail).

    The boundary is the first `{`, which makes the property self-maintaining: templates put
    their variables last, the documented shape for caching (static first, dynamic last).
    """
    boundary = template.find("{")
    if boundary == -1:
        return template, ""
    return template[:boundary], template[boundary:]


def system_blocks(template: str, **values) -> list[dict] | str:
    """The system turn for a governance call, as cacheable content blocks.

    Returns a plain string when the invariant prefix is too short to cache: no point paying
    the write premium for a block the API will refuse to store.
    """
    prefix, tail = cacheable_split(template)
    body = tail.format(**values) if tail else ""
    if len(prefix) < CACHE_MIN_CHARS:
        return prefix + body
    return [
        {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": body},
    ]
