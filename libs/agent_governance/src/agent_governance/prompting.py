"""Prompt hygiene shared by every governance call: untrusted content in its own
JSON turn, and a cacheable system prefix.

Both come from Anthropic's prompt-injection guidance. The rules belong in the
system turn and the data in a user turn, so the model has a structural signal
for which is which; and the data is JSON rather than tagged text, because XML
delimiters are guessable and escapable — a user who types `</user_query>` closes
the tag — while JSON escaping makes it impossible for a value to terminate its
own field.
"""

from __future__ import annotations

import json


def untrusted_turn(**fields) -> str:
    r"""JSON-encode untrusted content for its own user turn.

    `ensure_ascii=True` escapes non-ASCII to `\uXXXX`, which also neutralises the
    homoglyph and bidi-override tricks used to smuggle instructions past a
    reader's eye.
    """
    return json.dumps(fields, ensure_ascii=True, default=str)


# ── prompt caching ──────────────────────────────────────────────────────────
#
# Anthropic charges a cached prefix at ~10% of the input price on a read and
# ~125% on the write, so caching a system prompt pays for itself from the second
# use of that exact prefix within the TTL. A screen that asks the same rules
# once *per candidate agent* reuses the prefix inside a single turn.
#
# Verified against `databricks-claude-sonnet-4-5`: the workspace creates the
# cache (`cache_creation_input_tokens`) and reads it back. One caveat:
# `ChatDatabricks` passes `cache_control` through but does not surface the two
# cache counters in `response_metadata` — confirm caching in MLflow traces, not
# in LangChain usage numbers.
CACHE_MIN_TOKENS = 1024
# Sonnet/Opus will not cache a block below `CACHE_MIN_TOKENS`, and the failure
# is silent — full price, no error. Measured at ~4.12 chars per token on these
# templates, so this is the character floor with a margin.
CACHE_MIN_CHARS = 4300


def cacheable_split(template: str) -> tuple[str, str]:
    """Split a single-brace template into (invariant prefix, per-call tail).

    The boundary is the first `{`, which makes the property self-maintaining:
    whatever an author writes *before* the first variable is the cached prefix.
    Templates therefore put their variables last — the documented shape for
    caching (static first, dynamic last).
    """
    boundary = template.find("{")
    if boundary == -1:
        return template, ""
    return template[:boundary], template[boundary:]


def system_blocks(template: str, **values) -> list[dict] | str:
    """The system turn for a governance call, as cacheable content blocks.

    Returns a plain string when the invariant prefix is too short to cache —
    there is no point paying the write premium for a block the API will refuse
    to store. The tail is interpolated and sent uncached on every call.
    """
    prefix, tail = cacheable_split(template)
    body = tail.format(**values) if tail else ""
    if len(prefix) < CACHE_MIN_CHARS:
        return prefix + body
    return [
        {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": body},
    ]
