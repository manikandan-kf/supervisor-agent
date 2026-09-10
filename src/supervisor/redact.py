"""Secret and PII redaction at the persistence boundary — Blueprint §06.

Engineers paste credentials and personal data into queries. Before this module,
query-derived text — the model's paraphrase in `result.reason`, the resolved
context, block reasons, clarification questions — was copied verbatim into
three governed-but-not-secret sinks: the Postgres `decision_trail`, MLflow
trace metadata, and the review queue's `reason` / `query_excerpt`. Each has
different access controls and retention than a secret store, so a pasted token
outlived the conversation in three places at once.

This is a *persistence* redactor, applied where text is about to be written to
a sink a human queries later. It is deliberately not applied to the live
conversation: the worker needs the real content to do its job, and mangling a
user's own messages back at them is a different (and worse) failure. What the
worker actually receives, and what the user gets back, are screened separately
by `output_guard.py` — which decides per category whether to mask, withhold or
escalate, where this module only ever masks.

The shapes live in `sensitive.py`, one catalogue for every boundary. This
module keeps the two-tier split its callers rely on: `pii=False` applies the
**secret tier** only — credentials and secret assignments — for a caller that
wants a token masked without deciding personal-data policy for the text;
`pii=True`, the persistence default, applies the whole catalogue.

"The whole catalogue" rather than "the PII tier", deliberately.
`sensitive.PII_TIER` names a specific and smaller set — the categories
`OUTPUT_PII_MASKING` switches — and using one phrase for both meanings across
two modules that import each other is how a reader comes to believe a control
covers less than it does. The output guard reads the catalogue directly and
applies its own per-category policy.

Every replacement keeps a short label so a reviewer can still see *that* a
secret was present and what kind — which is itself signal (a query carrying
credentials is a different review proposition from one that does not).
"""

from __future__ import annotations

from . import sensitive

_SECRET_TIER = frozenset({sensitive.CREDENTIAL, sensitive.SECRET_ASSIGNMENT})


def redact_text(text: str, *, pii: bool = True) -> tuple[str, int]:
    """Redact known secret/PII shapes. Returns (clean text, replacement count).

    `pii=False` applies only the secret tier — for callers that want a
    credential masked without deciding personal-data policy for the text.
    """
    if not text:
        return text or "", 0
    categories = None if pii else _SECRET_TIER
    cleaned, findings = sensitive.redact(text, categories)
    return cleaned, len(findings)


def redact_structure(value):
    """Redact every string inside a JSON-shaped structure. Returns (value, count).

    Keys are left alone — they are schema, written by this codebase, and
    redacting them would break the queries the audit table exists to answer.
    """
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        total = 0
        out = {}
        for key, item in value.items():
            cleaned, count = redact_structure(item)
            out[key] = cleaned
            total += count
        return out, total
    if isinstance(value, (list, tuple)):
        total = 0
        items = []
        for item in value:
            cleaned, count = redact_structure(item)
            items.append(cleaned)
            total += count
        return (items if isinstance(value, list) else tuple(items)), total
    return value, 0
