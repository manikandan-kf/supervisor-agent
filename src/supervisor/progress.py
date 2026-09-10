"""Progress events — the task plan the calling UI renders while a turn runs.

Each governance stage announces itself as it starts and reports how it
resolved. A UI turns these into a live checklist:

    ✓ Checking your access
    ✓ Screening the request
    ✓ Resolving the details
    ✓ Drafting the answer

Emitted through LangGraph's custom stream channel, so they arrive *while* the
graph runs rather than being reconstructed from the finished result:

    graph.stream(..., stream_mode=["custom", "messages"])

A node calls `step(...)`; `get_stream_writer()` returns a no-op when nothing is
streaming, so the same node code works unchanged under `invoke()`.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Stage -> the label a user should see. Deliberately plain language: this is
# shown in the chat, not in an operations console.
LABELS: dict[str, str] = {
    "rbac_gate": "Checking your access",
    "guardrails": "Screening the request",
    "route": "Resolving the details",
    "dispatch": "Drafting the answer",
    "respond": "Finishing up",
}


def step(stage: str, status: str, detail: str = "", **extra) -> None:
    """Emit one progress event.

    `status` is one of:
        started   — the stage is running
        done      — completed normally
        blocked   — the stage refused the request
        clarify   — the stage needs more from the user
        error     — the stage failed
    """
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:
        # No active stream (a plain invoke, or a test). Progress is an
        # additive channel, so its absence must never affect the answer.
        return

    if writer is None:
        return

    try:
        writer(
            {
                "channel": "progress",
                "stage": stage,
                "label": LABELS.get(stage, stage.replace("_", " ").capitalize()),
                "status": status,
                "detail": detail,
                **extra,
            }
        )
    except Exception:
        logger.debug("progress emit failed for %s/%s", stage, status, exc_info=True)


def sources(items: list[dict]) -> None:
    """Emit the grounding sources the calling UI shows under the answer.

    Each item is `{"title": ..., "origin": ...}`. Real worker agents supply
    these from their retrieval step; the simulated worker declares what it
    would have consulted, clearly labelled.
    """
    if not items:
        return
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:
        return

    if writer is None:
        return

    try:
        writer({"channel": "sources", "items": items})
    except Exception:
        logger.debug("sources emit failed", exc_info=True)
