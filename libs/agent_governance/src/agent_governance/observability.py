"""Structured logs, correlated to the turn that produced them.

The decision trail answers "what did governance decide", but a row is only written by a turn
that *reached* `respond`; the failures worth diagnosing at 02:00 are the ones that did not, and
the container log carried no request id. So: one JSON object per line, and every line inside a
turn carries the turn's correlation, conversation and request ids — bound once in the serving
entrypoint, read from a `ContextVar` by the formatter, so no call site has to remember `extra=`.
Never log message text, worker replies or entitlement blocks here; ids are pseudonymous by design.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Optional

#: The ids every log line inside a turn carries. A ContextVar, not a thread-local: the entrypoint
#: streams, and a generator resumed on another thread would lose its correlation mid-turn.
_TURN: ContextVar[Optional[dict]] = ContextVar("turn_context", default=None)


def _turn() -> dict:
    return _TURN.get() or {}


# Attributes `logging` puts on every record. Anything *not* in here was passed
# by a call site as `extra=` and is worth carrying into the JSON.
_STANDARD = frozenset(
    """
    args asctime created exc_info exc_text filename funcName levelname levelno
    lineno module msecs message msg name pathname process processName
    relativeCreated stack_info thread threadName taskName
    """.split()
)


def bind(**fields: Any) -> None:
    """Attach ids to every log line emitted from here on in this context."""
    merged = dict(_turn())
    merged.update({k: v for k, v in fields.items() if v not in (None, "")})
    _TURN.set(merged)


def clear() -> None:
    _TURN.set(None)


def current() -> dict:
    return dict(_turn())


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with the turn's ids folded in."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            # UTC ISO 8601 from the epoch value rather than `formatTime`: the container's local
            # zone is not worth carrying into an aggregator, and `strftime` lacks milliseconds.
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_turn())
        for key, value in record.__dict__.items():
            if key not in _STANDARD and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # `default=str` so an unexpected object in `extra=` degrades to its repr: a formatter
        # that raises takes out the very line that was trying to report a problem.
        return json.dumps(payload, default=str)


def configure(level: Optional[str] = None, fmt: Optional[str] = None) -> None:
    """Install the formatter on the root handler. Safe to call more than once.

    `LOG_FORMAT=plain` opts out, for a workstation where JSON is just harder to
    read. Everything else gets JSON, because everything else is somewhere a
    machine reads the logs.
    """
    chosen_format: str = (fmt if fmt else os.environ.get("LOG_FORMAT") or "json").lower()
    chosen_level: str = (level if level else os.environ.get("LOG_LEVEL") or "INFO").upper()

    root = logging.getLogger()
    root.setLevel(getattr(logging, chosen_level, logging.INFO))
    handler = next((h for h in root.handlers if getattr(h, "_governance_handler", False)), None)
    if handler is None:
        handler = logging.StreamHandler(sys.stdout)
        handler._governance_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    handler.setFormatter(
        JsonFormatter()
        if chosen_format == "json"
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
