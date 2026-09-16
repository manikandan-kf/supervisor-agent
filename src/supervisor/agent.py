"""Databricks ResponsesAgent entrypoint (MLflow models-from-code).

The Governance Front Door forwards validated requests here with `custom_inputs`: `agent_id`
(the target worker), `permitted_agents` / `approvable_agents` (trusted because computed
server-side from a validated token — never accepted from a browser, §2.7), `user_role`,
`conversation_id`, `user_id` and `resume`. Identity reaches the graph as runtime context,
not state (see `state.py`); only the conversation is checkpointed. `predict_stream` is the
real path; `predict` collects the same stream for callers that cannot stream.
"""

from __future__ import annotations

import logging
import uuid
from typing import Generator

import mlflow
from agent_governance import observability
from agent_governance.locking import thread_lock
from agent_governance.output_guard import OutputGuard, StreamGuard
from agent_governance.sanitize import clean_inbound_text
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from supervisor.graph import build_graph
from supervisor.memory import lock_connection_source
from supervisor.messages import BUSY_MESSAGE
from supervisor.settings import Settings
from supervisor.state import SupervisorContext

logger = logging.getLogger(__name__)

# Structured logging carrying the turn's correlation ids. Configured here, not in
# `build_services`: this module is the one thing guaranteed to import in the container.
observability.configure()

mlflow.langchain.autolog()

_DURABILITY_MODES = ("sync", "async", "exit")


def _to_lc_messages(items) -> list[dict]:
    messages = []
    for item in items or []:
        data = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        role = data.get("role")
        content = data.get("content")
        if role not in ("user", "assistant") or content is None:
            continue
        if isinstance(content, list):
            content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        # Ingress hygiene (guardrail layer 1): control, zero-width and bidi characters are
        # stripped before the text enters checkpointed graph state.
        messages.append({"role": role, "content": clean_inbound_text(str(content))})
    return messages


class SupervisorAgent(ResponsesAgent):
    def __init__(self, graph=None, output_guard=None):
        if graph is None:
            # Built from one `Services` so the stream guard screens tokens with the *same*
            # governed policy the dispatch node applies to the finished reply.
            from supervisor.services import build_services

            services = build_services()
            graph = build_graph(services)
            output_guard = output_guard or services.output_guard
        self._graph = graph
        # A fake graph in a test gets a bare guard: the shipped defaults,
        # which is the production masking behaviour without configuration.
        self._output_guard = output_guard or OutputGuard()
        # "sync" holds each stage until its checkpoint is written — LangGraph's recommendation
        # for HITL flows. An unrecognised value falls back to it rather than raising.
        settings = Settings()
        # Refuse to serve a deployed environment whose safety-critical settings are disabled;
        # dev logs at ERROR instead — see Settings.enforce.
        settings.enforce()
        durability = settings.durability
        if durability not in _DURABILITY_MODES:
            durability = "sync"
        self._durability = durability
        self._settings = settings
        # For the busy-refusal audit row below, on the same pooled connection
        # source as the graph's own sink — no extra entitlement, no extra pool.
        self._audit = None

    # ── one execution per thread (§4.4) ─────────────────────────────────────

    def _lock(self, conversation_id: str):
        """The execution lock for this turn's conversation.

        Several replicas and worker processes with no conversation affinity means nothing
        else stops two turns on one thread racing each other's checkpoint. See `locking`.
        """
        return thread_lock(
            conversation_id,
            connection_source=lock_connection_source(),
            enabled=self._settings.thread_lock_enabled,
            timeout=self._settings.thread_lock_timeout_seconds,
            poll=self._settings.thread_lock_poll_seconds,
            namespace="supervisor",
        )

    def _busy(self, conversation_id: str) -> dict:
        """Custom outputs for a turn refused because its thread was already busy.

        A minimal audit row *is* written: a refusal is a governance outcome, and a table
        without busy refusals under-reports the concurrency a capacity investigation asks
        about. Best-effort — a refusal must not fail because the sink blinked.
        """
        try:
            mlflow.update_current_trace(
                session_id=conversation_id or None, tags={"outcome": "busy"}
            )
        except Exception:
            # Expected when there is no active trace (offline test, no autolog). Debug rather
            # than swallowed, so a *real* tagging failure in the endpoint is findable.
            logger.debug("busy-turn trace tagging skipped", exc_info=True)
        try:
            if self._audit is None:
                from supervisor.services import build_audit_logger

                self._audit = build_audit_logger(self._settings)
            self._audit.log(
                {
                    "conversation_id": conversation_id,
                    "outcome": "busy",
                    "decision_trail": [
                        {
                            "stage": "thread_lock",
                            "decision": "refused",
                            "detail": "another execution already holds this conversation "
                            "— turn refused, not run (§4.4)",
                        }
                    ],
                }
            )
        except Exception:
            logger.warning("busy-refusal audit row failed", exc_info=True)
        return {
            "conversation_id": conversation_id,
            "request_id": "",
            "outcome": "busy",
            "routed_agent_id": "",
            "routed_agent_name": "",
            "pending_clarification": None,
            "pending_approval": None,
            "sources": [],
        }

    # ── request -> graph inputs ─────────────────────────────────────────────

    def _prepare(self, request: ResponsesAgentRequest):
        custom = request.custom_inputs or {}
        conversation_id = custom.get("conversation_id") or str(uuid.uuid4())
        context = SupervisorContext.from_custom_inputs(custom)
        # Every log line from here to the end of this turn carries these ids,
        # bound once and read by the formatter. Nothing here is message content.
        observability.clear()
        observability.bind(
            conversation_id=conversation_id,
            correlation_id=context.correlation_id,
            agent_id=context.requested_agent_id,
            environment=context.environment,
        )
        config = {
            "configurable": {"thread_id": conversation_id},
            # LangGraph's default (10007 in 1.2) is no bound for a six-node acyclic graph; set
            # explicitly so a future looping edge fails fast.
            "recursion_limit": self._settings.graph_recursion_limit,
        }

        resume = custom.get("resume")
        if resume is not None:
            # `Command(resume=...)` resumes *inside* the interrupted node instead of replaying
            # the turn, so the worker is not called twice.
            from langgraph.types import Command

            return Command(resume=resume), config, context, conversation_id

        return (
            {"messages": _to_lc_messages(request.input), "conversation_id": conversation_id},
            config,
            context,
            conversation_id,
        )

    def _custom_outputs(self, state: dict, conversation_id: str, sources: list) -> dict:
        return {
            "conversation_id": conversation_id,
            "request_id": state.get("request_id", ""),
            "outcome": state.get("outcome", ""),
            "routed_agent_id": state.get("target_agent_id", ""),
            "routed_agent_name": state.get("routed_agent_name", ""),
            "pending_clarification": state.get("pending_clarification"),
            "pending_approval": state.get("pending_approval"),
            "sources": sources or state.get("sources") or [],
        }

    # ── non-streaming ───────────────────────────────────────────────────────

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        payload, config, context, conversation_id = self._prepare(request)

        with self._lock(conversation_id) as acquired:
            if not acquired:
                return ResponsesAgentResponse(
                    output=[self.create_text_output_item(text=BUSY_MESSAGE, id=str(uuid.uuid4()))],
                    custom_outputs=self._busy(conversation_id),
                )

            result = self._graph.invoke(
                payload, config=config, context=context, durability=self._durability
            )

        item = self.create_text_output_item(text=result.get("final_text", ""), id=str(uuid.uuid4()))
        return ResponsesAgentResponse(
            output=[item],
            custom_outputs=self._custom_outputs(result, conversation_id, result.get("sources", [])),
        )

    # ── streaming ───────────────────────────────────────────────────────────

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Stream governance progress, answer tokens and sources as they happen.

        Multiplexes LangGraph `custom` (progress, sources), `messages` (answer tokens) and
        `updates` (final state) onto the Responses stream. Progress rides on `custom_outputs`,
        which the schema passes through; a client ignoring it still gets a plain text stream.
        """
        payload, config, context, conversation_id = self._prepare(request)
        item_id = str(uuid.uuid4())

        final_state: dict = {}
        sources: list = []
        # Layer 7 on the live stream: tokens wait behind a hold-back window, are masked before
        # relay, and a withhold-tier finding stops relaying — so the closing item is not the
        # *first* place the guard applies. `OUTPUT_STREAM_WORKER_TOKENS=false` relays nothing.
        stream_guard = StreamGuard(
            self._output_guard, hold=self._settings.output_stream_holdback_chars
        )
        relay_tokens = self._settings.output_stream_worker_tokens

        # The lock spans the graph run only, so a slow client reading the last event cannot
        # hold the conversation against its own next turn.
        with self._lock(conversation_id) as acquired:
            if not acquired:
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=self.create_text_output_item(text=BUSY_MESSAGE, id=item_id),
                    custom_outputs=self._busy(conversation_id),
                )
                return

            # `version="v2"` (LangGraph 1.2) yields typed StreamParts of one shape instead of
            # positional tuples whose arity changed with the mode list.
            for part in self._graph.stream(
                payload,
                config=config,
                context=context,
                stream_mode=["custom", "messages", "updates"],
                durability=self._durability,
                version="v2",
            ):
                mode, chunk = part["type"], part["data"]
                if mode == "custom":
                    if not isinstance(chunk, dict):
                        continue
                    channel = chunk.get("channel")
                    if channel == "sources":
                        sources = chunk.get("items") or []
                        yield ResponsesAgentStreamEvent(
                            type="response.custom", custom_outputs={"sources": sources}
                        )
                    elif channel == "progress":
                        yield ResponsesAgentStreamEvent(
                            type="response.custom", custom_outputs={"progress": chunk}
                        )

                elif mode == "messages":
                    # (message_chunk, metadata). Only the worker's answer reaches the user:
                    # governance models produce structured verdicts, not prose.
                    message, metadata = chunk
                    if (metadata or {}).get("langgraph_node") != "dispatch":
                        continue
                    text = getattr(message, "content", "") or ""
                    if isinstance(text, list):
                        text = "".join(
                            part.get("text", "") for part in text if isinstance(part, dict)
                        )
                    if text and relay_tokens:
                        cleared = stream_guard.feed(text)
                        if cleared:
                            yield ResponsesAgentStreamEvent(
                                **self.create_text_delta(delta=cleared, item_id=item_id)
                            )

                elif mode == "updates":
                    for node_state in (chunk or {}).values():
                        if isinstance(node_state, dict):
                            final_state.update(node_state)

        # Whatever was still in the hold-back window, masked like the rest. Nothing is flushed
        # if the guard stopped the stream — the closing item carries the guarded outcome.
        if relay_tokens:
            tail = stream_guard.flush()
            if tail:
                yield ResponsesAgentStreamEvent(
                    **self.create_text_delta(delta=tail, item_id=item_id)
                )

        text = final_state.get("final_text", "")

        # The closing item always carries the full text — the assembled message when tokens
        # streamed, the whole answer when they did not (blocked turns never reach the model).
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(text=text, id=item_id),
            custom_outputs=self._custom_outputs(final_state, conversation_id, sources),
        )


AGENT = SupervisorAgent()
mlflow.models.set_model(AGENT)
