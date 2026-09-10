"""Databricks ResponsesAgent entrypoint (MLflow models-from-code).

The Governance Front Door (POST /agents/{agent-id}/invocation) forwards
validated requests here with custom_inputs:

    {
      "input": [{"role": "user", "content": "..."}],
      "custom_inputs": {
        "agent_id": "requirement-agent",    # the target worker, from the
                                            # invocation path — each chat widget
                                            # is scoped to one agent
        "permitted_agents": ["requirement-agent", "deployment-agent"],
                                            # derived from the identity token by
                                            # the front door — authoritative
        "approvable_agents": ["requirement-agent"],
                                            # the approve permission set, same
                                            # provenance; binds a sign-off to
                                            # the agent that staged the work
        "user_role": "BA",                  # persona, for display and audit
        "conversation_id": "thr_abc123",    # resolved thread; omit to start one
        "user_id": "usr_9f2c…",             # pseudonymous, keys long-term memory
        "resume": {"decision": "approved"}  # answers a pending approval
      }
    }

`permitted_agents` and `approvable_agents` are trusted because they are computed
server-side from a validated token against the environment catalogue. They are
never accepted from a browser (§2.7).

Identity is passed to the graph as **runtime context**, not state — see
`context.py`. Only the conversation is checkpointed.

`predict_stream` is the real path: it streams governance progress, answer
tokens and grounding sources as the graph runs. `predict` collects the same
stream into one response for callers that cannot stream.
"""

from __future__ import annotations

import logging
import uuid
from typing import Generator

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from supervisor.context import SupervisorContext
from supervisor.graph import build_graph
from supervisor.locking import BUSY_MESSAGE, thread_lock
from supervisor.output_guard import OutputGuard, StreamGuard
from supervisor.sanitize import clean_inbound_text
from supervisor.settings import Settings

logger = logging.getLogger(__name__)

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
        # Ingress hygiene (guardrail layer 1): control, zero-width and bidi
        # characters are stripped before the text enters graph state — once in
        # state it is checkpointed and echoed back through every later turn.
        # See sanitize.clean_inbound_text for what is (and is not) removed.
        messages.append({"role": role, "content": clean_inbound_text(str(content))})
    return messages


class SupervisorAgent(ResponsesAgent):
    def __init__(self, graph=None, output_guard=None):
        if graph is None:
            # Built from one `Services` so the stream guard below screens
            # tokens with the *same* governed policy the dispatch node applies
            # to the finished reply — two guards from two documents would
            # disagree exactly when it mattered.
            from supervisor.services import build_services

            services = build_services()
            graph = build_graph(services)
            output_guard = output_guard or services.output_guard
        self._graph = graph
        # A fake graph in a test gets a bare guard: the shipped defaults,
        # which is the production masking behaviour without configuration.
        self._output_guard = output_guard or OutputGuard()
        # "sync" holds each stage until its checkpoint is written — the setting
        # LangGraph recommends for production HITL flows, and an approval gate
        # is one (see settings.py). An unrecognised value falls back to the
        # safe mode rather than raising inside the serving container.
        settings = Settings()
        # Refuse to serve a deployed environment whose safety-critical settings
        # are disabled (a stray THREAD_LOCK_ENABLED=false, TURN_BUDGET_SECONDS=0
        # …). In dev this logs at ERROR instead — see Settings.enforce.
        settings.enforce()
        durability = settings.durability
        if durability not in _DURABILITY_MODES:
            durability = "sync"
        self._durability = durability
        self._settings = settings
        # For the busy-refusal audit row below. Shares the same pooled
        # connection source as the graph's own sink, so this costs no extra
        # entitlement and no extra pool.
        self._audit = None

    # ── one execution per thread (§4.4) ─────────────────────────────────────

    def _lock(self, conversation_id: str):
        """The execution lock for this turn's conversation. See locking.py."""
        return thread_lock(
            conversation_id,
            enabled=self._settings.thread_lock_enabled,
            timeout=self._settings.thread_lock_timeout_seconds,
            poll=self._settings.thread_lock_poll_seconds,
        )

    def _busy(self, conversation_id: str) -> dict:
        """Custom outputs for a turn refused because its thread was already busy.

        A minimal audit row *is* written: refusing a turn is a governance
        outcome, and a table with no trace of busy refusals under-reports
        exactly the concurrency behaviour a capacity investigation asks about.
        The row is best-effort — a refusal must not fail because the sink
        blinked — and the trace is tagged as well, where latency and
        concurrency are actually investigated.
        """
        try:
            mlflow.update_current_trace(
                session_id=conversation_id or None, tags={"outcome": "busy"}
            )
        except Exception:
            # Expected whenever there is no active trace — an offline test, a
            # local run without autolog. Logged at debug rather than swallowed
            # silently, so a *real* tagging failure in the endpoint is findable.
            logger.debug("busy-turn trace tagging skipped", exc_info=True)
        try:
            if self._audit is None:
                from supervisor.audit import build_audit_logger

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
        config = {
            "configurable": {"thread_id": conversation_id},
            # The graph's step ceiling, set explicitly rather than left at
            # LangGraph's default 25. The graph is acyclic and six nodes deep,
            # so this is defence in depth against a future edge — see
            # `settings.graph_recursion_limit`.
            "recursion_limit": self._settings.graph_recursion_limit,
        }

        resume = custom.get("resume")
        if resume is not None:
            # Answering a pending approval. `Command(resume=...)` picks up
            # *inside* the interrupted node rather than replaying the turn, so
            # the worker is not called twice.
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
                    output=[
                        self.create_text_output_item(text=BUSY_MESSAGE, id=str(uuid.uuid4()))
                    ],
                    custom_outputs=self._busy(conversation_id),
                )

            result = self._graph.invoke(
                payload, config=config, context=context, durability=self._durability
            )

        item = self.create_text_output_item(
            text=result.get("final_text", ""), id=str(uuid.uuid4())
        )
        return ResponsesAgentResponse(
            output=[item],
            custom_outputs=self._custom_outputs(result, conversation_id, result.get("sources", [])),
        )

    # ── streaming ───────────────────────────────────────────────────────────

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Stream governance progress, answer tokens and sources as they happen.

        Three LangGraph channels are multiplexed onto the Responses stream:

          custom    -> progress steps and sources, for the calling UI's task plan
          messages  -> answer tokens, as the model produces them
          updates   -> the final state, for the closing item and custom outputs

        Progress events ride on `custom_outputs`, which the Responses schema
        passes through untouched. A client that ignores them still receives a
        conventional text stream.
        """
        payload, config, context, conversation_id = self._prepare(request)
        item_id = str(uuid.uuid4())

        final_state: dict = {}
        sources: list = []
        # Layer 7 on the live stream. Tokens wait behind a hold-back window,
        # the part that clears it is masked before it is relayed, and a
        # withhold-tier finding stops relaying altogether — so the closing
        # item is no longer the *first* place the output guard applies. See
        # `output_guard.StreamGuard`; `OUTPUT_STREAM_WORKER_TOKENS=false`
        # relays nothing and lets the closing item carry the whole answer.
        stream_guard = StreamGuard(
            self._output_guard, hold=self._settings.output_stream_holdback_chars
        )
        relay_tokens = self._settings.output_stream_worker_tokens

        # The lock spans the graph run only — released as soon as the stream is
        # exhausted, before the closing item below, so a slow client reading the
        # last event cannot hold the conversation against its own next turn.
        with self._lock(conversation_id) as acquired:
            if not acquired:
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=self.create_text_output_item(text=BUSY_MESSAGE, id=item_id),
                    custom_outputs=self._busy(conversation_id),
                )
                return

            # `version="v2"` (LangGraph 1.2) yields typed StreamParts — one shape,
            # `{"type", "ns", "data"}`, however many modes are multiplexed — instead
            # of the positional tuples whose arity used to change with the mode list.
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
                    # (message_chunk, metadata). Only the worker's answer should
                    # reach the user — the guardrail and routing models produce
                    # structured governance verdicts, not prose.
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

        # Whatever was still inside the hold-back window when the graph
        # finished — masked like the rest. Nothing is flushed if the guard
        # stopped the stream; the closing item carries the guarded outcome.
        if relay_tokens:
            tail = stream_guard.flush()
            if tail:
                yield ResponsesAgentStreamEvent(
                    **self.create_text_delta(delta=tail, item_id=item_id)
                )

        text = final_state.get("final_text", "")

        # The closing item always carries the full text. When tokens streamed,
        # this is the assembled message the client replaces its buffer with;
        # when they did not — a blocked or clarifying turn, which never reaches
        # the model — it is the whole answer.
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item(text=text, id=item_id),
            custom_outputs=self._custom_outputs(final_state, conversation_id, sources),
        )


AGENT = SupervisorAgent()
mlflow.models.set_model(AGENT)
