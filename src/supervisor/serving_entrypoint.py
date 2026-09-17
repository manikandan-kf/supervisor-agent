"""Databricks ResponsesAgent entrypoint (MLflow models-from-code).

The calling application sends each request with `custom_inputs`: `agent_id` (the target
worker), `permitted_agents` / `approvable_agents` (the caller's entitlements; trusted because
only the caller's service principal holds CAN QUERY on this endpoint), `user_role`,
`conversation_id`, `user_id` and `resume`. Identity reaches the graph as runtime context, not
state (see `state.py`); only the conversation is checkpointed. The caller sends one turn per
conversation at a time, so nothing here serializes turns.

Solution §02 pattern 01 is a synchronous relay: the caller's request blocks until the graph
has an answer. `predict` is that path. `predict_stream` exists only because stream-capable
callers (the AI Playground, the SDK with `stream=True`) send every request as a stream; it
runs the same synchronous turn and emits the finished answer as one event.
"""

from __future__ import annotations

import logging
import uuid
from typing import Generator

import mlflow
from agent_governance.sanitize import clean_inbound_text
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from supervisor.graph import build_graph
from supervisor.settings import Settings
from supervisor.state import SupervisorContext

logger = logging.getLogger(__name__)

# MLflow Tracing instruments every LangGraph node and model call; the trace is the
# request-level log, tagged with the conversation, user and outcome by the audit sink.
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
    def __init__(self, graph=None):
        if graph is None:
            from supervisor.services import build_services

            graph = build_graph(build_services())
        self._graph = graph
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

    # ── request -> graph inputs ─────────────────────────────────────────────

    def _prepare(self, request: ResponsesAgentRequest):
        custom = request.custom_inputs or {}
        conversation_id = custom.get("conversation_id") or str(uuid.uuid4())
        context = SupervisorContext.from_custom_inputs(custom)
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

    @staticmethod
    def _custom_outputs(state: dict, conversation_id: str) -> dict:
        return {
            "conversation_id": conversation_id,
            "request_id": state.get("request_id", ""),
            "outcome": state.get("outcome", ""),
            "routed_agent_id": state.get("target_agent_id", ""),
            "routed_agent_name": state.get("routed_agent_name", ""),
            "pending_clarification": state.get("pending_clarification"),
            "pending_approval": state.get("pending_approval"),
            "sources": state.get("sources") or [],
        }

    # ── the synchronous relay ───────────────────────────────────────────────

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        payload, config, context, conversation_id = self._prepare(request)

        result = self._graph.invoke(
            payload, config=config, context=context, durability=self._durability
        )

        item = self.create_text_output_item(text=result.get("final_text", ""), id=str(uuid.uuid4()))
        return ResponsesAgentResponse(
            output=[item], custom_outputs=self._custom_outputs(result, conversation_id)
        )

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """The same synchronous turn, delivered as a one-event stream.

        No token relay and no progress channel: the answer is screened whole by the output
        guard inside the graph, and only the finished text leaves. A stream-capable client
        sees one `response.output_item.done` carrying exactly what `predict` returns.
        """
        response = self.predict(request)
        for item in response.output:
            yield ResponsesAgentStreamEvent(
                type="response.output_item.done",
                item=item,
                custom_outputs=response.custom_outputs,
            )


AGENT = SupervisorAgent()
mlflow.models.set_model(AGENT)
