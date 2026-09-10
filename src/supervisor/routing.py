"""Route / clarify stage.

The target agent is already fixed by the chat widget, so routing here means
resolving the context the worker needs (e.g. product line). When context is
ambiguous the supervisor asks one clarifying question and resumes on the
reply, escalating to a human after the configured number of loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .prompt_provider import get_prompt, untrusted_turn
from .resilience import invoke_with_retries


@dataclass(frozen=True)
class RouteResult:
    ready: bool
    resolved_context: dict[str, str] = field(default_factory=dict)
    clarifying_question: Optional[str] = None
    # Context keys carried in from an earlier turn or an earlier conversation
    # that this request is *not* about, and which were therefore dropped rather
    # than dispatched on. Kept on the result so the decision trail records the
    # drop: context silently surviving a change of subject and context silently
    # vanishing are both wrong, and only one of them is visible without this.
    dropped_context: tuple[str, ...] = ()
    # False when the request does not depend on the agent's required context at
    # all. Kept on the result so the decision trail records *why* nothing was
    # asked, rather than leaving a silent pass-through (BR-004).
    context_applies: bool = True


class _ContextItem(BaseModel):
    key: str
    value: str


class RouteDecision(BaseModel):
    """Structured output contract for the routing stage.

    Public because it is part of the prompt's registered surface:
    `register_prompts.py` binds it to `supervisor_routing` as the version's
    `response_format`, so the schema the model is held to ships with the same
    immutable prompt version that instructs it.
    """

    context_applies: bool = Field(
        default=True,
        description=(
            "False when the agent can produce something useful without the required "
            "context — nothing was said about it, and knowing it would decorate the "
            "artifact rather than change it. This is the common case: proceed"
        ),
    )
    # Asked before `ready`, and before anything else that could be answered
    # *from* the carried context — the order is the same load-bearing trick the
    # domain screen uses. A model that has already listed a product line as
    # resolved will not then decide the request was never about it.
    #
    # This is the fix for context outliving its subject. `known_context` holds
    # values resolved on earlier turns and, for a brand-new conversation, the
    # user's most recent values read back from long-term memory. Applied
    # unconditionally, that is how a fresh question about a different feature
    # gets answered in terms of the last one — the user starts what they think
    # is a clean conversation and the previous one's subject follows them into
    # it, with nothing in the transcript to say why.
    prior_context_applies: bool = Field(
        default=True,
        description=(
            "First, before anything else. Looking ONLY at what the user is asking for now: "
            "is the carried-over context in `known_context` still the context of THIS "
            "request? True when the request continues the same piece of work, or says "
            "nothing that contradicts it. False when the user has moved to a different "
            "subject that the carried values do not describe — a different product line, "
            "a different environment, a different feature area. When in doubt about a "
            "value the user has not mentioned in this conversation, answer false: "
            "re-resolving costs a question, applying the wrong context costs a wrong "
            "artifact that looks right."
        ),
    )
    ready: bool = Field(description="True when every required context item is resolved")
    resolved_context: list[_ContextItem] = Field(default_factory=list)
    clarifying_question: Optional[str] = Field(
        default=None,
        description="One short question about the single most important missing item; null when ready",
    )


# The name tests and older callers imported before the class went public.
_RouteDecision = RouteDecision

# §4.1: prompts load from MLflow Prompt Registry by name and environment alias.
_PROMPT_NAME = "supervisor_routing"


class Router:
    def __init__(self, llm, model_for=None):
        self._llm = llm
        # Multi-model support: optional `agent -> chat model`
        # resolver, so context resolution for an agent runs on the model its
        # registry entry names. None (tests, direct callers) keeps every call
        # on `llm`, unchanged.
        self._model_for = model_for

    def resolve(
        self,
        agent,
        history: list[str],
        prior_context: dict[str, str],
        deadline=None,
        carried_over: Optional[dict[str, str]] = None,
    ) -> RouteResult:
        """Resolve the agent's required context, or ask for what is missing.

        `deadline` is the turn's remaining time budget (§05 Stage 05). Checked
        before the model call and only when a call is actually needed — an agent
        with no `required_context` returns without spending anything, so an
        exhausted budget must not turn that free path into a failure. None
        disables the check.
        """
        if not agent.required_context:
            return RouteResult(True, dict(prior_context))
        carried_over = dict(carried_over or {})

        if deadline is not None:
            deadline.ensure(f"resolving context for {agent.id}")

        # Rules in the system turn, untrusted content in a JSON user turn.
        # `prior` is read back from long-term memory, which is user-influenced,
        # so it belongs on the untrusted side of that line too.
        rules = get_prompt(_PROMPT_NAME).format(
            agent_name=agent.name,
            description=agent.description,
            required=", ".join(agent.required_context) or "(none)",
        )
        # `carried_over` is named separately from the rest of `known_context`
        # because the two carry different weight. A value the user stated in
        # *this* conversation is something they can see in the transcript above
        # them; a value read back from another conversation is not, and applying
        # it silently is what makes an answer look like it belongs to a
        # different session. Naming it lets the model — and the audit row — tell
        # them apart.
        payload = untrusted_turn(
            known_context=dict(prior_context),
            carried_over_from_earlier=carried_over,
            conversation=history[-12:] or ["(start of conversation)"],
        )
        # Transient failures retry here, at the single call — see resilience.py
        # for why this replaced the graph-level RetryPolicy that never fired.
        llm = self._model_for(agent) if self._model_for is not None else self._llm
        decision = invoke_with_retries(
            lambda: llm.with_structured_output(RouteDecision).invoke(
                [SystemMessage(content=rules), HumanMessage(content=payload)]
            ),
            what=f"context resolution for {agent.id}",
            deadline=deadline,
        )

        # A change of subject drops what was carried in, and only what was
        # carried in: anything the model resolved from *this* request stands.
        dropped: tuple[str, ...] = ()
        base = dict(prior_context)
        if not decision.prior_context_applies and base:
            dropped = tuple(sorted(base))
            base = {}
        context = base
        context.update({item.key: item.value for item in decision.resolved_context})

        # `required_context` describes what the agent needs to *produce an
        # artifact*, not a toll on every message. Asking "which product line?"
        # before answering "what is an epic?" is noise, and after two of them
        # the user is escalated to a human for no reason.
        #
        # The bar was raised deliberately after watching a real session: a user
        # asked for user stories about a password reset and was asked which
        # product line it was for — a question whose answer would have decorated
        # the stories, not changed them. The design scopes clarification to
        # context that "is ambiguous", which is narrower than context that is
        # merely absent, and the prompt now tests for exactly that: ask when the
        # conversation puts several candidates in play, or when a wrong guess
        # would do real damage; otherwise proceed with what is known.
        if not decision.context_applies:
            return RouteResult(True, context, context_applies=False, dropped_context=dropped)

        if decision.ready:
            return RouteResult(True, context, dropped_context=dropped)
        question = decision.clarifying_question or (
            "Could you share a bit more detail so I can route your request correctly?"
        )
        return RouteResult(False, context, question, dropped_context=dropped)
