"""Two-tier guardrail engine — the screen `nodes/guardrails.py` calls.

Named apart from the `"guardrails"` node so the two are never the same string in a traceback.
Tier 1: deterministic rules (global + per-agent regex deny patterns). Tier 2: semantic LLM check
of the query against an agent's domain scope. `screen` is the whole public surface: it runs both
tiers across the agents a caller can reach and reports which owns the query, so the thing that
decides "is this in your domain?" is also the thing that decides "whose domain is this?" — no
separate selection stage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from agent_governance.deny_rules import compile_rules, first_match, kill_switch_message
from agent_governance.resilience import invoke_with_retries
from agent_governance.sanitize import system_blocks, untrusted_turn
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .prompt_provider import get_prompt
from .registry import WorkerAgent


@dataclass(frozen=True)
class GuardrailResult:
    passed: bool
    tier: str  # "deterministic" | "semantic" | "small_talk"
    reason: str
    # Which kind of small talk (greeting | thanks | ack | meta) when tier is "small_talk", so the
    # supervisor answers in one sentence instead of forwarding a greeting to a worker.
    small_talk: str = ""
    # The screen recognised this agent's subject but not which deliverable was wanted: the one
    # question that would settle it, asked instead of dispatching a guess or refusing.
    clarification: str = ""
    # A safety refusal rather than a scope refusal. Changes only whether the block offers the
    # appeal path: a reviewer can overturn "that isn't my agent's subject".
    safety_refusal: bool = False
    # Two or more reachable agents claimed this with confidences too close to separate; ids, best
    # first. The user is asked which deliverable they meant rather than candidate *order* deciding.
    contested: tuple[str, ...] = ()
    # A SECOND, separate deliverable, offered back as a question after the first is dispatched.
    # Verbatim, so the follow-up screens it exactly as a fresh request.
    additional_request: str = ""
    # A rule published with `action: escalate`: refused like a block, and handed to a reviewer with
    # the conversation held, for requests a refusal alone under-reports. OWASP Agentic T10.
    escalate: bool = False


@dataclass(frozen=True)
class ScreenResult:
    """Which agent owns the query, plus the verdict that decided it."""

    # None when nothing may be sent: a deterministic block, no agent's domain
    # covers it, or the message was small talk the supervisor answers itself.
    agent: object
    result: GuardrailResult
    # The agents actually evaluated, for the decision trail. Shorter than the
    # candidate list whenever an early match short-circuited the rest.
    considered: tuple[str, ...] = ()
    # `{agent_id: sdlc_reading}` for every candidate that claimed the request, best first. In the
    # contested case the readings are what the user chooses between; agent names mean nothing.
    claims: tuple[tuple[str, str], ...] = ()


class GuardrailVerdict(BaseModel):
    """Structured output for the semantic tier.

    **Field order is load-bearing**: output is generated in order, so `sdlc_reading` precedes
    `in_domain` — an in-domain reading is articulated before it can be refused (lexical
    overfitting, e.g. "how to reset the password" refused as IT support).
    """

    sdlc_reading: str = Field(
        description=(
            "First, before judging: if this request were about building software, what "
            "deliverable would it be asking for? Name it in a few words (e.g. 'user stories "
            "for a password-reset feature'). Write 'none' only if no software deliverable "
            "could plausibly be meant."
        )
    )
    in_domain: bool = Field(description="True if the query belongs to the agent's domain scope")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the verdict, 0 to 1")
    # Separate from confidence because they are different states: "clearly your subject, unclear
    # what they want produced" arrives at 0.95. Asked, models flag it; volunteered, under 5% do.
    underspecified: bool = Field(
        default=False,
        description=(
            "True if this agent could own the request once the user says what they want "
            "produced — the subject fits the scope but no deliverable is named"
        ),
    )
    clarification: str = Field(
        default="",
        description=(
            "When underspecified, the ONE short question that would settle it, phrased in "
            "terms of the deliverables in this agent's scope"
        ),
    )
    # Compound requests are ordinary; answering only the first half reads as being ignored.
    # Verbatim, because the follow-up re-enters the screen as a fresh request, not a paraphrase.
    additional_request: str = Field(
        default="",
        description=(
            "If the message asks for a SECOND, separate deliverable as well as the one in "
            "sdlc_reading — 'write the user stories AND the test cases' — quote that second "
            "request verbatim from the message. Empty when the message asks for one thing, "
            "however many sentences it takes. Elaboration, context and constraints on the "
            "first request are not a second request."
        ),
    )
    reason: str = Field(description="One-sentence justification, addressed to the user")
    # Kept LAST, so the harm question cannot prime `in_domain`: SDLC work is full of alarming
    # vocabulary (threat models, pen tests), and asked earlier it turns a request for security
    # *requirements* into one for harm. Blast radius: only the appeal line of a decided refusal.
    safety_refusal: bool = Field(
        default=False,
        description=(
            "Last. True ONLY if the request seeks real-world harm — weapons, explosives, "
            "violence, self-harm, illegal activity, or malware and attacks against systems "
            "the user does not own. False for every ordinary out-of-scope request (cooking, "
            "travel, sport, personal advice, IT support): those are simply not this agent's "
            "subject. False, too, for legitimate software work that merely sounds alarming — "
            "security requirements, threat models, abuse cases, authorised penetration-test "
            "planning, incident runbooks. If in doubt, answer false."
        ),
    )


# §4.1: prompts load from MLflow Prompt Registry by name and environment alias.
_PROMPT_NAME = "supervisor_domain_screen"

# What a refusal costs depends on how many agents the caller can reach — the one thing the screen
# cannot see, being asked about one agent at a time. "Do not stretch" is wrong for a sole agent.
_COMPETING = """\
You are asked this separately about each agent the user is allowed to reach, and the
answers are compared. Declining costs the user nothing: the agent that owns the
request is being asked about it too."""

_SOLE = """\
You are the ONLY agent this user can reach. Nothing you decline is picked up by
anyone else, so a wrong refusal is a dead end for them. Refuse what is clearly
outside this agent's speciality — never something that is merely vaguely worded."""

# Conversational openers and meta questions, classified so the supervisor answers them itself:
# blocking "hi" reads as a broken assistant; forwarding it returns a worker's capability list.
# Deterministic (nothing to inject into), anchored both ends: any extra word gets the full screen.
_ADDRESSEE = r"(?:\s+(?:there|team|folks|all|everyone|bot|supervisor|agent))?"
_SMALL_TALK_KINDS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    (
        "thanks",
        re.compile(
            r"^\W*(?:thanks?(?:\s+you)?|thank\s+you|ty|cheers|thx|"
            r"(?:that'?s\s+)?(?:great|perfect|awesome|helpful))"
            r"(?:\s+(?:a\s+lot|so\s+much|very\s+much))?\W*$",
            re.IGNORECASE,
        ),
    ),
    (
        # A closing, not an opening: answering "bye" with "What are you working
        # on?" reopens a conversation the user just ended.
        "farewell",
        re.compile(
            r"^\W*(?:bye|goodbye|good\s+bye|see\s+(?:you|ya)(?:\s+later)?|"
            r"that'?s\s+(?:all|it)(?:\s+for\s+now)?|(?:i'?m\s+)?done(?:\s+for\s+now)?|"
            r"no(?:thing)?\s+(?:thanks|thank\s+you))\W*$",
            re.IGNORECASE,
        ),
    ),
    (
        "meta",
        re.compile(
            r"^\W*(?:who\s+are\s+you|what\s+(?:can|do)\s+you\s+do|"
            r"what\s+are\s+you\s+for|help)\W*$",
            re.IGNORECASE,
        ),
    ),
    (
        # Split out from `greeting` because it is a *question*, and answering a
        # question with "Hello — I'm the Supervisor" ignores what was asked.
        "how_are_you",
        re.compile(
            # "hi, how are you?" is one move, not two, and the question is the
            # part that wants answering.
            r"^\W*(?:(?:hi|hey|hello|yo)\W+)?"
            r"(?:how\s+(?:are\s+(?:you|u|things|we)|r\s+u|is\s+it\s+going|"
            r"do\s+you\s+do)|how'?s\s+(?:it\s+going|things|life)|"
            r"(?:are\s+)?you\s+(?:ok|okay|well|there|alright)|"
            r"what'?s\s+up|wassup|sup)\W*$",
            re.IGNORECASE,
        ),
    ),
    (
        "greeting",
        re.compile(
            r"^\W*(?:hi|hii+|hey+|hello+|heya|yo|greetings|howdy|"
            r"good\s+(?:morning|afternoon|evening|day))" + _ADDRESSEE + r"\W*$",
            re.IGNORECASE,
        ),
    ),
    (
        "ack",
        re.compile(
            r"^\W*(?:ok(?:ay)?|cool|great|nice|got\s+it|sure|start|test)\W*$", re.IGNORECASE
        ),
    ),
    (
        # Questions about the *agents*: answered by the supervisor, never forwarded (RBAC gates
        # dispatch, not what a worker says *about* a neighbour). Explicit shapes rather than a
        # proximity rule: "agent" is product vocabulary, and a false positive here is a silent drop.
        "meta_agents",
        re.compile(
            # A polite preamble is the normal way people ask this, and without
            # it the shape below misses every such question.
            r"^\W*(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?(?:just\s+)?"
            r"(?:tell\s+me\s+|show\s+me\s+|let\s+me\s+know\s+|explain\s+|clarify\s+)?"
            r"(?:"
            # "does the Deployment Agent have access to production secrets"
            r"(?:does|do|can|could|is|are)\b[^.?!\n]{0,60}?\b(?:agent|workflow|assistant|bot|supervisor)s?\b"
            r"[^.?!\n]{0,40}?\b(?:have|has|get|hold|see|access|read|reach)\b"
            r"[^.?!\n]{0,40}?\b(?:access|secrets?|credentials?|permissions?|privileges?|"
            r"tokens?|keys?|production|prod\b|prompts?|memory|"
            r"my\s+(?:data|prompts?|messages?|history|context|files?|notes?|conversations?))"
            # "what data does the Requirement Agent have stored" — opens on the *thing asked about*,
            # unlike "what are the user stories for the agent permissions screen".
            r"|(?:what|which|how\s+much)\s+(?:data|information|context|memory|records?|secrets?|"
            r"permissions?|access|privileges?|credentials?|scopes?|tokens?|keys?)\b"
            r"[^.?!\n]{0,60}?\b(?:agent|workflow|assistant|supervisor)s?\b"
            r"|(?:what|which)\b[^.?!\n]{0,30}?\b(?:agent|workflow)s?\b[^.?!\n]{0,30}?"
            r"\b(?:store[ds]?|stored|hold[s]?|keep[s]?|remember[s]?|retain[s]?)\b"
            # "is there a legal review agent only certain people can see"
            r"|(?:is|are)\s+there\b[^.?!\n]{0,60}?\b(?:agent|workflow)s?\b"
            r"|(?:which|what)\s+(?:other\s+)?(?:agent|workflow)s?\b[^.?!\n]{0,40}?"
            r"\b(?:exist|are\s+there|am\s+i|can\s+i|hidden|restricted|available\s+to\s+me|"
            r"allowed|my\s+role)\b"
            r")",
            re.IGNORECASE,
        ),
    ),
)


# A message that asks for something to be *produced* is a request. Vetoes `meta_agents` only,
# which never dispatches, so a false positive there costs the request with nothing to appeal.
# `review`, `test`, `design`, `document`, `describe` are out: they are subject vocabulary too.
_NAMES_A_DELIVERABLE = re.compile(
    r"\b(?:write|draft|generate|produce|implement|refactor|spec(?:ify)?|"
    r"summari[sz]e|user\s+stor(?:y|ies)|test\s+cases?|test\s+plan|"
    r"acceptance\s+criteria|hld|lld|epic)\b",
    re.IGNORECASE,
)
# The conversational kinds are anchored, so nothing naming a deliverable can match them anyway;
# `meta_agents` is the one kind that matches a fragment of a sentence.
_VETOABLE = frozenset({"meta_agents"})


def small_talk_kind(query: str) -> str:
    """Which kind of small talk this message is, or "" if it is a real request."""
    query = query or ""
    for kind, pattern in _SMALL_TALK_KINDS:
        if pattern.match(query):
            if kind in _VETOABLE and _NAMES_A_DELIVERABLE.search(query):
                return ""
            return kind
    return ""


# ── Answering the "shall I do the second one too?" offer ────────────────────
#
# Deterministic and anchored at both ends, like `small_talk`: only a message that is *entirely* an
# answer is read as one. "yes, and also delete the staging database" is not a confirmation.
_ACCEPT_WORDS = r"yes|yeah|yep|yup|sure|ok(?:ay)?|affirmative"
_ACCEPT_VERBS = (
    r"please(?:\s+do)?|do\s+(?:it|that|both|the\s+second(?:\s+one|\s+task|\s+part)?)|"
    r"go\s+ahead|carry\s+on|continue|proceed|keep\s+going"
)
_ACCEPTS = re.compile(
    rf"^\W*(?:(?:{_ACCEPT_WORDS})\W*$"
    rf"|(?:(?:{_ACCEPT_WORDS})\W+)?(?:{_ACCEPT_VERBS})\W*$)",
    re.IGNORECASE,
)
_DECLINE_WORDS = r"no|nope|nah|no\s+thanks?|no\s+thank\s+you"
_DECLINE_VERBS = (
    r"not\s+(?:now|yet|today|for\s+now)|skip\s+(?:it|that|the\s+second(?:\s+one)?)|"
    r"leave\s+(?:it|that)|don'?t(?:\s+bother)?|cancel|that'?s\s+(?:all|it|everything)|"
    r"thanks?(?:\s+that'?s\s+all)?"
)
_DECLINES = re.compile(
    rf"^\W*(?:(?:{_DECLINE_WORDS})\W*$"
    rf"|(?:(?:{_DECLINE_WORDS})\W+)?(?:{_DECLINE_VERBS})\W*$)",
    re.IGNORECASE,
)


def followup_answer(text: str) -> str:
    """ "accept" | "decline" | "" — is this message an answer to the offer?

    Anything else is a new request, so a held-over task cannot trigger by accident: the offer
    expires the moment the user says something other than yes or no.
    """
    message = (text or "").strip()
    if not message:
        return ""
    if _ACCEPTS.match(message):
        return "accept"
    if _DECLINES.match(message):
        return "decline"
    return ""


def _second_request(verdict: "GuardrailVerdict") -> str:
    """The second deliverable in the message, if the verdict found one.

    Bounded and stripped rather than trusted: the field is model-written from untrusted text,
    and it becomes the *query* of a follow-up turn if the user accepts it.
    """
    found = (verdict.additional_request or "").strip()
    if len(found) < 8 or len(found) > 400:
        return ""
    return found


def _ownership_question(contested: list[tuple["WorkerAgent", "GuardrailVerdict"]]) -> str:
    """The one question that settles which agent a contested request belongs to.

    Phrased in deliverables, not agent names: "a unit test or a functional test case?" is
    answerable by someone never told the agents apart. Names are the fallback when a
    reading is missing.
    """
    readings: list[str] = []
    for agent, verdict in contested:
        reading = (verdict.sdlc_reading or "").strip().rstrip(".")
        readings.append(reading if reading and reading.lower() != "none" else agent.name)
    # Deduplicate while keeping order — two agents can name the same
    # deliverable, and "a unit test or a unit test?" helps nobody.
    seen: set[str] = set()
    unique: list[str] = []
    for reading in readings:
        if reading.lower() not in seen:
            seen.add(reading.lower())
            unique.append(reading)
    if len(unique) < 2:
        unique = [agent.name for agent, _ in contested]
    joined = (
        " or ".join((", ".join(unique[:-1]), unique[-1]))
        if len(unique) > 2
        else " or ".join(unique)
    )
    return f"More than one of your agents could take this — did you want {joined}?"


def _asks(verdict: "GuardrailVerdict") -> str:
    """The question to put to the user, when the verdict says one is needed.

    Gated on the flag, not on the text being non-empty: a model that volunteers an unrequested
    question must not be able to turn a settled verdict into an interrogation.
    """
    return (verdict.clarification or "").strip() if verdict.underspecified else ""


class GuardrailEngine:
    def __init__(
        self,
        llm,
        global_deny_patterns: list[dict],
        confidence_threshold: float = 0.7,
        model_for=None,
        kill_switch=None,
        decisive_threshold: float = 0.9,
        contested_margin: float = 0.15,
    ):
        self._llm = llm
        # Multi-model support: an optional `agent -> chat model` resolver, so a
        # candidate's verdict can run on the model its registry entry names.
        # None means every verdict uses `llm`.
        self._model_for = model_for
        self._rules = compile_rules(global_deny_patterns)
        self._threshold = confidence_threshold
        # ── Two thresholds ──────────────────────────────────────────────────
        # "Confident enough to act on" and "confident enough to stop asking" differ: since
        # `candidates` is target-first, only a `decisive_threshold` verdict short-circuits, and
        # a top two within `contested_margin` asks the user which deliverable they meant.
        self._decisive = max(confidence_threshold, decisive_threshold)
        self._contested_margin = max(0.0, contested_margin)
        # The emergency stop, carried on the engine because `Reloading` rebuilds it when the
        # governed document changes — so flipping the switch is a config publish, not a deploy.
        self.kill_switch_message = kill_switch_message(kill_switch)

    @classmethod
    def from_mapping(
        cls,
        llm,
        data: dict,
        confidence_threshold: float = 0.7,
        model_for=None,
        decisive_threshold: float = 0.9,
        contested_margin: float = 0.15,
    ) -> "GuardrailEngine":
        """Build from the parsed guardrails document (`config.SupervisorConfig`)."""
        return cls(
            llm,
            (data or {}).get("global_deny_patterns", []),
            confidence_threshold,
            model_for=model_for,
            kill_switch=(data or {}).get("kill_switch"),
            decisive_threshold=decisive_threshold,
            contested_margin=contested_margin,
        )

    def deterministic_block(self, query: str) -> str:
        """The reason a deterministic rule refuses this text, or "".

        Exposed so the supervisor can check a *held-over* request before offering to run it:
        offering something the rules would refuse, then refusing, is worse than not offering.
        """
        rule = first_match(self._rules, query)
        return rule.reason if rule else ""

    def screen(
        self,
        query: str,
        candidates: list["WorkerAgent"],
        history: Optional[list[str]] = None,
        deadline=None,
    ) -> "ScreenResult":
        """Which of these agents owns the query — and may it be sent at all?

        `candidates` is the caller's *permitted* agents, target first; nothing outside it is
        considered, so screening cannot surface or route to an inaccessible agent. `deadline`
        (§05 Stage 05) is checked before each verdict, because this is one model call per agent.
        """
        history = history or []

        # Global deterministic rules run once, before any agent is considered: a
        # blocked pattern must not become shoppable by retrying it against the
        # next agent in the list.
        rule = first_match(self._rules, query)
        if rule:
            return ScreenResult(
                None,
                GuardrailResult(False, "deterministic", rule.reason, escalate=rule.escalate),
                (),
            )

        kind = small_talk_kind(query)
        if kind:
            return ScreenResult(
                None,
                GuardrailResult(True, "small_talk", f"{kind}, no domain intent", small_talk=kind),
                (),
            )

        considered: list[str] = []
        in_domain: list[tuple[WorkerAgent, GuardrailVerdict]] = []
        unclear: list[tuple[WorkerAgent, GuardrailVerdict]] = []
        best_off: Optional[GuardrailVerdict] = None
        # Tracked across every candidate rather than read off `best_off`, which
        # is selected by confidence and could be a different agent's verdict.
        # Whether a request seeks harm is a property of the request.
        flagged_unsafe = False
        sole = len(candidates) == 1
        # With one reachable agent there is no ordering to be misled by, so the
        # higher bar buys nothing and would only relabel a settled verdict as a
        # "best available match".
        decisive = self._threshold if sole else self._decisive

        for agent in candidates:
            # An agent's own deny patterns rule *it* out, not the whole request:
            # another agent may legitimately handle the same topic.
            if any(re.search(p, query, flags=re.IGNORECASE) for p in agent.deny_patterns):
                continue

            # Before the call, not after: checked after it would only report an overrun that
            # already happened. A partial screen never becomes a pass (§06 fail-closed).
            if deadline is not None:
                deadline.ensure(f"screening against {agent.id}")

            considered.append(agent.id)
            verdict = self._semantic_verdict(query, agent, history, sole=sole, deadline=deadline)
            flagged_unsafe = flagged_unsafe or verdict.safety_refusal

            if verdict.in_domain and verdict.confidence >= decisive:
                # Only a *decisive* verdict short-circuits: a merely above-threshold one
                # leaves room for a better owner. The question rides along, or the second half
                # of "my subject, but what do you want produced?" is lost.
                return ScreenResult(
                    agent,
                    GuardrailResult(
                        True,
                        "semantic",
                        verdict.reason,
                        clarification=_asks(verdict),
                        additional_request=_second_request(verdict),
                    ),
                    tuple(considered),
                    claims=((agent.id, verdict.sdlc_reading),),
                )
            if verdict.in_domain:
                in_domain.append((agent, verdict))
            elif verdict.underspecified:
                # Not evidence of anything being out of domain, so it must not
                # feed `best_off`.
                unclear.append((agent, verdict))
            elif best_off is None or verdict.confidence > best_off.confidence:
                best_off = verdict

        # Nobody was decisive. Rank what did claim it, and decide whether the
        # ranking means anything.
        if in_domain:
            ranked = sorted(in_domain, key=lambda pair: pair[1].confidence, reverse=True)
            agent, verdict = ranked[0]
            claims = tuple((a.id, v.sdlc_reading) for a, v in ranked)

            # Contested: runner-up within the margin and both over the acting threshold. Two
            # genuine owners is not a tie to break by rounding — only the user knows which.
            contested = [
                (a, v)
                for a, v in ranked
                if v.confidence >= self._threshold
                and verdict.confidence - v.confidence <= self._contested_margin
            ]
            if len(contested) > 1:
                return ScreenResult(
                    agent,
                    GuardrailResult(
                        True,
                        "semantic",
                        "more than one agent you can reach owns this request: "
                        + ", ".join(f"{a.name} ({v.confidence:.2f})" for a, v in contested),
                        clarification=_ownership_question(contested),
                        contested=tuple(a.id for a, _ in contested),
                        additional_request=_second_request(verdict),
                    ),
                    tuple(considered),
                    claims=claims,
                )

            # One clear winner among the weaker verdicts. Still beats guessing,
            # and it is now a *comparison* rather than whoever was asked first.
            return ScreenResult(
                agent,
                GuardrailResult(
                    True,
                    "semantic",
                    f"best available match ({verdict.confidence:.2f}): {verdict.reason}",
                    clarification=_asks(verdict),
                    additional_request=_second_request(verdict),
                ),
                tuple(considered),
                claims=claims,
            )

        # Subject recognised but the ask unclear: a question, not a refusal, or a user whose
        # only reachable agent just declined their own topic is told to go away. The first
        # such agent is taken because `candidates` is target-first.
        if unclear:
            agent, verdict = unclear[0]
            return ScreenResult(
                agent,
                GuardrailResult(
                    True,
                    "semantic",
                    f"underspecified: {verdict.reason}",
                    clarification=_asks(verdict),
                ),
                tuple(considered),
                claims=tuple((a.id, v.sdlc_reading) for a, v in unclear),
            )

        # Everything came back off-domain. Low confidence is ambiguity rather
        # than a refusal, so it falls through to route/clarify on the agent the
        # turn was already addressed to — unchanged from the single-agent path.
        if best_off is not None and best_off.confidence < self._threshold:
            return ScreenResult(
                candidates[0] if candidates else None,
                GuardrailResult(
                    True,
                    "semantic",
                    f"low-confidence off-domain verdict ({best_off.confidence:.2f}): {best_off.reason}",
                ),
                tuple(considered),
            )

        if best_off is not None:
            return ScreenResult(
                None,
                GuardrailResult(False, "semantic", best_off.reason, safety_refusal=flagged_unsafe),
                tuple(considered),
            )

        # Nothing was evaluated at all, so every candidate was ruled out by its
        # own deny patterns. "No agent covers this topic" would contradict the
        # refusal's own "your role covers …" line — the agents exist.
        return ScreenResult(
            None,
            GuardrailResult(
                False, "deterministic", "this topic is blocked for every agent you can reach"
            ),
            tuple(considered),
        )

    def _semantic_verdict(
        self, query: str, agent, history: list[str], sole: bool = False, deadline=None
    ) -> GuardrailVerdict:
        """Two turns, not one: the rules are the system prompt, the untrusted
        content is a JSON user turn. See `sanitize.untrusted_turn`."""
        # Content blocks, not one string, so the invariant rules can be cached: this loop asks
        # the same ~1100 tokens once per candidate. `ChatDatabricks` reports no cache counters.
        rules = system_blocks(
            get_prompt(_PROMPT_NAME),
            agent_name=agent.name,
            domain_scope=agent.domain_scope,
            alternatives=_SOLE if sole else _COMPETING,
        )
        payload = untrusted_turn(
            conversation=history[-10:] or ["(start of conversation)"],
            user_query=query,
        )
        # The deadline gates *whether* this call runs; the call's own ceiling stays
        # `routing_llm_timeout_seconds`, since rebinding per call would defeat the client cache.
        # Retried here: a transient 429/5xx recovers without re-screening every candidate.
        llm = self._model_for(agent) if self._model_for is not None else self._llm
        return invoke_with_retries(
            lambda: llm.with_structured_output(GuardrailVerdict).invoke(
                [SystemMessage(content=rules), HumanMessage(content=payload)]
            ),
            what=f"the domain screen for {agent.id}",
            deadline=deadline,
        )
