"""Two-tier guardrail engine.

Tier 1: deterministic rules (global + per-agent regex deny patterns).
Tier 2: semantic LLM check of the query against an agent's domain scope.
Off-domain queries are blocked with a clear explanation.

`screen` is the whole public surface. It runs both tiers across the agents a
caller can actually reach and reports which one owns the query — that is how the
supervisor routes without a separate selection stage: the thing that decides "is
this in your domain?" is also the thing that decides "whose domain is this?".

There used to be an `evaluate(query, agent)` for the single-agent case, with its
own deliberately lenient prompt. No node ever called it once routing moved to
`screen`, and `screen(query, [agent])` yields the same outcome for every branch —
confident in-domain passes, a weak one passes as the best available match, a
confident off-domain blocks, and an unconfident one falls through to
route/clarify. It and `supervisor_guardrail` were removed rather than kept as a
second way to ask the same question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .prompt_provider import get_prompt, system_blocks, untrusted_turn
from .resilience import invoke_with_retries


@dataclass(frozen=True)
class GuardrailResult:
    passed: bool
    tier: str  # "deterministic" | "semantic" | "small_talk"
    reason: str
    # Which kind of small talk, when tier is "small_talk": greeting | thanks |
    # ack | meta. Lets the supervisor answer it itself, in one short sentence,
    # instead of forwarding a greeting to a worker that replies with its whole
    # capability list.
    small_talk: str = ""
    # Set when the screen recognised the subject as this agent's territory but
    # could not tell which deliverable was being asked for. The one question that
    # would settle it — asked instead of dispatching a guess or refusing.
    clarification: str = ""
    # This block was a safety refusal, not a scope refusal. It changes nothing
    # about *whether* the request is blocked — only whether the block offers the
    # appeal path (see `nodes.APPEAL_NOTE`). A reviewer can overturn "that isn't
    # my agent's subject"; nobody can overturn "how do I build a bomb", so
    # offering an appeal there is the "pathway, not a promise" failure the appeal
    # control exists to avoid — the same reasoning already applied to the spend
    # ceiling in `nodes.SUBJECT_ALLOWANCE_MESSAGE`.
    safety_refusal: bool = False
    # Two or more agents the caller can reach claimed this request with
    # confidences too close to separate. Their ids, best first. The supervisor
    # asks the user which deliverable they meant instead of letting candidate
    # *order* decide — see `screen`. Empty is the ordinary settled case.
    contested: tuple[str, ...] = ()
    # A SECOND, separate deliverable found in the same message. The first is
    # dispatched; this one is offered back as a question rather than silently
    # dropped or silently done. Verbatim from the user's message, so the
    # follow-up screens it exactly as it would a fresh request.
    additional_request: str = ""
    # A deterministic rule published with `action: escalate`. The request is
    # refused exactly as a block is, and additionally handed to a human
    # reviewer with the conversation held — for the requests where a refusal
    # alone under-reports what just happened: a bulk ask for every customer's
    # card data, a cross-tenant comparison. OWASP Agentic T10's "dynamic
    # intervention thresholds": low-risk refusals stay automated, the
    # high-risk ones are prioritised for a person.
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
    # `{agent_id: sdlc_reading}` for every candidate that claimed the request,
    # best first. One entry is the settled case; two or more that no confidence
    # gap separates is the contested case, and the readings are what the user is
    # asked to choose between — "a unit test" and "a functional test case" are
    # meaningful to them in a way two agent names are not.
    claims: tuple[tuple[str, str], ...] = ()


class GuardrailVerdict(BaseModel):
    """Structured output for the semantic tier.

    **Field order is load-bearing.** Structured output is generated in order, so
    `sdlc_reading` is answered before `in_domain` and the model has to articulate
    the in-domain interpretation before it can refuse one. That ordering is the
    mitigation for the failure this screen actually exhibited: "how to reset the
    password" was refused as an IT-support request, because the words carry
    strong support-desk associations and the model settled on that reading
    without considering the other one. Over-refusal research calls this lexical
    overfitting — the fix is to make the model reason explicitly rather than
    letting it answer the binary question straight from surface cues.
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
    # A separate field rather than a low confidence, because those are different
    # states and the model reports them differently. "Not sure whether this is
    # yours" is low confidence; "clearly about your subject, but I cannot tell
    # what they want produced" is a confident reading of an unclear request, and
    # it arrives with confidence 0.95. Asked to judge ambiguity explicitly,
    # models identify it 60-80% of the time; left to volunteer a clarifying
    # question instead of answering, under 5% do. So it is asked for.
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
    # Asked because a compound request is the ordinary way people write, and
    # answering only the first half is the failure that looks most like the
    # assistant ignoring you. Models volunteer a second task almost never when
    # left to; asked directly, they report it reliably — the same reason
    # `underspecified` is a field rather than a hope.
    #
    # Verbatim, because the follow-up re-enters the screen as a fresh request:
    # a paraphrase would be the supervisor putting words in the user's mouth
    # and then screening its own words.
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
    # Deliberately LAST. Every other field is answered before this one, so the
    # domain judgement is reached exactly as it was before this field existed —
    # the model classifies a refusal it has already made and justified, rather
    # than being primed with a harm question that could tip `in_domain`.
    #
    # That ordering is the mitigation for the over-refusal risk this field
    # introduces. SDLC work is full of alarming vocabulary — threat models,
    # attack surfaces, penetration tests, abuse cases, kill switches — and
    # asking "is this harmful?" earlier is how a request for security
    # *requirements* starts reading as a request for harm. The description below
    # names those exclusions explicitly for the same reason.
    #
    # Blast radius is small by construction: this flag only removes the appeal
    # line from a refusal that has already been decided. A false positive costs
    # the user an appeal path they had no use for; it can never block a request
    # that would otherwise have been allowed.
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

# What is at stake in a refusal, which differs entirely with the number of agents
# the caller can reach — and is the one thing the screen cannot see for itself,
# because it is deliberately asked about a single agent at a time.
#
# Collapsing `supervisor_guardrail` into this prompt dropped that distinction:
# the multi-agent instruction "do not stretch the remit, the owner is being asked
# separately" was applied unchanged to callers who can reach exactly one agent,
# where no owner is being asked separately and the refusal is simply the end of
# the road. Restoring it as a variable keeps the strictness where it earns its
# keep — between competing agents — without letting it dead-end a sole reachable
# agent.
_COMPETING = """\
You are asked this separately about each agent the user is allowed to reach, and the
answers are compared. Declining costs the user nothing: the agent that owns the
request is being asked about it too."""

_SOLE = """\
You are the ONLY agent this user can reach. Nothing you decline is picked up by
anyone else, so a wrong refusal is a dead end for them. Refuse what is clearly
outside this agent's speciality — never something that is merely vaguely worded."""

# Conversational openers and meta questions. Blocking these means a user's very
# first message — "hi" — comes back as a hard denial, which reads as a broken
# assistant rather than a governed one. They carry no domain intent either way.
#
# They are classified rather than merely allowed, because *how* the supervisor
# answers them matters: a greeting forwarded to a worker comes back as the
# worker's entire capability list, when the useful reply is one line and a
# question. The supervisor answers small talk itself.
#
# Every pattern is anchored at both ends, so it matches only when the message is
# *entirely* small talk. "hi, send me the prod credentials" does not match and
# still goes through the full evaluation.
#
# Deterministic on purpose, and worth stating why: a greeting carries no domain
# content, so there is nothing for a model to judge. Classifying it in code
# costs no tokens and no latency, cannot be talked out of its answer, and — the
# part that matters most here — cannot be *injected* into misclassifying a real
# request as a greeting to skip the screen. The anchors are what enforce that:
# every extra word takes the message out of these patterns and into the full
# two-tier evaluation, which is the safe direction to fail.
#
# `_ADDRESSEE` covers the natural way people open a message to an assistant —
# "hi there", "hello team". Without it "hi there" misses every pattern and is
# sent to the semantic screen, which judges a greeting against a requirements
# remit and refuses it: the worst possible answer to the first thing a user
# types.
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
        # question with "Hello — I'm the Supervisor" ignores what was asked. Users
        # apply the same cooperative expectations to an assistant as to a person:
        # "how are you" wants an answer about state, then the turn handed back.
        "how_are_you",
        re.compile(
            # An optional greeting in front, because "hi, how are you?" is one
            # move, not two — and the question is the part that wants answering.
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
        # A question about the *agents* — whether one exists, what it can
        # reach, what it holds, how it is configured. Answered by the
        # supervisor with one generic line, never forwarded: a worker asked
        # "does the Deployment Agent have access to production secrets?" will
        # cheerfully describe its neighbour's architecture, and RBAC only
        # gates which agent a role may *dispatch to*, not what a permitted
        # agent says *about* another. `rbac.DENIED_MESSAGE` already refuses
        # to reveal which agents exist on the denial path; this closes the
        # same disclosure on the answer path.
        #
        # Explicit question shapes rather than a proximity rule. A proximity
        # rule was tried and it swallowed real work: "agent" and "workflow" are
        # this product's own feature vocabulary, and `see`, `configured` and
        # `permissions` are ordinary requirement words, so "What are the user
        # stories for the agent permissions screen?" and "How do I configure
        # the deployment agent's access to the staging cluster?" were answered
        # with a canned non-disclosure line and never reached a worker. A
        # false positive here is not a refusal the user can appeal — it is a
        # silently unanswered request.
        #
        # Question shapes only, each asking about *the system* rather than
        # about a deliverable: whether an agent can reach something, what an
        # agent holds, and whether a restricted agent exists.
        # `_NAMES_A_DELIVERABLE` below vetoes every one of them.
        "meta_agents",
        re.compile(
            # A polite preamble is the normal way people ask this — "can
            # you tell me what data the Requirement Agent has stored" — and
            # without it the shape below misses every such question.
            r"^\W*(?:(?:can|could|would|will)\s+you\s+)?(?:please\s+)?(?:just\s+)?"
            r"(?:tell\s+me\s+|show\s+me\s+|let\s+me\s+know\s+|explain\s+|clarify\s+)?"
            r"(?:"
            # "does the Deployment Agent have access to production secrets"
            r"(?:does|do|can|could|is|are)\b[^.?!\n]{0,60}?\b(?:agent|workflow|assistant|bot|supervisor)s?\b"
            r"[^.?!\n]{0,40}?\b(?:have|has|get|hold|see|access|read|reach)\b"
            r"[^.?!\n]{0,40}?\b(?:access|secrets?|credentials?|permissions?|privileges?|"
            r"tokens?|keys?|production|prod\b|prompts?|memory|"
            r"my\s+(?:data|prompts?|messages?|history|context|files?|notes?|conversations?))"
            # "what data does the Requirement Agent have stored", and
            # "what permissions does the coding agent have". Both open on the
            # *thing being asked about*, which is what separates them from
            # "what are the user stories for the agent permissions screen".
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


# A message that asks for something to be *produced* is a request, whatever
# else it mentions. This vetoes `meta_agents` specifically: that kind answers
# with a fixed non-disclosure line and never dispatches, so a false positive
# costs the user their request with no refusal to appeal. "Can you draft an HLD
# for the assistant permissions model?" names a deliverable; "does the
# Deployment Agent have access to production secrets?" does not.
# Unambiguous production verbs and artifact nouns only. `review`, `test`,
# `design`, `document` and `describe` were tried and taken back out: they are
# as much *subject* vocabulary as request vocabulary in this domain, and
# "is there a 'legal review' agent…" was vetoed by the word "review" inside
# the agent's own name.
_NAMES_A_DELIVERABLE = re.compile(
    r"\b(?:write|draft|generate|produce|implement|refactor|spec(?:ify)?|"
    r"summari[sz]e|user\s+stor(?:y|ies)|test\s+cases?|test\s+plan|"
    r"acceptance\s+criteria|hld|lld|epic)\b",
    re.IGNORECASE,
)
# The kinds a deliverable veto applies to. The conversational kinds are
# anchored end to end, so nothing that names a deliverable can match them
# anyway; `meta_agents` is the one kind that matches a fragment of a sentence.
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
# Deterministic and anchored at both ends, for the same reasons `small_talk`
# patterns are: the answer carries no domain content, so there is nothing for a
# model to judge, and a message that is *entirely* an answer is the only thing
# that should be read as one. "yes, and also delete the staging database" is not
# a confirmation — every extra word takes the message out of these patterns and
# into the full screen, which is the safe direction to fail.
# A bare affirmative, or an affirmative lead-in followed by a "do it" phrase.
# Two arms rather than one so "yes" alone matches without also letting a bare
# verb phrase drift ("do the deployment" is a request, not a confirmation).
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
    """"accept" | "decline" | "" — is this message an answer to the offer?

    Anything else is a new request and is screened as one, which is what makes
    a held-over task impossible to trigger by accident: the offer expires the
    moment the user says something other than yes or no.
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

    Bounded and stripped here rather than trusted: the field is model-written
    from untrusted text, and it becomes the *query* of a follow-up turn if the
    user accepts it. Anything longer than a request is not a request.
    """
    found = (verdict.additional_request or "").strip()
    if len(found) < 8 or len(found) > 400:
        return ""
    return found


def _ownership_question(contested: list) -> str:
    """The one question that settles which agent a contested request belongs to.

    Phrased in deliverables, not agent names: "a unit test or a functional test
    case?" is answerable by someone who has never been told the agents apart,
    and "the Coding Agent or the Test Case Agent?" is not. Falls back to the
    names only when a reading is missing, which means the model returned an
    empty `sdlc_reading` and there is nothing better to offer.
    """
    readings: list[str] = []
    for agent, verdict in contested:
        reading = (verdict.sdlc_reading or "").strip().rstrip(".")
        readings.append(reading if reading and reading.lower() != "none" else agent.name)
    # Deduplicate while keeping order — two agents can name the same
    # deliverable, and "a unit test or a unit test?" helps nobody.
    seen: set[str] = set()
    unique = [r for r in readings if not (r.lower() in seen or seen.add(r.lower()))]
    if len(unique) < 2:
        unique = [agent.name for agent, _ in contested]
    joined = " or ".join((", ".join(unique[:-1]), unique[-1])) if len(unique) > 2 else " or ".join(unique)
    return f"More than one of your agents could take this — did you want {joined}?"


def _asks(verdict: "GuardrailVerdict") -> str:
    """The question to put to the user, when the verdict says one is needed.

    Gated on the flag rather than on the text being non-empty: a model that
    volunteers a question it did not ask for should not be able to turn a settled
    verdict into an interrogation.
    """
    return (verdict.clarification or "").strip() if verdict.underspecified else ""


# Shown when the kill switch is engaged and the operator supplied no message of
# their own. Worded as an operational hold, not a refusal — nothing was judged.
KILL_SWITCH_MESSAGE = (
    "The assistant is temporarily paused by the operations team. Nothing was sent "
    "to an agent — please try again later."
)


def _kill_switch_message(declared) -> str:
    """The operator's hold message, or "" when the switch is off.

    Accepts the two shapes the governed document may carry — `kill_switch: true`
    and `kill_switch: {enabled: true, message: "..."}` — so an emergency stop
    can be one line in a publish. Anything else (absent, false, enabled: false)
    means off, which is every existing document.
    """
    if declared is True:
        return KILL_SWITCH_MESSAGE
    if isinstance(declared, dict) and declared.get("enabled") is True:
        return str(declared.get("message") or "").strip() or KILL_SWITCH_MESSAGE
    return ""


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
        # Multi-model support: an optional `agent -> chat model`
        # resolver, so each candidate's verdict can run on the model its
        # registry entry names. None — every test and any direct caller —
        # means every verdict uses `llm`, exactly as before the seam existed.
        self._model_for = model_for
        self._rules = [
            (
                re.compile(rule["pattern"]),
                rule.get("reason", "matched a blocked pattern"),
                rule.get("action") == "escalate",
            )
            for rule in (global_deny_patterns or [])
        ]
        self._threshold = confidence_threshold
        # ── Two thresholds, because "confident enough to act on" and "confident
        # enough to stop asking" are different questions ─────────────────────
        #
        # The screen used to stop at the first verdict above `confidence_
        # threshold`, and `candidates` is ordered target-first — so which agent
        # got a request that two of them legitimately own was decided by which
        # chat widget the user happened to open, and the second agent was never
        # asked. "Write a unit test for this" is the standing example: the
        # Coding Agent's scope names unit tests and the Test Case Agent's names
        # test generation, both truthfully.
        #
        # Now the short-circuit needs `decisive_threshold`. Below it the
        # remaining candidates are asked, and if the top two land within
        # `contested_margin` of each other the user is asked which deliverable
        # they meant — Anthropic's own framing of the routing pattern is that it
        # works "where classification can be handled accurately", and the honest
        # reading of a 0.72-vs-0.71 split is that here it cannot.
        #
        # The cost is real and bounded: a decisive verdict still costs one call,
        # so the common case is unchanged; a borderline one costs up to one call
        # per reachable agent, stopped by the turn deadline and charged to the
        # turn's spend ledger as the exact number of verdicts asked for.
        self._decisive = max(confidence_threshold, decisive_threshold)
        self._contested_margin = max(0.0, contested_margin)
        # The emergency stop, carried on the engine because the engine is what
        # the `Reloading` proxy rebuilds when the governed guardrails document
        # changes — so flipping the switch is a config publish that reaches a
        # running endpoint within the config cache TTL, not a redeploy. Empty
        # string means off; non-empty is the message the user is shown.
        self.kill_switch_message = _kill_switch_message(kill_switch)

    @classmethod
    def from_yaml(
        cls,
        llm,
        path: Path,
        confidence_threshold: float = 0.7,
        model_for=None,
        decisive_threshold: float = 0.9,
        contested_margin: float = 0.15,
    ) -> "GuardrailEngine":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_mapping(
            llm,
            data,
            confidence_threshold,
            model_for=model_for,
            decisive_threshold=decisive_threshold,
            contested_margin=contested_margin,
        )

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
        """Build from an already-parsed document (file or governed table)."""
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

        Exposed so the supervisor can check a *held-over* request before
        offering to run it. Offering to do something the tier-1 rules would
        refuse, and only refusing once the user says yes, is a worse experience
        than not offering — and it costs a regex sweep rather than a model call
        to avoid.
        """
        for regex, reason, _escalate in self._rules:
            if regex.search(query or ""):
                return reason
        return ""

    def screen(
        self,
        query: str,
        candidates: list,
        history: Optional[list[str]] = None,
        deadline=None,
    ) -> "ScreenResult":
        """Which of these agents owns the query — and may it be sent at all?

        `candidates` is the caller's *permitted* agents, target first. Nothing
        outside that list is ever considered, so screening can neither surface
        nor route to an agent the caller has no access to.

        A confident in-domain verdict wins immediately, so the common case — the
        first candidate owns it — costs exactly one model call, as before. Only a
        query that does not obviously belong to the first agent pays for the
        others.

        `deadline` is the turn's remaining time budget (§05 Stage 05). This loop
        is the reason the budget exists: it makes up to one model call *per
        candidate agent*, so a "30-second timeout" is really a 30 x N-second
        node, and `graph.py` then retries the whole node three times. The
        deadline is checked before each verdict, so the fan-out stops instead of
        overrunning — see `deadline.py` for the arithmetic. None disables it,
        which is what every offline test passes.
        """
        history = history or []

        # Global deterministic rules run once, before any agent is considered: a
        # blocked pattern must not become shoppable by retrying it against the
        # next agent in the list.
        for regex, reason, escalate in self._rules:
            if regex.search(query):
                return ScreenResult(
                    None, GuardrailResult(False, "deterministic", reason, escalate=escalate), ()
                )

        kind = small_talk_kind(query)
        if kind:
            return ScreenResult(
                None,
                GuardrailResult(True, "small_talk", f"{kind}, no domain intent", small_talk=kind),
                (),
            )

        considered: list[str] = []
        in_domain: list[tuple[object, GuardrailVerdict]] = []
        unclear: list[tuple[object, GuardrailVerdict]] = []
        best_off: Optional[GuardrailVerdict] = None
        # Tracked across every candidate rather than read off `best_off`, which
        # is selected by confidence and could easily be a different agent's
        # verdict. Whether a request seeks harm is a property of the request, not
        # of whichever agent happened to answer most confidently about it — one
        # candidate recognising it is enough.
        flagged_unsafe = False
        sole = len(candidates) == 1
        # With one reachable agent there is no ordering to be misled by and
        # nothing to compare against, so the higher bar buys nothing and would
        # only relabel a settled verdict as a "best available match". The
        # decisive bar exists to stop candidate *order* deciding ownership; a
        # sole candidate has no rival.
        decisive = self._threshold if sole else self._decisive

        for agent in candidates:
            # An agent's own deny patterns rule *it* out, not the whole request:
            # another agent may legitimately handle the same topic.
            if any(re.search(p, query, flags=re.IGNORECASE) for p in agent.deny_patterns):
                continue

            # Before the call, not after: checked after, it would only report an
            # overrun that had already happened. A partial screen is never
            # allowed to become a pass — the exception propagates to the node,
            # which holds the request (§06 fail-closed).
            if deadline is not None:
                deadline.ensure(f"screening against {agent.id}")

            considered.append(agent.id)
            verdict = self._semantic_verdict(query, agent, history, sole=sole, deadline=deadline)
            flagged_unsafe = flagged_unsafe or verdict.safety_refusal

            if verdict.in_domain and verdict.confidence >= decisive:
                # Ownership settled beyond argument, so the remaining candidates
                # are not asked. Only a *decisive* verdict short-circuits: a
                # merely-above-threshold one leaves room for another agent to
                # own the request better, and skipping them there is how
                # candidate order decided ownership.
                #
                # The question rides along: "yes, this is my subject, but I
                # can't tell what you want produced" is a complete verdict, and
                # dropping its second half here is what sent an unclear request
                # to route to be asked which *product line* it was for.
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
                # feed `best_off` — an unclear request would otherwise accumulate
                # into the refusal that this branch exists to prevent.
                unclear.append((agent, verdict))
            elif best_off is None or verdict.confidence > best_off.confidence:
                best_off = verdict

        # Nobody was decisive. Rank what did claim it, and decide whether the
        # ranking means anything.
        if in_domain:
            ranked = sorted(in_domain, key=lambda pair: pair[1].confidence, reverse=True)
            agent, verdict = ranked[0]
            claims = tuple((a.id, v.sdlc_reading) for a, v in ranked)

            # Contested: the runner-up is within a margin of the winner, and
            # both cleared the acting threshold. Two agents that both genuinely
            # own a request is not a tie to be broken by rounding — the user is
            # the only one who knows which deliverable they wanted, and they can
            # answer in one word. Asked rather than guessed, and asked in terms
            # of the deliverables rather than the agent names.
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

        # Nobody claimed it outright, but an agent recognised the subject as its
        # own and only needs to know what is being asked for. That is a question,
        # not a refusal: "can you provide the password reset" is the requirements
        # agent's subject with its deliverable left unsaid, and blocking it tells
        # a user whose only reachable agent just declined their own topic to go
        # away. The first such agent is taken because `candidates` is ordered
        # target-first, so the question comes from the agent already in the
        # conversation.
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
                GuardrailResult(
                    False, "semantic", best_off.reason, safety_refusal=flagged_unsafe
                ),
                tuple(considered),
            )

        # Nothing was evaluated at all, so every candidate was ruled out by its
        # own deny patterns. Saying "no agent covers this topic" would contradict
        # the refusal's own "your role covers …" line — the agents exist, they
        # are barred from this subject.
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
        content is a JSON user turn. See `prompt_provider.untrusted_turn`."""
        # Content blocks, not one string, so the invariant rules can be cached.
        # This loop asks the same ~1100 tokens of rules once per candidate
        # agent, so the reuse that makes caching pay happens inside a single
        # turn — it does not depend on request volume or on the cache surviving
        # between turns. See `prompt_provider.system_blocks`, and note that
        # `ChatDatabricks` does not report the cache counters back.
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
        # The deadline gates *whether* this call runs (checked by the caller);
        # the call's own ceiling stays `routing_llm_timeout_seconds`. Rebinding
        # the timeout per call would defeat `model_provider`'s client cache and
        # rebuild an HTTP client on every verdict, so the guarantee is stated
        # rather than over-engineered: worst case is the budget plus one
        # in-flight call, which is why the budget sits well under the gateway's
        # bound. See `deadline.py` and `settings.turn_budget_seconds`.
        #
        # Retried here, per verdict, because this is the one place a transient
        # 429/5xx can be recovered without re-screening every candidate — see
        # `resilience.py` for why the graph-level RetryPolicy never did this.
        llm = self._model_for(agent) if self._model_for is not None else self._llm
        return invoke_with_retries(
            lambda: llm.with_structured_output(GuardrailVerdict).invoke(
                [SystemMessage(content=rules), HumanMessage(content=payload)]
            ),
            what=f"the domain screen for {agent.id}",
            deadline=deadline,
        )
