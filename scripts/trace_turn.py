"""Show the whole flow of a supervisor turn — every stage, prompt and response.

This answers "what actually happened on that request?": which agent was chosen
and why, what each governance model was sent verbatim, what it returned, what
the worker produced, and what landed in the decision trail.

Two modes.

**Run a turn locally** — the graph in this working tree, real routing model,
simulated workers:

    python scripts/trace_turn.py "Write acceptance criteria for the billing epic"
    python scripts/trace_turn.py "Draft an HLD" "product line alpha"   # two turns, one thread
    python scripts/trace_turn.py "Review this function" --agent coding-agent --role Developer
    python scripts/trace_turn.py "Refactor this" --agent coding-agent --role BA \
        --permitted requirement-agent                                  # an RBAC denial

**Replay a trace that already happened** — including one from the deployed
endpoint, so a user's real request can be inspected after the fact:

    python scripts/trace_turn.py --replay last --tracking databricks
    python scripts/trace_turn.py --replay last:5 --tracking databricks
    python scripts/trace_turn.py --replay tr-5c7ef15df0e02fe518832d9cf5b4286b --tracking databricks

Both modes print the same waterfall, because both read MLflow spans. The
deployed endpoint traces because `src/supervisor/agent.py` calls
`mlflow.langchain.autolog()`; a local run traces because this script makes the
same call.

Where the spans live:

    --tracking local        .mlflow-local/traces.db, browsable with
                            `mlflow ui --backend-store-uri sqlite:///…`
    --tracking databricks   the workspace experiment (default
                            /Shared/supervisor-agent) — the same traces the
                            Databricks Experiments UI shows

Things worth knowing, each of which cost real debugging time:

  * **Async trace export must be off to read a trace back in-process.** With
    MLflow's default async queue, `get_trace()` straight after a run reports
    "span data is corrupted" because the write has not landed. Pinned below.
  * **MLflow's `input_tokens` and cost are inflated on a streamed call** — it
    sums a figure every chunk repeats. `total_tokens` stays right, so input is
    derived from it. See `token_usage`.
  * **Workers are simulated by default**, matching the `dev` bundle target,
    because the worker endpoints do not exist yet. `--live-workers` when they do.
  * **The prompt source is printed per prompt**, registry or bundled fallback. A
    silent fallback is the failure mode §4.1 is most exposed to.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import textwrap
from pathlib import Path


def _repo_root() -> Path:
    try:
        here = Path(__file__)
    except NameError:  # spark_python_task execs the source; no __file__
        import inspect

        here = Path(inspect.currentframe().f_code.co_filename)
    return here.resolve().parents[1]


ROOT = _repo_root()
sys.path.insert(0, str(ROOT / "src"))

W = 96  # rule width; narrow enough to survive a split pane

# The graph's nodes. Used to attribute a model call to a governance stage and to
# pick the node spans out of a trace.
NODES = (
    "rbac_gate",
    "guardrails",
    "route",
    "dispatch",
    "approval",
    "respond",
)


# ── output helpers ──────────────────────────────────────────────────────────


def rule(title: str = "", char: str = "─") -> None:
    if not title:
        print(char * W)
        return
    print(f"{char * 3} {title} " + char * max(0, W - len(title) - 5))


def banner(title: str) -> None:
    print()
    rule(title, "━")


def block(text: str, indent: str = "  │ ", limit: int = 0) -> None:
    """Print a multi-line payload, indented and optionally truncated."""
    text = text if isinstance(text, str) else str(text)
    if limit and len(text) > limit:
        text = text[:limit] + f"\n… [{len(text) - limit} more chars — raise --max-chars]"
    for line in text.splitlines() or [""]:
        for wrapped in textwrap.wrap(line, W - len(indent)) or [""]:
            print(f"{indent}{wrapped}")


def as_text(value) -> str:
    """Render a span payload the way a human wants to read it.

    Prompts arrive as a bare string or as a `messages` list; both should print
    as the prompt text, not as escaped JSON.
    """
    if value is None:
        return "(none)"
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        messages = value.get("messages")
        if isinstance(messages, list):
            parts = []
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role") or msg.get("type") or "?"
                    content = msg.get("content")
                    if isinstance(content, list):
                        content = "".join(
                            c.get("text", "") for c in content if isinstance(c, dict)
                        )
                    parts.append(f"[{role}]\n{content}")
            if parts:
                return "\n\n".join(parts)
    return json.dumps(value, indent=2, default=str, ensure_ascii=False)


def structured_decision(outputs) -> str:
    """The governance verdict itself, out of the chat-completion envelope.

    Every governance stage asks for structured output, so the model replies with
    a tool call rather than prose. The arguments are the decision; the
    surrounding `choices[0].message.tool_calls[0]` scaffolding is noise.
    """
    if not isinstance(outputs, dict):
        return ""
    choices = outputs.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = (choices[0] or {}).get("message") or {}
    rendered = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        arguments = function.get("arguments", "")
        try:
            arguments = json.dumps(json.loads(arguments), indent=2, ensure_ascii=False)
        except Exception:
            pass
        rendered.append(f"{function.get('name', '?')} ->\n{arguments}")
    return "\n".join(rendered)


def plain_content(outputs) -> str:
    """The assistant's prose out of a chat-completion envelope.

    The dispatch stage is the one call that returns text rather than a verdict —
    the worker's answer, or the simulated stand-in for it.
    """
    if not isinstance(outputs, dict):
        return ""
    choices = outputs.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    content = ((choices[0] or {}).get("message") or {}).get("content")
    if isinstance(content, list):
        content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
    return content if isinstance(content, str) else ""


def token_usage(attrs: dict) -> tuple[dict, str]:
    """Trustworthy token counts, plus a note when MLflow's differ.

    MLflow sums `input_tokens` across streamed chunks, and every chunk carries
    the *cumulative* prompt count — so on a streamed call the input figure comes
    out multiplied by the chunk count, and `mlflow.llm.cost` inherits the error.
    `total_tokens` and `output_tokens` stay correct, so input is derived from
    those. The graph streams (a chat UI needs `messages` mode), so the deployed
    endpoint's traces carry the same inflation.
    """
    usage = dict(attrs.get("mlflow.chat.tokenUsage") or {})
    if not usage:
        return {}, ""

    total = usage.get("total_tokens")
    output = usage.get("output_tokens")
    reported = usage.get("input_tokens")
    if not isinstance(total, int) or not isinstance(output, int):
        return usage, ""

    derived = total - output
    note = ""
    if isinstance(reported, int) and derived > 0 and reported > derived * 1.5:
        note = (
            f"MLflow reported input_tokens={reported} ({reported / derived:.0f}x) — "
            "summed over streamed chunks; its cost figure is inflated by the same factor"
        )
        usage["input_tokens"] = derived
    return usage, note


# ── trace rendering, shared by both modes ───────────────────────────────────


def span_index(trace):
    """Spans in start order, with a parent lookup and a stage resolver."""
    # `trace.data.spans` is in completion order, which puts a child before its
    # parent. Sorting by start time makes the stages read top to bottom.
    spans = sorted(trace.data.spans, key=lambda s: s.start_time_ns)
    by_id = {s.span_id: s for s in spans}

    def stage_of(span) -> str:
        cursor = span
        while cursor is not None:
            if cursor.name in NODES:
                return cursor.name
            cursor = by_id.get(cursor.parent_id)
        return "-"

    return spans, stage_of


def ms_of(span) -> float:
    return (span.end_time_ns - span.start_time_ns) / 1e6


def render_llm_calls(spans, stage_of, limit: int) -> list[dict]:
    """Every model call: the prompt as sent, the verdict as returned."""
    records: list[dict] = []
    llm_spans = [s for s in spans if str(s.span_type) in ("CHAT_MODEL", "LLM")]

    if not llm_spans:
        print("  No model call on this turn — the stage that ended it is")
        print("  deterministic (RBAC denial, tier-1 guardrail, or small talk).")
        return records

    for span in llm_spans:
        attrs = span.attributes or {}
        usage, usage_note = token_usage(attrs)
        stage = stage_of(span)
        record = {
            "stage": stage,
            "model": attrs.get("mlflow.llm.model", ""),
            "provider": attrs.get("mlflow.llm.provider", ""),
            "ms": round(ms_of(span)),
            "tokens": usage,
            "token_note": usage_note,
            "prompt": as_text(span.inputs),
            "verdict": structured_decision(span.outputs),
        }
        records.append(record)

        rule()
        print(f"  stage    : {stage}")
        print(f"  model    : {record['provider']}/{record['model']}")
        print(
            "  tokens   : "
            + (
                f"in {usage.get('input_tokens', '?')} / out {usage.get('output_tokens', '?')}"
                f" / total {usage.get('total_tokens', '?')}"
                if usage
                else "not reported"
            )
        )
        if usage_note:
            print(f"  ⚠ tokens : {usage_note}")
        print(f"  latency  : {record['ms']} ms")

        # The tool schema is how a structured verdict is actually requested.
        tools = attrs.get("mlflow.chat.tools")
        names = [
            (t.get("function", t) or {}).get("name")
            for t in (tools if isinstance(tools, list) else [])
            if isinstance(t, dict)
        ]
        names = [n for n in names if n]
        if names:
            print(f"  schema   : {', '.join(names)}  (structured output)")

        print("  ── PROMPT SENT ──")
        block(record["prompt"], limit=limit)
        if record["verdict"]:
            print("  ── VERDICT RETURNED (structured output) ──")
            block(record["verdict"], limit=limit)
        else:
            print("  ── RESPONSE ──")
            prose = plain_content(span.outputs)
            block(prose or as_text(span.outputs), limit=limit)

    return records


def render_graph_path(spans, stage_of) -> None:
    """The node path, with model time separated from everything else."""
    print(f"  {'node':<14} {'total':>8} {'model':>8} {'other':>8}  next           state keys set")
    slow: list[str] = []

    for span in spans:
        if span.name not in NODES:
            continue
        total = ms_of(span)
        model = sum(
            ms_of(s)
            for s in spans
            if str(s.span_type) in ("CHAT_MODEL", "LLM") and stage_of(s) == span.name
        )
        outputs = span.outputs if isinstance(span.outputs, dict) else {}
        update = outputs.get("update")
        goto = outputs.get("goto")
        changed = ", ".join(sorted(update)) if isinstance(update, dict) else "-"
        print(
            f"  {span.name:<14} {total:>6.0f}ms {model:>6.0f}ms {total - model:>6.0f}ms"
            f"  {str(goto or 'END'):<14} {changed[:33]}"
        )
        if total - model > 2000:
            slow.append(span.name)

    if slow:
        print()
        print("  ⚠ Non-model time above 2s in: " + ", ".join(slow))
        print("    With the prompt registry refused, `get_prompt` retries the Unity")
        print("    Catalog lookup before falling back — see the prompt-source section.")
        print("    That retry is paid once per prompt per process, then suppressed for")
        print("    PROMPT_FAILURE_TTL_SECONDS (default 300).")


def state_from_spans(spans) -> dict:
    """Reconstruct the final graph state from the node spans.

    Replay has no live `updates` channel, so the state is rebuilt by applying
    each node's `update` in order — which is what LangGraph itself did.
    """
    state: dict = {}
    for span in spans:
        if span.name not in NODES or not isinstance(span.outputs, dict):
            continue
        update = span.outputs.get("update")
        if isinstance(update, dict):
            state.update(update)
        elif span.name == "respond":
            # `respond` returns a plain dict, not a Command.
            state.update({k: v for k, v in span.outputs.items() if k != "messages"})
    # The inputs to the last node carry the accumulated state, which fills in
    # anything a node read but never rewrote.
    for span in reversed(spans):
        if span.name in NODES and isinstance(span.inputs, dict):
            for key, value in span.inputs.items():
                state.setdefault(key, value)
            break
    return state


def render_decision_trail(trail: list) -> None:
    for entry in trail or []:
        print(
            f"    {str(entry.get('ts', ''))[:19]}  {entry.get('stage', ''):<13}"
            f" {entry.get('decision', ''):<10} {str(entry.get('detail', ''))[:58]}"
        )


# ── prompt provenance ───────────────────────────────────────────────────────


class PromptSourceCapture(logging.Handler):
    """Records whether each prompt came from the registry or the fallback.

    `prompt_provider` logs the registry hit at INFO and the fallback at WARNING.
    Reading those records reports the prompt source without a second MLflow call.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.events: list[tuple[str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:
            return
        if "loaded prompt" in message or "prompt registry unavailable" in message:
            self.events.append((record.levelname, message))


def load_dotenv(path: Path) -> list[str]:
    """Minimal .env reader — Databricks auth for the routing endpoint.

    Existing environment variables win, so an explicitly exported value is not
    silently overridden by the file.
    """
    loaded = []
    if not path.exists():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


# ── main ────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show the whole flow of a supervisor turn.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "message", nargs="*", help="one message per turn, same thread (omit with --replay)"
    )
    parser.add_argument(
        "--replay",
        default="",
        metavar="TRACE",
        help="render an existing trace instead of running one: a trace id, "
        "'last', or 'last:N' for the newest N",
    )
    parser.add_argument("--role", default="BA", help="persona, for display and audit")
    parser.add_argument(
        "--agent",
        default="requirement-agent",
        help="target agent id — the widget the turn is scoped to",
    )
    parser.add_argument(
        "--permitted",
        default="",
        help="comma-separated permitted agent ids (what the Platform API derives "
        "from the token). Defaults to every registered agent",
    )
    parser.add_argument("--thread", default="trace-1", help="conversation/thread id")
    parser.add_argument(
        "--approve",
        choices=("approved", "rejected"),
        help="answer a pending approval interrupt after the last turn",
    )
    parser.add_argument(
        "--tracking",
        choices=("local", "databricks"),
        default="local",
        help="local SQLite store (default) or the workspace MLflow experiment",
    )
    parser.add_argument(
        "--experiment",
        default=os.getenv("MLFLOW_EXPERIMENT", "/Shared/supervisor-agent"),
        help="experiment name when --tracking databricks",
    )
    parser.add_argument(
        "--live-workers",
        action="store_true",
        help="call the real worker Model Serving endpoints instead of simulating",
    )
    parser.add_argument(
        "--no-registry",
        action="store_true",
        help="skip MLflow Prompt Registry and use the bundled templates",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=4000,
        help="truncate each payload at this many characters (0 = unlimited)",
    )
    parser.add_argument(
        "--json", dest="json_out", default="", help="also write the trace to this file as JSON"
    )
    parser.add_argument(
        "--quiet-logs", action="store_true", help="hide the library log lines during the run"
    )
    args = parser.parse_args()

    if not args.message and not args.replay:
        parser.error("give a message to run, or --replay to render an existing trace")

    # ── environment ─────────────────────────────────────────────────────────
    loaded = load_dotenv(ROOT / ".env")
    if os.getenv("DATABRICKS_CLIENT_ID") and os.getenv("DATABRICKS_CLIENT_SECRET"):
        # An OAuth M2M service principal is non-interactive; a CLI profile may
        # want a browser, which a script cannot provide.
        os.environ.setdefault("DATABRICKS_AUTH_TYPE", "oauth-m2m")
    if not args.live_workers:
        os.environ["SUPERVISOR_MOCK_WORKERS"] = "true"
    if args.no_registry:
        os.environ["PROMPT_REGISTRY_ENABLED"] = "false"
    # Reading a trace back in the same process needs the synchronous exporter;
    # see the module docstring.
    os.environ["MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"] = "false"

    logging.basicConfig(
        level=logging.WARNING if args.quiet_logs else logging.INFO,
        format="  · %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    prompts = PromptSourceCapture()
    prompt_logger = logging.getLogger("supervisor.prompt_provider")
    # Capture regardless of the console level; with --quiet-logs keep the records
    # off the console so a multi-line REST error does not break up the stream.
    prompt_logger.setLevel(logging.INFO)
    prompt_logger.addHandler(prompts)
    if args.quiet_logs:
        prompt_logger.propagate = False

    import mlflow

    if args.tracking == "databricks":
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment(args.experiment)
        store_hint = f"Databricks Experiments UI → {args.experiment}"
    else:
        store = ROOT / ".mlflow-local"
        store.mkdir(exist_ok=True)
        db = (store / "traces.db").as_posix()
        mlflow.set_tracking_uri(f"sqlite:///{db}")
        mlflow.set_experiment("supervisor-local-traces")
        store_hint = f"mlflow ui --backend-store-uri sqlite:///{db}"

    limit = args.max_chars or 0

    # ── replay mode ─────────────────────────────────────────────────────────
    if args.replay:
        return replay(mlflow, args, limit, store_hint)

    # ── run mode ────────────────────────────────────────────────────────────
    # The same call the deployed entrypoint makes (`src/supervisor/agent.py`), so
    # a local trace has the same shape as a production one.
    mlflow.langchain.autolog()

    from supervisor.context import SupervisorContext
    from supervisor.graph import build_graph
    from supervisor.prompt_provider import prompt_names, prompt_uri
    from supervisor.registry import AgentRegistry
    from supervisor.settings import Settings

    settings = Settings()
    graph = build_graph()
    registry = AgentRegistry.from_yaml(settings.agents_config)
    permitted = (
        tuple(a.strip() for a in args.permitted.split(",") if a.strip())
        if args.permitted
        else tuple(registry.ids())
    )

    context = SupervisorContext(
        user_role=args.role,
        user_key=f"usr_local_{args.role.lower().replace(' ', '_')}",
        permitted_agents=permitted,
        requested_agent_id=args.agent,
        correlation_id=f"corr-local-{args.thread}",
        environment=settings.environment,
    )
    config = {"configurable": {"thread_id": args.thread}}

    # ── 1. what the agent receives ──────────────────────────────────────────
    banner("1. INBOUND — what the endpoint receives")
    print("The Platform API posts this to the serving endpoint's ResponsesAgent:")
    print()
    block(
        json.dumps(
            {
                "input": [{"role": "user", "content": args.message[0]}],
                "custom_inputs": {
                    "agent_id": args.agent,
                    "user_role": args.role,
                    "permitted_agents": list(permitted),
                    "conversation_id": args.thread,
                    "user_id": context.user_key,
                    "correlation_id": context.correlation_id,
                },
            },
            indent=2,
        )
    )
    print()
    print("  Split by `agent.py:_prepare`:")
    print("    state    (checkpointed)  -> messages, conversation_id")
    print("    context  (per-run only)  -> user_role, user_key, permitted_agents,")
    print("                                requested_agent_id, correlation_id")
    print(f"    config                   -> thread_id={args.thread!r}")
    print()
    print(
        f"  Routing model : {settings.routing_llm_provider}:{settings.routing_llm_endpoint}"
        f" (temp {settings.routing_temperature})"
    )
    print(
        "  Workers       : "
        + (
            "SIMULATED (routing model stands in)"
            if settings.mock_workers
            else "LIVE Model Serving endpoints"
        )
    )
    print(
        "  Prompt source : "
        + ("bundled defaults (registry disabled)" if args.no_registry else prompt_uri("<name>"))
    )
    if loaded:
        print(f"  Loaded from .env: {', '.join(loaded)}")

    # ── 2. run the turns, streaming ─────────────────────────────────────────
    from langgraph.types import Command

    payloads: list = [
        {"messages": [{"role": "user", "content": m}], "conversation_id": args.thread}
        for m in args.message
    ]
    if args.approve:
        payloads.append(Command(resume={"decision": args.approve}))

    turns: list[dict] = []
    trace_ids: list[str] = []

    for index, payload in enumerate(payloads, start=1):
        label = (
            f"resume: {args.approve}"
            if not isinstance(payload, dict)
            else payload["messages"][0]["content"]
        )
        banner(f"2.{index} LIVE STREAM — turn {index}: {label!r}")
        print("  What the caller sees while the graph runs (progress + tokens):")
        print()

        progress_seen: list[dict] = []
        deltas: list[str] = []
        final_state: dict = {}

        with mlflow.start_span(name=f"turn-{index}") as span:
            trace_ids.append(span.trace_id)
            for mode, chunk in graph.stream(
                payload,
                config=config,
                context=context,
                stream_mode=["custom", "messages", "updates"],
            ):
                if mode == "custom" and isinstance(chunk, dict):
                    if chunk.get("channel") == "progress":
                        progress_seen.append(chunk)
                        mark = {
                            "started": "…",
                            "done": "✓",
                            "blocked": "✗",
                            "clarify": "?",
                            "error": "!",
                        }.get(chunk.get("status", ""), "·")
                        print(
                            f"    {mark} {chunk.get('label', ''):<26} "
                            f"{chunk.get('status', ''):<8} {str(chunk.get('detail') or '')[:44]}"
                        )
                    elif chunk.get("channel") == "sources":
                        titles = [
                            s.get("title") for s in (chunk.get("items") or []) if isinstance(s, dict)
                        ]
                        print(f"    ⌸ sources: {titles}")
                elif mode == "messages":
                    message, metadata = chunk
                    if (metadata or {}).get("langgraph_node") == "dispatch":
                        text = getattr(message, "content", "") or ""
                        if isinstance(text, list):
                            text = "".join(
                                p.get("text", "") for p in text if isinstance(p, dict)
                            )
                        if text:
                            deltas.append(text)
                elif mode == "updates":
                    for node_state in (chunk or {}).values():
                        if isinstance(node_state, dict):
                            final_state.update(node_state)

        if deltas:
            print(f"    ▸ {len(deltas)} answer token chunk(s) streamed from `dispatch`")
        turns.append(
            {
                "turn": index,
                "input": label,
                "progress": progress_seen,
                "state": final_state,
                "streamed_chars": sum(len(d) for d in deltas),
            }
        )

    # ── 3 & 4. per-turn spans ───────────────────────────────────────────────
    all_llm: list[dict] = []
    for index, trace_id in enumerate(trace_ids, start=1):
        trace = mlflow.get_trace(trace_id)
        if trace is None:
            print(f"\n  (trace {trace_id} not retrievable — was the export flushed?)")
            continue
        spans, stage_of = span_index(trace)

        banner(f"3.{index} LLM CALLS — turn {index}: exactly what each model received")
        all_llm += render_llm_calls(spans, stage_of, limit)

        banner(f"4.{index} GRAPH PATH — turn {index}")
        render_graph_path(spans, stage_of)

    # ── 5. decision trail ───────────────────────────────────────────────────
    banner("5. DECISION TRAIL — the audit record (BR-006)")
    last = turns[-1]["state"] if turns else {}
    for turn in turns:
        if turn["state"].get("audit_trail"):
            print(f"  turn {turn['turn']}:")
            render_decision_trail(turn["state"]["audit_trail"])
    print()
    print("  Written by the `respond` node as one row:")
    block(
        json.dumps(
            {
                "request_id": last.get("request_id", ""),
                "conversation_id": args.thread,
                "correlation_id": context.correlation_id,
                "user_role": context.user_role,
                "user_key": context.user_key,
                "target_agent_id": last.get("target_agent_id", ""),
                "outcome": last.get("outcome", ""),
                "decision_trail": "[…]",
            },
            indent=2,
        )
    )
    from supervisor.audit import build_audit_logger

    print(f"  sink: {type(build_audit_logger(settings)).__name__}")

    # ── 6. prompt provenance ────────────────────────────────────────────────
    banner("6. PROMPT SOURCE — registry or bundled fallback (§4.1)")
    if args.no_registry:
        print("  Registry disabled by --no-registry; bundled templates used.")
    elif not prompts.events and not all_llm:
        print("  No prompt was needed — the turn ended at a deterministic stage")
        print("  before any model-backed stage ran.")
    elif not prompts.events:
        print("  No load logged this run — MLflow's own prompt cache served the")
        print("  templates (60s for an alias URI, indefinite for a pinned version).")
    for level, message in prompts.events:
        print(f"  {'✓ REGISTRY ' if level == 'INFO' else '✗ FALLBACK '}{message}")
    print()
    for name in prompt_names():
        print(f"  {name:<24} {prompt_uri(name)}")

    # ── 7. the response ─────────────────────────────────────────────────────
    banner("7. RESPONSE — what the caller renders")
    render_response(last, limit)

    # ── 8. totals ───────────────────────────────────────────────────────────
    banner("8. TOTALS")
    render_totals(all_llm, trace_ids, store_hint)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "context": {
                        "role": args.role,
                        "requested_agent": args.agent,
                        "permitted_agents": list(permitted),
                        "thread": args.thread,
                        "routing_model": f"{settings.routing_llm_provider}:{settings.routing_llm_endpoint}",
                    },
                    "turns": turns,
                    "llm_calls": all_llm,
                    "prompt_events": prompts.events,
                    "trace_ids": trace_ids,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"  JSON written to {args.json_out}")

    return 0


def render_response(state: dict, limit: int) -> None:
    print(f"  outcome           : {state.get('outcome', '')}")
    print(
        f"  routed agent      : {state.get('routed_agent_name', '') or '-'}"
        f" ({state.get('target_agent_id', '') or '-'})"
    )
    if state.get("pending_clarification"):
        print(f"  awaiting reply to : {state['pending_clarification']}")
    if state.get("pending_approval"):
        print(
            f"  awaiting approval : stage {state['pending_approval'].get('stage')}"
            " — re-run with --approve approved"
        )
    print()
    block(state.get("final_text", ""), limit=limit)


def render_totals(all_llm: list[dict], trace_ids: list[str], store_hint: str) -> None:
    total_in = sum((r["tokens"] or {}).get("input_tokens", 0) for r in all_llm)
    total_out = sum((r["tokens"] or {}).get("output_tokens", 0) for r in all_llm)
    print(f"  model calls : {len(all_llm)}")
    print(f"  tokens      : in {total_in} / out {total_out}")
    print(f"  model time  : {sum(r['ms'] for r in all_llm)} ms")
    if any(r["token_note"] for r in all_llm):
        print("  note        : MLflow's own input-token and cost figures are inflated on")
        print("                streamed calls; the counts above are corrected from")
        print("                total_tokens - output_tokens. Do not build usage")
        print("                reporting (BR-006) on the raw span cost attribute.")
    print(f"  trace ids   : {', '.join(trace_ids)}")
    print()
    print(f"  Browse the same spans in:\n    {store_hint}")


def replay(mlflow, args, limit: int, store_hint: str) -> int:
    """Render traces that already happened, local or from the workspace."""
    selector = args.replay.strip()

    if selector.startswith("last"):
        _, _, count = selector.partition(":")
        try:
            wanted = max(1, int(count)) if count else 1
        except ValueError:
            wanted = 1
        found = mlflow.search_traces(max_results=wanted, return_type="list")
        traces = list(found or [])
        if not traces:
            print(f"No traces found in {store_hint}.")
            print("Send a request through the calling UI (or run a turn locally) first.")
            return 1
    else:
        trace = mlflow.get_trace(selector)
        if trace is None:
            print(f"Trace {selector} not found in {store_hint}.")
            return 1
        traces = [trace]

    banner(f"REPLAY — {len(traces)} trace(s) from {store_hint}")

    all_llm: list[dict] = []
    trace_ids: list[str] = []

    for index, trace in enumerate(traces, start=1):
        spans, stage_of = span_index(trace)
        trace_ids.append(trace.info.trace_id)
        state = state_from_spans(spans)

        # `predict_stream` / `predict` is the endpoint entrypoint; its inputs are
        # the request as it arrived, custom_inputs included.
        entry = next(
            (s for s in spans if s.name in ("predict_stream", "predict")), None
        )

        banner(f"{index}. TRACE {trace.info.trace_id}")
        print(f"  state     : {trace.info.state}")
        print(f"  ms        : {getattr(trace.info, 'execution_duration', '') or ''}")
        if entry is not None:
            print(f"  entrypoint: {entry.name}  ({ms_of(entry):.0f} ms)")
            print("  ── REQUEST AS RECEIVED ──")
            block(as_text(entry.inputs), limit=limit)

        banner(f"{index}.1 LLM CALLS — exactly what each model received")
        all_llm += render_llm_calls(spans, stage_of, limit)

        banner(f"{index}.2 GRAPH PATH")
        render_graph_path(spans, stage_of)

        banner(f"{index}.3 DECISION TRAIL")
        trail = state.get("audit_trail")
        if trail:
            render_decision_trail(trail)
        else:
            print("  (no audit trail in this trace's node spans)")

        banner(f"{index}.4 RESPONSE")
        render_response(state, limit)

    banner("TOTALS")
    render_totals(all_llm, trace_ids, store_hint)
    return 0


if __name__ == "__main__":
    # Windows consoles default to a codepage that cannot encode the box-drawing
    # characters above; force UTF-8 rather than crash on the first rule.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    _exit_code = main()
    if _exit_code:
        sys.exit(_exit_code)
