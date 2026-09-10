"""Does the deployed endpoint actually stream? Ask it and count what arrives.

    python scripts/verify_stream.py
    python scripts/verify_stream.py "What is a good recipe for carbonara?"

Calls the serving endpoint directly with `{"stream": true}` and tallies the
three channels a chat UI depends on:

    progress   the governance task plan, stage by stage
    deltas     answer tokens, as the worker writes them
    sources    what the answer was grounded on

A governed refusal is a *successful* run with zero deltas — the request never
reached a model. So read the two numbers together: progress > 0 with deltas = 0
and outcome=blocked is the guardrail working, not a broken stream.

This authenticates as **you**, not as the platform service principal, so it
still passes when the caller is getting 502s for want of a CAN QUERY grant.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import _environment  # noqa: E402  — needs the path insert above
from trace_turn import load_dotenv  # noqa: E402

DEFAULT_QUESTION = "Write an HLD for the billing service on product line alpha"


def main() -> int:
    # The SDK does not read `.env` (the CLI finds credentials on its own), so
    # without this the script dies on "cannot configure default credentials" in
    # any shell that has not exported them by hand. Loaded before the parser is
    # built, because `--environment` defaults to $ENVIRONMENT.
    load_dotenv(ROOT / ".env")
    if os.getenv("DATABRICKS_CLIENT_ID") and os.getenv("DATABRICKS_CLIENT_SECRET"):
        os.environ.setdefault("DATABRICKS_AUTH_TYPE", "oauth-m2m")

    parser = argparse.ArgumentParser()
    _environment.add_arguments(parser)
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--endpoint", default="", help="override the derived endpoint name")
    parser.add_argument("--role", default="Engineering Manager")
    parser.add_argument(
        "--agent",
        default="requirement-agent",
        help="target agent id — the widget the turn is scoped to",
    )
    parser.add_argument(
        "--permitted",
        default="requirement-agent,deployment-agent",
        help="comma-separated agent ids the caller may use",
    )
    parser.add_argument(
        "--conversation",
        default="",
        help=(
            "thread id. Defaults to a fresh one per run — a fixed id accumulates "
            "history in the checkpoint, and an earlier turn can change how this "
            "one is routed. Pass one explicitly to test a multi-turn flow."
        ),
    )
    args = parser.parse_args()

    conversation = args.conversation or f"verify-stream-{uuid.uuid4().hex[:8]}"
    endpoint = args.endpoint or _environment.resolve(args).endpoint

    import requests
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    url = f"{w.config.host.rstrip('/')}/serving-endpoints/{endpoint}/invocations"
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    headers.update(w.config.authenticate() or {})

    payload = {
        "input": [{"role": "user", "content": args.question}],
        "custom_inputs": {
            "agent_id": args.agent,
            "user_role": args.role,
            "user_id": "usr_verify_stream",
            "permitted_agents": [a for a in args.permitted.split(",") if a],
            "conversation_id": conversation,
        },
        "stream": True,
    }

    print(f"> {args.question}")
    print(f"  thread: {conversation}\n")

    progress: list[dict] = []
    deltas: list[str] = []
    sources: list[dict] = []
    final: dict = {}

    with requests.post(url, headers=headers, json=payload, stream=True, timeout=300) as response:
        print(f"HTTP {response.status_code}  {response.headers.get('content-type')}\n")
        if response.status_code != 200:
            print(response.text[:600])
            return 1

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                event = json.loads(body)
            except json.JSONDecodeError:
                continue

            kind = event.get("type", "")
            if kind == "response.custom":
                outputs = event.get("custom_outputs") or {}
                if "progress" in outputs:
                    entry = outputs["progress"]
                    progress.append(entry)
                    detail = f" — {entry['detail']}" if entry.get("detail") else ""
                    print(f"  [progress] {entry['status']:8} {entry['label']}{detail}")
                if "sources" in outputs:
                    sources = outputs["sources"]
            elif kind == "response.output_text.delta":
                deltas.append(event.get("delta", ""))
            elif kind == "response.output_item.done":
                final = event.get("custom_outputs") or {}

    print()
    print("=" * 68)
    print(f"progress events : {len(progress)}")
    print(f"token deltas    : {len(deltas)}")
    print(f"sources         : {[s.get('title') for s in sources]}")
    print(f"outcome         : {final.get('outcome')}")
    print(f"routed to       : {final.get('routed_agent_name')}")

    answer = "".join(deltas)
    if answer:
        print()
        print("first 200 chars of the streamed answer:")
        print(" ", " ".join(answer.split())[:200])

    if progress:
        return 0

    # An empty progress channel is normally a broken stream — but small talk the
    # supervisor answers itself runs no gate worth narrating and deliberately
    # emits no plan, so "no plan" is only a failure when work actually happened.
    answered_directly = (
        final.get("outcome") == "answer" and not final.get("routed_agent_name") and not deltas
    )
    if answered_directly:
        print()
        print("No task plan, and none expected: the supervisor answered this itself")
        print("(no routing, no worker, no model call).")
        return 0

    print()
    print("FAIL: the stream carried no task plan for a turn that did real work.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
