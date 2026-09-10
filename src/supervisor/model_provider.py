"""Model client creation (§4.1).

The framework forbids hardcoding model identifiers in `graph.py`, LangGraph
nodes or tools — all model access is isolated here, so changing the routing
model is a configuration change with no code edit.

Provider selection is configuration too. `init_chat_model` resolves a
`provider:model` string at runtime, so the same graph runs against Anthropic,
OpenAI, Bedrock or a local Ollama without touching a node. ASM-04 in the BRD
leaves the LLM choice to the client; this is what keeps that choice cheap.

    ROUTING_LLM_PROVIDER=databricks
    ROUTING_LLM_ENDPOINT=databricks-claude-sonnet-4-5

**`databricks` is not one of `init_chat_model`'s providers**, so that setting
always takes the `ChatDatabricks` fallback below. The endpoint log says so on
every startup:

    init_chat_model unavailable for databricks (ValueError: Unsupported
    provider='databricks'. Supported model providers are: anthropic, azure_ai,
    azure_openai, bedrock, ... ) — using ChatDatabricks

That is the fallback doing its job, not a defect. It does mean the provider
switch is only exercised for the non-Databricks providers; moving off Databricks
would use the `init_chat_model` path for real.

Multi-model support is the second seam here: a registry entry may
carry its own `model` (a serving-endpoint name), and `get_agent_model` resolves
it through the same cached factory — so a cheaper endpoint can screen a simple
domain while a stronger one screens a complex one, per configuration. The
feature ships dark: until `MULTI_MODEL_ENABLED=true`, every agent resolves to
the one routing model regardless of what its entry declares.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


# 16, not 8: one client for the routing model plus one per agent that declares
# its own `model` once multi-model support is enabled. The registry is small
# (four agents today) but the cache must not start evicting live clients the
# day a fifth is onboarded.
@lru_cache(maxsize=16)
def _cached_chat_model(
    provider: str, model: str, temperature: float, timeout: float, max_retries: int
):
    logger.info(
        "creating chat model provider=%s model=%s temp=%s timeout=%ss max_retries=%s",
        provider,
        model,
        temperature,
        timeout,
        max_retries,
    )

    try:
        from langchain.chat_models import init_chat_model

        return init_chat_model(
            model,
            model_provider=provider,
            temperature=temperature,
            # Tokens must reach the caller as they are produced, not in one
            # block at the end — the Task Plan and the answer stream together.
            disable_streaming=False,
            timeout=timeout,
            max_retries=max_retries,
        )
    except Exception as exc:
        if provider != "databricks":
            raise
        # `init_chat_model` needs the provider integration installed. Inside
        # Model Serving `databricks_langchain` is always present, so fall back
        # to it rather than failing a governed request over a packaging detail.
        logger.warning(
            "init_chat_model unavailable for databricks (%s: %s) — using ChatDatabricks",
            type(exc).__name__,
            exc,
        )
        from databricks_langchain import ChatDatabricks

        return ChatDatabricks(
            endpoint=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )


def get_routing_model(settings):
    """The model behind guardrail verdicts and context resolution.

    Cached per (provider, model, temperature, timeout, max_retries): the graph
    is rebuilt per worker process but the client is reusable, and creating one
    per node call would add needless connection setup.

    **`timeout` and `max_retries` are passed explicitly because both libraries
    default to the wrong thing here**, which is only visible if you check what
    those defaults actually resolve to:

      * `ChatDatabricks.timeout` defaults to `None`, which does not mean "no
        limit chosen, use something sane" — it means the parameter is never
        forwarded, so the underlying OpenAI client's own default applies, and
        that is a **600-second read timeout**. A routing model that accepted the
        connection and then stalled held a serving worker for ten minutes.
      * `ChatDatabricks.max_retries` also defaults to `None`, and the OpenAI
        client's default is 2 retries — three attempts. `graph.py` already wraps
        these nodes in `RetryPolicy(max_attempts=3)`, so the two layers
        multiplied to **nine calls** for one governed turn, with the inner eight
        invisible to the node span latency is measured from.

    Hence 30s and 0 by default: one bound, one retry authority. See
    `settings.routing_llm_max_retries` for why the graph keeps the retry budget
    rather than the client.
    """
    return _cached_chat_model(
        settings.routing_llm_provider,
        settings.routing_llm_endpoint,
        settings.routing_temperature,
        settings.routing_llm_timeout_seconds,
        settings.routing_llm_max_retries,
    )


# Agents whose declared `model` was ignored because the feature flag is off,
# so the notice logs once per (agent, model) rather than on every verdict of
# every turn. Process-local, like the client cache above.
_ignored_overrides: set[tuple[str, str]] = set()


def get_agent_model(settings, agent):
    """The model behind one agent's governance calls.

    Resolution, in order:

      1. The entry declares no `model` — the shared routing model. This is
         every agent today, and the whole path costs one attribute read.
      2. `MULTI_MODEL_ENABLED` is not true — the shared routing model, with a
         one-time notice naming the override being ignored. The support code
         is integrated ahead of the feature being switched on, and a disabled
         feature must be *visibly* disabled: an operator
         who publishes `model:` entries and sees no cost shift should find the
         reason in the log, not in a debugger.
      3. Both present — a client for the agent's own endpoint, built by the
         same cached factory as the routing model and sharing its provider,
         temperature, timeout and retry settings. Only the endpoint varies:
         the bounds in `get_routing_model`'s docstring exist to protect the
         serving worker and the retry budget, and letting a registry row relax
         them would let a config publish undo a deliberate operational choice.

    Per-agent resolution is also what makes token spend attributable per agent:
    once each agent's verdicts run on its own endpoint,
    the endpoint name on the trace span identifies the agent.
    """
    reference = (getattr(agent, "model", "") or "").strip()
    if not reference or reference == settings.routing_llm_endpoint:
        return get_routing_model(settings)

    if not settings.multi_model_enabled:
        key = (getattr(agent, "id", "?"), reference)
        if key not in _ignored_overrides:
            _ignored_overrides.add(key)
            logger.info(
                "agent %s declares model %s but MULTI_MODEL_ENABLED is not true — "
                "using the routing model %s",
                key[0],
                reference,
                settings.routing_llm_endpoint,
            )
        return get_routing_model(settings)

    return _cached_chat_model(
        settings.routing_llm_provider,
        reference,
        settings.routing_temperature,
        settings.routing_llm_timeout_seconds,
        settings.routing_llm_max_retries,
    )
