"""Model client creation (§4.1).

The framework forbids hardcoding model identifiers in `graph.py`, LangGraph
nodes or tools — all model access is isolated here, so changing the routing
model is a configuration change with no code edit:

    ROUTING_LLM_ENDPOINT=databricks-claude-sonnet-4-5

Every client is a `ChatDatabricks` over a Model Serving endpoint, and that is
also how a different vendor is reached: an external-model endpoint (OpenAI,
Anthropic, Bedrock, …) behind the AI Gateway looks the same to this module as
a foundation-model endpoint, so ASM-04's "the client chooses the LLM" stays a
serving-endpoint decision rather than a code path. There used to be an
`init_chat_model` provider switch in front of this; `databricks` is not one of
its providers, so on the only platform this runs on it raised and fell back to
`ChatDatabricks` on every start, logging a warning that read as a defect.

Multi-model support is the second seam here: a registry entry may carry its
own `model` (a serving-endpoint name), and `get_agent_model` resolves it
through the same cached factory — so a cheaper endpoint can screen a simple
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
def _cached_chat_model(endpoint: str, temperature: float, timeout: float, max_retries: int):
    from databricks_langchain import ChatDatabricks

    logger.info(
        "creating chat model endpoint=%s temp=%s timeout=%ss max_retries=%s",
        endpoint,
        temperature,
        timeout,
        max_retries,
    )
    return ChatDatabricks(
        endpoint=endpoint,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )


def get_routing_model(settings):
    """The model behind guardrail verdicts and context resolution.

    Cached per (endpoint, temperature, timeout, max_retries): the graph is
    rebuilt per worker process but the client is reusable, and creating one per
    node call would add needless connection setup.

    **`timeout` and `max_retries` are passed explicitly because the defaults
    are the wrong thing here**, which is only visible if you check what those
    defaults actually resolve to:

      * `ChatDatabricks.timeout` defaults to `None`, which does not mean "no
        limit chosen, use something sane" — it means the parameter is never
        forwarded, so the underlying OpenAI client's own default applies, and
        that is a **600-second read timeout**. A routing model that accepted the
        connection and then stalled held a serving worker for ten minutes.
      * `ChatDatabricks.max_retries` also defaults to `None`, and the OpenAI
        client's default is 2 retries — three attempts, invisible to the node
        span latency is measured from, and multiplied by any retry layer above.

    Hence 30s and 0 by default: one bound, one retry authority
    (`resilience.invoke_with_retries`). See `settings.routing_llm_max_retries`.
    """
    return _cached_chat_model(
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
         one-time notice naming the override being ignored. A disabled feature
         must be *visibly* disabled: an operator who publishes `model:` entries
         and sees no cost shift should find the reason in the log, not in a
         debugger.
      3. Both present — a client for the agent's own endpoint, built by the
         same cached factory as the routing model and sharing its temperature,
         timeout and retry settings. Only the endpoint varies: the bounds in
         `get_routing_model`'s docstring exist to protect the serving worker
         and the retry budget, and letting a registry row relax them would let
         a config publish undo a deliberate operational choice.

    Per-agent resolution is also what makes token spend attributable per agent:
    once each agent's verdicts run on its own endpoint, the endpoint name on
    the trace span identifies the agent.
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
        reference,
        settings.routing_temperature,
        settings.routing_llm_timeout_seconds,
        settings.routing_llm_max_retries,
    )
