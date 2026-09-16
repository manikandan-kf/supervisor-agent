"""Model client creation (§4.1).

No model identifier is hardcoded in `graph.py`, nodes or tools; every client is a
`ChatDatabricks` over a Model Serving endpoint, so changing the routing model is config
(`ROUTING_LLM_ENDPOINT=...`) and another vendor is an external-model endpoint behind the AI
Gateway (ASM-04). `databricks` is not an `init_chat_model` provider, hence no provider switch.
Multi-model support ships dark: until `MULTI_MODEL_ENABLED=true` every agent resolves to the
routing model regardless of what its registry entry declares.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


# 16, not 8: one client for the routing model plus one per agent declaring its own `model`.
# The cache must not start evicting live clients the day a fifth agent is onboarded.
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

    Cached per (endpoint, temperature, timeout, max_retries). `timeout` and `max_retries` are
    passed explicitly because `ChatDatabricks` defaults of `None` fall through to the OpenAI
    client's 600s read timeout and 2 retries — a stalled model held a serving worker ten
    minutes. Hence 30s and 0: one bound, one retry authority (`resilience.invoke_with_retries`).
    """
    return _cached_chat_model(
        settings.routing_llm_endpoint,
        settings.routing_temperature,
        settings.routing_llm_timeout_seconds,
        settings.routing_llm_max_retries,
    )


# (agent, model) pairs whose declared `model` was ignored because the flag is off, so the
# notice logs once rather than on every verdict. Process-local, like the client cache.
_ignored_overrides: set[tuple[str, str]] = set()


def get_agent_model(settings, agent):
    """The model behind one agent's governance calls.

    No `model` in the entry, or `MULTI_MODEL_ENABLED` off: the shared routing model — with a
    one-time notice, because a disabled feature must be *visibly* disabled. Otherwise a client
    for the agent's endpoint sharing the routing model's temperature, timeout and retries:
    a registry row must not relax bounds that protect the serving worker and the retry budget.
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
