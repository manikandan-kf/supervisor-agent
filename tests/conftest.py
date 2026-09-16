import os
import sys
from pathlib import Path

# Bare `pytest` works in a fresh clone: both source trees go on the path here,
# before anything imports them.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "libs" / "agent_governance" / "src"))

# The suite runs offline. Without this every `get_prompt` attempts a Unity Catalog
# round-trip and falls back anyway, so tests pin the bundled templates either way.
os.environ.setdefault("PROMPT_REGISTRY_ENABLED", "false")

import pytest  # noqa: E402
from helpers import (  # noqa: E402
    RBAC,
    REGISTRY,
    StubAudit,
    StubGuardrails,
    StubReviews,
    StubRouter,
    StubWorkers,
)
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.store.memory import InMemoryStore  # noqa: E402

from supervisor.graph import build_graph  # noqa: E402
from supervisor.memory import LongTermMemory  # noqa: E402
from supervisor.services import Services  # noqa: E402
from supervisor.settings import Settings  # noqa: E402


@pytest.fixture
def make_graph():
    def _make(
        guardrails=None,
        router=None,
        workers=None,
        audit=None,
        reviews=None,
        settings=None,
        memory=None,
        output_guard=None,
    ):
        services = Services(
            settings=settings or Settings(),
            registry=REGISTRY,
            rbac=RBAC,
            guardrails=guardrails or StubGuardrails(),
            router=router or StubRouter(),
            workers=workers or StubWorkers(),
            audit=audit or StubAudit(),
            # Same derivation as production (services.build_services): the
            # allowlist is the registry's declared context keys, so tests
            # exercise the real §04 validation rather than an open store.
            memory=memory or LongTermMemory(InMemoryStore(), allowed_keys=REGISTRY.context_keys()),
            reviews=reviews if reviews is not None else StubReviews(),
        )
        if output_guard is not None:
            services.output_guard = output_guard
        graph = build_graph(services, checkpointer=MemorySaver(), store=InMemoryStore())
        return graph, services

    return _make
