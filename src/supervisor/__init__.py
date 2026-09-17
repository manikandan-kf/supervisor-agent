"""Custom LangGraph Supervisor Agent (Option B).

RBAC gate -> Guardrails -> Route/Clarify -> Dispatch -> Respond & Audit,
exposed as a Databricks ResponsesAgent on Model Serving.
"""

# Source version; `pyproject.toml` reads it (`dynamic = ["version"]`) so they cannot
# drift. Not the deployed version — that is the UC registered-model version.
__version__ = "1.3.0"
