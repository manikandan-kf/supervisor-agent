"""Custom LangGraph Supervisor Agent (Option B).

RBAC gate -> Guardrails -> Route/Clarify -> Dispatch -> Respond & Audit,
exposed as a Databricks ResponsesAgent on Model Serving.
"""

# The source version. `pyproject.toml` reads it from here (`dynamic = ["version"]`)
# so the two cannot drift. This is not the deployed version — a deployment is
# identified by its Unity Catalog registered-model version, assigned at deploy
# time — but it is what a checkout can answer, and what CHANGELOG.md tracks.
__version__ = "1.1.0"
