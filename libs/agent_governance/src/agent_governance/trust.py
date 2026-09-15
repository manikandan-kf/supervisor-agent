"""Verifying that the entitlement block really came from the gateway.

The graph builds every downstream control — the RBAC gate, the guardrail
candidate set, the approval permission — on `user_role`, `user_id`,
`permitted_agents` and `approvable_agents` from `custom_inputs`. The gateway
derives those from the validated identity token and never accepts them from a
browser; but nothing in the graph could *check* that, so the only thing between
a direct endpoint caller and full impersonation was the serving endpoint's ACL.

This module closes that gap with an HMAC over the entitlement block, keyed by a
secret only the gateway and the supervisor hold:

  * the gateway signs the block it derived (`sign_entitlements`) and sends the
    digest as `custom_inputs["entitlement_signature"]`;
  * `SupervisorContext.from_custom_inputs` verifies it when
    `SUPERVISOR_TRUST_SECRET` is set, and the RBAC gate refuses the turn when
    verification fails.

**Ships dark.** With no secret configured, behaviour is exactly what it was —
the endpoint ACL remains the boundary, as today. Turning the control on is
configuration on both deployables (the same secret in the gateway's and the
endpoint's environment, via Databricks secrets), not a redeploy of either.
The gateway keeps its own copy of `sign_entitlements` (it is a separate
deployable), and the two implementations must be kept in step.

What the signature binds and what it does not: it covers the entitlement
fields plus `conversation_id`, so a signed block cannot be edited or replayed
onto a different conversation. It deliberately carries no timestamp — the
signed payload never transits a browser (gateway → Model Serving only), so the
replay window is an attacker already inside that channel, at which point the
secret itself is the thing lost.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets

# Every field the graph treats as authoritative about the caller. `agent_id`
# rides along so a signature minted for one widget cannot be replayed against
# another agent's invocation path.
SIGNED_FIELDS = (
    "user_role",
    "user_id",
    "user_reference",
    "permitted_agents",
    "approvable_agents",
    "agent_id",
    "conversation_id",
)

SIGNATURE_FIELD = "entitlement_signature"

# ── The other direction: supervisor -> worker ───────────────────────────────
# The gateway->supervisor boundary above is cryptographically verified. The
# adjacent one was not: the dispatch payload carries the conversation, the
# resolved context and the correlation set, and nothing signed it, so only the
# serving endpoint's ACL separated a forged dispatch from a real one. That is
# the gap OWASP's ASI07 (insecure inter-agent communication) names — no message
# integrity, no replay protection.
#
# The supervisor signs; a worker that wants the guarantee calls
# `verify_dispatch` with the same secret. Like the entitlement signature this
# **ships dark** — with no secret configured nothing is attached and nothing
# changes — and it is deliberately the *sending* half only, because the worker
# agents are a separate deliverable (ASM-03). Publishing the signer and the
# verifier together is what lets a worker team adopt it without this repo
# guessing at their framework.
#
# `nonce` is what the entitlement signature deliberately omits and this one
# needs: the entitlement block never transits a browser, whereas a dispatch is
# a call a worker could be induced to replay. A worker enforcing replay
# protection keeps recently-seen nonces for a short window; one that only wants
# integrity can ignore it and still verify.
DISPATCH_SIGNED_FIELDS = (
    "conversation_id",
    "agent_id",
    "request_id",
    "correlation_id",
    "pseudonymous_user_reference",
    "nonce",
)

DISPATCH_SIGNATURE_FIELD = "dispatch_signature"


def trust_secret() -> str:
    """The shared secret, or "" when the control is dark.

    Read from the environment per call rather than captured at import, for the
    same reason `prompt_provider._failure_ttl` is: a serving container's
    environment is not necessarily final when modules first import.
    """
    return os.getenv("SUPERVISOR_TRUST_SECRET", "")


def _canonical(custom: dict, fields: tuple[str, ...] = SIGNED_FIELDS) -> bytes:
    """The exact bytes signed — sorted keys, no incidental whitespace.

    Lists are passed through as-is (order is meaningful to the gateway and
    preserved end to end); absent fields are distinct from empty ones, matching
    how `SupervisorContext` treats None vs ().

    `fields` defaults to the entitlement set so the digest this produces for an
    entitlement block is byte-identical to what it produced before the dispatch
    signature existed — the gateway keeps its own copy of `sign_entitlements`
    and a test pins the two together, so a change here that shifted the
    entitlement bytes would break a live deployment mid-rollout.
    """
    payload = {name: custom[name] for name in fields if name in custom}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def sign_entitlements(custom: dict, secret: str) -> str:
    """The hex HMAC-SHA256 the gateway attaches as `entitlement_signature`."""
    if not secret:
        raise ValueError("cannot sign entitlements with an empty secret")
    return hmac.new(secret.encode("utf-8"), _canonical(custom), hashlib.sha256).hexdigest()


def verify_entitlements(custom: dict, secret: str) -> bool:
    """Whether `custom` carries a valid signature over its entitlement block.

    `compare_digest`, so a byte-by-byte comparison cannot be timed. A missing
    signature is simply invalid — when the secret is configured, an unsigned
    request is by definition one that did not come through the gateway.
    """
    if not secret:
        return True
    supplied = str(custom.get(SIGNATURE_FIELD, "") or "")
    if not supplied:
        return False
    expected = sign_entitlements(custom, secret)
    return hmac.compare_digest(supplied, expected)


def new_nonce() -> str:
    """A single-use value for one dispatch. `secrets`, not `random`.

    A predictable nonce is not a nonce: it is the field a replay defence keys
    on, so it has to be unguessable by whoever might replay the call.
    """
    return secrets.token_hex(16)


def sign_dispatch(custom: dict, secret: str) -> str:
    """The hex HMAC-SHA256 the supervisor attaches as `dispatch_signature`."""
    if not secret:
        raise ValueError("cannot sign a dispatch with an empty secret")
    return hmac.new(
        secret.encode("utf-8"), _canonical(custom, DISPATCH_SIGNED_FIELDS), hashlib.sha256
    ).hexdigest()


def verify_dispatch(custom: dict, secret: str) -> bool:
    """Whether `custom` carries a valid supervisor dispatch signature.

    **For worker agents to call**, not used inside this repo — the workers are
    a separate deliverable, and shipping the verifier beside the signer is what
    keeps the two from drifting the way a re-implementation would.

    Returns True when no secret is configured, exactly like
    `verify_entitlements`: the control ships dark, so a worker adopting this
    does not have to be deployed in lockstep with the secret rollout. A worker
    that also wants replay protection checks `custom["nonce"]` against a
    recently-seen set *after* this returns True.
    """
    if not secret:
        return True
    supplied = str(custom.get(DISPATCH_SIGNATURE_FIELD, "") or "")
    if not supplied:
        return False
    return hmac.compare_digest(supplied, sign_dispatch(custom, secret))
