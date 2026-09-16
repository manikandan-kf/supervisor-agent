"""Verifying that the entitlement block really came from the gateway.

Nothing in the graph could check that `custom_inputs` entitlements came from the gateway: only the
endpoint ACL stood between a direct caller and impersonation. `sign_entitlements` HMAC-SHA256s the
compact sorted-JSON of `SIGNED_FIELDS` (entitlements plus `agent_id` and `conversation_id`) under
a shared secret, hex-encoded into `entitlement_signature`; `verify_entitlements` checks it. Ships
dark. The gateway holds its own copy of the signer, so the canonical bytes must not drift.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets

# Every field the graph treats as authoritative about the caller. `agent_id` rides along so a
# signature minted for one widget cannot be replayed against another agent's invocation path.
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
# supervisor->worker had no message integrity and no replay protection (OWASP ASI07). It signs and
# ships dark, carrying the *sending* half plus a published `verify_dispatch` because the workers
# are a separate deliverable (ASM-03). `nonce` is here because a worker can be induced to replay.
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

    Read from the environment per call, not captured at import: a serving container's
    environment is not necessarily final when modules first import.
    """
    return os.getenv("SUPERVISOR_TRUST_SECRET", "")


def _canonical(custom: dict, fields: tuple[str, ...] = SIGNED_FIELDS) -> bytes:
    """The exact bytes signed — sorted keys, no incidental whitespace.

    Lists pass through as-is (order is meaningful to the gateway); absent fields stay distinct
    from empty ones. `fields` defaults to the entitlement set so the digest stays byte-identical
    to the gateway's own copy of `sign_entitlements` — drift would break a live rollout.
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

    `compare_digest`, so the comparison cannot be timed. A missing signature is invalid: with
    the secret configured, an unsigned request did not come through the gateway.
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

    A predictable nonce is not a nonce: it is the field a replay defence keys on.
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

    *For worker agents to call*, not used in this repo: shipping the verifier beside the signer
    is what keeps the two from drifting. Returns True when no secret is configured, so adoption
    need not be in lockstep with the rollout. Replay protection is the caller's `nonce` check.
    """
    if not secret:
        return True
    supplied = str(custom.get(DISPATCH_SIGNATURE_FIELD, "") or "")
    if not supplied:
        return False
    return hmac.compare_digest(supplied, sign_dispatch(custom, secret))
