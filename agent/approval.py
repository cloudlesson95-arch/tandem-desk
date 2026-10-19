"""Approval tokens: an HMAC over the exact pending action, issued by the server when the
graph pauses and required before a write tool runs."""
import hashlib
import hmac
import json
import os


def _secret() -> bytes:
    secret = os.environ.get("APPROVAL_SECRET")
    if not secret:
        raise RuntimeError("APPROVAL_SECRET is not set")
    return secret.encode()


def sign(pending: dict) -> str:
    payload = json.dumps(
        {k: pending[k] for k in ("action_id", "thread_id", "customer_id", "action", "args")},
        sort_keys=True, separators=(",", ":"),
    )
    return hmac.new(_secret(), payload.encode(), hashlib.sha256).hexdigest()


def verify(pending: dict, token: str | None) -> bool:
    return bool(token) and hmac.compare_digest(sign(pending), token)
