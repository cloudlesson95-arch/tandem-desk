"""Scripted conversation against a running API.

    uv run python scripts/smoke.py [base_url]    (default http://127.0.0.1:8000)

Resets the demo data, then: a policy question that states a preference, an extension
that pauses for approval, the approval and updated rental, an off-topic refusal, and
the preference recalled in a new thread."""
import sys
import uuid
from datetime import date, timedelta

import httpx

CUSTOMER = "C-1001"


def fail(msg: str, data: dict | None = None) -> None:
    print("FAILED:", msg)
    if data:
        print({k: data.get(k) for k in ("status", "answer", "topic", "cached")})
    sys.exit(1)


def main(base: str) -> None:
    client = httpx.Client(base_url=base.rstrip("/"), timeout=240)

    def chat(thread: str, message: str) -> dict:
        r = client.post("/api/agent/chat", json={"customer_id": CUSTOMER, "thread_id": thread, "message": message})
        if r.status_code != 200:
            fail(f"chat returned {r.status_code}: {r.text[:300]}")
        data = r.json()
        if data.get("cached"):
            fail("got the cached example instead of a live answer", data)
        print(f"> {message}\n  [{data['status']}] {data['answer']}")
        return data

    def rental(rid: str) -> dict:
        customers = client.get("/api/agent/customers").json()["customers"]
        return next(r for c in customers for r in c["rentals"] if r["id"] == rid)

    r = client.post("/api/agent/reset")
    if r.status_code != 200:
        fail(f"reset returned {r.status_code}: {r.text[:300]}")
    thread = f"smoke-{uuid.uuid4()}"

    data = chat(thread, "I always pick up my bikes at the Harbour Desk. What time does it open?")
    if data["status"] not in ("answered", "blocked"):
        fail("policy question was not answered", data)
    print("policy ok")

    before = rental("R-2001")
    data = chat(thread, "Please extend my rental R-2001 by 2 days.")
    pending = data.get("pending_action") or {}
    if data["status"] != "pending" or pending.get("args") != {"rental_id": "R-2001", "days": 2}:
        fail("extension did not pause for approval with the requested arguments", data)
    print("interrupt ok")

    r = client.post("/api/agent/resume", json={
        "customer_id": CUSTOMER, "thread_id": thread, "decision": "approve", "token": pending["token"],
    })
    data = r.json()
    expected = (date.fromisoformat(before["end_date"]) + timedelta(days=2)).isoformat()
    if r.status_code != 200 or data["status"] != "done" or rental("R-2001")["end_date"] != expected:
        fail(f"approval did not update the rental to end on {expected}", data)
    print(f"  [{data['status']}] {data['answer']}")
    print("resume ok")

    data = chat(thread, "What's the best pizza topping?")
    if data["status"] != "refused":
        fail("off-topic message was not refused", data)
    print("refusal ok")

    data = chat(f"smoke-{uuid.uuid4()}", "Which desk do I usually use, and when does it open?")
    recalled = next((t.get("recalled", []) for t in data["trace"] if t["node"] == "supervisor"), [])
    if not any("harbour" in m.lower() for m in recalled):
        fail(f"the Harbour Desk preference was not recalled (recalled: {recalled})", data)
    print("memory ok")
    print("smoke ok")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000")
