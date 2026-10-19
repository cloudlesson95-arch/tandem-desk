"""FastAPI app served as a Vercel Python function and, locally, by uvicorn on :8000."""
import asyncio
import hashlib
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import groq  # noqa: E402
import psycopg  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from agent import approval, memory, tools  # noqa: E402
from agent.config import database_url, keys_present  # noqa: E402
from agent.graph import Ctx, build_graph, turn_input  # noqa: E402
from agent.seed import reset_demo  # noqa: E402

app = FastAPI(title="Tandem Desk agent API")
BUILDER = build_graph()

# Demo guard: keeps public visitors from using up the free model quotas.
HOURLY_PER_IP = 20
DAILY_TOTAL = 200
EXAMPLE = ROOT / "data" / "example_conversation.json"

CustomerId = Field(pattern=r"^C-\d{4}$")
ThreadId = Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9-]+$")


class ChatIn(BaseModel):
    customer_id: str = CustomerId
    thread_id: str = ThreadId
    message: str = Field(min_length=1, max_length=2000)
    verify: bool = True
    memory: bool = True


class ResumeIn(BaseModel):
    customer_id: str = CustomerId
    thread_id: str = ThreadId
    decision: Literal["approve", "reject"]
    token: str = Field(max_length=128)


async def _connect() -> psycopg.AsyncConnection:
    return await psycopg.AsyncConnection.connect(
        database_url(), autocommit=True, prepare_threshold=0, row_factory=dict_row
    )


@asynccontextmanager
async def session():
    """Two Postgres connections per request: the checkpointer writes in the background
    while nodes run, so the tools get their own connection."""
    saver_conn, conn = await asyncio.gather(_connect(), _connect())
    try:
        yield BUILDER.compile(checkpointer=AsyncPostgresSaver(saver_conn)), conn
    finally:
        await asyncio.gather(saver_conn.close(), conn.close())


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}


async def _snapshot(graph, thread_id: str, customer_id: str):
    snap = await graph.aget_state(_config(thread_id))
    owner = (snap.values or {}).get("customer_id")
    if owner and owner != customer_id:
        raise HTTPException(403, "This conversation belongs to another customer.")
    return snap


def _response(snap, thread_id: str) -> dict:
    v = snap.values or {}
    pending = None
    if snap.interrupts:
        p = snap.interrupts[0].value
        pending = {**p, "token": approval.sign(p)}
    return {
        "thread_id": thread_id,
        "status": "pending" if pending else v.get("status"),
        "answer": v.get("answer"),
        "draft": v.get("draft"),
        "topic": (v.get("guard") or {}).get("topic"),
        "pending_action": pending,
        "action_result": v.get("action_result"),
        "verification": v.get("verification"),
        "redactions": v.get("redactions", []),
        "evidence": [
            {k: e.get(k) for k in ("pid", "kind", "title", "content", "score")}
            for e in v.get("evidence", [])
        ],
        "trace": v.get("trace", []),
        "cached": False,
    }


@app.get("/api/agent/health")
async def health() -> dict:
    return {"ok": True, "keys": keys_present()}


def _client_ip(request: Request) -> str:
    # Vercel overwrites X-Forwarded-For with the real client address.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.headers.get("x-real-ip") or (request.client.host if request.client else "unknown")


async def _over_limit(conn: psycopg.AsyncConnection, ip: str) -> str | None:
    """Count this message per IP per hour and overall per day; IPs are stored hashed."""
    ip_key = "ip:" + hashlib.sha256(ip.encode()).hexdigest()[:16]
    cur = await conn.execute(
        "INSERT INTO rate_limits (key, window_start, count) VALUES"
        " (%s, date_trunc('hour', now()), 1), ('all', date_trunc('day', now()), 1)"
        " ON CONFLICT (key, window_start) DO UPDATE SET count = rate_limits.count + 1"
        " RETURNING key, count",
        (ip_key,),
    )
    counts = {r["key"]: r["count"] for r in await cur.fetchall()}
    if counts[ip_key] > HOURLY_PER_IP:
        return f"more than {HOURLY_PER_IP} messages from this address in the last hour"
    if counts["all"] > DAILY_TOTAL:
        return f"the demo reached its limit of {DAILY_TOTAL} messages today"
    return None


def _cached(reason: str) -> dict:
    """A recorded real conversation, returned instead of a live answer."""
    return {
        "thread_id": None, "status": "cached", "cached": True, "reason": reason,
        "answer": f"Live answers are paused ({reason}). Here is a recorded conversation instead.",
        "example": json.loads(EXAMPLE.read_text(encoding="utf-8")),
        "pending_action": None, "trace": [], "redactions": [], "evidence": [],
    }


@app.post("/api/agent/chat")
async def chat(req: ChatIn, request: Request) -> dict:
    if not keys_present():
        return _cached("an API key is missing on the server")
    async with session() as (graph, conn):
        reason = await _over_limit(conn, _client_ip(request))
        if reason:
            return _cached(reason)
        snap = await _snapshot(graph, req.thread_id, req.customer_id)
        ctx = Ctx(db=conn)
        try:
            if snap.interrupts:
                # A new message while an action waits for approval rejects that action.
                await graph.ainvoke(Command(resume={"decision": "reject"}), _config(req.thread_id), context=ctx)
            inp = turn_input(req.customer_id, req.thread_id, req.message, req.verify, req.memory)
            await graph.ainvoke(inp, _config(req.thread_id), context=ctx)
        except groq.RateLimitError:
            return _cached("the model provider's free rate limit was reached")
        return _response(await graph.aget_state(_config(req.thread_id)), req.thread_id)


@app.post("/api/agent/resume")
async def resume(req: ResumeIn) -> dict:
    async with session() as (graph, conn):
        snap = await _snapshot(graph, req.thread_id, req.customer_id)
        if not snap.interrupts:
            raise HTTPException(409, "No action is waiting for approval in this conversation.")
        if req.decision == "approve" and not approval.verify(snap.interrupts[0].value, req.token):
            raise HTTPException(403, "The approval token doesn't match the pending action.")
        await graph.ainvoke(
            Command(resume={"decision": req.decision, "token": req.token}),
            _config(req.thread_id), context=Ctx(db=conn),
        )
        return _response(await graph.aget_state(_config(req.thread_id)), req.thread_id)


@app.get("/api/agent/customers")
async def customers() -> dict:
    async with session() as (_, conn):
        cur = await conn.execute("SELECT id, name, tier FROM customers ORDER BY id")
        rows = await cur.fetchall()
        for row in rows:
            row["rentals"] = await tools.fetch_rentals(conn, row["id"])
        return {"customers": rows}


@app.get("/api/agent/memories")
async def memories(customer_id: str) -> dict:
    async with session() as (_, conn):
        return await memory.list_memories(conn, customer_id)


@app.delete("/api/agent/memories/{memory_id}")
async def forget(memory_id: int, customer_id: str) -> dict:
    async with session() as (_, conn):
        if not await memory.forget(conn, customer_id, memory_id):
            raise HTTPException(404, "Memory not found.")
        return {"deleted": memory_id}


@app.post("/api/agent/reset")
async def reset() -> dict:
    def run() -> int:
        with psycopg.connect(database_url()) as conn:
            return reset_demo(conn)

    return {"customers": await asyncio.to_thread(run)}
