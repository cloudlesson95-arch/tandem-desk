"""Long-term customer memory in Postgres + pgvector: durable preferences and one episode
summary per conversation, all written from already-redacted text."""
import re

from langchain_core.messages import HumanMessage, SystemMessage
from psycopg import AsyncConnection
from pydantic import BaseModel, Field

from agent.embed import embed_documents, embed_query, to_pgvector
from agent.llm import small_llm
from agent.pii import redact

MAX_FACTS = 3
DUPLICATE_SCORE = 0.92
PLACEHOLDER = re.compile(r"\[[A-Z]+_\d+\]")
# One-off details that are not durable preferences: rental ids, dates, prices.
ONE_OFF = re.compile(r"\bR-\d{4}\b|\b\d{4}-\d{2}-\d{2}\b|\$\d")

MEMORY_PROMPT = """You keep long-term notes about a customer of Velomarsh, a fictional bike-rental service.
From the latest exchange, extract at most 3 durable facts worth remembering in future conversations:
preferences (bike types, desks, riding plans, accessibility needs, group size) or ongoing situations.
Do not record one-off questions, policy facts, rental ids, dates, prices, or personal data
(names, contacts, payment details, or placeholders like [EMAIL_1]). Skip facts already in the known notes.
Return an empty list when nothing is worth keeping. Also write a one-sentence summary of the whole
conversation so far, updating the previous summary."""


class MemoryUpdate(BaseModel):
    preferences: list[str] = Field(description="Up to 3 short durable facts, e.g. 'Prefers e-bikes'.")
    summary: str = Field(description="One sentence summarizing the conversation so far.")


async def recall(db: AsyncConnection, customer_id: str, text: str) -> list[dict]:
    """Top-5 memories by similarity to the message plus the last 3 episode summaries."""
    cur = await db.execute("SELECT count(*) AS n FROM memories WHERE customer_id = %s", (customer_id,))
    if (await cur.fetchone())["n"] == 0:
        return []
    v = to_pgvector(await embed_query(text))
    cur = await db.execute(
        "SELECT id, kind, content, round((1 - (embedding <=> %s::vector))::numeric, 3)::float AS score"
        " FROM memories WHERE customer_id = %s ORDER BY embedding <=> %s::vector LIMIT 5",
        (v, customer_id, v),
    )
    similar = await cur.fetchall()
    cur = await db.execute(
        "SELECT id, kind, content, NULL::float AS score FROM memories"
        " WHERE customer_id = %s AND kind = 'episode' ORDER BY created_at DESC LIMIT 3",
        (customer_id,),
    )
    seen, out = set(), []
    for row in [*similar, *await cur.fetchall()]:
        if row["id"] not in seen:
            seen.add(row["id"])
            out.append(row)
    return out


def _clean(fact: str) -> str | None:
    """Drop facts that carry personal data, even as placeholders."""
    fact = fact.strip()
    if not fact or PLACEHOLDER.search(fact) or redact(fact).redactions:
        return None
    return fact[:200]


async def write(
    db: AsyncConnection, customer_id: str, thread_id: str, message: str, answer: str
) -> list[dict]:
    cur = await db.execute(
        "SELECT kind, content, thread_id FROM memories WHERE customer_id = %s", (customer_id,)
    )
    rows = await cur.fetchall()
    known = [r["content"] for r in rows if r["kind"] == "preference"]
    previous = next((r["content"] for r in rows if r["kind"] == "episode" and r["thread_id"] == thread_id), "")
    update = await small_llm().with_structured_output(MemoryUpdate, method="json_schema").ainvoke([
        SystemMessage(MEMORY_PROMPT),
        HumanMessage(
            f"Previous summary: {previous or 'none'}\nKnown notes: {known or 'none'}\n\n"
            f"Customer: {message}\nAssistant: {answer}"
        ),
    ])
    facts = [f for f in (_clean(p) for p in update.preferences[:MAX_FACTS]) if f and not ONE_OFF.search(f)]
    facts = [f for f in facts if f.lower() not in {k.lower() for k in known}]
    summary = _clean(update.summary)
    texts = facts + ([summary] if summary else [])
    if not texts:
        return []
    vectors = await embed_documents(texts)
    stored = []
    for fact, vec in zip(facts, vectors):
        v = to_pgvector(vec)
        cur = await db.execute(
            "SELECT 1 FROM memories WHERE customer_id = %s AND kind = 'preference'"
            " AND 1 - (embedding <=> %s::vector) > %s LIMIT 1",
            (customer_id, v, DUPLICATE_SCORE),
        )
        if await cur.fetchone():
            continue
        await db.execute(
            "INSERT INTO memories (customer_id, kind, content, embedding, thread_id)"
            " VALUES (%s, 'preference', %s, %s::vector, %s)",
            (customer_id, fact, v, thread_id),
        )
        stored.append({"kind": "preference", "content": fact})
    if summary:
        await db.execute(
            "DELETE FROM memories WHERE customer_id = %s AND kind = 'episode' AND thread_id = %s",
            (customer_id, thread_id),
        )
        await db.execute(
            "INSERT INTO memories (customer_id, kind, content, embedding, thread_id)"
            " VALUES (%s, 'episode', %s, %s::vector, %s)",
            (customer_id, summary, to_pgvector(vectors[-1]), thread_id),
        )
        stored.append({"kind": "episode", "content": summary})
    return stored


async def list_memories(db: AsyncConnection, customer_id: str) -> dict:
    cur = await db.execute(
        "SELECT id, kind, content, created_at FROM memories WHERE customer_id = %s ORDER BY created_at DESC",
        (customer_id,),
    )
    memories = [{**r, "created_at": r["created_at"].isoformat()} for r in await cur.fetchall()]
    cur = await db.execute(
        "SELECT id, action, args, decision, result, created_at FROM action_log"
        " WHERE customer_id = %s ORDER BY created_at DESC LIMIT 20",
        (customer_id,),
    )
    actions = [{**r, "created_at": r["created_at"].isoformat()} for r in await cur.fetchall()]
    return {"memories": memories, "actions": actions}


async def forget(db: AsyncConnection, customer_id: str, memory_id: int) -> bool:
    cur = await db.execute(
        "DELETE FROM memories WHERE id = %s AND customer_id = %s", (memory_id, customer_id)
    )
    return cur.rowcount > 0
