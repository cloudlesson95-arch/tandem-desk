"""Agent tools: Pydantic argument schemas plus the deterministic business rules that
decide whether a proposed action may go to the customer for approval."""
import asyncio
from datetime import date, timedelta

from langchain_core.utils.function_calling import convert_to_openai_tool
from psycopg import AsyncConnection
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from agent.embed import embed_query, to_pgvector

MAX_EXTENSION_DAYS = 3
WEAK_SCORE = 0.62
TOP_K = 5


class SearchPolicies(BaseModel):
    """Search the Velomarsh rental policy documents. The queries run in parallel."""

    queries: list[str] = Field(
        min_length=1, max_length=3,
        description="One to three differently worded search queries for the same question.",
    )


class GetRentals(BaseModel):
    """List the current customer's rentals: id, bike, dates, status and daily rate."""


class ExtendRental(BaseModel):
    """Propose extending one of the customer's rentals. Nothing changes until the customer approves."""

    rental_id: str = Field(pattern=r"^R-\d{4}$", description="Rental id from get_rentals, e.g. R-2001.")
    days: int = Field(ge=1, description="Number of extra days the customer asked for.")


class CancelRental(BaseModel):
    """Propose cancelling one of the customer's booked rentals. Nothing changes until the customer approves."""

    rental_id: str = Field(pattern=r"^R-\d{4}$", description="Rental id from get_rentals, e.g. R-2002.")


def spec(name: str, model: type[BaseModel]) -> dict:
    tool = convert_to_openai_tool(model)
    tool["function"]["name"] = name
    return tool


KNOWLEDGE_TOOLS = {"search_policies": SearchPolicies}
ACCOUNT_TOOLS = {"get_rentals": GetRentals, "extend_rental": ExtendRental, "cancel_rental": CancelRental}
WRITE_TOOLS = {"extend_rental", "cancel_rental"}


class RuleViolation(Exception):
    """A deterministic business rule rejected the action."""


# ---- retrieval -------------------------------------------------------------

async def _search_one(db: AsyncConnection, vector: list[float]) -> list[dict]:
    v = to_pgvector(vector)
    cur = await db.execute(
        "SELECT id, doc, title, content, 1 - (embedding <=> %s::vector) AS score"
        " FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s",
        (v, v, TOP_K),
    )
    return await cur.fetchall()


async def search_policies(db: AsyncConnection, queries: list[str]) -> list[dict]:
    """Embed all queries at once, search each, and merge by best score per chunk."""
    vectors = await asyncio.gather(*(embed_query(q) for q in queries))
    results = await asyncio.gather(*(_search_one(db, v) for v in vectors))
    best: dict[str, dict] = {}
    for rows in results:
        for row in rows:
            if row["id"] not in best or row["score"] > best[row["id"]]["score"]:
                best[row["id"]] = {**row, "score": round(float(row["score"]), 3)}
    return sorted(best.values(), key=lambda r: r["score"], reverse=True)[:TOP_K]


# ---- rentals ---------------------------------------------------------------

def _rental(row: dict) -> dict:
    return {
        "id": row["id"], "bike": row["bike"], "status": row["status"],
        "start_date": row["start_date"].isoformat(), "end_date": row["end_date"].isoformat(),
        "daily_rate": float(row["daily_rate"]),
    }


RENTAL_COLUMNS = "id, bike, start_date, end_date, status, daily_rate"


async def fetch_rentals(db: AsyncConnection, customer_id: str) -> list[dict]:
    cur = await db.execute(
        f"SELECT {RENTAL_COLUMNS} FROM rentals WHERE customer_id = %s ORDER BY start_date",
        (customer_id,),
    )
    return [_rental(r) for r in await cur.fetchall()]


async def fetch_tier(db: AsyncConnection, customer_id: str) -> str:
    cur = await db.execute("SELECT tier FROM customers WHERE id = %s", (customer_id,))
    row = await cur.fetchone()
    return row["tier"] if row else "standard"


def check_action(name: str, args: BaseModel, rental: dict | None, last_lookup: list[str]) -> None:
    """Business rules for write tools. Raises RuleViolation with a reason the model can relay."""
    rid = args.rental_id
    if rental is None:
        raise RuleViolation(f"{rid} is not one of this customer's rentals.")
    if rid not in last_lookup:
        raise RuleViolation(f"{rid} was not returned by the last get_rentals call; call get_rentals first.")
    if name == "extend_rental":
        if args.days > MAX_EXTENSION_DAYS:
            raise RuleViolation(
                f"Extensions are limited to {MAX_EXTENSION_DAYS} days per request; "
                f"{args.days} days needs a new booking."
            )
        if rental["status"] not in ("booked", "active"):
            raise RuleViolation(f"{rid} is {rental['status']}; only booked or active rentals can be extended.")
    if name == "cancel_rental" and rental["status"] != "booked":
        raise RuleViolation(
            f"{rid} is {rental['status']}; only booked rentals that have not started can be cancelled."
        )


def describe_proposal(name: str, args: BaseModel, rental: dict, tier: str, today: date) -> tuple[str, dict]:
    """A human-readable summary and the rental as it would look after approval."""
    label = f"{rental['id']} ({rental['bike']})"
    if name == "extend_rental":
        new_end = date.fromisoformat(rental["end_date"]) + timedelta(days=args.days)
        cost = args.days * rental["daily_rate"]
        price = f"{args.days} × ${rental['daily_rate']:.2f} = ${cost:.2f}"
        if tier == "plus":
            price += f", ${cost * 0.9:.2f} after the 10% Plus discount"
        summary = (
            f"Extend {label} by {args.days} day{'s' if args.days > 1 else ''}: the end date moves "
            f"from {rental['end_date']} to {new_end.isoformat()}. Extra cost: {price}, plus tax."
        )
        return summary, {**rental, "end_date": new_end.isoformat()}
    hours = (date.fromisoformat(rental["start_date"]) - today).days * 24
    free_window = 24 if tier == "plus" else 48
    if hours > free_window:
        fee = f"none, because the start is more than {free_window} hours away"
    else:
        fee = f"${rental['daily_rate'] * 0.5:.2f} (50% of the first day's rate), because the start is within {free_window} hours"
    summary = (
        f"Cancel {label}, booked {rental['start_date']} to {rental['end_date']}. Cancellation fee: {fee}."
    )
    return summary, {**rental, "status": "cancelled"}


async def execute_action(db: AsyncConnection, pending: dict) -> dict:
    """Run an approved action. The WHERE clause re-checks ownership and status, so a rental
    that changed since the proposal is not touched."""
    args, cid = pending["args"], pending["customer_id"]
    if pending["action"] == "extend_rental":
        if not 1 <= args["days"] <= MAX_EXTENSION_DAYS:
            raise RuleViolation("extension length out of range")
        cur = await db.execute(
            "UPDATE rentals SET end_date = end_date + %s, extended_days = extended_days + %s"
            f" WHERE id = %s AND customer_id = %s AND status IN ('booked', 'active') RETURNING {RENTAL_COLUMNS}",
            (args["days"], args["days"], args["rental_id"], cid),
        )
    else:
        cur = await db.execute(
            "UPDATE rentals SET status = 'cancelled'"
            f" WHERE id = %s AND customer_id = %s AND status = 'booked' RETURNING {RENTAL_COLUMNS}",
            (args["rental_id"], cid),
        )
    row = await cur.fetchone()
    if row is None:
        raise RuleViolation("The rental changed since the proposal; nothing was updated.")
    return _rental(row)


# ---- action log ------------------------------------------------------------

async def log_pending(db: AsyncConnection, customer_id: str, thread_id: str, action: str, args: dict) -> int:
    cur = await db.execute(
        "INSERT INTO action_log (customer_id, thread_id, action, args, decision)"
        " VALUES (%s, %s, %s, %s, 'pending') RETURNING id",
        (customer_id, thread_id, action, Jsonb(args)),
    )
    return (await cur.fetchone())["id"]


async def log_decision(db: AsyncConnection, action_id: int, decision: str, result: dict | None) -> None:
    await db.execute(
        "UPDATE action_log SET decision = %s, result = %s WHERE id = %s",
        (decision, Jsonb(result) if result is not None else None, action_id),
    )
