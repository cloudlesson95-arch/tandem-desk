"""Apply db/schema.sql, create the LangGraph checkpoint tables and seed demo data."""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psycopg  # noqa: E402

from agent.config import database_url  # noqa: E402
from agent.seed import reset_demo  # noqa: E402


async def setup_checkpointer() -> None:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(database_url()) as saver:
        await saver.setup()


def main() -> None:
    with psycopg.connect(database_url()) as conn:
        conn.execute((ROOT / "db" / "schema.sql").read_text(encoding="utf-8"))
        conn.commit()
        n = reset_demo(conn)
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables"
            " WHERE table_schema = 'public' ORDER BY table_name"
        ).fetchall()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(setup_checkpointer())
    print("tables:", ", ".join(t[0] for t in tables))
    print(f"seeded {n} customers")


if __name__ == "__main__":
    main()
