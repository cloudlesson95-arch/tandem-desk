"""FastAPI app served as a Vercel Python function and, locally, by uvicorn on :8000."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402

from agent.config import keys_present  # noqa: E402

app = FastAPI(title="Tandem Desk agent API")


@app.get("/api/agent/health")
async def health() -> dict:
    return {"ok": True, "keys": keys_present()}
