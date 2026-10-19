"""Chat models on Groq, shared by the agents.

The agents and the verifier use AGENT_MODEL. When Groq reports that its daily token quota
is spent, calls switch to FALLBACK_MODEL for the next ten minutes, so the demo keeps
answering; the trace records which model answered. The eval turns this off
(MODEL_FALLBACK=off) so its numbers come from one model."""
import time
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import groq
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq

from agent.config import AGENT_MODEL, FALLBACK_MODEL, GUARD_MODEL, MODEL_FALLBACK

FALLBACK_SECONDS = 600
_primary_off_until = 0.0


@lru_cache
def agent_llm() -> ChatGroq:
    # Few retries: a spent daily quota should reach the fallback quickly.
    return ChatGroq(
        model=AGENT_MODEL, temperature=0, reasoning_effort="low", max_tokens=1500, max_retries=3
    )


@lru_cache
def fallback_llm() -> ChatGroq:
    return ChatGroq(
        model=FALLBACK_MODEL, temperature=0, reasoning_effort="low", max_tokens=1500, max_retries=6
    )


@lru_cache
def small_llm() -> ChatGroq:
    return ChatGroq(
        model=GUARD_MODEL, temperature=0, reasoning_effort="low", max_tokens=800, max_retries=6
    )


def _daily_limit(e: Exception) -> bool:
    return "per day" in str(e) or "(TPD)" in str(e) or "(RPD)" in str(e)


async def agent_call(make: Callable[[ChatGroq], Runnable], messages: list) -> tuple[Any, str]:
    """Run `make(model)` on the agent model, or on the fallback model while the agent
    model's daily quota is spent. Returns the result and the model that produced it."""
    global _primary_off_until
    if MODEL_FALLBACK and time.time() < _primary_off_until:
        return await make(fallback_llm()).ainvoke(messages), FALLBACK_MODEL
    try:
        return await make(agent_llm()).ainvoke(messages), AGENT_MODEL
    except groq.RateLimitError as e:
        if not (MODEL_FALLBACK and _daily_limit(e)):
            raise
        _primary_off_until = time.time() + FALLBACK_SECONDS
        return await make(fallback_llm()).ainvoke(messages), FALLBACK_MODEL
