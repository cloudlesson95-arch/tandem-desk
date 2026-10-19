"""Chat models on Groq, shared by the agents."""
from functools import lru_cache

from langchain_groq import ChatGroq

from agent.config import AGENT_MODEL, GUARD_MODEL


@lru_cache
def agent_llm() -> ChatGroq:
    return ChatGroq(
        model=AGENT_MODEL, temperature=0, reasoning_effort="low", max_tokens=1500, max_retries=6
    )


@lru_cache
def small_llm() -> ChatGroq:
    return ChatGroq(
        model=GUARD_MODEL, temperature=0, reasoning_effort="low", max_tokens=800, max_retries=6
    )
