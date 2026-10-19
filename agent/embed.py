"""Gemini embeddings (gemini-embedding-001 reduced to 768 dimensions)."""
from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from agent.config import EMBED_DIM, EMBED_MODEL


@lru_cache
def embedder() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, output_dimensionality=EMBED_DIM)


async def embed_query(text: str) -> list[float]:
    return await embedder().aembed_query(text)


async def embed_documents(texts: list[str]) -> list[list[float]]:
    return await embedder().aembed_documents(texts)


def to_pgvector(values: list[float]) -> str:
    """Text form of a vector for a `%s::vector` parameter."""
    return "[" + ",".join(f"{v:.6f}" for v in values) + "]"
