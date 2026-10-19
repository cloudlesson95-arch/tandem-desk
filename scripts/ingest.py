"""Split data/policies/*.md, embed new chunks with gemini-embedding-001 and upsert them
into `chunks` by content hash. Chunks whose text no longer exists are removed."""
import asyncio
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psycopg  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from agent.config import database_url  # noqa: E402
from agent.embed import embed_documents, to_pgvector  # noqa: E402


def load_chunks() -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450, chunk_overlap=60, separators=["\n## ", "\n\n", "\n", ". ", " "]
    )
    chunks = []
    for path in sorted((ROOT / "data" / "policies").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip()
        for piece in splitter.split_text(text):
            piece = piece.strip()
            # Every chunk carries its document title so it reads on its own.
            content = piece if piece.startswith("# ") else f"# {title}\n{piece}"
            digest = hashlib.sha256(f"{path.stem}\n{content}".encode()).hexdigest()[:16]
            chunks.append({"id": digest, "doc": path.stem, "title": title, "content": content})
    return chunks


def main() -> None:
    chunks = load_chunks()
    with psycopg.connect(database_url()) as conn:
        existing = {r[0] for r in conn.execute("SELECT id FROM chunks").fetchall()}
        new = [c for c in chunks if c["id"] not in existing]
        if new:
            vectors = asyncio.run(embed_documents([c["content"] for c in new]))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (id, doc, title, content, embedding)"
                    " VALUES (%s, %s, %s, %s, %s::vector) ON CONFLICT (id) DO NOTHING",
                    [(c["id"], c["doc"], c["title"], c["content"], to_pgvector(v))
                     for c, v in zip(new, vectors)],
                )
        stale = conn.execute(
            "DELETE FROM chunks WHERE NOT (id = ANY(%s))", ([c["id"] for c in chunks],)
        ).rowcount
        conn.commit()
    docs = len({c["doc"] for c in chunks})
    print(f"{len(chunks)} chunks from {docs} documents: {len(new)} new, {stale} removed")


if __name__ == "__main__":
    main()
