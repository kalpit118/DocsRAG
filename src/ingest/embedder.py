"""
Embedding and vector-store write step in the ingestion phase.

Called by scripts/ingest.py after src.ingest.chunker. It embeds chunks with
OpenAI or local sentence-transformers and upserts them into Chroma; the next phase is querying via src.api.main.
"""

from __future__ import annotations

import os
import uuid

import chromadb
from openai import OpenAI

from src.llm import get_chat_client, get_local_embedder

_client: OpenAI | None = None
_chroma: chromadb.PersistentClient | None = None


def _openai() -> OpenAI:
    global _client
    if _client is None:
        _client = get_chat_client()
    return _client


def _chroma_client(db_path: str) -> chromadb.PersistentClient:
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=db_path)
    return _chroma


def _embed_batch(texts: list[str], model: str) -> list[list[float]]:
    if _client is not None or model.startswith("text-embedding-") or model.startswith("openai/"):
        resp = _openai().embeddings.create(input=texts, model=model)
        return [r.embedding for r in resp.data]
    embedder = get_local_embedder(model)
    embeddings = embedder.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def upsert_chunks(
    chunks: list[dict],
    collection_name: str,
    db_path: str = "db/chroma",
    embed_model: str | None = None,
    batch_size: int = 100,
) -> int:
    """Embed and upsert chunks into a named Chroma collection. Returns count upserted."""
    embed_model = embed_model or os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    client = _chroma_client(db_path)
    col = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    upserted = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = _embed_batch(texts, embed_model)
        ids = [str(uuid.uuid4()) for _ in batch]
        metadatas = [c["metadata"] for c in batch]
        col.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        upserted += len(batch)

    return upserted
