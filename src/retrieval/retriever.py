"""
Vector retrieval step in the query phase.

Called by src.api.main or src.agent.loop. It embeds the user question, searches
the selected Chroma collection, and returns Source objects for optional reranking.
"""

from __future__ import annotations

import os

import chromadb
from openai import OpenAI

from src.api.schemas import Source
from src.llm import get_chat_client, get_local_embedder

_chroma: chromadb.PersistentClient | None = None
_openai: OpenAI | None = None


def _get_chroma(db_path: str) -> chromadb.PersistentClient:
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=db_path)
    return _chroma


def _get_openai() -> OpenAI:
    global _openai
    if _openai is None:
        _openai = get_chat_client()
    return _openai


def _embed_query(query: str, model: str) -> list[float]:
    if _openai is not None or model.startswith("text-embedding-") or model.startswith("openai/"):
        resp = _get_openai().embeddings.create(input=[query], model=model)
        return resp.data[0].embedding
    embedder = get_local_embedder(model)
    emb = embedder.encode(query, normalize_embeddings=True)
    return emb.tolist()


def retrieve(
    query: str,
    collection_name: str | list[str],
    top_k: int = 20,
    db_path: str = "db/chroma",
    embed_model: str = "all-MiniLM-L6-v2",
) -> list[Source]:
    """Dense retrieval from Chroma. Returns top_k Sources sorted by score desc."""
    col_names = [collection_name] if isinstance(collection_name, str) else collection_name
    client = _get_chroma(db_path)
    qemb = _embed_query(query, embed_model)

    all_sources: list[Source] = []
    for cname in col_names:
        try:
            col = client.get_collection(cname)
        except Exception:
            continue

        count = col.count()
        if count == 0:
            continue

        results = col.query(
            query_embeddings=[qemb],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        if not results or not results.get("documents") or not results["documents"][0]:
            continue

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        for doc, meta, dist in zip(docs, metas, dists):
            all_sources.append(
                Source(
                    doc_id=meta.get("source", ""),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    text=doc,
                    score=float(1 - dist),  # cosine distance → similarity
                    header_path=meta.get("header_path"),
                )
            )

    all_sources.sort(key=lambda s: s.score, reverse=True)
    return all_sources[:top_k]

