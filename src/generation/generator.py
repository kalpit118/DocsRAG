"""
Answer generation step in the query phase.

Called by src.api.main after retrieval/reranking. It builds the context prompt,
calls the chat LLM, asks src.eval.judge to check groundedness, then logs through
src.eval.logger before returning a RAGAnswer.
"""

from __future__ import annotations

import json
import os
import time

from openai import OpenAI

from typing import Generator

from src.api.schemas import RAGAnswer, Source
from src.eval.judge import score_faithfulness
from src.eval.logger import log_query, save_chat_message
from src.generation.prompts import SYSTEM_PROMPT
from src.llm import get_chat_client, get_default_chat_model

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = get_chat_client()
    return _client


def _build_context(sources: list[Source]) -> str:
    parts = []
    for s in sources:
        tag = f"[{s.doc_id}#{s.chunk_index}]"
        parts.append(f"{tag}\n{s.text}")
    return "\n\n---\n\n".join(parts)


def _build_messages(
    question: str,
    sources: list[Source],
    history: list[dict] | None = None,
) -> list[dict[str, str]]:
    context = _build_context(sources)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for h in history[-6:]:
            role = "user" if h.get("role") == "user" else "assistant"
            content = h.get("content", "")
            if content:
                messages.append({"role": role, "content": content})
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}",
    })
    return messages


def generate(
    question: str,
    sources: list[Source],
    chunking: str = "recursive",
    use_reranker: bool = True,
    model: str | None = None,
    history: list[dict] | None = None,
    session_id: str | None = None,
) -> RAGAnswer:
    chat_model = model or get_default_chat_model()
    messages = _build_messages(question, sources, history)

    t0 = time.perf_counter()
    resp = _get_client().chat.completions.create(
        model=chat_model,
        messages=messages,
        temperature=0,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    answer_text = resp.choices[0].message.content or ""

    grounded, _ = score_faithfulness(question, answer_text, sources)

    log_query(
        question=question,
        chunking=chunking,
        use_reranker=use_reranker,
        answer=answer_text,
        grounded=grounded,
        latency_ms=latency_ms,
    )

    if session_id:
        save_chat_message(session_id=session_id, role="user", content=question)
        save_chat_message(
            session_id=session_id,
            role="assistant",
            content=answer_text,
            sources=[s.model_dump() for s in sources],
            grounded=grounded,
            latency_ms=latency_ms,
        )

    return RAGAnswer(
        answer=answer_text,
        sources=sources,
        grounded=grounded,
        latency_ms=round(latency_ms, 1),
        session_id=session_id,
    )


def generate_stream(
    question: str,
    sources: list[Source],
    model: str | None = None,
    history: list[dict] | None = None,
) -> Generator[str, None, None]:
    """Yield incremental text tokens for Server-Sent Events (SSE) streaming."""
    chat_model = model or get_default_chat_model()
    messages = _build_messages(question, sources, history)

    stream = _get_client().chat.completions.create(
        model=chat_model,
        messages=messages,
        temperature=0,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

