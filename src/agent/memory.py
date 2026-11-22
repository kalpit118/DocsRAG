"""
Multi-turn conversational memory and query reformulation utilities.

Takes the recent conversation context to rewrite follow-up questions
into standalone search queries before vector retrieval.
"""

from __future__ import annotations

from openai import OpenAI

from src.llm import get_chat_client, get_default_chat_model

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = get_chat_client()
    return _client


REWRITE_PROMPT = """\
Given a chat conversation history and a follow-up question from the user, rewrite the follow-up question into a clear, standalone technical search query that includes all necessary context and entity names mentioned earlier.

Rules:
- Do NOT answer the question.
- Do NOT add explanations or preamble.
- Output ONLY the standalone search query.
- If the question is already complete and standalone, return it exactly as is.

Chat History:
{history}

Follow-up Question: {question}

Standalone Query:"""


def reformulate_query(
    question: str,
    history: list[dict],
    model: str | None = None,
) -> str:
    """
    Rewrite a follow-up question into a standalone query using session history.
    If history is empty or reformulation fails, returns the original question.
    """
    if not history:
        return question

    # Format history turns (last 6 turns max)
    recent = history[-6:]
    history_lines = []
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        # Truncate assistant content to avoid excessive prompt size
        content = msg.get("content", "")[:300].strip()
        history_lines.append(f"{role}: {content}")

    formatted_history = "\n".join(history_lines)
    prompt = REWRITE_PROMPT.format(history=formatted_history, question=question)

    try:
        chat_model = model or get_default_chat_model()
        resp = _get_client().chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        # Sanity check on rewritten query
        if rewritten and len(rewritten) > 3 and not rewritten.lower().startswith("here"):
            return rewritten
    except Exception:
        pass

    return question
