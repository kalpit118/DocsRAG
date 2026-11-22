"""
LLM and embedding utilities supporting OpenAI, Groq, and local sentence-transformers.
"""

from __future__ import annotations

import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer

_chat_client: OpenAI | None = None
_local_embedder: SentenceTransformer | None = None
_local_model_name: str | None = None


def get_chat_client() -> OpenAI:
    global _chat_client
    if _chat_client is None:
        api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GITHUB_TOKEN")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not base_url:
            if os.getenv("GROQ_API_KEY") or (api_key and api_key.startswith("gsk_")):
                base_url = "https://api.groq.com/openai/v1"
        _chat_client = OpenAI(api_key=api_key, base_url=base_url)
    return _chat_client


def get_default_chat_model() -> str:
    if os.getenv("CHAT_MODEL"):
        return os.environ["CHAT_MODEL"]
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "")
    if os.getenv("GROQ_API_KEY") or (api_key and api_key.startswith("gsk_")) or "groq.com" in base_url:
        return "qwen/qwen3.8-27b"
    return "gpt-4o-mini"


def get_local_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _local_embedder, _local_model_name
    if _local_embedder is None or _local_model_name != model_name:
        _local_embedder = SentenceTransformer(model_name)
        _local_model_name = model_name
    return _local_embedder
