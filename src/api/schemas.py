"""
Shared API data contracts.

Used by src.api.main to validate incoming requests and shape outgoing responses.
The same Source and RAGAnswer models are passed through retrieval and generation.
"""

from typing import Literal

from pydantic import BaseModel, Field


class Source(BaseModel):
    doc_id: str
    chunk_index: int
    text: str
    score: float
    header_path: str | None = None


class RAGAnswer(BaseModel):
    answer: str
    sources: list[Source]
    grounded: bool = Field(description="True when the answer is fully supported by retrieved context")
    latency_ms: float
    session_id: str | None = None


class QueryRequest(BaseModel):
    question: str
    chunking: Literal["recursive", "header_aware"] = "recursive"
    use_reranker: bool = True
    top_k: int = Field(default=5, ge=1, le=20)
    corpus: Literal["all", "langchain", "user"] = "all"
    session_id: str | None = None


class DocumentItem(BaseModel):
    filename: str
    size_bytes: int
    chunks: int
    modified_at: float


class ChatMessageItem(BaseModel):
    id: int
    session_id: str
    role: str
    content: str
    sources: list[Source] = []
    grounded: bool | None = None
    latency_ms: float | None = None
    timestamp: str | None = None


class RawDocumentResponse(BaseModel):
    doc_id: str
    filename: str
    source_type: str
    content: str
    size_bytes: int


class EvalResult(BaseModel):
    id: int
    config: str
    hit_at_1: bool
    hit_at_5: bool
    hit_at_10: bool
    mrr: float
    faithfulness: float
    avg_latency_ms: float
    timestamp: str
