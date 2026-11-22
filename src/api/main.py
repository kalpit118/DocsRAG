"""
FastAPI entry point for the DocsRAG system.

Exposes REST and SSE streaming endpoints for documentation Q&A, multi-turn
sessions, file uploads, document management, and evaluation logs.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

load_dotenv()

from src.agent.memory import reformulate_query
from src.api.schemas import (
    ChatMessageItem,
    DocumentItem,
    EvalResult,
    QueryRequest,
    RAGAnswer,
    RawDocumentResponse,
)
from src.eval.judge import score_faithfulness
from src.eval.logger import (
    clear_session_messages,
    get_latest_evals,
    get_session_messages,
    log_query,
    save_chat_message,
)
from src.generation.generator import generate, generate_stream
from src.ingest.user_docs import (
    USER_COLLECTION_HEADER,
    USER_COLLECTION_RECURSIVE,
    delete_user_document,
    get_raw_document,
    ingest_user_document,
    list_user_documents,
)
from src.retrieval.reranker import rerank
from src.retrieval.retriever import retrieve

app = FastAPI(
    title="DocsRAG",
    description="Retrieval-Augmented Documentation Assistant",
    version="0.2.0",
)

_COLLECTION = {
    "recursive": os.getenv("COLLECTION_RECURSIVE", "langchain_docs_recursive"),
    "header_aware": os.getenv("COLLECTION_HEADER", "langchain_docs_header"),
}
_DB_PATH = os.getenv("CHROMA_DB_PATH", "db/chroma")
_EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
_HTML_PATH = Path(__file__).parent / "index.html"


def _get_target_collections(chunking: str, corpus: str = "all") -> list[str]:
    base_col = _COLLECTION.get(chunking) or "langchain_docs_recursive"
    user_col = (
        USER_COLLECTION_RECURSIVE
        if chunking == "recursive"
        else USER_COLLECTION_HEADER
    )
    if corpus == "langchain":
        return [base_col]
    elif corpus == "user":
        return [user_col]
    return [base_col, user_col]


@app.get("/", response_class=HTMLResponse)
def home():
    if _HTML_PATH.exists():
        return _HTML_PATH.read_text(encoding="utf-8")
    return "<h1>DocsRAG API is running. Visit <a href='/docs'>/docs</a> for Swagger UI.</h1>"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=RAGAnswer)
def ask(req: QueryRequest) -> RAGAnswer:
    collections = _get_target_collections(req.chunking, req.corpus)
    session_id = req.session_id or str(uuid.uuid4())

    history = get_session_messages(session_id, limit=6) if req.session_id else []
    search_query = reformulate_query(req.question, history) if history else req.question

    try:
        sources = retrieve(
            search_query,
            collections if len(collections) > 1 else collections[0],
            top_k=req.top_k * 4 if req.use_reranker else req.top_k,
            db_path=_DB_PATH,
            embed_model=_EMBED_MODEL,
        )
        if req.use_reranker and sources:
            sources = rerank(search_query, sources, keep_top_k=req.top_k)
        else:
            sources = sources[: req.top_k]

        return generate(
            question=req.question,
            sources=sources,
            chunking=req.chunking,
            use_reranker=req.use_reranker,
            history=history,
            session_id=session_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask/stream")
def ask_stream(req: QueryRequest):
    """
    Server-Sent Events (SSE) streaming endpoint.
    Emits metadata (sources), text delta tokens, and a final verification done event.
    """
    collections = _get_target_collections(req.chunking, req.corpus)
    session_id = req.session_id or str(uuid.uuid4())
    history = get_session_messages(session_id, limit=6) if req.session_id else []
    search_query = reformulate_query(req.question, history) if history else req.question

    def event_generator():
        t0 = time.perf_counter()
        try:
            sources = retrieve(
                search_query,
                collections if len(collections) > 1 else collections[0],
                top_k=req.top_k * 4 if req.use_reranker else req.top_k,
                db_path=_DB_PATH,
                embed_model=_EMBED_MODEL,
            )
            if req.use_reranker and sources:
                sources = rerank(search_query, sources, keep_top_k=req.top_k)
            else:
                sources = sources[: req.top_k]

            retrieval_latency = (time.perf_counter() - t0) * 1000

            # 1. Emit metadata event
            meta_payload = {
                "session_id": session_id,
                "sources": [s.model_dump() for s in sources],
                "retrieval_latency_ms": round(retrieval_latency, 1),
            }
            yield f"event: metadata\ndata: {json.dumps(meta_payload)}\n\n"

            # 2. Emit streaming tokens
            full_text_chunks = []
            for token in generate_stream(
                question=req.question,
                sources=sources,
                history=history,
            ):
                full_text_chunks.append(token)
                yield f"event: token\ndata: {json.dumps({'delta': token})}\n\n"

            full_answer = "".join(full_text_chunks)
            total_latency = (time.perf_counter() - t0) * 1000

            # 3. Faithfulness score & persistence
            grounded, faith_score = score_faithfulness(req.question, full_answer, sources)

            log_query(
                question=req.question,
                chunking=req.chunking,
                use_reranker=req.use_reranker,
                answer=full_answer,
                grounded=grounded,
                latency_ms=total_latency,
            )
            save_chat_message(
                session_id=session_id,
                role="user",
                content=req.question,
            )
            save_chat_message(
                session_id=session_id,
                role="assistant",
                content=full_answer,
                sources=[s.model_dump() for s in sources],
                grounded=grounded,
                latency_ms=total_latency,
            )

            # 4. Emit done event
            done_payload = {
                "session_id": session_id,
                "grounded": grounded,
                "faithfulness_score": faith_score,
                "total_latency_ms": round(total_latency, 1),
            }
            yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"

        except Exception as err:
            err_payload = {"error": str(err)}
            yield f"event: error\ndata: {json.dumps(err_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- Document Upload & Management Endpoints ---


@app.post("/documents/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    """Upload one or more user documents (.pdf, .md, .txt) and index into Chroma."""
    results = []
    for file in files:
        if not file.filename:
            continue
        ext = Path(file.filename).suffix.lower()
        if ext not in (".pdf", ".md", ".mdx", ".txt", ".json"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format '{ext}'. Allowed: .pdf, .md, .mdx, .txt",
            )
        content = await file.read()
        if len(content) > 25 * 1024 * 1024:  # 25 MB limit
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' exceeds 25 MB size limit.",
            )
        try:
            doc_meta = ingest_user_document(
                filename=file.filename,
                content_bytes=content,
                db_path=_DB_PATH,
                embed_model=_EMBED_MODEL,
            )
            results.append(doc_meta)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to ingest '{file.filename}': {e}") from e

    return {"uploaded": results, "total_files": len(results)}


@app.get("/documents", response_model=list[DocumentItem])
def get_documents():
    """List all uploaded user documents, sizes, and indexed chunk counts."""
    return list_user_documents(db_path=_DB_PATH)


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """Delete a user document from ChromaDB collections and storage."""
    deleted = delete_user_document(filename, db_path=_DB_PATH)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")
    return {"status": "deleted", "filename": filename}


@app.get("/documents/raw", response_model=RawDocumentResponse)
def get_raw_doc(doc_id: str = Query(..., description="Document ID or path")):
    """
    Retrieve full raw text for in-app context reading and highlight inspection.
    Enforces strict path traversal prevention.
    """
    try:
        return get_raw_document(doc_id)
    except PermissionError as pe:
        raise HTTPException(status_code=403, detail=str(pe)) from pe
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=str(fe)) from fe
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# --- Session History & Evaluation Endpoints ---


@app.get("/sessions/{session_id}/messages")
def get_session_history(session_id: str):
    """Retrieve message history for an ongoing multi-turn session."""
    return get_session_messages(session_id)


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    """Clear chat messages for a session."""
    count = clear_session_messages(session_id)
    return {"status": "cleared", "messages_removed": count}


@app.get("/eval/latest", response_model=list[EvalResult])
def eval_latest(n: int = Query(default=10, ge=1, le=100)):
    return get_latest_evals(n)
