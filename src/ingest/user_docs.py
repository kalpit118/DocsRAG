"""
User document ingestion and management.

Handles file upload persistence, parsing (.pdf, .md, .txt), chunking,
embedding into Chroma user collections, and document deletion.
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Any

import chromadb
from pypdf import PdfReader

from src.ingest.chunker import header_aware_chunks, recursive_chunks
from src.ingest.embedder import upsert_chunks

USER_UPLOAD_DIR = Path("data/user_uploads")
USER_COLLECTION_RECURSIVE = "user_docs_recursive"
USER_COLLECTION_HEADER = "user_docs_header"


def get_user_upload_dir() -> Path:
    """Ensure and return the user uploads directory."""
    USER_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return USER_UPLOAD_DIR


def extract_text(filename: str, content_bytes: bytes) -> str:
    """Extract plain text from bytes based on file extension."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(content_bytes))
        pages_text = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(f"--- [Page {i}] ---\n{text.strip()}")
        return "\n\n".join(pages_text)
    
    # Text / Markdown / Code fallback
    try:
        return content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return content_bytes.decode("latin-1", errors="replace")


def ingest_user_document(
    filename: str,
    content_bytes: bytes,
    db_path: str = "db/chroma",
    embed_model: str | None = None,
) -> dict[str, Any]:
    """
    Save uploaded file to disk and index into both recursive and header-aware
    user Chroma collections.
    """
    upload_dir = get_user_upload_dir()
    # Sanitize filename
    safe_name = Path(filename).name
    target_path = upload_dir / safe_name

    target_path.write_bytes(content_bytes)
    extracted_text = extract_text(safe_name, content_bytes)

    if not extracted_text.strip():
        raise ValueError(f"No extractable text found in '{safe_name}'.")

    # Generate chunks
    # For markdown, header_aware works nicely; for non-markdown, fallback to recursive chunks
    rec_chunks = recursive_chunks(extracted_text, source=safe_name)
    
    ext = Path(safe_name).suffix.lower()
    if ext in (".md", ".mdx"):
        hdr_chunks = header_aware_chunks(extracted_text, source=safe_name)
    else:
        hdr_chunks = rec_chunks

    # Index into Chroma
    count_rec = upsert_chunks(
        rec_chunks,
        collection_name=USER_COLLECTION_RECURSIVE,
        db_path=db_path,
        embed_model=embed_model,
    )
    count_hdr = upsert_chunks(
        hdr_chunks,
        collection_name=USER_COLLECTION_HEADER,
        db_path=db_path,
        embed_model=embed_model,
    )

    return {
        "filename": safe_name,
        "size_bytes": len(content_bytes),
        "chunks_recursive": count_rec,
        "chunks_header": count_hdr,
        "indexed_at": time.time(),
    }


def list_user_documents(db_path: str = "db/chroma") -> list[dict[str, Any]]:
    """List all uploaded user documents, their file sizes, and indexed chunk counts."""
    upload_dir = get_user_upload_dir()
    files = list(upload_dir.glob("*"))
    if not files:
        return []

    client = chromadb.PersistentClient(path=db_path)
    try:
        col = client.get_collection(USER_COLLECTION_RECURSIVE)
        all_metadata = col.get(include=["metadatas"])["metadatas"] or []
    except Exception:
        all_metadata = []

    # Count chunks per source filename
    counts: dict[str, int] = {}
    for meta in all_metadata:
        src = meta.get("source")
        if src:
            counts[src] = counts.get(src, 0) + 1

    docs = []
    for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
        if f.is_file():
            stat = f.stat()
            docs.append({
                "filename": f.name,
                "size_bytes": stat.st_size,
                "chunks": counts.get(f.name, 0),
                "modified_at": stat.st_mtime,
            })
    return docs


def delete_user_document(filename: str, db_path: str = "db/chroma") -> bool:
    """Delete a user document from ChromaDB collections and from disk."""
    safe_name = Path(filename).name
    client = chromadb.PersistentClient(path=db_path)

    for col_name in [USER_COLLECTION_RECURSIVE, USER_COLLECTION_HEADER]:
        try:
            col = client.get_collection(col_name)
            col.delete(where={"source": safe_name})
        except Exception:
            pass

    file_path = get_user_upload_dir() / safe_name
    if file_path.exists():
        file_path.unlink()
        return True
    return False


def get_raw_document(doc_id: str, base_docs_path: str = "data/docs") -> dict[str, Any]:
    """
    Safely retrieve the raw content of a document by doc_id.
    Guards strictly against directory traversal.
    """
    # Check in user_uploads first
    user_file = (get_user_upload_dir() / Path(doc_id).name).resolve()
    if user_file.exists() and user_file.is_file():
        # Verify it is inside USER_UPLOAD_DIR
        if USER_UPLOAD_DIR.resolve() in user_file.parents:
            content = extract_text(user_file.name, user_file.read_bytes())
            return {
                "doc_id": doc_id,
                "filename": user_file.name,
                "source_type": "user_upload",
                "content": content,
                "size_bytes": user_file.stat().st_size,
            }

    # Check in base docs corpus
    docs_base = Path(base_docs_path).resolve()
    target = (docs_base / doc_id).resolve()
    
    # Path traversal check
    if docs_base not in target.parents and target != docs_base:
        raise PermissionError(f"Access denied to path: {doc_id}")

    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"Document not found: {doc_id}")

    content = extract_text(target.name, target.read_bytes())
    return {
        "doc_id": doc_id,
        "filename": target.name,
        "source_type": "corpus",
        "content": content,
        "size_bytes": target.stat().st_size,
    }
