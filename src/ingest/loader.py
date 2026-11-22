"""
Document loading step in the ingestion phase.

Called by scripts/ingest.py. It finds Markdown files under data/docs and returns
their path/text pairs; the next step is src.ingest.chunker.
"""

from pathlib import Path


def load_docs(docs_dir: str | Path) -> list[dict]:
    """Walk docs_dir and return list of {path, text} for every .md and .mdx file."""
    root = Path(docs_dir)
    docs = []
    files = sorted(set(list(root.rglob("*.md")) + list(root.rglob("*.mdx"))))
    for p in files:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            continue
        if text:
            rel_path = str(p.relative_to(root)).replace("\\", "/")
            docs.append({"path": rel_path, "text": text})
    return docs
