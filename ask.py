"""
Interactive CLI for asking questions to DocsRAG directly from the terminal.

Usage:
    python ask.py "How do I install LangChain?"
    python ask.py
"""

import sys
from dotenv import load_dotenv

load_dotenv()

from src.generation.generator import generate
from src.retrieval.reranker import rerank
from src.retrieval.retriever import retrieve


def ask_question(question: str):
    print(f"\nSearching documentation for: '{question}'...")
    sources = retrieve(question, "langchain_docs_recursive", top_k=10)
    sources = rerank(question, sources, keep_top_k=3)
    ans = generate(question, sources)

    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(ans.answer)
    print("=" * 60)
    print(f"Grounded in Docs: {ans.grounded} | Latency: {ans.latency_ms} ms")
    print(f"Sources cited ({len(ans.sources)}):")
    for s in ans.sources:
        print(f"  • [{s.doc_id}#{s.chunk_index}] (score: {s.score:.2f})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ask_question(" ".join(sys.argv[1:]))
    else:
        print("=" * 60)
        print(" DocsRAG Terminal Assistant")
        print(" Type your question and press Enter. Type 'exit' to quit.")
        print("=" * 60 + "\n")
        while True:
            try:
                q = input("Question > ").strip()
                if not q or q.lower() in ("exit", "quit"):
                    break
                ask_question(q)
            except (KeyboardInterrupt, EOFError):
                break
        print("\nGoodbye!")
