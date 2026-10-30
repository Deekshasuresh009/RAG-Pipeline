"""
Phase 2, Step 7: BM25 Retrieval Demo
Interactive demonstration script displaying top 5 BM25 results for a natural language query.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.phase2.bm25_retriever import BM25Retriever, DEFAULT_CHUNKS_PATH


def run_demo(query: str, top_k: int = 5, chunks_path: Path = DEFAULT_CHUNKS_PATH) -> None:
    """Run BM25 query and print formatted results to stdout."""
    print("=" * 80)
    print(" Aventro Motors RAG - BM25 Keyword Retrieval Demo (Phase 2, Step 7)")
    print("=" * 80)
    print(f"Index Source : {chunks_path}")
    print(f"Sample Query : {query!r}")
    print(f"Top K        : {top_k}")
    print("-" * 80)

    try:
        retriever = BM25Retriever.from_jsonl(chunks_path)
    except Exception as e:
        print(f"Error loading index: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Total Chunks Indexed: {len(retriever.chunks)}")
    print("-" * 80)

    results = retriever.retrieve(query, top_k=top_k)

    if not results:
        print("No matching chunks found for this query.")
        print("=" * 80)
        return

    for item in results:
        meta = item["citation_metadata"]
        # Extract a clean, meaningful 2-3 line excerpt from actual text
        raw_lines = [line.strip() for line in item["text"].split("\n") if line.strip()]
        # Skip top markdown header if body lines exist
        excerpt_lines = raw_lines[1:4] if len(raw_lines) > 1 else raw_lines[:2]
        excerpt = " ".join(excerpt_lines)
        if len(excerpt) > 220:
            excerpt = excerpt[:217] + "..."

        # Format output safely for any Windows terminal encoding
        safe_source = meta['source_document'].encode("ascii", "replace").decode("ascii")
        safe_heading = meta['section_heading'].encode("ascii", "replace").decode("ascii")
        safe_excerpt = excerpt.encode("ascii", "replace").decode("ascii")

        print(f"[Rank {item['bm25_rank']}] Score: {item['bm25_score']:.4f}")
        print(f"  Source File    : {safe_source}")
        print(f"  Section Heading: {safe_heading}")
        print(f"  Chunk ID       : {item['chunk_id']}")
        print(f"  Excerpt        : {safe_excerpt}")
        print("-" * 80)

    print(f"Demo complete: {len(results)} chunks retrieved.")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 Keyword Retrieval Demo")
    parser.add_argument(
        "query",
        nargs="?",
        default="What are the 380 checkpoints for pre-owned cars?",
        help="Natural language query to retrieve against knowledge chunks"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to return (default: 5)"
    )
    parser.add_argument(
        "--chunks-path",
        type=str,
        default=str(DEFAULT_CHUNKS_PATH),
        help="Path to document chunks JSONL file"
    )
    args = parser.parse_args()

    run_demo(query=args.query, top_k=args.top_k, chunks_path=Path(args.chunks_path))


if __name__ == "__main__":
    main()
