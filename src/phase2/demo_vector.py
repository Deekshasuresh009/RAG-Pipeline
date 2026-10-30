"""
Phase 2, Step 8: ChromaDB Vector Retrieval Demo
Demonstration script retrieving top k semantic nearest neighbor chunks for a query.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.phase2.vector_retriever import (
    VectorRetriever,
    DEFAULT_DB_DIR,
    DEFAULT_COLLECTION_NAME
)


def run_demo(
    query: str,
    top_k: int = 5,
    persist_directory: str = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME
) -> None:
    """Run vector retrieval query and print formatted results to stdout."""
    print("=" * 80)
    print(" Aventro Motors RAG - Vector Retrieval Demo (Phase 2, Step 8)")
    print("=" * 80)
    print(f"Database Directory : {persist_directory}")
    print(f"Collection Name    : {collection_name}")
    print(f"Sample Query       : {query!r}")
    print(f"Top K              : {top_k}")
    print("-" * 80)

    try:
        retriever = VectorRetriever(
            persist_directory=persist_directory,
            collection_name=collection_name
        )
    except Exception as e:
        print(f"Error initializing vector retriever: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Total Chunks in Collection: {retriever.collection.count()}")
    print("-" * 80)

    try:
        results = retriever.retrieve(query, top_k=top_k)
    except Exception as e:
        print(f"Retrieval error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Retrieved Results: {len(results)} chunks")
    print("-" * 80)

    if not results:
        print("No matching chunks retrieved.")
        print("=" * 80)
        return

    for item in results:
        meta = item["citation_metadata"]
        raw_lines = [line.strip() for line in item["text"].split("\n") if line.strip()]
        # Skip top markdown heading if body lines exist for a better excerpt
        excerpt_lines = raw_lines[1:4] if len(raw_lines) > 1 else raw_lines[:2]
        excerpt = " ".join(excerpt_lines)
        if len(excerpt) > 220:
            excerpt = excerpt[:217] + "..."

        # Safe encoding for Windows consoles
        safe_source = meta.get("source_document", "").encode("ascii", "replace").decode("ascii")
        safe_title = meta.get("document_title", "").encode("ascii", "replace").decode("ascii")
        safe_heading = meta.get("section_heading", "").encode("ascii", "replace").decode("ascii")
        safe_excerpt = excerpt.encode("ascii", "replace").decode("ascii")

        print(f"[Rank {item['vector_rank']}] Cosine Distance: {item['vector_distance']:.6f}")
        print(f"  Source Document : {safe_source}")
        print(f"  Document Title  : {safe_title}")
        print(f"  Section Heading : {safe_heading}")
        print(f"  Chunk ID        : {item['chunk_id']}")
        print(f"  Excerpt         : {safe_excerpt}")
        print("-" * 80)

    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="ChromaDB Vector Retrieval Demo")
    parser.add_argument(
        "query",
        nargs="?",
        default="What are the 380 checkpoints for pre-owned cars?",
        help="Natural language query to retrieve semantic nearest neighbor chunks"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to return (default: 5)"
    )
    parser.add_argument(
        "--db-dir",
        type=str,
        default=DEFAULT_DB_DIR,
        help="Path to ChromaDB persistent storage directory"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help="ChromaDB collection name"
    )
    args = parser.parse_args()

    run_demo(
        query=args.query,
        top_k=args.top_k,
        persist_directory=args.db_dir,
        collection_name=args.collection
    )


if __name__ == "__main__":
    main()
