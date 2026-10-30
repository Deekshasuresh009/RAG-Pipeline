"""Phase 2, Step 9: Reciprocal Rank Fusion (RRF) Demo

Interactive and command-line demonstration of hybrid BM25 + Vector retrieval fused using RRF.

Usage:
    python -m src.phase2.demo_rrf "What are the 380 checkpoints for pre-owned cars?"
"""

from __future__ import annotations

import sys
from typing import Optional

from src.phase2.rrf_fusion import (
    RRFFusion,
    DEFAULT_BM25_TOP_K,
    DEFAULT_VECTOR_TOP_K,
    DEFAULT_FINAL_TOP_K,
    DEFAULT_K,
)


def run_demo(
    query: str,
    bm25_top_k: int = DEFAULT_BM25_TOP_K,
    vector_top_k: int = DEFAULT_VECTOR_TOP_K,
    final_top_k: int = DEFAULT_FINAL_TOP_K,
    k: int = DEFAULT_K,
) -> None:
    print("=" * 80)
    print(" Aventro Motors RAG — Phase 2, Step 9: Reciprocal Rank Fusion (RRF) Demo")
    print("=" * 80)
    print(f"Query: {query!r}")
    print(f"Configuration: bm25_top_k={bm25_top_k}, vector_top_k={vector_top_k}, final_top_k={final_top_k}, k={k}")
    print("-" * 80)

    fusion = RRFFusion()

    # Retrieve components individually to show contributor stats
    bm25_results = fusion.bm25.retrieve(query, top_k=bm25_top_k)
    vector_results = fusion.vector.retrieve(query, top_k=vector_top_k)

    fused_results = fusion.fuse(
        query=query,
        bm25_top_k=bm25_top_k,
        vector_top_k=vector_top_k,
        final_top_k=final_top_k,
        k=k,
    )

    print(f"BM25 retriever matches      : {len(bm25_results)}")
    print(f"Vector retriever matches    : {len(vector_results)}")
    print(f"Unique fused results count  : {len(fused_results)}")
    print("=" * 80)

    if not fused_results:
        print("No matching results found for this query.")
        return

    for res in fused_results:
        meta = res.get("citation_metadata", {})
        contributors = ", ".join(res.get("contributors", []))
        text = res.get("text", "").strip()
        first_line = text.split("\n")[0][:120] if text else ""

        ranks_str = []
        if "bm25_rank" in res:
            score = res.get("bm25_score")
            score_str = f" (score: {score:.4f})" if score is not None else ""
            ranks_str.append(f"BM25 rank #{res['bm25_rank']}{score_str}")
        if "vector_rank" in res:
            dist = res.get("vector_distance")
            dist_str = f" (dist: {dist:.4f})" if dist is not None else ""
            ranks_str.append(f"Vector rank #{res['vector_rank']}{dist_str}")

        ranks_desc = "; ".join(ranks_str) if ranks_str else "N/A"

        print(f"Fused Rank #{res['rrf_rank']} | RRF Score: {res['rrf_score']:.6f} | Contributors: [{contributors}]")
        print(f"  Contributing Ranks : {ranks_desc}")
        print(f"  Chunk ID           : {res['chunk_id']}")
        print(f"  Document Title     : {meta.get('document_title', 'N/A')}")
        print(f"  Source Document    : {meta.get('source_document', 'N/A')}")
        print(f"  Section Heading    : {meta.get('section_heading', 'N/A')}")
        print(f"  Heading Path       : {meta.get('heading_path', 'N/A')}")
        print(f"  Text Excerpt       : {first_line}...")
        print("-" * 80)


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "What are the 380 checkpoints for pre-owned cars?"
    bm25_k = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_BM25_TOP_K
    vector_k = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_VECTOR_TOP_K
    final_k = int(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_FINAL_TOP_K
    k_val = int(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_K

    run_demo(
        query=query,
        bm25_top_k=bm25_k,
        vector_top_k=vector_k,
        final_top_k=final_k,
        k=k_val,
    )


if __name__ == "__main__":
    main()
