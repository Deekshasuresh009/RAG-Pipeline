"""
Phase 2, Step 10: Evidence Relevance and Sufficiency Demonstration
Accepts a user query, runs hybrid RRF retrieval, evaluates evidence relevance and sufficiency,
and displays a structured assessment report with citation details.

Usage:
    python -m src.phase2.demo_evidence "What are the 380 checkpoints for pre-owned cars?"
"""

from __future__ import annotations

import sys
from typing import Optional

from src.phase2.rrf_fusion import RRFFusion
from src.phase2.evidence_evaluator import EvidenceEvaluator


def run_demo(query: str, top_k: int = 5) -> None:
    print("=" * 80)
    print(" Aventro Motors RAG — Phase 2, Step 10: Evidence Relevance & Sufficiency Demo")
    print("=" * 80)
    print(f"Query: {query!r}")
    print(f"Retrieval Pool: top {top_k} from BM25 and Vector fused with RRF (k=60)")
    print("-" * 80)

    # 1. Hybrid Retrieval via RRF
    rrf = RRFFusion()
    retrieved_results = rrf.fuse(
        query=query,
        bm25_top_k=top_k,
        vector_top_k=top_k,
        final_top_k=top_k,
    )

    print(f"Retrieved candidate chunks : {len(retrieved_results)}")

    # 2. Evidence Relevance and Sufficiency Evaluation
    evaluator = EvidenceEvaluator()
    assessment = evaluator.evaluate(query=query, retrieved_results=retrieved_results)

    status = assessment["status"]
    is_sufficient = assessment["is_sufficient"]
    reason = assessment["reason"]
    relevant = assessment["relevant_evidence"]
    irrelevant = assessment["irrelevant_evidence"]
    missing = assessment["missing_information"]
    conflicts = assessment["conflicts"]

    print(f"Evidence Status            : {status.upper()}")
    print(f"Is Evidence Sufficient     : {is_sufficient}")
    print(f"Decision Reason            : {reason}")
    print(f"Relevant Chunks Count      : {len(relevant)} (IDs: {[c['chunk_id'] for c in relevant]})")
    print(f"Irrelevant Chunks Count    : {len(irrelevant)} (IDs: {[c['chunk_id'] for c in irrelevant]})")

    if missing:
        print(f"Missing Information        : {missing}")

    if conflicts:
        print(f"Detected Conflicts ({len(conflicts)}) :")
        for conf in conflicts:
            print(f"  - [{conf.get('type')}] {conf.get('description')}")

    print("=" * 80)
    print(" EVALUATED CHUNKS BREAKDOWN")
    print("=" * 80)

    all_assessed = relevant + irrelevant
    if not all_assessed:
        print("No chunks to display.")
        return

    for idx, c in enumerate(all_assessed, start=1):
        rel_flag = "RELEVANT" if c["is_relevant"] else "IRRELEVANT"
        meta = c.get("citation_metadata", {})
        first_line = c["text"].strip().split("\n")[0][:110]
        rank_str = f"RRF Rank #{c.get('rrf_rank')}" if c.get("rrf_rank") else "N/A"
        score_str = f"Score: {c.get('rrf_score', 0):.5f}" if c.get("rrf_score") else ""

        print(f"[{idx}] {rel_flag} | {c['chunk_id']} | {rank_str} {score_str}")
        print(f"    Source Document : {c.get('source_document')}")
        print(f"    Section Heading : {c.get('section_heading')}")
        print(f"    Heading Path    : {c.get('heading_path')}")
        print(f"    Assessment Note : {c.get('relevance_reason')}")
        print(f"    Text Excerpt    : {first_line}...")
        print("-" * 80)


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "What are the 380 checkpoints for pre-owned cars?"
    run_demo(query)


if __name__ == "__main__":
    main()
