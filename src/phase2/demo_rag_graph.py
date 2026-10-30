"""
Phase 2, Step 11: LangGraph RAG Pipeline Demonstration
Executes the compiled LangGraph workflow against the real Aventro Motors corpus.

Usage:
    python -m src.phase2.demo_rag_graph "What are the 380 checkpoints for pre-owned cars?"
"""

from __future__ import annotations

import sys
from typing import Optional

from src.phase2.rag_graph import build_rag_graph
from src.phase2.rag_state import (
    WORKFLOW_READY_FOR_GENERATION,
    WORKFLOW_REQUIRES_ABSTENTION,
    WORKFLOW_REQUIRES_FURTHER_RETRIEVAL,
    WORKFLOW_REQUIRES_CONFLICT_RESOLUTION,
    WORKFLOW_VALIDATION_FAILED,
    WORKFLOW_ERROR,
)


def run_demo(query: str, top_k: int = 5) -> None:
    print("=" * 80)
    print(" Aventro Motors RAG — Phase 2, Step 11: LangGraph Pipeline Demo")
    print("=" * 80)
    print(f"User Query : {query!r}")
    print(f"Top K      : {top_k}")
    print("-" * 80)

    # 1. Build and compile the LangGraph workflow
    app = build_rag_graph(top_k=top_k)

    # 2. Execute graph
    initial_state = {
        "query": query,
        "trace": [],
    }
    final_state = app.invoke(initial_state)

    workflow_status = final_state.get("workflow_status", "UNKNOWN")
    error = final_state.get("error")
    retrieved = final_state.get("retrieved_results", [])
    assessment = final_state.get("evidence_assessment") or {}
    evidence_status = assessment.get("status", "N/A")
    relevant = assessment.get("relevant_evidence", [])
    irrelevant = assessment.get("irrelevant_evidence", [])
    missing = assessment.get("missing_information", [])
    conflicts = assessment.get("conflicts", [])
    trace = final_state.get("trace", [])

    print(f"Final Workflow Status    : {workflow_status.upper()}")
    print(f"Evidence Assessment Status: {evidence_status.upper()}")
    print(f"Retrieved Candidates     : {len(retrieved)}")
    print(f"Relevant Evidence Count  : {len(relevant)}")
    print(f"Irrelevant Evidence Count: {len(irrelevant)}")

    if error:
        print(f"Error Message            : {error}")

    # Explanation when not ready for generation
    if workflow_status == WORKFLOW_READY_FOR_GENERATION:
        print("\n>>> Pipeline Decision: Evidence is SUFFICIENT. Ready for grounded answer generation.")
    elif workflow_status == WORKFLOW_REQUIRES_ABSTENTION:
        print("\n>>> Pipeline Decision: ABSTAIN FROM GENERATING ANSWER.")
        print(f"    Explanation: The retrieved documents do not provide sufficient factual support.")
        if missing:
            print(f"    Missing Information: {missing}")
    elif workflow_status == WORKFLOW_REQUIRES_FURTHER_RETRIEVAL:
        print("\n>>> Pipeline Decision: PARTIAL EVIDENCE DETECTED.")
        print(f"    Explanation: The query has multiple parts and some requested facts are missing.")
        if missing:
            print(f"    Missing Information: {missing}")
    elif workflow_status == WORKFLOW_REQUIRES_CONFLICT_RESOLUTION:
        print("\n>>> Pipeline Decision: CONFLICT DETECTED.")
        print(f"    Explanation: Retrieved evidence contains contradictory claims that require resolution.")
        if conflicts:
            print(f"    Conflicts: {conflicts}")
    elif workflow_status in (WORKFLOW_VALIDATION_FAILED, WORKFLOW_ERROR):
        print(f"\n>>> Pipeline Decision: WORKFLOW HALTED due to error: {error}")

    # Display relevant evidence details
    if relevant:
        print("\n" + "=" * 80)
        print(" RELEVANT EVIDENCE EXCERPTS")
        print("=" * 80)
        for idx, item in enumerate(relevant, start=1):
            meta = item.get("citation_metadata", {})
            first_line = item.get("text", "").strip().split("\n")[0][:110]
            print(f"[{idx}] Chunk ID        : {item['chunk_id']}")
            print(f"    Source Document : {item.get('source_document')}")
            print(f"    Section Heading : {item.get('section_heading')}")
            print(f"    Heading Path    : {item.get('heading_path')}")
            print(f"    Excerpt         : {first_line}...")
            print("-" * 80)

    # Display execution trace summary
    print("\n" + "=" * 80)
    print(" EXECUTION TRACE")
    print("=" * 80)
    for step in trace:
        print(f"  • {step}")
    print("=" * 80)


def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "What are the 380 checkpoints for pre-owned cars?"
    run_demo(query)


if __name__ == "__main__":
    main()
