"""
Phase 2, Step 12: Grounded Answer Generation with Citations Demonstration
Executes the full LangGraph RAG pipeline against the real Aventro Motors corpus,
generating grounded answers with inline citations for sufficient evidence,
or executing explicit abstentions for missing/unsupported/partial topics.

Usage:
    python -m src.phase2.demo_rag_answer "What are the 380 checkpoints for pre-owned cars?"
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from src.phase2.rag_graph import build_rag_graph
from src.phase2.answer_generator import (
    AnswerGenerator,
    FakeLLMClient,
    OllamaClient,
    format_rag_response,
)
from src.phase2.rag_state import (
    WORKFLOW_COMPLETED,
    WORKFLOW_REQUIRES_ABSTENTION,
    WORKFLOW_REQUIRES_FURTHER_RETRIEVAL,
    WORKFLOW_REQUIRES_CONFLICT_RESOLUTION,
    WORKFLOW_VALIDATION_FAILED,
    WORKFLOW_ERROR,
)


def get_configured_generator() -> tuple[AnswerGenerator, str]:
    """
    Configure AnswerGenerator based on environment or availability.
    Attempts local Ollama first; if unreachable, falls back to FakeLLMClient
    with clear diagnostic labeling so offline demos function transparently.
    """
    prov_env = os.getenv("LLM_PROVIDER")
    if prov_env == "fake":
        return AnswerGenerator(llm_client=FakeLLMClient()), "FakeLLMClient (explicitly configured via LLM_PROVIDER=fake)"

    # Test if Ollama is accessible
    ollama_client = OllamaClient()
    try:
        import urllib.request
        req = urllib.request.Request(f"{ollama_client.host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.status == 200:
                return AnswerGenerator(llm_client=ollama_client), f"Ollama ({ollama_client.model} @ {ollama_client.host})"
    except Exception:
        pass

    # When Ollama is unreachable, use FakeLLMClient with explicit notice
    notice = (
        "FakeLLMClient (Ollama is not running locally; using deterministic offline fake model. "
        "To use real local LLM, start Ollama with 'ollama serve' and pull a model)."
    )
    return AnswerGenerator(llm_client=FakeLLMClient()), notice


def run_demo(query: str, top_k: int = 5) -> None:
    print("=" * 80)
    print(" Aventro Motors RAG — Phase 2, Step 12: Grounded Answer Generation Demo")
    print("=" * 80)
    print(f"User Query : {query!r}")
    print(f"Top K      : {top_k}")

    generator, provider_info = get_configured_generator()
    print(f"LLM Provider: {provider_info}")
    print("-" * 80)

    # Build graph with generator
    app = build_rag_graph(answer_generator=generator, top_k=top_k)

    # Execute graph
    initial_state = {
        "query": query,
        "trace": [],
    }
    final_state = app.invoke(initial_state)
    response = format_rag_response(final_state)

    workflow_status = response["status"]
    answer = response["answer"]
    citations = response["citations"]
    abstention_reason = response["abstention_reason"]
    missing_info = response["missing_information"]
    error = response["error"]
    trace = response["trace"]

    print(f"Workflow Status : {workflow_status.upper()}")

    # Display Answer or Abstention
    if workflow_status == WORKFLOW_COMPLETED:
        print("\n" + "=" * 80)
        print(" GENERATED GROUNDED ANSWER")
        print("=" * 80)
        print(answer)
        print("-" * 80)

        if citations:
            print(f"RESOLVED INLINE CITATIONS ({len(citations)}):")
            for cit in citations:
                print(f"  • [{cit['evidence_id']}] -> Chunk ID: {cit['chunk_id']}")
                print(f"      Document Title : {cit['document_title']}")
                print(f"      Source Document: {cit['source_document']}")
                print(f"      Section Heading: {cit['section_heading']}")
                print(f"      Heading Path   : {cit['heading_path']}")
        else:
            print("No inline citations were found in the answer.")

    elif workflow_status == WORKFLOW_REQUIRES_ABSTENTION:
        print("\n" + "=" * 80)
        print(" WORKFLOW DECISION: ABSTENTION (No Grounded Answer Generated)")
        print("=" * 80)
        print(f"Reason: {abstention_reason}")
        if missing_info:
            print(f"Missing Information: {missing_info}")

    elif workflow_status == WORKFLOW_REQUIRES_FURTHER_RETRIEVAL:
        print("\n" + "=" * 80)
        print(" WORKFLOW DECISION: PARTIAL EVIDENCE (No Premature Answer Generated)")
        print("=" * 80)
        print(f"Reason: {abstention_reason}")
        if missing_info:
            print(f"Uncovered Sub-Aspects: {missing_info}")

    elif workflow_status == WORKFLOW_REQUIRES_CONFLICT_RESOLUTION:
        print("\n" + "=" * 80)
        print(" WORKFLOW DECISION: CONFLICT RESOLUTION REQUIRED")
        print("=" * 80)
        print(f"Reason: {abstention_reason}")

    elif workflow_status in (WORKFLOW_VALIDATION_FAILED, WORKFLOW_ERROR):
        print("\n" + "=" * 80)
        print(f" WORKFLOW DECISION: HALTED DUE TO ERROR")
        print("=" * 80)
        print(f"Error: {error}")

    # Display Execution Trace
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
