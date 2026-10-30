"""
Phase 2, Step 11 & 12: LangGraph Shared State Schema
Defines the typed graph state for the Aventro Motors RAG pipeline,
including retrieval, evidence assessment, and grounded answer generation with citations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


# Standard workflow status constants
WORKFLOW_READY_FOR_GENERATION = "ready_for_generation"
WORKFLOW_COMPLETED = "completed"
WORKFLOW_VERIFIED = "verified"
WORKFLOW_REQUIRES_REVIEW = "requires_review"
WORKFLOW_REQUIRES_ABSTENTION = "requires_abstention"
WORKFLOW_REQUIRES_FURTHER_RETRIEVAL = "requires_further_retrieval"
WORKFLOW_REQUIRES_CONFLICT_RESOLUTION = "requires_conflict_resolution"
WORKFLOW_VALIDATION_FAILED = "validation_failed"
WORKFLOW_ERROR = "workflow_error"


class RAGState(TypedDict, total=False):
    """
    Shared typed state passed between LangGraph nodes in the RAG pipeline.
    Preserves full citation metadata, evidence assessment, execution trace,
    grounded generated answers, and claim-level citation verification reports.
    """
    query: str
    retrieved_results: List[Dict[str, Any]]
    evidence_assessment: Optional[Dict[str, Any]]
    workflow_status: str
    error: Optional[str]
    trace: List[str]

    # Step 12 Grounded Generation extensions
    generated_answer: Optional[str]
    answer_citations: List[Dict[str, Any]]
    evidence_registry: Optional[Dict[str, Dict[str, Any]]]
    generation_status: Optional[str]
    generation_error: Optional[str]

    # Step 13 Claim-Level Citation Verification extensions
    verification_report: Optional[Dict[str, Any]]
    verification_status: Optional[str]
    is_verified: Optional[bool]
