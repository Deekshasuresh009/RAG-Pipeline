"""
Phase 2, Steps 11, 12 & 13: LangGraph Orchestration Pipeline
Constructs the LangGraph state graph connecting:
1. Query Validation
2. Hybrid RRF Retrieval
3. Evidence Relevance and Sufficiency Assessment
4. Explicit Conditional Routing (Abstention vs Generation)
5. Grounded Answer Generation with Citations
6. Claim-Level Citation Verification
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from src.phase2.rag_state import (
    RAGState,
    WORKFLOW_READY_FOR_GENERATION,
    WORKFLOW_COMPLETED,
    WORKFLOW_VERIFIED,
    WORKFLOW_REQUIRES_REVIEW,
    WORKFLOW_REQUIRES_ABSTENTION,
    WORKFLOW_REQUIRES_FURTHER_RETRIEVAL,
    WORKFLOW_REQUIRES_CONFLICT_RESOLUTION,
    WORKFLOW_VALIDATION_FAILED,
    WORKFLOW_ERROR,
)
from src.phase2.rrf_fusion import RRFFusion
from src.phase2.evidence_evaluator import EvidenceEvaluator
from src.phase2.answer_generator import AnswerGenerator, format_rag_response
from src.phase2.citation_verifier import CitationVerifier


def validate_query_node(state: RAGState) -> Dict[str, Any]:
    """
    Validate that the incoming query is a non-empty, non-whitespace string.
    """
    query = state.get("query")
    trace = list(state.get("trace", []))
    trace.append("validate_query: Validating input query.")

    if not isinstance(query, str):
        err_msg = f"Invalid query type: expected string, got {type(query).__name__}."
        trace.append(f"validate_query: FAILED - {err_msg}")
        return {
            "workflow_status": WORKFLOW_VALIDATION_FAILED,
            "error": err_msg,
            "trace": trace,
        }

    clean_query = query.strip()
    if not clean_query:
        err_msg = "Query must not be empty or whitespace-only."
        trace.append(f"validate_query: FAILED - {err_msg}")
        return {
            "workflow_status": WORKFLOW_VALIDATION_FAILED,
            "error": err_msg,
            "trace": trace,
        }

    trace.append(f"validate_query: SUCCESS - Normalized query: {clean_query!r}")
    return {
        "query": clean_query,
        "trace": trace,
    }


def make_retrieve_node(rrf_retriever: RRFFusion, top_k: int = 5) -> Callable[[RAGState], Dict[str, Any]]:
    """Factory returning a node function that executes hybrid RRF retrieval."""
    def retrieve_node(state: RAGState) -> Dict[str, Any]:
        query = state["query"]
        trace = list(state.get("trace", []))
        trace.append(f"retrieve_hybrid: Querying BM25 and Vector with top_k={top_k}.")

        try:
            results = rrf_retriever.fuse(
                query=query,
                bm25_top_k=top_k,
                vector_top_k=top_k,
                final_top_k=top_k,
            )
            trace.append(f"retrieve_hybrid: SUCCESS - Retrieved {len(results)} candidate chunk(s).")
            return {
                "retrieved_results": results,
                "trace": trace,
            }
        except Exception as exc:
            err_msg = f"Retrieval component failure: {exc}"
            trace.append(f"retrieve_hybrid: FAILED - {err_msg}")
            return {
                "workflow_status": WORKFLOW_ERROR,
                "error": err_msg,
                "retrieved_results": [],
                "trace": trace,
            }

    return retrieve_node


def make_evaluate_node(evaluator: EvidenceEvaluator) -> Callable[[RAGState], Dict[str, Any]]:
    """Factory returning a node function that evaluates evidence relevance and sufficiency."""
    def evaluate_node(state: RAGState) -> Dict[str, Any]:
        query = state["query"]
        retrieved_results = state.get("retrieved_results", [])
        trace = list(state.get("trace", []))
        trace.append(f"evaluate_evidence: Assessing {len(retrieved_results)} chunk(s) for relevance & sufficiency.")

        try:
            assessment = evaluator.evaluate(query=query, retrieved_results=retrieved_results)
            status = assessment.get("status", "unknown")
            trace.append(f"evaluate_evidence: SUCCESS - Assessment status: {status.upper()}.")
            return {
                "evidence_assessment": assessment,
                "trace": trace,
            }
        except Exception as exc:
            err_msg = f"Evidence evaluation failure: {exc}"
            trace.append(f"evaluate_evidence: FAILED - {err_msg}")
            return {
                "workflow_status": WORKFLOW_ERROR,
                "error": err_msg,
                "trace": trace,
            }

    return evaluate_node


def make_generate_node(generator: AnswerGenerator) -> Callable[[RAGState], Dict[str, Any]]:
    """Factory returning a node function that executes grounded answer generation with citations."""
    def generate_answer_node(state: RAGState) -> Dict[str, Any]:
        query = state["query"]
        assessment = state.get("evidence_assessment") or {}
        relevant_evidence = assessment.get("relevant_evidence", [])
        trace = list(state.get("trace", []))
        trace.append(f"generate_answer: Generating answer using {len(relevant_evidence)} cited evidence chunk(s).")

        if not relevant_evidence:
            err_msg = "Cannot generate answer: no relevant evidence available."
            trace.append(f"generate_answer: FAILED - {err_msg}")
            return {
                "workflow_status": WORKFLOW_REQUIRES_ABSTENTION,
                "generation_status": "not_attempted",
                "generated_answer": None,
                "answer_citations": [],
                "evidence_registry": {},
                "error": err_msg,
                "trace": trace,
            }

        try:
            res = generator.generate(query=query, evidence=relevant_evidence)
            answer = res["answer"]
            citations = res["citations"]
            unknown_citations = res.get("unknown_citations", [])
            evidence_registry = res.get("evidence_registry", {})

            trace.append(f"generate_answer: SUCCESS - Generated answer with {len(citations)} citation(s).")
            if unknown_citations:
                trace.append(f"generate_answer: WARNING - Detected unknown citation ID(s): {unknown_citations}")

            return {
                "workflow_status": WORKFLOW_COMPLETED,
                "generation_status": "success",
                "generated_answer": answer,
                "answer_citations": citations,
                "evidence_registry": evidence_registry,
                "trace": trace,
            }
        except Exception as exc:
            err_msg = f"Answer generation failure: {exc}"
            trace.append(f"generate_answer: FAILED - {err_msg}")
            return {
                "workflow_status": WORKFLOW_ERROR,
                "generation_status": "failed",
                "generation_error": err_msg,
                "generated_answer": None,
                "answer_citations": [],
                "evidence_registry": {},
                "error": err_msg,
                "trace": trace,
            }

    return generate_answer_node


def make_verify_node(verifier: CitationVerifier) -> Callable[[RAGState], Dict[str, Any]]:
    """Factory returning a node function that performs claim-level citation verification."""
    def verify_citations_node(state: RAGState) -> Dict[str, Any]:
        answer = state.get("generated_answer") or ""
        evidence_registry = state.get("evidence_registry") or {}
        trace = list(state.get("trace", []))
        trace.append("verify_citations: Segmenting answer into claims and verifying cited evidence support.")

        # If evidence_registry is empty, try reconstructing from relevant evidence
        if not evidence_registry:
            relevant = state.get("evidence_assessment", {}).get("relevant_evidence", [])
            for idx, item in enumerate(relevant, start=1):
                eid = f"E{idx}"
                meta = item.get("citation_metadata", {})
                evidence_registry[eid] = {
                    "evidence_id": eid,
                    "chunk_id": item.get("chunk_id", ""),
                    "source_document": meta.get("source_document") or item.get("source_document", ""),
                    "document_title": meta.get("document_title") or item.get("document_title", ""),
                    "section_heading": meta.get("section_heading") or item.get("section_heading", ""),
                    "heading_path": meta.get("heading_path") or item.get("heading_path", ""),
                    "text": item.get("text", ""),
                }

        try:
            report = verifier.verify(answer=answer, evidence_registry=evidence_registry)
            v_status = report["verification_status"]
            is_verified = report["is_verified"]

            if is_verified:
                final_workflow_status = WORKFLOW_VERIFIED
                trace.append("verify_citations: SUCCESS - All claims verified by cited evidence.")
            elif v_status == "verification_error":
                final_workflow_status = WORKFLOW_ERROR
                trace.append(f"verify_citations: FAILED - {report.get('verification_error')}")
            else:
                final_workflow_status = WORKFLOW_REQUIRES_REVIEW
                trace.append(
                    f"verify_citations: REVIEW REQUIRED - Status '{v_status}'. "
                    f"Supported: {report['supported_claim_count']}, "
                    f"Unsupported: {report['unsupported_claim_count']}, "
                    f"Uncited: {report['uncited_claim_count']}, "
                    f"Invalid citations: {report['invalid_citation_count']}."
                )

            return {
                "workflow_status": final_workflow_status,
                "verification_report": report,
                "verification_status": v_status,
                "is_verified": is_verified,
                "trace": trace,
            }
        except Exception as exc:
            err_msg = f"Citation verification failure: {exc}"
            trace.append(f"verify_citations: FAILED - {err_msg}")
            return {
                "workflow_status": WORKFLOW_ERROR,
                "verification_report": None,
                "verification_status": "verification_error",
                "is_verified": False,
                "error": err_msg,
                "trace": trace,
            }

    return verify_citations_node


def handle_partially_sufficient_node(state: RAGState) -> Dict[str, Any]:
    """Route when evidence is partially sufficient: Mark requiring further retrieval or abstention."""
    trace = list(state.get("trace", []))
    trace.append(f"route: Evidence is partially sufficient -> {WORKFLOW_REQUIRES_FURTHER_RETRIEVAL}.")
    return {
        "workflow_status": WORKFLOW_REQUIRES_FURTHER_RETRIEVAL,
        "generation_status": "not_attempted",
        "generated_answer": None,
        "answer_citations": [],
        "trace": trace,
    }


def handle_insufficient_or_no_evidence_node(state: RAGState) -> Dict[str, Any]:
    """Route when evidence is insufficient or no evidence found: Mark requiring abstention."""
    status = state.get("evidence_assessment", {}).get("status", "insufficient")
    trace = list(state.get("trace", []))
    trace.append(f"route: Evidence status is '{status}' -> {WORKFLOW_REQUIRES_ABSTENTION}.")
    return {
        "workflow_status": WORKFLOW_REQUIRES_ABSTENTION,
        "generation_status": "not_attempted",
        "generated_answer": None,
        "answer_citations": [],
        "trace": trace,
    }


def handle_conflicting_node(state: RAGState) -> Dict[str, Any]:
    """Route when evidence is conflicting: Mark requiring conflict resolution."""
    trace = list(state.get("trace", []))
    trace.append(f"route: Evidence contains contradictions -> {WORKFLOW_REQUIRES_CONFLICT_RESOLUTION}.")
    return {
        "workflow_status": WORKFLOW_REQUIRES_CONFLICT_RESOLUTION,
        "generation_status": "not_attempted",
        "generated_answer": None,
        "answer_citations": [],
        "trace": trace,
    }


# Conditional Edge Router Functions
def after_validation_router(state: RAGState) -> str:
    if state.get("workflow_status") == WORKFLOW_VALIDATION_FAILED:
        return END
    return "retrieve_hybrid"


def after_retrieval_router(state: RAGState) -> str:
    if state.get("workflow_status") == WORKFLOW_ERROR:
        return END
    return "evaluate_evidence"


def evidence_sufficiency_router(state: RAGState) -> str:
    """Conditional router selecting next node based on evidence assessment status."""
    if state.get("workflow_status") == WORKFLOW_ERROR:
        return END

    assessment = state.get("evidence_assessment", {})
    status = assessment.get("status")

    if status == "sufficient":
        return "generate_answer"
    elif status == "partially_sufficient":
        return "handle_partially_sufficient"
    elif status in ("insufficient", "no_evidence"):
        return "handle_insufficient_or_no_evidence"
    elif status == "conflicting":
        return "handle_conflicting"
    else:
        return "handle_insufficient_or_no_evidence"


def after_generation_router(state: RAGState) -> str:
    """Route after answer generation: proceed to verification only on successful generation."""
    if state.get("generation_status") == "success" and state.get("generated_answer"):
        return "verify_citations"
    return END


def build_rag_graph(
    rrf_retriever: Optional[RRFFusion] = None,
    evidence_evaluator: Optional[EvidenceEvaluator] = None,
    answer_generator: Optional[AnswerGenerator] = None,
    citation_verifier: Optional[CitationVerifier] = None,
    top_k: int = 5,
) -> CompiledStateGraph:
    """
    Construct and compile the LangGraph RAG pipeline.
    Connects:
    1. Query Validation
    2. Hybrid RRF Retrieval
    3. Evidence Relevance and Sufficiency Assessment
    4. Explicit Conditional Routing based on Evidence Status
    5. Grounded Answer Generation with Citations (for sufficient evidence)
    6. Claim-Level Citation Verification (for generated answers)
    """
    retriever_inst = rrf_retriever if rrf_retriever is not None else RRFFusion()
    evaluator_inst = evidence_evaluator if evidence_evaluator is not None else EvidenceEvaluator()
    generator_inst = answer_generator if answer_generator is not None else AnswerGenerator()
    verifier_inst = citation_verifier if citation_verifier is not None else CitationVerifier()

    workflow = StateGraph(RAGState)

    # 1. Register Nodes
    workflow.add_node("validate_query", validate_query_node)
    workflow.add_node("retrieve_hybrid", make_retrieve_node(retriever_inst, top_k=top_k))
    workflow.add_node("evaluate_evidence", make_evaluate_node(evaluator_inst))

    # Action / Generation Node
    workflow.add_node("generate_answer", make_generate_node(generator_inst))

    # Verification Node
    workflow.add_node("verify_citations", make_verify_node(verifier_inst))

    # Decision / Handler Nodes for non-sufficient outcomes
    workflow.add_node("handle_partially_sufficient", handle_partially_sufficient_node)
    workflow.add_node("handle_insufficient_or_no_evidence", handle_insufficient_or_no_evidence_node)
    workflow.add_node("handle_conflicting", handle_conflicting_node)

    # 2. Add Edges & Conditional Routing
    workflow.add_edge(START, "validate_query")

    workflow.add_conditional_edges(
        "validate_query",
        after_validation_router,
        {
            "retrieve_hybrid": "retrieve_hybrid",
            END: END,
        }
    )

    workflow.add_conditional_edges(
        "retrieve_hybrid",
        after_retrieval_router,
        {
            "evaluate_evidence": "evaluate_evidence",
            END: END,
        }
    )

    workflow.add_conditional_edges(
        "evaluate_evidence",
        evidence_sufficiency_router,
        {
            "generate_answer": "generate_answer",
            "handle_partially_sufficient": "handle_partially_sufficient",
            "handle_insufficient_or_no_evidence": "handle_insufficient_or_no_evidence",
            "handle_conflicting": "handle_conflicting",
            END: END,
        }
    )

    # After generation, route conditionally to verification or terminate on error
    workflow.add_conditional_edges(
        "generate_answer",
        after_generation_router,
        {
            "verify_citations": "verify_citations",
            END: END,
        }
    )

    workflow.add_edge("verify_citations", END)
    workflow.add_edge("handle_partially_sufficient", END)
    workflow.add_edge("handle_insufficient_or_no_evidence", END)
    workflow.add_edge("handle_conflicting", END)

    return workflow.compile()
