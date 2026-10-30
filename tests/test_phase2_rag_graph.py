"""
Unit and Integration Tests for Phase 2, Steps 11, 12 & 13: LangGraph Orchestration Pipeline
Tests validation, retrieval, evidence evaluation, conditional routing, grounded answer generation,
and claim-level citation verification.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest
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
from src.phase2.rag_graph import (
    build_rag_graph,
    validate_query_node,
    make_retrieve_node,
    make_evaluate_node,
    make_generate_node,
    make_verify_node,
)
from src.phase2.evidence_evaluator import EvidenceEvaluator
from src.phase2.rrf_fusion import RRFFusion
from src.phase2.answer_generator import (
    AnswerGenerator,
    FakeLLMClient,
    ModelUnavailableError,
)
from src.phase2.citation_verifier import CitationVerifier, FakeEntailmentVerifier


class MockRRF:
    """Mock RRF retriever returning predetermined results or raising exceptions."""
    def __init__(self, results: Optional[List[Dict[str, Any]]] = None, exc: Optional[Exception] = None):
        self.results = results if results is not None else []
        self.exc = exc
        self.called_query = None

    def fuse(self, query: str, bm25_top_k: int = 5, vector_top_k: int = 5, final_top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        self.called_query = query
        if self.exc:
            raise self.exc
        return copy.deepcopy(self.results[:final_top_k])


class MockEvaluator:
    """Mock evidence evaluator returning predetermined assessment or raising exceptions."""
    def __init__(self, assessment: Optional[Dict[str, Any]] = None, exc: Optional[Exception] = None):
        self.assessment = assessment
        self.exc = exc
        self.called_query = None
        self.called_results = None

    def evaluate(self, query: str, retrieved_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.called_query = query
        self.called_results = retrieved_results
        if self.exc:
            raise self.exc
        if self.assessment is not None:
            return copy.deepcopy(self.assessment)
        return {
            "status": "sufficient",
            "is_sufficient": True,
            "reason": "Mock sufficient reason",
            "relevant_evidence": retrieved_results,
            "irrelevant_evidence": [],
            "missing_information": [],
            "conflicts": [],
        }


def make_mock_chunk(chunk_id: str, text: str = "Test chunk text") -> Dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "citation_metadata": {
            "document_id": "doc_01",
            "source_document": "TestDoc.md",
            "document_title": "Test Title",
            "section_heading": "Section 1",
            "heading_path": "Test Title > Section 1",
            "chunk_index": 0,
            "character_count": len(text),
            "estimated_token_count": len(text) // 4,
            "content_hash": "hash_abc",
            "is_sub_chunk": False,
        },
        "source_document": "TestDoc.md",
        "document_title": "Test Title",
        "section_heading": "Section 1",
        "heading_path": "Test Title > Section 1",
        "rrf_rank": 1,
        "rrf_score": 0.032,
        "contributors": ["bm25", "vector"],
    }


# ==============================================================================
# 1. Graph Compilation and Structure
# ==============================================================================

def test_graph_compiles_successfully():
    graph = build_rag_graph(
        rrf_retriever=MockRRF(),
        evidence_evaluator=MockEvaluator(),
        answer_generator=AnswerGenerator(llm_client=FakeLLMClient()),
    )
    assert isinstance(graph, CompiledStateGraph)


# ==============================================================================
# 2. Input Validation Routing
# ==============================================================================

def test_invalid_non_string_query_routes_to_validation_failed():
    graph = build_rag_graph(
        rrf_retriever=MockRRF(),
        evidence_evaluator=MockEvaluator(),
        answer_generator=AnswerGenerator(llm_client=FakeLLMClient()),
    )
    state = graph.invoke({"query": 12345, "trace": []})  # type: ignore

    assert state["workflow_status"] == WORKFLOW_VALIDATION_FAILED
    assert "Invalid query type" in state["error"]
    assert "retrieved_results" not in state or state["retrieved_results"] == []
    assert state.get("generated_answer") is None


def test_empty_and_whitespace_query_routes_to_validation_failed():
    graph = build_rag_graph(
        rrf_retriever=MockRRF(),
        evidence_evaluator=MockEvaluator(),
        answer_generator=AnswerGenerator(llm_client=FakeLLMClient()),
    )

    state_empty = graph.invoke({"query": "", "trace": []})
    assert state_empty["workflow_status"] == WORKFLOW_VALIDATION_FAILED
    assert "Query must not be empty" in state_empty["error"]

    state_ws = graph.invoke({"query": "   \n\t  ", "trace": []})
    assert state_ws["workflow_status"] == WORKFLOW_VALIDATION_FAILED
    assert "Query must not be empty" in state_ws["error"]


# ==============================================================================
# 3. Grounded Answer Generation & Routing
# ==============================================================================

def test_routing_sufficient_evidence_triggers_grounded_answer():
    chunk = make_mock_chunk("chunk_1", "380 inspection checkpoints for pre-owned cars.")
    mock_rrf = MockRRF(results=[chunk])
    mock_eval = MockEvaluator(assessment={
        "status": "sufficient",
        "is_sufficient": True,
        "reason": "Evidence is fully sufficient.",
        "relevant_evidence": [chunk],
        "irrelevant_evidence": [],
        "missing_information": [],
        "conflicts": [],
    })
    generator = AnswerGenerator(
        llm_client=FakeLLMClient(canned_response="Vehicles undergo 380+ quality checkpoints [E1].")
    )

    graph = build_rag_graph(
        rrf_retriever=mock_rrf,
        evidence_evaluator=mock_eval,
        answer_generator=generator,
    )
    state = graph.invoke({"query": "380 checkpoints", "trace": []})

    # After Step 13 verification is wired in, sufficient evidence + generated answer
    # transitions to WORKFLOW_VERIFIED (all claims pass) or WORKFLOW_REQUIRES_REVIEW.
    assert state["workflow_status"] in (WORKFLOW_VERIFIED, WORKFLOW_REQUIRES_REVIEW)
    assert state["generation_status"] == "success"
    assert state["generated_answer"] == "Vehicles undergo 380+ quality checkpoints [E1]."
    assert len(state["answer_citations"]) == 1
    assert state["answer_citations"][0]["evidence_id"] == "E1"
    assert state["answer_citations"][0]["chunk_id"] == "chunk_1"
    assert state["answer_citations"][0]["source_document"] == "TestDoc.md"
    # Verification report must always be populated after a successful generation
    assert state.get("verification_report") is not None
    assert "claim_results" in state["verification_report"]


def test_routing_partially_sufficient_evidence_abstains_from_generation():
    chunk = make_mock_chunk("chunk_1")
    mock_rrf = MockRRF(results=[chunk])
    mock_eval = MockEvaluator(assessment={
        "status": "partially_sufficient",
        "is_sufficient": False,
        "reason": "Missing second part of query.",
        "relevant_evidence": [chunk],
        "irrelevant_evidence": [],
        "missing_information": ["doorstep return policy"],
        "conflicts": [],
    })
    generator = AnswerGenerator(llm_client=FakeLLMClient())

    graph = build_rag_graph(
        rrf_retriever=mock_rrf,
        evidence_evaluator=mock_eval,
        answer_generator=generator,
    )
    state = graph.invoke({"query": "checkpoints and return policy", "trace": []})

    assert state["workflow_status"] == WORKFLOW_REQUIRES_FURTHER_RETRIEVAL
    assert state.get("generation_status") == "not_attempted"
    assert state.get("generated_answer") is None
    assert state.get("answer_citations") == []


def test_routing_insufficient_evidence_abstains_from_generation():
    chunk = make_mock_chunk("chunk_1", "General history of company.")
    mock_rrf = MockRRF(results=[chunk])
    mock_eval = MockEvaluator(assessment={
        "status": "insufficient",
        "is_sufficient": False,
        "reason": "Evidence lacks required technical specs.",
        "relevant_evidence": [chunk],
        "irrelevant_evidence": [],
        "missing_information": ["battery kWh capacity"],
        "conflicts": [],
    })
    generator = AnswerGenerator(llm_client=FakeLLMClient())

    graph = build_rag_graph(
        rrf_retriever=mock_rrf,
        evidence_evaluator=mock_eval,
        answer_generator=generator,
    )
    state = graph.invoke({"query": "battery kWh capacity", "trace": []})

    assert state["workflow_status"] == WORKFLOW_REQUIRES_ABSTENTION
    assert state.get("generation_status") == "not_attempted"
    assert state.get("generated_answer") is None


def test_routing_no_evidence_abstains_from_generation():
    mock_rrf = MockRRF(results=[])
    mock_eval = MockEvaluator(assessment={
        "status": "no_evidence",
        "is_sufficient": False,
        "reason": "No evidence was retrieved.",
        "relevant_evidence": [],
        "irrelevant_evidence": [],
        "missing_information": ["All requested facts."],
        "conflicts": [],
    })
    generator = AnswerGenerator(llm_client=FakeLLMClient())

    graph = build_rag_graph(
        rrf_retriever=mock_rrf,
        evidence_evaluator=mock_eval,
        answer_generator=generator,
    )
    state = graph.invoke({"query": "rocket engines", "trace": []})

    assert state["workflow_status"] == WORKFLOW_REQUIRES_ABSTENTION
    assert state.get("generation_status") == "not_attempted"
    assert state.get("generated_answer") is None


def test_routing_conflicting_evidence_abstains_from_generation():
    c1 = make_mock_chunk("chunk_1", "Warranty is 1 year.")
    c2 = make_mock_chunk("chunk_2", "Warranty is 3 years.")
    mock_rrf = MockRRF(results=[c1, c2])
    mock_eval = MockEvaluator(assessment={
        "status": "conflicting",
        "is_sufficient": False,
        "reason": "Conflicting warranty specifications.",
        "relevant_evidence": [c1, c2],
        "irrelevant_evidence": [],
        "missing_information": ["Resolution of conflicting claims."],
        "conflicts": [{"type": "numeric_specification_conflict"}],
    })
    generator = AnswerGenerator(llm_client=FakeLLMClient())

    graph = build_rag_graph(
        rrf_retriever=mock_rrf,
        evidence_evaluator=mock_eval,
        answer_generator=generator,
    )
    state = graph.invoke({"query": "warranty duration", "trace": []})

    assert state["workflow_status"] == WORKFLOW_REQUIRES_CONFLICT_RESOLUTION
    assert state.get("generation_status") == "not_attempted"
    assert state.get("generated_answer") is None


# ==============================================================================
# 4. Error Handling and Node Failure Propagation
# ==============================================================================

def test_retrieval_failure_handling():
    mock_rrf = MockRRF(exc=RuntimeError("Vector database disk I/O failure"))
    mock_eval = MockEvaluator()

    graph = build_rag_graph(
        rrf_retriever=mock_rrf,
        evidence_evaluator=mock_eval,
        answer_generator=AnswerGenerator(llm_client=FakeLLMClient()),
    )
    state = graph.invoke({"query": "valid query", "trace": []})

    assert state["workflow_status"] == WORKFLOW_ERROR
    assert "Retrieval component failure" in state["error"]
    assert "evidence_assessment" not in state or state["evidence_assessment"] is None
    assert state.get("generated_answer") is None


def test_evidence_evaluator_failure_handling():
    mock_rrf = MockRRF(results=[make_mock_chunk("chunk_1")])
    mock_eval = MockEvaluator(exc=RuntimeError("Evaluation exception"))

    graph = build_rag_graph(
        rrf_retriever=mock_rrf,
        evidence_evaluator=mock_eval,
        answer_generator=AnswerGenerator(llm_client=FakeLLMClient()),
    )
    state = graph.invoke({"query": "valid query", "trace": []})

    assert state["workflow_status"] == WORKFLOW_ERROR
    assert "Evidence evaluation failure" in state["error"]
    assert state.get("generated_answer") is None


def test_model_provider_failure_handling_in_graph():
    chunk = make_mock_chunk("chunk_1", "Test evidence")
    mock_rrf = MockRRF(results=[chunk])
    mock_eval = MockEvaluator()
    failing_generator = AnswerGenerator(
        llm_client=FakeLLMClient(exc_to_raise=ModelUnavailableError("Local LLM provider not reachable."))
    )

    graph = build_rag_graph(
        rrf_retriever=mock_rrf,
        evidence_evaluator=mock_eval,
        answer_generator=failing_generator,
    )
    state = graph.invoke({"query": "valid query", "trace": []})

    assert state["workflow_status"] == WORKFLOW_ERROR
    assert state["generation_status"] == "failed"
    assert "Answer generation failure" in state["error"]
    assert "Local LLM provider not reachable" in state["error"]
    assert state.get("generated_answer") is None


# ==============================================================================
# 5. Citation Metadata Preservation & State Updates
# ==============================================================================

def test_citation_metadata_preserved_across_graph():
    chunk = make_mock_chunk("chunk_meta", "Original body text.")
    chunk["citation_metadata"]["source_document"] = "WarrantyPolicy.md"
    chunk["citation_metadata"]["section_heading"] = "Coverage Terms"

    mock_rrf = MockRRF(results=[chunk])
    evaluator = EvidenceEvaluator()
    generator = AnswerGenerator(llm_client=FakeLLMClient(canned_response="Warranty terms apply [E1]."))

    graph = build_rag_graph(
        rrf_retriever=mock_rrf,
        evidence_evaluator=evaluator,
        answer_generator=generator,
    )
    state = graph.invoke({"query": "coverage terms warranty", "trace": []})

    retrieved = state["retrieved_results"][0]
    assert retrieved["chunk_id"] == "chunk_meta"
    assert retrieved["citation_metadata"]["source_document"] == "WarrantyPolicy.md"

    relevant = state["evidence_assessment"]["relevant_evidence"][0]
    assert relevant["chunk_id"] == "chunk_meta"

    citations = state["answer_citations"]
    assert len(citations) == 1
    assert citations[0]["evidence_id"] == "E1"
    assert citations[0]["chunk_id"] == "chunk_meta"
    assert citations[0]["source_document"] == "WarrantyPolicy.md"


# ==============================================================================
# 6. Integration Test with Actual Corpus Components
# ==============================================================================

def test_integration_full_graph_on_real_corpus():
    chunks_path = Path("artifacts/chunks/document_chunks.jsonl")
    db_dir = Path("artifacts/chroma_db")
    if not chunks_path.is_file() or not db_dir.is_dir():
        pytest.skip("Phase 1 artifacts not present.")

    generator = AnswerGenerator(
        llm_client=FakeLLMClient(canned_response="All Aventro pre-owned certified cars undergo 380+ quality inspection checkpoints [E1][E2].")
    )
    graph = build_rag_graph(top_k=5, answer_generator=generator)
    query = "What are the 380 checkpoints for pre-owned cars?"
    state = graph.invoke({"query": query, "trace": []})

    # After Step 13: workflow ends in VERIFIED or REQUIRES_REVIEW (not COMPLETED)
    assert state["workflow_status"] in (WORKFLOW_VERIFIED, WORKFLOW_REQUIRES_REVIEW)
    assert state["generation_status"] == "success"
    assert state["evidence_assessment"]["status"] == "sufficient"
    assert len(state["retrieved_results"]) == 5
    assert len(state["evidence_assessment"]["relevant_evidence"]) >= 1

    assert state["generated_answer"] is not None
    assert len(state["answer_citations"]) >= 1
    assert state["answer_citations"][0]["chunk_id"].startswith("doc_")
    assert state["answer_citations"][0]["source_document"].endswith(".md")
    assert any("generate_answer: SUCCESS" in t for t in state["trace"])

    # Step 13: Verification report must be populated
    assert state.get("verification_report") is not None
    assert "claim_results" in state["verification_report"]
    assert any("verify_citations:" in t for t in state["trace"])


def test_phase1_and_retrievers_immutability():
    """Ensure that Phase 1 artifacts and prior Phase 2 components remain intact."""
    chunks_path = Path("artifacts/chunks/document_chunks.jsonl")
    assert chunks_path.is_file()
    with open(chunks_path, "r", encoding="utf-8") as f:
        line_count = sum(1 for line in f if line.strip())
    assert line_count == 459

    assert Path("src/phase2/bm25_retriever.py").is_file()
    assert Path("src/phase2/vector_retriever.py").is_file()
    assert Path("src/phase2/rrf_fusion.py").is_file()
    assert Path("src/phase2/evidence_evaluator.py").is_file()
    assert Path("src/phase2/answer_generator.py").is_file()


# ==============================================================================
# 7. Step 13 — Citation Verification Integration Tests
# ==============================================================================

def test_verification_node_produces_verified_status_on_supported_claims():
    """Graph returns WORKFLOW_VERIFIED when all answer claims are supported by cited evidence."""
    evidence_text = "380 inspection checkpoints for pre-owned cars including engine, brakes, and electricals."
    chunk = make_mock_chunk("chunk_1", evidence_text)
    mock_rrf = MockRRF(results=[chunk])
    mock_eval = MockEvaluator(assessment={
        "status": "sufficient",
        "is_sufficient": True,
        "reason": "Sufficient evidence.",
        "relevant_evidence": [chunk],
        "irrelevant_evidence": [],
        "missing_information": [],
        "conflicts": [],
    })
    # FakeLLMClient with all-supporting verifier -> should yield VERIFIED
    generator = AnswerGenerator(
        llm_client=FakeLLMClient(canned_response="Pre-owned cars undergo 380 inspection checkpoints [E1].")
    )
    fake_verifier = CitationVerifier(entailment_verifier=FakeEntailmentVerifier(default_support=True))
    graph = build_rag_graph(
        rrf_retriever=mock_rrf,
        evidence_evaluator=mock_eval,
        answer_generator=generator,
        citation_verifier=fake_verifier,
    )
    state = graph.invoke({"query": "380 checkpoints", "trace": []})

    assert state["workflow_status"] == WORKFLOW_VERIFIED
    assert state["is_verified"] is True
    assert state["verification_report"]["verification_status"] == "verified"
    assert state["verification_report"]["supported_claim_count"] == 1
    assert any("verify_citations: SUCCESS" in t for t in state["trace"])


def test_verification_node_produces_requires_review_on_unsupported_claim():
    """Graph returns WORKFLOW_REQUIRES_REVIEW when answer contains an unsupported claim."""
    chunk = make_mock_chunk("chunk_1", "General vehicle information text.")
    mock_rrf = MockRRF(results=[chunk])
    mock_eval = MockEvaluator(assessment={
        "status": "sufficient",
        "is_sufficient": True,
        "reason": "Sufficient.",
        "relevant_evidence": [chunk],
        "irrelevant_evidence": [],
        "missing_information": [],
        "conflicts": [],
    })
    generator = AnswerGenerator(
        llm_client=FakeLLMClient(canned_response="Aventro offers free lunar shuttle service for all customers [E1].")
    )
    # Force all claims to be unsupported
    fake_verifier = CitationVerifier(
        entailment_verifier=FakeEntailmentVerifier(
            default_support=False,
            default_reason="Claim is not supported by evidence."
        )
    )
    graph = build_rag_graph(
        rrf_retriever=mock_rrf,
        evidence_evaluator=mock_eval,
        answer_generator=generator,
        citation_verifier=fake_verifier,
    )
    state = graph.invoke({"query": "lunar shuttle service", "trace": []})

    assert state["workflow_status"] == WORKFLOW_REQUIRES_REVIEW
    assert state["is_verified"] is False
    assert state["verification_report"]["unsupported_claim_count"] >= 1
    assert any("REVIEW REQUIRED" in t for t in state["trace"])


def test_verification_skipped_on_abstention_pathways():
    """Verify node is NOT invoked for abstention branches (insufficient, no_evidence, etc.)."""
    mock_rrf = MockRRF(results=[])
    mock_eval = MockEvaluator(assessment={
        "status": "no_evidence",
        "is_sufficient": False,
        "reason": "No evidence found.",
        "relevant_evidence": [],
        "irrelevant_evidence": [],
        "missing_information": ["all"],
        "conflicts": [],
    })
    generator = AnswerGenerator(llm_client=FakeLLMClient())
    graph = build_rag_graph(
        rrf_retriever=mock_rrf,
        evidence_evaluator=mock_eval,
        answer_generator=generator,
    )
    state = graph.invoke({"query": "something obscure", "trace": []})

    assert state["workflow_status"] == WORKFLOW_REQUIRES_ABSTENTION
    assert state.get("verification_report") is None
    assert state.get("is_verified") is None
    # No verify_citations trace entry expected
    assert not any("verify_citations" in t for t in state.get("trace", []))


def test_verification_node_handles_uncited_claims():
    """Graph returns WORKFLOW_REQUIRES_REVIEW when answer has uncited factual claims."""
    chunk = make_mock_chunk("chunk_1", "Aventro pre-owned cars feature 380 checkpoints.")
    mock_rrf = MockRRF(results=[chunk])
    mock_eval = MockEvaluator(assessment={
        "status": "sufficient",
        "is_sufficient": True,
        "reason": "Sufficient.",
        "relevant_evidence": [chunk],
        "irrelevant_evidence": [],
        "missing_information": [],
        "conflicts": [],
    })
    # Answer has one uncited claim (no [E1] tag on second sentence)
    generator = AnswerGenerator(
        llm_client=FakeLLMClient(
            canned_response="Pre-owned cars undergo 380 checkpoints [E1]. Delivery takes 3 business days."
        )
    )
    fake_verifier = CitationVerifier(entailment_verifier=FakeEntailmentVerifier(default_support=True))
    graph = build_rag_graph(
        rrf_retriever=mock_rrf,
        evidence_evaluator=mock_eval,
        answer_generator=generator,
        citation_verifier=fake_verifier,
    )
    state = graph.invoke({"query": "checkpoints and delivery", "trace": []})

    assert state["workflow_status"] == WORKFLOW_REQUIRES_REVIEW
    assert state["is_verified"] is False
    assert state["verification_report"]["uncited_claim_count"] >= 1


def test_citation_verifier_module_is_wired_into_graph():
    """Confirm that citation_verifier.py is accessible and wired as a graph node."""
    assert Path("src/phase2/citation_verifier.py").is_file()
    graph = build_rag_graph(
        rrf_retriever=MockRRF(),
        evidence_evaluator=MockEvaluator(),
        answer_generator=AnswerGenerator(llm_client=FakeLLMClient()),
    )
    assert isinstance(graph, CompiledStateGraph)
    # Verify the verify_citations node name exists in the graph node set
    node_names = set(graph.get_graph().nodes.keys())
    assert "verify_citations" in node_names
