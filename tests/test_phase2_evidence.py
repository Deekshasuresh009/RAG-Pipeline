"""
Unit and Integration Tests for Phase 2, Step 10: Evidence Relevance and Sufficiency Checking
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import pytest

from src.phase2.evidence_evaluator import (
    EvidenceEvaluator,
    detect_conflicts,
    extract_salient_terms,
    split_query_sub_aspects,
)
from src.phase2.rrf_fusion import RRFFusion


def make_test_chunk(
    chunk_id: str,
    text: str,
    source_document: str = "TestDoc.md",
    document_title: str = "Test Document",
    section_heading: str = "Overview",
    heading_path: str = "Test Document > Overview",
    rrf_rank: int = 1,
    rrf_score: float = 0.03,
) -> Dict[str, Any]:
    """Helper to produce well-formed mock chunk matching RRF output schema."""
    return {
        "chunk_id": chunk_id,
        "text": text,
        "citation_metadata": {
            "document_id": "doc_test",
            "source_document": source_document,
            "document_title": document_title,
            "section_heading": section_heading,
            "heading_path": heading_path,
            "chunk_index": 0,
            "character_count": len(text),
            "estimated_token_count": len(text) // 4,
            "content_hash": "hash_xyz",
            "is_sub_chunk": False,
            "heading_hierarchy": [document_title, section_heading],
        },
        "rrf_rank": rrf_rank,
        "rrf_score": rrf_score,
        "contributors": ["bm25", "vector"],
        "bm25_rank": rrf_rank,
        "vector_rank": rrf_rank,
    }


# ==============================================================================
# 1. Input Validation Tests
# ==============================================================================

def test_evaluator_rejects_non_string_query():
    evaluator = EvidenceEvaluator()
    with pytest.raises(TypeError, match="Query must be a string"):
        evaluator.evaluate(12345, [])  # type: ignore


def test_evaluator_rejects_empty_and_whitespace_query():
    evaluator = EvidenceEvaluator()
    with pytest.raises(ValueError, match="Query must not be empty"):
        evaluator.evaluate("", [])
    with pytest.raises(ValueError, match="Query must not be empty"):
        evaluator.evaluate("   \n\t ", [])


def test_evaluator_rejects_non_list_results():
    evaluator = EvidenceEvaluator()
    with pytest.raises(TypeError, match="retrieved_results must be a list"):
        evaluator.evaluate("valid query", "not a list")  # type: ignore


def test_evaluator_rejects_malformed_item():
    evaluator = EvidenceEvaluator()
    with pytest.raises(ValueError, match="is not a dictionary"):
        evaluator.evaluate("valid query", ["string_item"])  # type: ignore


def test_evaluator_rejects_missing_chunk_id():
    evaluator = EvidenceEvaluator()
    malformed = {"text": "Some text", "citation_metadata": {}}
    with pytest.raises(ValueError, match="missing or has invalid 'chunk_id'"):
        evaluator.evaluate("valid query", [malformed])


def test_evaluator_rejects_missing_text():
    evaluator = EvidenceEvaluator()
    malformed = {"chunk_id": "c1", "citation_metadata": {}}
    with pytest.raises(ValueError, match="missing or has invalid 'text'"):
        evaluator.evaluate("valid query", [malformed])


def test_evaluator_rejects_missing_citation_metadata():
    evaluator = EvidenceEvaluator()
    malformed = {"chunk_id": "c1", "text": "Valid text"}
    with pytest.raises(ValueError, match="missing 'citation_metadata'"):
        evaluator.evaluate("valid query", [malformed])


def test_evaluator_handles_duplicate_chunk_ids_safely():
    evaluator = EvidenceEvaluator()
    c1 = make_test_chunk("chunk_1", "Aventro 380 checkpoints for pre-owned cars.")
    c1_dup = make_test_chunk("chunk_1", "Duplicate entry text.")

    res = evaluator.evaluate("380 checkpoints pre-owned", [c1, c1_dup])
    all_chunks = res["relevant_evidence"] + res["irrelevant_evidence"]
    assert len(all_chunks) == 1
    assert all_chunks[0]["chunk_id"] == "chunk_1"
    assert "Aventro 380 checkpoints" in all_chunks[0]["text"]


def test_evaluator_rejects_evaluation_question_leakage():
    evaluator = EvidenceEvaluator()
    leaked = make_test_chunk("chunk_leak", "Ground truth answer")
    leaked["citation_metadata"]["query_text"] = "What is the warranty?"
    with pytest.raises(ValueError, match="Evaluation question detected"):
        evaluator.evaluate("warranty", [leaked])


# ==============================================================================
# 2. Relevance Assessment Tests
# ==============================================================================

def test_clearly_relevant_chunk():
    evaluator = EvidenceEvaluator()
    chunk = make_test_chunk(
        "c_rel",
        "Aventro certified pre-owned vehicles undergo a rigorous 380+ quality checkpoints inspection covering engine, brakes, and electricals."
    )
    res = evaluator.evaluate("What are the 380 checkpoints for pre-owned cars?", [chunk])

    assert len(res["relevant_evidence"]) == 1
    assert len(res["irrelevant_evidence"]) == 0
    rel = res["relevant_evidence"][0]
    assert rel["chunk_id"] == "c_rel"
    assert rel["is_relevant"] is True
    assert "380" in rel["matched_terms"]


def test_clearly_irrelevant_chunk():
    evaluator = EvidenceEvaluator()
    chunk = make_test_chunk(
        "c_irrel",
        "The company cafeteria in Gurugram offers vegetarian lunch options for corporate employees from Monday to Friday."
    )
    res = evaluator.evaluate("What are the 380 checkpoints for pre-owned cars?", [chunk])

    assert len(res["relevant_evidence"]) == 0
    assert len(res["irrelevant_evidence"]) == 1
    irrel = res["irrelevant_evidence"][0]
    assert irrel["chunk_id"] == "c_irrel"
    assert irrel["is_relevant"] is False


def test_stop_words_do_not_create_false_relevance():
    evaluator = EvidenceEvaluator()
    # Chunk only contains common stop words and no domain concepts
    chunk = make_test_chunk(
        "c_stopwords",
        "What is it that we can do for you and how will this be about?"
    )
    # Query also contains stop words plus specific domain words
    res = evaluator.evaluate("What is the transmission fluid replacement interval?", [chunk])

    assert len(res["relevant_evidence"]) == 0
    assert len(res["irrelevant_evidence"]) == 1
    assert res["status"] in ("no_evidence", "insufficient")


def test_case_and_punctuation_normalization():
    evaluator = EvidenceEvaluator()
    chunk = make_test_chunk(
        "c_case",
        "The ADAPTIVE CRUISE CONTROL (ACC) utilizes front RADAR SENSORS."
    )
    res = evaluator.evaluate("adaptive cruise control radar sensors", [chunk])
    assert len(res["relevant_evidence"]) == 1
    assert res["relevant_evidence"][0]["is_relevant"] is True


def test_original_text_and_citation_metadata_remain_unchanged():
    evaluator = EvidenceEvaluator()
    original_text = "Exact text: Battery warranty is 8 years or 160,000 km."
    chunk = make_test_chunk("c_orig", original_text, section_heading="Battery Care")

    res = evaluator.evaluate("Battery warranty duration", [chunk])
    evaluated = res["relevant_evidence"][0]

    assert evaluated["text"] == original_text
    assert evaluated["citation_metadata"]["section_heading"] == "Battery Care"
    assert evaluated["citation_metadata"]["source_document"] == "TestDoc.md"


# ==============================================================================
# 3. Sufficiency Assessment Tests
# ==============================================================================

def test_sufficiency_no_evidence_on_empty_input():
    evaluator = EvidenceEvaluator()
    res = evaluator.evaluate("Any question", [])
    assert res["status"] == "no_evidence"
    assert res["is_sufficient"] is False
    assert res["relevant_evidence"] == []
    assert len(res["missing_information"]) > 0


def test_sufficiency_no_evidence_when_all_irrelevant():
    evaluator = EvidenceEvaluator()
    chunk = make_test_chunk("c_irr", "Aventro design philosophy emphasizes aerodynamic contours.")
    res = evaluator.evaluate("What is the battery warranty in years?", [chunk])

    assert res["status"] == "no_evidence"
    assert res["is_sufficient"] is False
    assert len(res["relevant_evidence"]) == 0
    assert len(res["irrelevant_evidence"]) == 1


def test_sufficiency_fully_sufficient():
    evaluator = EvidenceEvaluator()
    chunk = make_test_chunk(
        "c_suff",
        "Aventro certified pre-owned cars come with a comprehensive 380+ checkpoint inspection, 1-year complimentary warranty, and 24/7 roadside assistance."
    )
    res = evaluator.evaluate("What are the 380 checkpoints for pre-owned cars?", [chunk])

    assert res["status"] == "sufficient"
    assert res["is_sufficient"] is True
    assert len(res["relevant_evidence"]) == 1
    assert res["missing_information"] == []


def test_sufficiency_partially_sufficient_multi_part_query():
    evaluator = EvidenceEvaluator()
    # Chunk only answers part 1 (380 checkpoints), but not part 2 (home delivery return policy)
    chunk = make_test_chunk(
        "c_part",
        "Aventro 380 checkpoints cover engine compression, transmission shifting, and chassis alignment for pre-owned cars."
    )
    res = evaluator.evaluate("What are the 380 checkpoints for pre-owned cars and what is the 7-day doorstep return policy?", [chunk])

    assert res["status"] == "partially_sufficient"
    assert res["is_sufficient"] is False
    assert len(res["relevant_evidence"]) == 1
    assert len(res["missing_information"]) >= 1


def test_sufficiency_insufficient_when_keyword_mention_is_incomplete():
    evaluator = EvidenceEvaluator()
    # Chunk mentions the model name, but doesn't have the charging specs, pricing, or battery capacity
    chunk = make_test_chunk("c_vague", "The Aventro Spark Electric was introduced in 2019.")
    res = evaluator.evaluate("What is the fast charging time from 10% to 80% and battery kWh capacity of the Aventro Spark Electric?", [chunk])

    assert res["status"] in ("partially_sufficient", "insufficient")
    assert res["is_sufficient"] is False


def test_sufficiency_collective_multi_chunk_sufficient():
    evaluator = EvidenceEvaluator()
    # Part 1 in chunk 1
    c1 = make_test_chunk("c1", "Aventro Spark Electric has a 42 kWh lithium-ion battery pack.")
    # Part 2 in chunk 2
    c2 = make_test_chunk("c2", "Fast DC charging takes 45 minutes from 10% to 80% on a 50 kW charger.")

    res = evaluator.evaluate("What is the battery kWh capacity and fast charging time for Aventro Spark Electric?", [c1, c2])

    assert res["status"] == "sufficient"
    assert res["is_sufficient"] is True
    assert len(res["relevant_evidence"]) == 2


def test_conflict_detection_numeric_specification():
    evaluator = EvidenceEvaluator()
    c1 = make_test_chunk("c1", "Aventro pre-owned vehicles include a comprehensive warranty of 1 year.")
    c2 = make_test_chunk("c2", "All certified pre-owned models feature an extended warranty of 3 years.")

    res = evaluator.evaluate("What is the pre-owned warranty duration?", [c1, c2])

    assert res["status"] == "conflicting"
    assert res["is_sufficient"] is False
    assert len(res["conflicts"]) >= 1
    assert res["conflicts"][0]["type"] == "numeric_specification_conflict"


# ==============================================================================
# 4. Integration Tests with Actual Corpus & Step 7-9 Retrievers
# ==============================================================================

def test_integration_evaluator_with_real_rrf_retrieval():
    """Verify end-to-end integration: query -> RRF -> EvidenceEvaluator on real corpus."""
    chunks_path = Path("artifacts/chunks/document_chunks.jsonl")
    db_dir = Path("artifacts/chroma_db")
    if not chunks_path.is_file() or not db_dir.is_dir():
        pytest.skip("Phase 1 artifacts not present.")

    rrf = RRFFusion()
    query = "What are the 380 checkpoints for pre-owned cars?"
    retrieved = rrf.fuse(query=query, bm25_top_k=5, vector_top_k=5, final_top_k=5)

    evaluator = EvidenceEvaluator()
    assessment = evaluator.evaluate(query=query, retrieved_results=retrieved)

    assert isinstance(assessment, dict)
    assert assessment["status"] in ("sufficient", "partially_sufficient")
    assert assessment["relevant_evidence"]
    for rec in assessment["relevant_evidence"]:
        assert rec["chunk_id"].startswith("doc_")
        assert rec["source_document"].endswith(".md")
        assert rec["text"]
        assert rec["is_relevant"] is True


def test_integration_evaluator_with_unanswerable_or_absent_topic():
    """Verify that an absent topic results in insufficient or no_evidence without crash."""
    chunks_path = Path("artifacts/chunks/document_chunks.jsonl")
    if not chunks_path.is_file():
        pytest.skip("Phase 1 artifacts not present.")

    rrf = RRFFusion()
    query = "What is the orbital velocity and rocket fuel specification of Aventro satellite launchers?"
    retrieved = rrf.fuse(query=query, bm25_top_k=5, vector_top_k=5, final_top_k=5)

    evaluator = EvidenceEvaluator()
    assessment = evaluator.evaluate(query=query, retrieved_results=retrieved)

    assert assessment["status"] in ("no_evidence", "insufficient")
    assert assessment["is_sufficient"] is False


def test_phase1_and_retrievers_immutability():
    """Verify that Phase 1 artifacts and Step 7-9 modules remain untouched."""
    chunks_path = Path("artifacts/chunks/document_chunks.jsonl")
    assert chunks_path.is_file()
    with open(chunks_path, "r", encoding="utf-8") as f:
        line_count = sum(1 for line in f if line.strip())
    assert line_count == 459

    assert Path("src/phase2/bm25_retriever.py").is_file()
    assert Path("src/phase2/vector_retriever.py").is_file()
    assert Path("src/phase2/rrf_fusion.py").is_file()
