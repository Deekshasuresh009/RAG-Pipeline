"""Unit and Integration Tests for Phase 2, Step 9: Reciprocal Rank Fusion (RRF)

Tests cover:
- Standard RRF score calculations
- Chunks appearing in both retrievers
- Deduplication by chunk_id
- Correct descending fused-score ordering
- Deterministic tie-breaking
- Preservation of original text and citation metadata
- Preservation of retriever-specific ranks and scores
- Results present in only one retriever
- Empty results from one retriever
- Empty results from both retrievers
- Invalid queries, limits, and k
- Duplicate IDs within an individual retriever's results
- Malformed results
- Retriever failures and degraded mode
- Verification of Phase 1 and retriever file immutability
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List
import pytest

from src.phase2.rrf_fusion import (
    RRFFusion,
    compute_rrf_score,
    fuse_rankings,
    validate_k,
    validate_limit,
)


class MockRetriever:
    """Mock retriever returning predetermined results or raising exceptions."""

    def __init__(self, results: Optional[List[Dict[str, Any]]] = None, exc: Optional[Exception] = None):
        self.results = results if results is not None else []
        self.exc = exc
        self.called_with_query = None
        self.called_with_top_k = None

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        self.called_with_query = query
        self.called_with_top_k = top_k
        if self.exc is not None:
            raise self.exc
        return copy.deepcopy(self.results[:top_k])


def make_mock_chunk(
    chunk_id: str,
    text: str = "Sample chunk text",
    source_document: str = "doc.md",
    section_heading: str = "Intro",
    heading_path: str = "Doc > Intro",
    rank_type: str = "bm25",
    rank: int = 1,
    score: float = 1.0,
) -> Dict[str, Any]:
    """Helper to create valid mock chunk results matching retriever output schema."""
    item: Dict[str, Any] = {
        "chunk_id": chunk_id,
        "text": text,
        "citation_metadata": {
            "document_id": "doc_01",
            "source_document": source_document,
            "document_title": "Doc Title",
            "section_heading": section_heading,
            "heading_path": heading_path,
            "chunk_index": 0,
            "character_count": len(text),
            "estimated_token_count": len(text) // 4,
            "content_hash": "hash123",
            "is_sub_chunk": False,
            "heading_hierarchy": ["Doc", "Intro"],
        },
    }
    if rank_type == "bm25":
        item["bm25_rank"] = rank
        item["bm25_score"] = score
    elif rank_type == "vector":
        item["vector_rank"] = rank
        item["vector_distance"] = score
    return item


# ==============================================================================
# Unit Tests: Core Math & Validation
# ==============================================================================

def test_validate_k_and_limit():
    assert validate_k(60) == 60
    assert validate_k(1) == 1

    # Booleans are subclasses of int in Python, must be explicitly rejected
    with pytest.raises(ValueError, match="RRF constant k must be a positive integer"):
        validate_k(True)
    with pytest.raises(ValueError, match="RRF constant k must be a positive integer"):
        validate_k(False)
    with pytest.raises(ValueError, match="RRF constant k must be a positive integer"):
        validate_k(0)
    with pytest.raises(ValueError, match="RRF constant k must be a positive integer"):
        validate_k(-5)
    with pytest.raises(ValueError, match="RRF constant k must be a positive integer"):
        validate_k(60.5)

    assert validate_limit(5, "top_k") == 5
    assert validate_limit(None, "final_top_k", allow_none=True) is None

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        validate_limit(0, "top_k")
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        validate_limit(-1, "top_k")
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        validate_limit(True, "top_k")


def test_standard_rrf_score_calculation():
    # Rank 1 alone: 1 / (60 + 1) = 1/61 ≈ 0.016393
    score = compute_rrf_score([1], k=60)
    assert pytest.approx(score, rel=1e-5) == 1.0 / 61.0

    # Ranks 1 and 2: 1/61 + 1/62 ≈ 0.0163934 + 0.0161290 = 0.032522
    score_multi = compute_rrf_score([1, 2], k=60)
    assert pytest.approx(score_multi, rel=1e-5) == (1.0 / 61.0) + (1.0 / 62.0)


# ==============================================================================
# Unit Tests: Fusion Logic, Deduplication & Tie-Breaking
# ==============================================================================

def test_fusion_chunk_in_both_retrievers():
    bm25 = [
        make_mock_chunk("chunk_1", rank_type="bm25", rank=1, score=3.2),
        make_mock_chunk("chunk_2", rank_type="bm25", rank=2, score=2.1),
    ]
    vector = [
        make_mock_chunk("chunk_1", rank_type="vector", rank=2, score=0.35),
        make_mock_chunk("chunk_3", rank_type="vector", rank=1, score=0.20),
    ]

    fused = fuse_rankings(bm25, vector, k=60)

    # 3 unique chunks
    assert len(fused) == 3
    chunk_ids = [r["chunk_id"] for r in fused]
    assert chunk_ids[0] == "chunk_1"  # Score: 1/61 + 1/62 = 0.032522, highest!

    c1 = fused[0]
    assert c1["chunk_id"] == "chunk_1"
    assert c1["rrf_rank"] == 1
    assert pytest.approx(c1["rrf_score"], rel=1e-5) == (1.0 / 61.0) + (1.0 / 62.0)
    assert c1["contributors"] == ["bm25", "vector"]
    assert c1["bm25_rank"] == 1
    assert c1["vector_rank"] == 2
    assert c1["bm25_score"] == 3.2
    assert c1["vector_distance"] == 0.35
    assert c1["citation_metadata"]["source_document"] == "doc.md"


def test_fusion_descending_score_ordering():
    bm25 = [
        make_mock_chunk("c_low", rank_type="bm25", rank=5, score=1.0),
        make_mock_chunk("c_high", rank_type="bm25", rank=1, score=4.0),
    ]
    vector = [
        make_mock_chunk("c_high", rank_type="vector", rank=1, score=0.1),
    ]
    fused = fuse_rankings(bm25, vector, k=60)

    assert fused[0]["chunk_id"] == "c_high"
    assert fused[1]["chunk_id"] == "c_low"
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]


def test_fusion_deterministic_tie_breaking():
    # cA has BM25 rank 1 (1/61)
    # cB has Vector rank 1 (1/61)
    # Scores are identical. Tie-break rule: order of first appearance (bm25 first, then vector).
    bm25 = [make_mock_chunk("chunk_A", rank_type="bm25", rank=1)]
    vector = [make_mock_chunk("chunk_B", rank_type="vector", rank=1)]

    fused = fuse_rankings(bm25, vector, k=60)
    assert len(fused) == 2
    assert fused[0]["rrf_score"] == fused[1]["rrf_score"]
    assert fused[0]["chunk_id"] == "chunk_A"
    assert fused[1]["chunk_id"] == "chunk_B"


def test_fusion_single_retriever_empty_or_both_empty():
    bm25 = [make_mock_chunk("c1", rank_type="bm25", rank=1)]
    vector: List[Dict[str, Any]] = []

    # One empty
    fused1 = fuse_rankings(bm25, vector, k=60)
    assert len(fused1) == 1
    assert fused1[0]["chunk_id"] == "c1"
    assert fused1[0]["contributors"] == ["bm25"]
    assert "vector_rank" not in fused1[0]

    # Both empty
    fused_empty = fuse_rankings([], [], k=60)
    assert fused_empty == []


def test_fusion_final_top_k_limiting():
    bm25 = [make_mock_chunk(f"c{i}", rank_type="bm25", rank=i) for i in range(1, 10)]
    fused = fuse_rankings(bm25, [], k=60, final_top_k=3)
    assert len(fused) == 3
    assert [r["rrf_rank"] for r in fused] == [1, 2, 3]


def test_fusion_handles_duplicate_chunk_ids_in_same_retriever():
    # If a retriever erroneously returns duplicate chunk_id records, deduplicate gracefully
    bm25 = [
        make_mock_chunk("c1", rank_type="bm25", rank=1, score=3.0),
        make_mock_chunk("c1", rank_type="bm25", rank=1, score=3.0),
    ]
    fused = fuse_rankings(bm25, [], k=60)
    assert len(fused) == 1
    assert fused[0]["chunk_id"] == "c1"
    assert fused[0]["contributors"] == ["bm25"]
    assert pytest.approx(fused[0]["rrf_score"], rel=1e-5) == 1.0 / 61.0


def test_fusion_rejects_malformed_results():
    # Non-dict item
    with pytest.raises(ValueError, match="item is not a dictionary"):
        fuse_rankings(["not a dict"], [])  # type: ignore

    # Missing chunk_id
    with pytest.raises(ValueError, match="missing or invalid 'chunk_id'"):
        fuse_rankings([{"text": "hi"}], [])

    # Missing text
    with pytest.raises(ValueError, match="missing or invalid 'text'"):
        fuse_rankings([{"chunk_id": "c1"}], [])

    # Missing citation_metadata
    with pytest.raises(ValueError, match="missing 'citation_metadata'"):
        fuse_rankings([{"chunk_id": "c1", "text": "sample"}], [])


def test_fusion_rejects_evaluation_dataset_leakage():
    leak = make_mock_chunk("c1")
    leak["citation_metadata"]["query_text"] = "What is the warranty?"
    with pytest.raises(ValueError, match="Evaluation question detected"):
        fuse_rankings([leak], [])


# ==============================================================================
# Unit Tests: RRFFusion Class & Degraded Mode
# ==============================================================================

def test_rrf_fusion_query_validation():
    fusion = RRFFusion(
        bm25_retriever=MockRetriever(),
        vector_retriever=MockRetriever(),
    )
    # Non-string queries
    with pytest.raises(TypeError, match="Query must be a string"):
        fusion.fuse(12345)  # type: ignore

    # Whitespace or empty query returns empty list
    assert fusion.fuse("") == []
    assert fusion.fuse("   \n\t  ") == []


def test_rrf_fusion_propagates_retriever_failure_by_default():
    bm25_mock = MockRetriever(exc=RuntimeError("BM25 index corrupted"))
    vector_mock = MockRetriever(results=[make_mock_chunk("c1", rank_type="vector")])

    fusion = RRFFusion(bm25_retriever=bm25_mock, vector_retriever=vector_mock, allow_degraded_mode=False)

    with pytest.raises(RuntimeError, match="BM25 retrieval failed"):
        fusion.fuse("valid query")


def test_rrf_fusion_supports_degraded_mode():
    bm25_mock = MockRetriever(exc=RuntimeError("BM25 timeout"))
    vector_mock = MockRetriever(results=[make_mock_chunk("c1", rank_type="vector", rank=1, score=0.25)])

    fusion = RRFFusion(bm25_retriever=bm25_mock, vector_retriever=vector_mock, allow_degraded_mode=True)
    results = fusion.fuse("valid query")

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["contributors"] == ["vector"]


def test_rrf_fusion_degraded_mode_fails_if_both_fail():
    bm25_mock = MockRetriever(exc=RuntimeError("BM25 failed"))
    vector_mock = MockRetriever(exc=RuntimeError("Vector failed"))

    fusion = RRFFusion(bm25_retriever=bm25_mock, vector_retriever=vector_mock, allow_degraded_mode=True)
    with pytest.raises(RuntimeError, match="Both retrievers failed in degraded mode"):
        fusion.fuse("valid query")


# ==============================================================================
# Integration Test: Real Corpus & File Integrity
# ==============================================================================

def test_integration_real_corpus():
    chunks_path = Path("artifacts/chunks/document_chunks.jsonl")
    db_dir = Path("artifacts/chroma_db")
    if not chunks_path.is_file() or not db_dir.is_dir():
        pytest.skip("Phase 1 artifacts not present.")

    fusion = RRFFusion()
    query = "What are the 380 checkpoints for pre-owned cars?"
    results = fusion.fuse(query, bm25_top_k=5, vector_top_k=5, final_top_k=5)

    assert isinstance(results, list)
    assert len(results) > 0
    assert len(results) <= 5

    for rank, r in enumerate(results, start=1):
        assert r["rrf_rank"] == rank
        assert isinstance(r["chunk_id"], str) and r["chunk_id"]
        assert isinstance(r["text"], str) and r["text"]
        assert isinstance(r["rrf_score"], float) and r["rrf_score"] > 0
        assert isinstance(r["contributors"], list) and len(r["contributors"]) >= 1

        meta = r["citation_metadata"]
        assert "source_document" in meta and meta["source_document"]
        assert "document_title" in meta and meta["document_title"]
        assert "section_heading" in meta
        assert "heading_path" in meta


def test_phase1_and_retrievers_immutability():
    """Verify that existing Phase 1 artifacts and Step 7 & 8 retrievers remain intact."""
    chunks_path = Path("artifacts/chunks/document_chunks.jsonl")
    assert chunks_path.is_file()

    # Count lines in chunk file
    with open(chunks_path, "r", encoding="utf-8") as f:
        line_count = sum(1 for line in f if line.strip())
    assert line_count == 459

    # Verify retriever files exist
    assert Path("src/phase2/bm25_retriever.py").is_file()
    assert Path("src/phase2/vector_retriever.py").is_file()
