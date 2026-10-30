"""
Pytest Test Suite for Phase 2, Step 7: BM25 Keyword Retriever
Tests chunk indexing, deterministic scoring, citation metadata integrity,
input validation, boundary cases, and non-modification of Phase 1 artifacts.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from src.phase2.bm25_retriever import (
    BM25Retriever,
    DEFAULT_CHUNKS_PATH,
    tokenize_text,
    validate_chunks_data
)


# -------------------------------------------------------------------------
# 1. Tests for Production Dataset Loading & Verification
# -------------------------------------------------------------------------

def test_load_production_chunks_count_and_unique_ids():
    """Verify production chunks JSONL loads cleanly with 459 unique chunk IDs."""
    retriever = BM25Retriever.from_jsonl(DEFAULT_CHUNKS_PATH)
    assert len(retriever.chunks) == 459
    assert len(retriever.tokenized_corpus) == 459
    assert len(retriever.chunk_ids) == 459

    # Verify uniqueness of chunk IDs
    unique_ids = set(retriever.chunk_ids)
    assert len(unique_ids) == 459


# -------------------------------------------------------------------------
# 2. Tests for Keyword Retrieval on Known Terms
# -------------------------------------------------------------------------

def test_keyword_retrieval_known_terms():
    """Verify BM25 accurately retrieves relevant chunks for distinct domain keywords."""
    retriever = BM25Retriever.from_jsonl(DEFAULT_CHUNKS_PATH)

    # Test 1: Founder name
    res_founder = retriever.retrieve("Amit Bansal", top_k=3)
    assert len(res_founder) > 0
    assert any("About Aventro Motors.md" in r["citation_metadata"]["source_document"] for r in res_founder)
    assert res_founder[0]["bm25_score"] > 0

    # Test 2: Pre-owned checkpoints inspection
    res_checkpoints = retriever.retrieve("380+ checkpoints inspection", top_k=5)
    assert len(res_checkpoints) > 0
    top_doc = res_checkpoints[0]["citation_metadata"]["source_document"]
    assert "Pre-Owned Certified Cars.md" in top_doc
    assert res_checkpoints[0]["bm25_score"] > 5.0

    # Test 3: Warning alerts
    res_alert = retriever.retrieve("Brake System Warning Light", top_k=3)
    assert len(res_alert) > 0
    assert any("Warning" in r["citation_metadata"]["section_heading"] or "Brake" in r["citation_metadata"]["section_heading"] for r in res_alert)


# -------------------------------------------------------------------------
# 3. Tests for Controlled Synthetic Corpus Ranking
# -------------------------------------------------------------------------

def test_controlled_synthetic_corpus_ranking():
    """Verify BM25 ranking order and zero-score exclusion on a controlled 3-document dataset."""
    controlled_chunks = [
        {
            "chunk_id": "c_doc_001",
            "document_id": "doc_001",
            "source_document": "turbo_guide.md",
            "document_title": "Turbo Guide",
            "section_heading": "Turbocharger Architecture",
            "heading_path": "Turbo Guide > Turbocharger Architecture",
            "text": "The turbocharger delivers boost pressure. A secondary turbocharger enhances twin turbocharger spooling.",
            "chunk_index": 0
        },
        {
            "chunk_id": "c_doc_002",
            "document_id": "doc_002",
            "source_document": "specs.md",
            "document_title": "Vehicle Specs",
            "section_heading": "Engine Options",
            "heading_path": "Vehicle Specs > Engine Options",
            "text": "The 1.5L variant is equipped with an integrated turbocharger.",
            "chunk_index": 0
        },
        {
            "chunk_id": "c_doc_003",
            "document_id": "doc_003",
            "source_document": "chassis.md",
            "document_title": "Chassis Overview",
            "section_heading": "Suspension Setup",
            "heading_path": "Chassis Overview > Suspension Setup",
            "text": "MacPherson strut front suspension paired with multi-link rear suspension.",
            "chunk_index": 0
        }
    ]

    retriever = BM25Retriever(controlled_chunks)
    results = retriever.retrieve("turbocharger", top_k=5)

    # Chunk 1 (3 occurrences) must rank #1
    # Chunk 2 (1 occurrence) must rank #2
    # Chunk 3 (0 occurrences) must NOT be returned (score == 0)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "c_doc_001"
    assert results[0]["bm25_rank"] == 1
    assert results[1]["chunk_id"] == "c_doc_002"
    assert results[1]["bm25_rank"] == 2
    assert results[0]["bm25_score"] > results[1]["bm25_score"]


# -------------------------------------------------------------------------
# 4. Tests for Citation Metadata Preservation
# -------------------------------------------------------------------------

def test_citation_metadata_preservation():
    """Verify returned structured result preserves all original citation metadata fields."""
    retriever = BM25Retriever.from_jsonl(DEFAULT_CHUNKS_PATH)
    results = retriever.retrieve("battery warranty", top_k=2)

    assert len(results) > 0
    top = results[0]

    # Required top-level keys
    assert "chunk_id" in top
    assert "text" in top
    assert "citation_metadata" in top
    assert "bm25_rank" in top
    assert "bm25_score" in top
    assert top["bm25_rank"] == 1
    assert isinstance(top["bm25_score"], float)

    # Required metadata keys preserved from chunk
    meta = top["citation_metadata"]
    assert "document_id" in meta
    assert "source_document" in meta
    assert "document_title" in meta
    assert "section_heading" in meta
    assert "heading_path" in meta
    assert meta["source_document"].endswith(".md")


# -------------------------------------------------------------------------
# 5. Tests for Query Edge Cases
# -------------------------------------------------------------------------

def test_empty_and_whitespace_queries_return_empty_list():
    """Verify empty or whitespace-only queries cleanly return an empty list."""
    retriever = BM25Retriever.from_jsonl(DEFAULT_CHUNKS_PATH)
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []
    assert retriever.retrieve("\t\n  \r\n") == []


def test_unmatched_tokens_return_empty_list():
    """Verify queries with completely unindexed tokens return an empty list."""
    retriever = BM25Retriever.from_jsonl(DEFAULT_CHUNKS_PATH)
    assert retriever.retrieve("nonexistentgibberish987654321xyz") == []


# -------------------------------------------------------------------------
# 6. Tests for top_k Validation
# -------------------------------------------------------------------------

def test_top_k_validation():
    """Verify top_k parameter enforces valid positive integers."""
    retriever = BM25Retriever.from_jsonl(DEFAULT_CHUNKS_PATH)

    # Valid top_k values
    res_1 = retriever.retrieve("Aventro", top_k=1)
    assert len(res_1) == 1

    res_3 = retriever.retrieve("Aventro", top_k=3)
    assert len(res_3) == 3

    # Invalid top_k values
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve("Aventro", top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve("Aventro", top_k=-5)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve("Aventro", top_k="3")  # type: ignore

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve("Aventro", top_k=True)  # type: ignore (bool check)


# -------------------------------------------------------------------------
# 7. Tests for Missing, Malformed, and Duplicate Chunk Inputs
# -------------------------------------------------------------------------

def test_missing_chunks_file_raises_filenotfound():
    """Verify FileNotFoundError is raised when file does not exist."""
    with pytest.raises(FileNotFoundError, match="Chunks file not found"):
        BM25Retriever.from_jsonl("non_existent_path_xyz123.jsonl")


def test_empty_chunks_file_raises_value_error(tmp_path):
    """Verify empty file raises ValueError."""
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No records found"):
        BM25Retriever.from_jsonl(empty_file)


def test_malformed_jsonl_raises_value_error(tmp_path):
    """Verify corrupted JSON line raises ValueError with line number."""
    corrupt_file = tmp_path / "corrupt.jsonl"
    corrupt_file.write_text('{"chunk_id": "1", "text": "ok"}\n{bad json}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed JSONL on line 2"):
        BM25Retriever.from_jsonl(corrupt_file)


def test_missing_required_chunk_fields_raises_value_error():
    """Verify validator checks for all mandatory chunk fields."""
    incomplete_chunks = [
        {
            "chunk_id": "c1",
            "text": "Some text",
            # missing source_document, document_title, etc.
        }
    ]
    with pytest.raises(ValueError, match="missing required field"):
        validate_chunks_data(incomplete_chunks)


def test_duplicate_chunk_ids_raises_value_error():
    """Verify validator catches duplicate chunk IDs."""
    duplicate_chunks = [
        {
            "chunk_id": "dup_01",
            "document_id": "d1",
            "source_document": "d1.md",
            "document_title": "D1",
            "section_heading": "H1",
            "heading_path": "D1 > H1",
            "text": "Sample text one"
        },
        {
            "chunk_id": "dup_01",  # duplicate ID
            "document_id": "d2",
            "source_document": "d2.md",
            "document_title": "D2",
            "section_heading": "H2",
            "heading_path": "D2 > H2",
            "text": "Sample text two"
        }
    ]
    with pytest.raises(ValueError, match="Duplicate chunk_id detected"):
        validate_chunks_data(duplicate_chunks)


# -------------------------------------------------------------------------
# 8. Strict Boundary: Rejection of Evaluation Dataset
# -------------------------------------------------------------------------

def test_rejects_evaluation_dataset():
    """Verify that evaluation questions or expected answers are strictly rejected."""
    eval_chunks = [
        {
            "chunk_id": "eval_01",
            "query_text": "What is the warranty period?",
            "expected_answer": "3 years / 100,000 km",
            "text": "What is the warranty period?",
            "document_id": "d1",
            "source_document": "d1.md",
            "document_title": "D1",
            "section_heading": "H1",
            "heading_path": "D1 > H1"
        }
    ]
    with pytest.raises(ValueError, match="Evaluation dataset detected"):
        validate_chunks_data(eval_chunks)


# -------------------------------------------------------------------------
# 9. Determinism Test
# -------------------------------------------------------------------------

def test_deterministic_retrieval_output():
    """Verify that repeated index construction yields identical scores and ranks."""
    r1 = BM25Retriever.from_jsonl(DEFAULT_CHUNKS_PATH)
    r2 = BM25Retriever.from_jsonl(DEFAULT_CHUNKS_PATH)

    query = "electric motor battery warranty service schedule"
    res1 = r1.retrieve(query, top_k=5)
    res2 = r2.retrieve(query, top_k=5)

    assert len(res1) == len(res2)
    for item1, item2 in zip(res1, res2):
        assert item1["chunk_id"] == item2["chunk_id"]
        assert item1["bm25_rank"] == item2["bm25_rank"]
        assert item1["bm25_score"] == item2["bm25_score"]


# -------------------------------------------------------------------------
# 10. Phase 1 Artifacts Integrity Check
# -------------------------------------------------------------------------

def test_phase1_artifacts_integrity_unchanged():
    """Verify that Step 7 has not modified or deleted any Phase 1 artifacts."""
    chunks_path = Path("artifacts/chunks/document_chunks.jsonl")
    npy_path = Path("artifacts/embeddings/chunk_embeddings.npy")
    meta_path = Path("artifacts/embeddings/embedding_metadata.json")
    chroma_db_dir = Path("artifacts/chroma_db")

    assert chunks_path.is_file()
    assert npy_path.is_file()
    assert meta_path.is_file()
    assert chroma_db_dir.is_dir()

    # Verify embeddings matrix shape
    embeddings = np.load(npy_path)
    assert embeddings.shape == (459, 384)

    # Verify metadata JSON
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["number_of_chunks"] == 459
    assert meta["embedding_dimension"] == 384

    # Verify ChromaDB SQLite database
    sqlite_file = chroma_db_dir / "chroma.sqlite3"
    assert sqlite_file.is_file()
    assert sqlite_file.stat().st_size > 0
