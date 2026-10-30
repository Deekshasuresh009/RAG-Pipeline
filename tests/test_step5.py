"""
Pytest Test Suite for Phase 1, Step 5
Tests local embedding generation, input validation, vector alignment,
normalization, metadata schema, and model failure handling.
"""

import json
from pathlib import Path
import numpy as np
import pytest

from src.generate_embeddings import (
    validate_chunks_data,
    load_embedding_model,
    verify_cache,
    compute_file_sha256,
    ModelUnavailableError,
    DEFAULT_MODEL_NAME,
    REQUIRED_CHUNK_METADATA,
)


CHUNKS_PATH = "artifacts/chunks/document_chunks.jsonl"
EMBEDDINGS_NPY_PATH = "artifacts/embeddings/chunk_embeddings.npy"
EMBEDDINGS_META_PATH = "artifacts/embeddings/embedding_metadata.json"
REPORT_PATH = "reports/embedding_report.md"


# -------------------------------------------------------------------------
# 1. Tests for Input Validation & Evaluation Quarantine
# -------------------------------------------------------------------------

def test_input_validation_valid_chunks():
    """Verify that correctly formatted chunks pass validation."""
    valid_chunks = [
        {
            "chunk_id": f"chunk_{i:03d}",
            "document_id": "doc_test",
            "source_document": "test.md",
            "document_title": "Test Document",
            "section_heading": "Overview",
            "chunk_index": i,
            "text": f"This is sample chunk text number {i}.",
            "character_count": 35,
            "estimated_token_count": 9,
            "content_hash": "a" * 64
        }
        for i in range(5)
    ]
    assert validate_chunks_data(valid_chunks) is True


def test_input_validation_rejects_empty_and_duplicates():
    """Verify validator catches empty lists, missing text, and duplicate chunk IDs."""
    with pytest.raises(ValueError, match="non-empty list"):
        validate_chunks_data([])

    duplicate_chunks = [
        {
            "chunk_id": "dup_001",
            "document_id": "doc_test",
            "source_document": "test.md",
            "document_title": "Test",
            "section_heading": "Overview",
            "chunk_index": 0,
            "text": "Text 1",
            "character_count": 6,
            "estimated_token_count": 2,
            "content_hash": "a" * 64
        },
        {
            "chunk_id": "dup_001",  # duplicate ID
            "document_id": "doc_test",
            "source_document": "test.md",
            "document_title": "Test",
            "section_heading": "Overview",
            "chunk_index": 1,
            "text": "Text 2",
            "character_count": 6,
            "estimated_token_count": 2,
            "content_hash": "b" * 64
        }
    ]
    with pytest.raises(ValueError, match="Duplicate chunk_id"):
        validate_chunks_data(duplicate_chunks)


def test_input_validation_rejects_evaluation_dataset():
    """Verify validator strictly rejects evaluation questions if passed as chunks."""
    eval_record = [
        {
            "chunk_id": "eval_001",
            "query_text": "What is the price?",
            "expected_answer": "11.2 Lakh",
            "text": "What is the price?",
            "document_id": "doc_test",
            "source_document": "test.md",
            "document_title": "Test",
            "section_heading": "Overview",
            "chunk_index": 0,
            "character_count": 18,
            "estimated_token_count": 4,
            "content_hash": "a" * 64
        }
    ]
    with pytest.raises(ValueError, match="Evaluation dataset detected"):
        validate_chunks_data(eval_record)


def test_input_validation_rejects_missing_metadata():
    """Verify validator checks for all required metadata fields."""
    incomplete_chunk = [
        {
            "chunk_id": "c_001",
            "text": "Valid text here",
            # missing document_id, section_heading, etc.
        }
    ]
    with pytest.raises(ValueError, match="missing required metadata field"):
        validate_chunks_data(incomplete_chunk)


# -------------------------------------------------------------------------
# 2. Tests for Model Loading & Download Failure Handling
# -------------------------------------------------------------------------

def test_missing_model_raises_model_unavailable_error():
    """Verify that an invalid or non-existent model name raises ModelUnavailableError."""
    with pytest.raises(ModelUnavailableError) as exc_info:
        load_embedding_model("non_existent_model_xyz_12345", cache_dir="model_cache", local_files_only=True)
    assert "Failed to load or download embedding model" in str(exc_info.value)


# -------------------------------------------------------------------------
# 3. Tests for Cache Invalidation & Verification
# -------------------------------------------------------------------------

def test_cache_invalidation_logic():
    """Verify that verify_cache returns False when input hash or chunk count mismatches."""
    assert verify_cache("fake_hash", 459, DEFAULT_MODEL_NAME, True, "nonexistent.npy", "nonexistent.json") is False


# -------------------------------------------------------------------------
# 4. Tests for Output Artifacts (When Generated)
# -------------------------------------------------------------------------

def test_embeddings_artifacts_if_present():
    """
    If embedding artifacts have been generated, verify:
    - Matrix shape (459, 384)
    - Row-to-chunk alignment
    - Finite numerical values
    - Unit L2 normalization
    - Metadata schema completeness
    """
    npy_path = Path(EMBEDDINGS_NPY_PATH)
    meta_path = Path(EMBEDDINGS_META_PATH)
    chunks_path = Path(CHUNKS_PATH)

    if not npy_path.exists() or not meta_path.exists():
        pytest.skip("Embedding artifacts have not been generated yet.")

    embeddings = np.load(npy_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    # 1. Shape & Dimension
    assert embeddings.ndim == 2
    assert embeddings.shape[0] == 459
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] == meta["embedding_dimension"]
    assert meta["embedding_dimension"] == 384

    # 2. Finite values
    assert np.all(np.isfinite(embeddings)), "Embedding matrix contains NaN or Inf values!"

    # 3. Row alignment with chunk IDs
    assert meta["chunk_ids"] == [c["chunk_id"] for c in chunks]

    # 4. Normalization consistency
    if meta.get("normalization_enabled"):
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3), "Embeddings are not unit L2 normalized!"

    # 5. Metadata completeness
    assert meta["model_name"] == DEFAULT_MODEL_NAME
    assert meta["number_of_chunks"] == 459
    assert "input_artifact_sha256" in meta
    assert meta["vector_dtype"] == "float32"
