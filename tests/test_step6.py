"""
Pytest Test Suite for Phase 1, Step 6
Tests ChromaDB vector indexing, input validation, metadata scalar conversion,
idempotency, persistence in a fresh client, and evaluation data quarantine.
"""

import json
from pathlib import Path
import numpy as np
import pytest
import chromadb

from src.build_vector_index import (
    validate_indexing_inputs,
    convert_chunk_metadata_to_scalars,
    build_vector_index,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_CHUNKS_PATH,
    DEFAULT_EMBEDDINGS_NPY_PATH,
    DEFAULT_EMBEDDINGS_META_PATH,
    DEFAULT_DB_DIR,
    DEFAULT_REPORT_PATH
)


# -------------------------------------------------------------------------
# 1. Tests for Input Validation & Fingerprint Verification
# -------------------------------------------------------------------------

def test_validate_indexing_inputs_success():
    """Verify production artifacts pass validation cleanly."""
    chunks, embeddings, metadata = validate_indexing_inputs(
        DEFAULT_CHUNKS_PATH, DEFAULT_EMBEDDINGS_NPY_PATH, DEFAULT_EMBEDDINGS_META_PATH
    )
    assert len(chunks) == 459
    assert embeddings.shape == (459, 384)
    assert metadata["number_of_chunks"] == 459
    assert metadata["embedding_dimension"] == 384


def test_validation_rejects_row_count_mismatch(tmp_path):
    """Verify validation fails if chunks count does not match embedding matrix rows."""
    # Write a chunk file with only 2 chunks
    short_chunks_file = tmp_path / "short_chunks.jsonl"
    with open(DEFAULT_CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = [f.readline(), f.readline()]
    short_chunks_file.write_text("".join(chunks), encoding="utf-8")

    with pytest.raises(ValueError, match="Row count mismatch"):
        validate_indexing_inputs(
            str(short_chunks_file), DEFAULT_EMBEDDINGS_NPY_PATH, DEFAULT_EMBEDDINGS_META_PATH
        )


def test_validation_rejects_dimension_mismatch(tmp_path):
    """Verify validation fails if embedding dimension differs from metadata."""
    bad_npy = tmp_path / "bad_dim.npy"
    fake_matrix = np.ones((459, 128), dtype=np.float32)
    np.save(bad_npy, fake_matrix)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        validate_indexing_inputs(
            DEFAULT_CHUNKS_PATH, str(bad_npy), DEFAULT_EMBEDDINGS_META_PATH
        )


def test_validation_rejects_fingerprint_mismatch(tmp_path):
    """Verify validation catches modified chunk files with fingerprint mismatch."""
    meta_copy = tmp_path / "meta_mismatch.json"
    with open(DEFAULT_EMBEDDINGS_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["input_artifact_sha256"] = "0" * 64
    with open(meta_copy, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    with pytest.raises(ValueError, match="Fingerprint mismatch"):
        validate_indexing_inputs(
            DEFAULT_CHUNKS_PATH, DEFAULT_EMBEDDINGS_NPY_PATH, str(meta_copy)
        )


def test_validation_rejects_evaluation_dataset(tmp_path):
    """Verify validation strictly blocks evaluation questions from being indexed."""
    eval_chunks_file = tmp_path / "eval_chunks.jsonl"
    eval_record = {
        "chunk_id": "eval_001",
        "query_text": "What is the warranty?",
        "expected_answer": "3 years",
        "text": "Warranty info"
    }
    eval_chunks_file.write_text(json.dumps(eval_record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Evaluation record detected"):
        validate_indexing_inputs(
            str(eval_chunks_file), DEFAULT_EMBEDDINGS_NPY_PATH, DEFAULT_EMBEDDINGS_META_PATH
        )


# -------------------------------------------------------------------------
# 2. Tests for Metadata Scalar Conversion
# -------------------------------------------------------------------------

def test_convert_chunk_metadata_to_scalars():
    """Verify that nested structures are converted to Chroma-compatible scalar values."""
    chunk = {
        "chunk_id": "c_001",
        "document_id": "doc_test",
        "source_document": "test.md",
        "document_title": "Test Title",
        "section_heading": "Overview",
        "heading_path": "H1 > Overview",
        "heading_hierarchy": {"H1": "Test Title", "H2": "Overview"},
        "chunk_index": 0,
        "character_count": 150,
        "estimated_token_count": 35,
        "content_hash": "abc123hash",
        "is_sub_chunk": False
    }

    scalar_meta = convert_chunk_metadata_to_scalars(chunk)

    # All values must be scalar types supported by ChromaDB (str, int, float, bool)
    for k, v in scalar_meta.items():
        assert isinstance(v, (str, int, float, bool)), f"Non-scalar type for {k}: {type(v)}"

    assert scalar_meta["document_id"] == "doc_test"
    assert "heading_hierarchy_json" in scalar_meta
    parsed_json = json.loads(scalar_meta["heading_hierarchy_json"])
    assert parsed_json["H1"] == "Test Title"


# -------------------------------------------------------------------------
# 3. Tests for Index Creation, Idempotency & Persistence in Temp Directory
# -------------------------------------------------------------------------

def test_build_vector_index_in_temp_dir(tmp_path):
    """Test building ChromaDB index in an isolated temporary directory."""
    temp_db_dir = tmp_path / "temp_chroma"
    temp_report = tmp_path / "temp_report.md"

    summary = build_vector_index(
        chunks_path=DEFAULT_CHUNKS_PATH,
        embeddings_npy_path=DEFAULT_EMBEDDINGS_NPY_PATH,
        embeddings_meta_path=DEFAULT_EMBEDDINGS_META_PATH,
        db_dir=str(temp_db_dir),
        collection_name="test_temp_collection",
        report_path=str(temp_report),
        batch_size=150
    )

    assert summary["number_of_indexed_records"] == 459
    assert summary["embedding_dimension"] == 384
    assert summary["distance_metric"] == "cosine"
    assert temp_report.exists()

    # Verify fresh client can open and read the collection
    client = chromadb.PersistentClient(path=str(temp_db_dir))
    col = client.get_collection("test_temp_collection")
    assert col.count() == 459

    sample = col.get(ids=["doc_about_aventro_motors_chunk_000"], include=["documents", "metadatas", "embeddings"])
    assert len(sample["documents"]) == 1
    assert "Aventro Motors Pvt. Ltd." in sample["documents"][0]
    assert sample["metadatas"][0]["document_id"] == "doc_about_aventro_motors"
    assert len(sample["embeddings"][0]) == 384


def test_idempotent_rerun_in_temp_dir(tmp_path):
    """Verify repeated indexing runs recreate collection cleanly without duplicate IDs or stale records."""
    temp_db_dir = tmp_path / "temp_chroma_idempotent"
    temp_report = tmp_path / "temp_report.md"

    # Run 1
    summary1 = build_vector_index(
        db_dir=str(temp_db_dir),
        collection_name="test_idempotent_col",
        report_path=str(temp_report)
    )
    assert summary1["number_of_indexed_records"] == 459

    # Run 2 (immediate re-run)
    summary2 = build_vector_index(
        db_dir=str(temp_db_dir),
        collection_name="test_idempotent_col",
        report_path=str(temp_report)
    )
    assert summary2["number_of_indexed_records"] == 459

    # Verify count is strictly 459, not 918
    client = chromadb.PersistentClient(path=str(temp_db_dir))
    col = client.get_collection("test_idempotent_col")
    assert col.count() == 459


# -------------------------------------------------------------------------
# 4. Tests for Production ChromaDB Artifacts
# -------------------------------------------------------------------------

def test_production_chroma_db_integrity():
    """Verify production ChromaDB collection in artifacts/chroma_db exists, is persistent, and contains all 459 records."""
    prod_db_path = Path(DEFAULT_DB_DIR)
    prod_report_path = Path(DEFAULT_REPORT_PATH)

    assert prod_db_path.exists(), f"Missing production database directory: {DEFAULT_DB_DIR}"
    assert prod_report_path.exists(), f"Missing production report: {DEFAULT_REPORT_PATH}"

    client = chromadb.PersistentClient(path=str(prod_db_path))
    collection = client.get_collection(DEFAULT_COLLECTION_NAME)

    # 1. Record count
    assert collection.count() == 459

    # 2. Metadata configuration
    assert collection.metadata.get("hnsw:space") == "cosine"

    # 3. Sample integrity check
    sample_res = collection.get(
        ids=["doc_about_aventro_motors_chunk_000", "doc_what_is_abs_chunk_007"],
        include=["embeddings", "documents", "metadatas"]
    )
    assert len(sample_res["ids"]) == 2
    for emb in sample_res["embeddings"]:
        assert len(emb) == 384
        assert np.all(np.isfinite(emb))

    # 4. Report content
    report_text = prod_report_path.read_text(encoding="utf-8")
    assert "459" in report_text
    assert "aventro_motors_knowledge" in report_text
    assert "cosine" in report_text
