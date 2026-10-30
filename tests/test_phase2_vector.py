"""
Pytest Test Suite for Phase 2, Step 8: ChromaDB Vector Retriever
Tests vector retrieval, embedding generation, input validation, citation metadata integrity,
error handling for missing databases/collections, and non-modification of Phase 1 artifacts.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import chromadb

from src.phase2.vector_retriever import (
    VectorRetriever,
    ModelUnavailableError,
    load_local_embedding_model,
    DEFAULT_DB_DIR,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_MODEL_NAME,
    EXPECTED_EMBEDDING_DIM,
    EXPECTED_COLLECTION_COUNT
)


# -------------------------------------------------------------------------
# Fixtures for Fast Mocked Unit Testing and Real Integration Testing
# -------------------------------------------------------------------------

class DummyEmbeddingModel:
    """Fast deterministic mock for unit tests avoiding repeated PyTorch loading."""
    def __init__(self, dim: int = EXPECTED_EMBEDDING_DIM):
        self.dim = dim

    def encode(self, text, normalize_embeddings=True, show_progress_bar=False):
        vec = np.ones((self.dim,), dtype=np.float32)
        if normalize_embeddings:
            vec = vec / np.linalg.norm(vec)
        return vec


@pytest.fixture
def mock_embedding_model():
    return DummyEmbeddingModel()


@pytest.fixture(scope="module")
def prod_retriever():
    """Module-scoped production retriever to avoid redundant model reloads."""
    return VectorRetriever(
        persist_directory=DEFAULT_DB_DIR,
        collection_name=DEFAULT_COLLECTION_NAME,
        expected_count=EXPECTED_COLLECTION_COUNT
    )


# -------------------------------------------------------------------------
# 1. Tests for Initialization & Database Connectivity
# -------------------------------------------------------------------------

def test_production_collection_opens_successfully(prod_retriever):
    """Verify existing production collection opens and contains exactly 459 records."""
    assert prod_retriever.collection is not None
    assert prod_retriever.collection.count() == 459
    assert prod_retriever.collection_name == DEFAULT_COLLECTION_NAME


def test_missing_database_directory_raises_filenotfound():
    """Verify missing database directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="ChromaDB persistent directory not found"):
        VectorRetriever(
            persist_directory="artifacts/non_existent_chroma_dir_xyz",
            collection_name=DEFAULT_COLLECTION_NAME,
            model=DummyEmbeddingModel()
        )


def test_missing_collection_raises_value_error(tmp_path):
    """Verify attempting to access a missing collection raises ValueError."""
    fake_db = tmp_path / "fake_chroma"
    fake_db.mkdir()
    client = chromadb.PersistentClient(path=str(fake_db))
    # Create a different collection
    client.create_collection(name="different_collection")

    with pytest.raises(ValueError, match="Collection 'missing_collection' not found"):
        VectorRetriever(
            persist_directory=str(fake_db),
            collection_name="missing_collection",
            model=DummyEmbeddingModel()
        )


def test_empty_collection_raises_value_error(tmp_path):
    """Verify accessing an empty collection raises ValueError."""
    fake_db = tmp_path / "empty_chroma"
    fake_db.mkdir()
    client = chromadb.PersistentClient(path=str(fake_db))
    client.create_collection(name="empty_collection")

    with pytest.raises(ValueError, match="is empty"):
        VectorRetriever(
            persist_directory=str(fake_db),
            collection_name="empty_collection",
            model=DummyEmbeddingModel()
        )


# -------------------------------------------------------------------------
# 2. Tests for Query Embedding Generation
# -------------------------------------------------------------------------

def test_embed_query_dimension_and_finiteness(prod_retriever):
    """Verify query embeddings have 384 dimensions and contain only finite numbers."""
    vec = prod_retriever.embed_query("What is the price of Aventro Grand SUV?")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert np.all(np.isfinite(vec))
    # Normalized vector should have unit L2 norm
    norm = np.linalg.norm(vec)
    assert np.isclose(norm, 1.0, atol=1e-4)


def test_model_is_reused_across_queries(prod_retriever):
    """Verify that retriever reuses the same initialized model object across queries."""
    model_ref = prod_retriever.model
    vec1 = prod_retriever.embed_query("First query")
    vec2 = prod_retriever.embed_query("Second query")
    assert prod_retriever.model is model_ref
    assert vec1.shape == (384,)
    assert vec2.shape == (384,)


def test_embed_query_rejects_non_finite_values(tmp_path):
    """Verify embed_query raises ValueError when embedding contains NaN or Inf."""
    class NanModel:
        def encode(self, text, **kwargs):
            arr = np.ones((384,), dtype=np.float32)
            arr[0] = np.nan
            return arr

    fake_db = tmp_path / "chroma"
    fake_db.mkdir()
    client = chromadb.PersistentClient(path=str(fake_db))
    col = client.create_collection("col1")
    col.add(ids=["1"], documents=["doc"], embeddings=[[0.1]*384])

    retriever = VectorRetriever(
        persist_directory=str(fake_db),
        collection_name="col1",
        model=NanModel()
    )
    with pytest.raises(ValueError, match="non-finite values"):
        retriever.embed_query("test query")


def test_embed_query_rejects_dimension_mismatch(tmp_path):
    """Verify embed_query raises ValueError when model produces wrong dimension."""
    class BadDimModel:
        def encode(self, text, **kwargs):
            return np.ones((128,), dtype=np.float32)

    fake_db = tmp_path / "chroma_dim"
    fake_db.mkdir()
    client = chromadb.PersistentClient(path=str(fake_db))
    col = client.create_collection("col1")
    col.add(ids=["1"], documents=["doc"], embeddings=[[0.1]*384])

    retriever = VectorRetriever(
        persist_directory=str(fake_db),
        collection_name="col1",
        model=BadDimModel()
    )
    with pytest.raises(ValueError, match="dimension mismatch"):
        retriever.embed_query("test query")


def test_model_loading_failure_raises_model_unavailable_error():
    """Verify that trying to load an invalid model locally raises ModelUnavailableError."""
    with pytest.raises(ModelUnavailableError, match="Failed to load embedding model"):
        load_local_embedding_model(
            model_name="non_existent_invalid_model_12345",
            cache_dir="model_cache",
            local_files_only=True
        )


# -------------------------------------------------------------------------
# 3. Tests for Retrieval Behavior against Real Corpus
# -------------------------------------------------------------------------

def test_real_corpus_retrieval_pre_owned_checkpoints(prod_retriever):
    """Verify querying '380 checkpoints for pre-owned cars' returns relevant chunks."""
    results = prod_retriever.retrieve("What are the 380 checkpoints for pre-owned cars?", top_k=5)
    assert len(results) == 5

    # Check top result
    top = results[0]
    assert top["vector_rank"] == 1
    assert isinstance(top["vector_distance"], float)
    assert top["vector_distance"] < 0.6  # Cosine distance indicates high similarity

    # Verify that Pre-Owned Certified Cars.md is among the top retrieved chunks
    doc_sources = [r["citation_metadata"]["source_document"] for r in results]
    assert "Pre-Owned Certified Cars.md" in doc_sources

    # Check that excerpt/text mentions 380+ Checkpoints
    top_sources_text = " ".join([r["text"] for r in results[:3]])
    assert "380+" in top_sources_text or "Check" in top_sources_text


def test_retrieval_result_structure_and_metadata_preservation(prod_retriever):
    """Verify returned dictionary matches required schema and preserves Phase 1 metadata."""
    results = prod_retriever.retrieve("battery warranty", top_k=3)
    assert len(results) == 3

    for rank, res in enumerate(results, start=1):
        assert "chunk_id" in res
        assert "text" in res
        assert "citation_metadata" in res
        assert "vector_rank" in res
        assert "vector_distance" in res

        assert res["vector_rank"] == rank
        assert isinstance(res["vector_distance"], float)
        assert res["chunk_id"].startswith("doc_")

        # Citation metadata checks
        meta = res["citation_metadata"]
        assert "source_document" in meta
        assert "document_title" in meta
        assert "section_heading" in meta
        assert "heading_path" in meta
        assert meta["source_document"].endswith(".md")
        assert len(res["text"].strip()) > 0


def test_top_k_limiting_parameter(prod_retriever):
    """Verify top_k parameter strictly limits result set size."""
    res_1 = prod_retriever.retrieve("engine", top_k=1)
    assert len(res_1) == 1

    res_4 = prod_retriever.retrieve("engine", top_k=4)
    assert len(res_4) == 4


def test_retrieval_order_is_ascending_distance(prod_retriever):
    """Verify results are sorted in ascending distance order (most similar first)."""
    results = prod_retriever.retrieve("hybrid vehicle fuel economy", top_k=5)
    distances = [r["vector_distance"] for r in results]
    assert distances == sorted(distances)


# -------------------------------------------------------------------------
# 4. Tests for Input Validation & Error Handling
# -------------------------------------------------------------------------

def test_empty_and_whitespace_query_raises_value_error(prod_retriever):
    """Verify empty or whitespace-only query raises ValueError."""
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        prod_retriever.retrieve("")

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        prod_retriever.retrieve("   ")

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        prod_retriever.retrieve("\n\t  \r\n")


def test_non_string_query_raises_type_error(prod_retriever):
    """Verify non-string query raises TypeError."""
    with pytest.raises(TypeError, match="Query must be a string"):
        prod_retriever.retrieve(12345)  # type: ignore

    with pytest.raises(TypeError, match="Query must be a string"):
        prod_retriever.retrieve(None)  # type: ignore

    with pytest.raises(TypeError, match="Query must be a string"):
        prod_retriever.retrieve(["sample query"])  # type: ignore


def test_invalid_top_k_raises_value_error(prod_retriever):
    """Verify invalid top_k values raise ValueError."""
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        prod_retriever.retrieve("query", top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        prod_retriever.retrieve("query", top_k=-3)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        prod_retriever.retrieve("query", top_k="5")  # type: ignore

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        prod_retriever.retrieve("query", top_k=True)  # type: ignore


# -------------------------------------------------------------------------
# 5. Tests for Data Integrity & Non-Modification of Phase 1
# -------------------------------------------------------------------------

def test_evaluation_data_is_never_returned_as_knowledge(tmp_path):
    """Verify that if evaluation question fields are detected in retrieved metadata, an error is raised."""
    fake_db = tmp_path / "eval_chroma"
    fake_db.mkdir()
    client = chromadb.PersistentClient(path=str(fake_db))
    col = client.create_collection("col_eval")
    col.add(
        ids=["eval_chunk_001"],
        documents=["What is the price of Aventro Vista?"],
        metadatas=[{"query_text": "What is the price?", "expected_answer": "11.2 Lakh", "source_document": "eval.csv"}],
        embeddings=[[0.1]*384]
    )

    retriever = VectorRetriever(
        persist_directory=str(fake_db),
        collection_name="col_eval",
        model=DummyEmbeddingModel()
    )
    with pytest.raises(ValueError, match="Evaluation question detected in retrieved chunk"):
        retriever.retrieve("What is the price?", top_k=1)


def test_phase1_artifacts_integrity_unchanged():
    """Verify that VectorRetriever execution did not modify or corrupt Phase 1 artifacts."""
    chunks_path = Path("artifacts/chunks/document_chunks.jsonl")
    npy_path = Path("artifacts/embeddings/chunk_embeddings.npy")
    meta_path = Path("artifacts/embeddings/embedding_metadata.json")
    db_path = Path("artifacts/chroma_db/chroma.sqlite3")

    assert chunks_path.is_file()
    assert npy_path.is_file()
    assert meta_path.is_file()
    assert db_path.is_file()

    # Verify embeddings matrix
    embeddings = np.load(npy_path)
    assert embeddings.shape == (459, 384)

    # Verify ChromaDB record count is still 459
    client = chromadb.PersistentClient(path="artifacts/chroma_db")
    col = client.get_collection("aventro_motors_knowledge")
    assert col.count() == 459
