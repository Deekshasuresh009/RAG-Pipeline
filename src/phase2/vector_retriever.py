"""
Phase 2, Step 8: ChromaDB Vector Retriever
Retrieves canonical knowledge chunks from persistent ChromaDB using local dense embeddings.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
import chromadb


DEFAULT_DB_DIR = "artifacts/chroma_db"
DEFAULT_COLLECTION_NAME = "aventro_motors_knowledge"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = "model_cache"
EXPECTED_EMBEDDING_DIM = 384
EXPECTED_COLLECTION_COUNT = 459


class ModelUnavailableError(Exception):
    """Raised when the embedding model cannot be loaded or executed."""
    pass


def load_local_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    local_files_only: bool = True
) -> Any:
    """
    Load local embedding model using sentence_transformers.
    Reuses cached weights without external API calls or network retries.
    Raises ModelUnavailableError on failure.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ModelUnavailableError(f"sentence-transformers package is not installed: {e}") from e

    cache_path = Path(cache_dir).resolve() if cache_dir else None
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_path)

    try:
        # Default to local_files_only=True for instant offline loading
        model = SentenceTransformer(
            model_name,
            cache_folder=str(cache_path) if cache_path else None,
            local_files_only=local_files_only
        )
        return model
    except Exception as e:
        raise ModelUnavailableError(
            f"Failed to load embedding model '{model_name}' locally from '{cache_dir}'. "
            f"Ensure weights are cached in '{cache_dir}'. Details: {e}"
        ) from e


class VectorRetriever:
    """
    Persistent ChromaDB vector retriever for Aventro Motors RAG.
    Generates local embeddings and retrieves nearest neighbor chunks by cosine distance.
    """

    def __init__(
        self,
        persist_directory: Union[str, Path] = DEFAULT_DB_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
        normalize_embeddings: bool = True,
        expected_dim: int = EXPECTED_EMBEDDING_DIM,
        expected_count: Optional[int] = EXPECTED_COLLECTION_COUNT,
        model: Optional[Any] = None,
        client: Optional[Any] = None
    ) -> None:
        self.persist_directory = Path(persist_directory).resolve()
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.cache_dir = cache_dir
        self.normalize_embeddings = normalize_embeddings
        self.expected_dim = expected_dim
        self.expected_count = expected_count

        # 1. Initialize & validate ChromaDB database connection
        if client is not None:
            self.client = client
        else:
            if not self.persist_directory.exists():
                raise FileNotFoundError(
                    f"ChromaDB persistent directory not found at: '{self.persist_directory}'."
                )
            try:
                self.client = chromadb.PersistentClient(path=str(self.persist_directory))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to open ChromaDB database at '{self.persist_directory}': {e}"
                ) from e

        # 2. Access and validate collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except Exception as e:
            raise ValueError(
                f"Collection '{self.collection_name}' not found in ChromaDB at '{self.persist_directory}': {e}"
            ) from e

        total_docs = self.collection.count()
        if total_docs == 0:
            raise ValueError(
                f"Collection '{self.collection_name}' in '{self.persist_directory}' is empty."
            )

        if self.expected_count is not None and total_docs != self.expected_count:
            # Report discrepancy if count differs from expected canonical count
            print(
                f"[VectorRetriever] Note: Collection record count ({total_docs}) "
                f"differs from expected count ({self.expected_count}).",
                file=sys.stderr
            )

        # 3. Load embedding model once during initialization
        if model is not None:
            self.model = model
        else:
            self.model = load_local_embedding_model(
                model_name=self.embedding_model_name,
                cache_dir=self.cache_dir,
                local_files_only=True
            )

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate a dense embedding vector for a user query.
        Validates dimension, finiteness, and applies normalization.
        """
        try:
            vec = self.model.encode(
                query,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False
            )
        except Exception as e:
            raise RuntimeError(f"Embedding generation failed for query: {e}") from e

        if not isinstance(vec, np.ndarray):
            vec = np.array(vec, dtype=np.float32)

        # Validate finiteness
        if not np.all(np.isfinite(vec)):
            raise ValueError("Generated query embedding contains non-finite values (NaN or Inf).")

        # Validate dimension
        if vec.ndim == 1:
            actual_dim = vec.shape[0]
        elif vec.ndim == 2:
            actual_dim = vec.shape[1]
            vec = vec.flatten()
        else:
            raise ValueError(f"Unexpected embedding shape: {vec.shape}")

        if actual_dim != self.expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.expected_dim}, got {actual_dim}."
            )

        return vec.astype(np.float32)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve up to top_k most relevant chunks for a user query from ChromaDB.
        Returns results in ascending cosine distance order with complete citation metadata.
        """
        # Validate query string
        if not isinstance(query, str):
            raise TypeError(f"Query must be a string, got {type(query).__name__}.")

        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query must not be empty or whitespace-only.")

        # Validate top_k
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}.")

        # Generate query embedding
        query_embedding = self.embed_query(clean_query)

        # Query ChromaDB collection
        try:
            response = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            raise RuntimeError(f"ChromaDB query failed: {e}") from e

        if not response or "ids" not in response or not response["ids"] or not response["ids"][0]:
            return []

        ids_list = response["ids"][0]
        docs_list = response.get("documents", [[]])[0]
        metas_list = response.get("metadatas", [[]])[0]
        dists_list = response.get("distances", [[]])[0]

        results: List[Dict[str, Any]] = []
        for idx in range(len(ids_list)):
            chunk_id = ids_list[idx]
            text = docs_list[idx] if idx < len(docs_list) else ""
            meta = metas_list[idx] if idx < len(metas_list) else {}
            raw_dist = float(dists_list[idx]) if idx < len(dists_list) else 0.0

            # Guard against evaluation question leakage
            if "query_text" in meta or "expected_answer" in meta:
                raise ValueError(
                    f"Evaluation question detected in retrieved chunk '{chunk_id}'. "
                    "Evaluation data must never be indexed or retrieved as knowledge."
                )

            # Reconstruct heading hierarchy if stored as JSON string
            heading_hierarchy = meta.get("heading_hierarchy")
            if "heading_hierarchy_json" in meta and meta["heading_hierarchy_json"]:
                try:
                    heading_hierarchy = json.loads(meta["heading_hierarchy_json"])
                except Exception:
                    heading_hierarchy = meta.get("heading_hierarchy")

            citation_metadata = {
                "document_id": meta.get("document_id", ""),
                "source_document": meta.get("source_document", ""),
                "document_title": meta.get("document_title", ""),
                "section_heading": meta.get("section_heading", ""),
                "heading_path": meta.get("heading_path", ""),
                "chunk_index": meta.get("chunk_index"),
                "character_count": meta.get("character_count"),
                "estimated_token_count": meta.get("estimated_token_count"),
                "content_hash": meta.get("content_hash"),
                "is_sub_chunk": meta.get("is_sub_chunk", False),
                "heading_hierarchy": heading_hierarchy
            }

            results.append({
                "chunk_id": chunk_id,
                "text": text,
                "citation_metadata": citation_metadata,
                "vector_rank": idx + 1,
                "vector_distance": round(raw_dist, 6)
            })

        # Ensure sorted in ascending distance order
        results.sort(key=lambda r: r["vector_distance"])
        # Re-assign 1-based ranks in case of sorting adjustments
        for rank, res in enumerate(results, start=1):
            res["vector_rank"] = rank

        return results


def main() -> None:
    """CLI testing entrypoint."""
    query = sys.argv[1] if len(sys.argv) > 1 else "What are the 380 checkpoints for pre-owned cars?"
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"Initializing VectorRetriever on '{DEFAULT_DB_DIR}'...")
    retriever = VectorRetriever()
    print(f"Collection: '{retriever.collection_name}' (count: {retriever.collection.count()})")
    print(f"Query: {query!r} (top_k={top_k})")
    print("=" * 80)

    results = retriever.retrieve(query, top_k=top_k)
    for res in results:
        meta = res["citation_metadata"]
        first_line = res["text"].strip().split("\n")[0][:100]
        print(f"Rank {res['vector_rank']} | Cosine Distance: {res['vector_distance']:.6f}")
        print(f"  Source File    : {meta['source_document']}")
        print(f"  Document Title : {meta['document_title']}")
        print(f"  Section Heading: {meta['section_heading']}")
        print(f"  Chunk ID       : {res['chunk_id']}")
        print(f"  Text Excerpt   : {first_line}...")
        print("-" * 80)


if __name__ == "__main__":
    main()
