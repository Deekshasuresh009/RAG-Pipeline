"""Phase 2, Step 9: Reciprocal Rank Fusion (RRF)

Provides a clean, modular component that combines the BM25 keyword retriever
(`src.phase2.bm25_retriever.BM25Retriever`) and the dense vector retriever
(`src.phase2.vector_retriever.VectorRetriever`).

Uses the standard Reciprocal Rank Fusion formula:
    RRF_score(d) = sum(1 / (k + rank_i(d)))
where rank_i is the 1-based rank from retriever i, and k is a configurable
parameter (default 60).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.phase2.bm25_retriever import BM25Retriever, DEFAULT_CHUNKS_PATH
from src.phase2.vector_retriever import (
    VectorRetriever,
    DEFAULT_DB_DIR,
    DEFAULT_COLLECTION_NAME,
)

DEFAULT_BM25_TOP_K = 5
DEFAULT_VECTOR_TOP_K = 5
DEFAULT_FINAL_TOP_K = 10
DEFAULT_K = 60


def validate_k(k: Any) -> int:
    """Validate that k is a positive integer and explicitly reject booleans."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError(f"RRF constant k must be a positive integer, got {k!r}.")
    return k


def validate_limit(limit: Any, name: str = "limit", allow_none: bool = False) -> Optional[int]:
    """Validate limit / top_k values as positive integers."""
    if allow_none and limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError(f"{name} must be a positive integer, got {limit!r}.")
    return limit


def compute_rrf_score(ranks: List[int], k: int = DEFAULT_K) -> float:
    """Compute RRF score from a list of 1-based ranks."""
    k = validate_k(k)
    return sum(1.0 / (k + r) for r in ranks)


def fuse_rankings(
    bm25_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    k: int = DEFAULT_K,
    final_top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Core deterministic fusion function.
    Fuses two lists of retriever outputs according to RRF rules:
    - Deduplicate chunks using canonical chunk_id.
    - Compute RRF score from 1-based ranks.
    - Merge citation metadata and retriever-specific ranks/scores.
    - Handle duplicate chunk IDs in an individual retriever gracefully.
    - Detect and validate malformed retriever records.
    - Deterministic tie-breaking: descending rrf_score, then order of first appearance.
    """
    k = validate_k(k)
    final_top_k = validate_limit(final_top_k, "final_top_k", allow_none=True)

    if not isinstance(bm25_results, list) or not isinstance(vector_results, list):
        raise TypeError("Retriever results must be provided as lists of dictionaries.")

    fused_map: Dict[str, Dict[str, Any]] = {}
    first_appearance: List[str] = []

    def process_result_item(item: Any, source: str) -> None:
        if not isinstance(item, dict):
            raise ValueError(f"Malformed result from {source}: item is not a dictionary ({item!r}).")

        chunk_id = item.get("chunk_id")
        if not chunk_id or not isinstance(chunk_id, str):
            raise ValueError(f"Malformed result from {source}: missing or invalid 'chunk_id'.")

        text = item.get("text")
        if text is None or not isinstance(text, str):
            raise ValueError(f"Malformed result from {source} for chunk '{chunk_id}': missing or invalid 'text'.")

        # Check citation metadata
        citation_metadata = item.get("citation_metadata")
        if not isinstance(citation_metadata, dict):
            raise ValueError(f"Malformed result from {source} for chunk '{chunk_id}': missing 'citation_metadata'.")

        # Strict security guard against evaluation dataset leakage
        if "query_text" in citation_metadata or "expected_answer" in citation_metadata:
            raise ValueError(f"Evaluation question detected in retrieved chunk '{chunk_id}'.")

        if chunk_id not in fused_map:
            fused_map[chunk_id] = {
                "chunk_id": chunk_id,
                "text": text,
                "citation_metadata": dict(citation_metadata),
                "contributors": [],
                "ranks": [],
                "bm25_rank": None,
                "vector_rank": None,
                "bm25_score": None,
                "vector_distance": None,
            }
            first_appearance.append(chunk_id)

        entry = fused_map[chunk_id]

        if source == "bm25":
            if "bm25" not in entry["contributors"]:
                entry["contributors"].append("bm25")
                rank = item.get("bm25_rank")
                if rank is not None:
                    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
                        raise ValueError(f"Invalid bm25_rank: {rank!r}")
                    entry["bm25_rank"] = rank
                    entry["ranks"].append(rank)
                if "bm25_score" in item and item["bm25_score"] is not None:
                    entry["bm25_score"] = float(item["bm25_score"])
        elif source == "vector":
            if "vector" not in entry["contributors"]:
                entry["contributors"].append("vector")
                rank = item.get("vector_rank")
                if rank is not None:
                    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
                        raise ValueError(f"Invalid vector_rank: {rank!r}")
                    entry["vector_rank"] = rank
                    entry["ranks"].append(rank)
                if "vector_distance" in item and item["vector_distance"] is not None:
                    entry["vector_distance"] = float(item["vector_distance"])
        else:
            raise ValueError(f"Unknown retriever source: {source}")

    for b in bm25_results:
        process_result_item(b, "bm25")
    for v in vector_results:
        process_result_item(v, "vector")

    # Calculate final RRF scores
    for cid, entry in fused_map.items():
        entry["rrf_score"] = compute_rrf_score(entry["ranks"], k=k)
        entry["contributors"] = sorted(entry["contributors"])

    # Deterministic sorting:
    # 1. Descending RRF score
    # 2. First appearance index in retriever lists
    appearance_index = {cid: idx for idx, cid in enumerate(first_appearance)}
    sorted_cids = sorted(
        fused_map.keys(),
        key=lambda cid: (-fused_map[cid]["rrf_score"], appearance_index[cid]),
    )

    final_results: List[Dict[str, Any]] = []
    for rank, cid in enumerate(sorted_cids, start=1):
        raw_entry = fused_map[cid]
        res: Dict[str, Any] = {
            "chunk_id": raw_entry["chunk_id"],
            "text": raw_entry["text"],
            "citation_metadata": raw_entry["citation_metadata"],
            "rrf_score": raw_entry["rrf_score"],
            "rrf_rank": rank,
            "contributors": raw_entry["contributors"],
        }
        if raw_entry["bm25_rank"] is not None:
            res["bm25_rank"] = raw_entry["bm25_rank"]
        if raw_entry["vector_rank"] is not None:
            res["vector_rank"] = raw_entry["vector_rank"]
        if raw_entry["bm25_score"] is not None:
            res["bm25_score"] = raw_entry["bm25_score"]
        if raw_entry["vector_distance"] is not None:
            res["vector_distance"] = raw_entry["vector_distance"]

        final_results.append(res)

    if final_top_k is not None:
        final_results = final_results[:final_top_k]

    return final_results


class RRFFusion:
    """
    Reciprocal Rank Fusion coordinator that queries both BM25 and Vector retrievers,
    fuses and deduplicates their results, preserves citation metadata, and handles errors.
    """

    def __init__(
        self,
        bm25_retriever: Optional[BM25Retriever] = None,
        vector_retriever: Optional[VectorRetriever] = None,
        bm25_chunks_path: Union[str, Path] = DEFAULT_CHUNKS_PATH,
        vector_db_dir: Union[str, Path] = DEFAULT_DB_DIR,
        vector_collection_name: str = DEFAULT_COLLECTION_NAME,
        default_k: int = DEFAULT_K,
        allow_degraded_mode: bool = False,
    ) -> None:
        self.default_k = validate_k(default_k)
        self.allow_degraded_mode = allow_degraded_mode

        self._bm25_retriever = bm25_retriever
        self._vector_retriever = vector_retriever
        self._bm25_chunks_path = Path(bm25_chunks_path).resolve()
        self._vector_db_dir = Path(vector_db_dir).resolve()
        self._vector_collection_name = vector_collection_name

    @property
    def bm25(self) -> BM25Retriever:
        if self._bm25_retriever is None:
            self._bm25_retriever = BM25Retriever.from_jsonl(self._bm25_chunks_path)
        return self._bm25_retriever

    @property
    def vector(self) -> VectorRetriever:
        if self._vector_retriever is None:
            self._vector_retriever = VectorRetriever(
                persist_directory=self._vector_db_dir,
                collection_name=self._vector_collection_name,
            )
        return self._vector_retriever

    def fuse(
        self,
        query: str,
        bm25_top_k: int = DEFAULT_BM25_TOP_K,
        vector_top_k: int = DEFAULT_VECTOR_TOP_K,
        final_top_k: Optional[int] = DEFAULT_FINAL_TOP_K,
        k: Optional[int] = None,
        allow_degraded: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute BM25 and vector retrieval, then fuse results using RRF.
        Returns empty list for valid empty/whitespace query when treated as empty input.
        """
        if not isinstance(query, str):
            raise TypeError(f"Query must be a string, got {type(query).__name__}.")

        clean_query = query.strip()
        if not clean_query:
            # Query is whitespace or empty
            return []

        validate_limit(bm25_top_k, "bm25_top_k")
        validate_limit(vector_top_k, "vector_top_k")
        validate_limit(final_top_k, "final_top_k", allow_none=True)

        k_val = validate_k(k if k is not None else self.default_k)
        degraded = allow_degraded if allow_degraded is not None else self.allow_degraded_mode

        bm25_results: List[Dict[str, Any]] = []
        vector_results: List[Dict[str, Any]] = []
        bm25_error: Optional[Exception] = None
        vector_error: Optional[Exception] = None

        try:
            bm25_results = self.bm25.retrieve(clean_query, top_k=bm25_top_k)
        except Exception as exc:
            bm25_error = exc
            if not degraded:
                raise RuntimeError(f"BM25 retrieval failed: {exc}") from exc

        try:
            vector_results = self.vector.retrieve(clean_query, top_k=vector_top_k)
        except Exception as exc:
            vector_error = exc
            if not degraded:
                raise RuntimeError(f"Vector retrieval failed: {exc}") from exc

        if bm25_error and vector_error:
            raise RuntimeError(
                f"Both retrievers failed in degraded mode. BM25 error: {bm25_error}; Vector error: {vector_error}"
            )

        fused = fuse_rankings(
            bm25_results=bm25_results,
            vector_results=vector_results,
            k=k_val,
            final_top_k=final_top_k,
        )

        return fused
