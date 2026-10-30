"""
Phase 2, Step 7: BM25 Keyword Retriever
Deterministic keyword retrieval component using BM25Okapi over pre-computed chunk artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Union

from rank_bm25 import BM25Okapi


DEFAULT_CHUNKS_PATH = Path("artifacts/chunks/document_chunks.jsonl")

REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "text",
    "source_document",
    "document_title",
    "section_heading",
    "heading_path"
}


def tokenize_text(text: str) -> List[str]:
    """
    Deterministically tokenizes input text into lowercase terms.
    Preserves numbers with decimals (e.g. '11.2', '1.5') and alphanumeric terms,
    splitting punctuation and whitespace.
    """
    if not text or not isinstance(text, str):
        return []
    return re.findall(r"\d+\.\d+|[a-zA-Z0-9_]+", text.lower())


def validate_chunks_data(chunks: List[Dict[str, Any]]) -> bool:
    """
    Validates that chunks data is well-formed, non-empty, contains required metadata,
    has unique chunk IDs, and does not contain evaluation questions.
    Raises ValueError on validation failure.
    """
    if not isinstance(chunks, list) or len(chunks) == 0:
        raise ValueError("Input chunks data must be a non-empty list of dictionaries.")

    seen_ids = set()
    for idx, c in enumerate(chunks):
        if not isinstance(c, dict):
            raise ValueError(f"Chunk at index {idx} is not a valid dictionary.")

        # Strict boundary: Guard against indexing evaluation data
        if "expected_answer" in c or "query_text" in c:
            raise ValueError(
                f"Evaluation dataset detected at index {idx} ({c.get('chunk_id')}). "
                "Evaluation questions and expected answers must never be indexed."
            )

        for field in REQUIRED_CHUNK_FIELDS:
            if field not in c or c[field] is None:
                raise ValueError(f"Chunk at index {idx} is missing required field: '{field}'.")
            if isinstance(c[field], str) and not c[field].strip():
                raise ValueError(f"Chunk at index {idx} has an empty value for field: '{field}'.")

        chunk_id = c["chunk_id"]
        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk_id detected at index {idx}: '{chunk_id}'.")
        seen_ids.add(chunk_id)

    return True


class BM25Retriever:
    """
    In-memory BM25 keyword retriever indexing pre-extracted chunk documents.
    Provides deterministic relevance scoring and structured citation output.
    """

    def __init__(self, chunks: List[Dict[str, Any]]) -> None:
        validate_chunks_data(chunks)
        self.chunks: List[Dict[str, Any]] = list(chunks)
        self.chunk_ids: List[str] = [c["chunk_id"] for c in self.chunks]

        # Tokenize corpus deterministically
        self.tokenized_corpus: List[List[str]] = [
            tokenize_text(c["text"]) for c in self.chunks
        ]
        self.bm25: BM25Okapi = BM25Okapi(self.tokenized_corpus)

    @classmethod
    def from_jsonl(cls, jsonl_path: Union[str, Path] = DEFAULT_CHUNKS_PATH) -> "BM25Retriever":
        """
        Construct a BM25Retriever from a chunks JSONL file.
        Raises FileNotFoundError if file is missing, ValueError if invalid.
        """
        path = Path(jsonl_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Chunks file not found at: {path}")

        chunks: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                clean_line = line.strip()
                if not clean_line:
                    continue
                try:
                    record = json.loads(clean_line)
                    chunks.append(record)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Malformed JSONL on line {line_num} in '{path}': {e}") from e

        if not chunks:
            raise ValueError(f"No records found in chunks file: '{path}'.")

        return cls(chunks=chunks)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top_k most relevant chunks for a keyword query.
        Returns a list of structured dictionaries ranked by BM25 score.
        Returns [] if query is empty, has no matching tokens, or produces no positive scores.
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")

        if not isinstance(query, str) or not query.strip():
            return []

        tokens = tokenize_text(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        # Filter to only documents with positive relevance scores
        # Tie-breaker: sort descending by score, ascending by original corpus index for determinism
        candidates: List[tuple[float, int]] = [
            (float(scores[i]), i) for i in range(len(scores)) if scores[i] > 0.0
        ]

        if not candidates:
            return []

        candidates.sort(key=lambda item: (-item[0], item[1]))
        top_candidates = candidates[:top_k]

        results: List[Dict[str, Any]] = []
        for rank, (score, idx) in enumerate(top_candidates, start=1):
            chunk = self.chunks[idx]
            citation_metadata = {
                "document_id": chunk["document_id"],
                "source_document": chunk["source_document"],
                "document_title": chunk["document_title"],
                "section_heading": chunk["section_heading"],
                "heading_path": chunk["heading_path"],
                "chunk_index": chunk.get("chunk_index"),
                "character_count": chunk.get("character_count"),
                "estimated_token_count": chunk.get("estimated_token_count"),
                "content_hash": chunk.get("content_hash"),
                "is_sub_chunk": chunk.get("is_sub_chunk", False),
                "heading_hierarchy": chunk.get("heading_hierarchy")
            }

            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "citation_metadata": citation_metadata,
                "bm25_rank": rank,
                "bm25_score": round(score, 4)
            })

        return results


def main() -> None:
    """CLI test entrypoint for BM25 retrieval."""
    query = sys.argv[1] if len(sys.argv) > 1 else "What are the 380 checkpoints for pre-owned cars?"
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"Loading chunks from: {DEFAULT_CHUNKS_PATH}...")
    retriever = BM25Retriever.from_jsonl(DEFAULT_CHUNKS_PATH)
    print(f"Loaded {len(retriever.chunks)} chunks into BM25 index.")
    print(f"\nQuery: {query!r} (top_k={top_k})")
    print("=" * 80)

    results = retriever.retrieve(query, top_k=top_k)
    if not results:
        print("No matching chunks found.")
        return

    for res in results:
        meta = res["citation_metadata"]
        first_line = res["text"].strip().split("\n")[0][:100]
        print(f"Rank {res['bm25_rank']} | BM25 Score: {res['bm25_score']:.4f}")
        print(f"  Source File    : {meta['source_document']}")
        print(f"  Section Heading: {meta['section_heading']}")
        print(f"  Heading Path   : {meta['heading_path']}")
        print(f"  Chunk ID       : {res['chunk_id']}")
        print(f"  Text Excerpt   : {first_line}...")
        print("-" * 80)


if __name__ == "__main__":
    main()
