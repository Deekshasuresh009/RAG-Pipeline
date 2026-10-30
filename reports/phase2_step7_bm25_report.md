# Aventro Motors RAG - Phase 2, Step 7: BM25 Keyword Retriever Report

## 1. Executive Summary

- **Module**: `src/phase2/bm25_retriever.py`
- **Demo Script**: `src/phase2/demo_bm25.py`
- **Test Suite**: `tests/test_phase2_bm25.py` (15 passing tests)
- **Dependency Added**: `rank-bm25>=0.2.2` (added to `requirements.txt`)
- **Indexed Corpus**: 459 chunks from `artifacts/chunks/document_chunks.jsonl`
- **Underlying Algorithm**: `BM25Okapi` with deterministic regex tokenization
- **Phase 1 Artifact Preservation**: All Phase 1 artifacts (`document_chunks.jsonl`, `chunk_embeddings.npy`, `embedding_metadata.json`, `chroma_db/`) remain completely intact and unmodified.
- **Evaluation Dataset Quarantine**: Verified that 314 evaluation questions and answers are strictly blocked from BM25 indexing.

---

## 2. Component Design & Implementation

### A. Deterministic Tokenization (`tokenize_text`)
- Converts text to lowercase and extracts alphanumeric terms and decimal numbers (e.g., `11.2`, `1.5`, `380`, `BS6`, `turbocharger`).
- Ensures clean matching across vehicle specifications, prices, part codes, and natural language query terms.

### B. Index Construction & Validation (`BM25Retriever`)
- **Factory Method**: `BM25Retriever.from_jsonl(jsonl_path)` loads and validates chunk records.
- **Strict Validation**:
  - Validates non-empty chunk lists and dictionary schemas.
  - Requires mandatory fields: `chunk_id`, `text`, `source_document`, `document_title`, `section_heading`, `heading_path`.
  - Enforces uniqueness of `chunk_id`.
  - Rejects evaluation records containing `expected_answer` or `query_text` with explicit error handling.
  - Catches missing files (`FileNotFoundError`) and malformed lines (`ValueError`).

### C. Retrieval Behavior & Scoring
- **Positive Relevance Filter**: Retains only chunks with BM25 score $> 0.0$. Chunks with zero relevance are excluded.
- **Deterministic Ranking**: Sorts descending by BM25 score, using original corpus order as secondary tie-breaker.
- **Structured Output**: Returns structured records containing:
  - `chunk_id`: Original chunk identifier.
  - `text`: Verbatim chunk Markdown text.
  - `citation_metadata`: Complete preserved metadata (`source_document`, `document_title`, `section_heading`, `heading_path`, `chunk_index`, etc.).
  - `bm25_rank`: 1-based integer rank ($1, 2, \dots$).
  - `bm25_score`: Rounded BM25 floating-point score.
- **Boundary Handling**:
  - Empty or whitespace queries return `[]`.
  - Queries with unseen/unmatched terms return `[]`.
  - Invalid `top_k` (non-integer, $\le 0$, bool) raises `ValueError`.

---

## 3. Retrieval Demonstration

Sample execution on actual corpus:
```powershell
& ".\.venv\Scripts\python.exe" -m src.phase2.demo_bm25 "What are the 380 checkpoints for pre-owned cars?"
```

**Retrieved Results**:
1. **Rank 1** (Score: `19.7134`):
   - **Source**: `Pre-Owned Certified Cars.md`
   - **Section**: `Why Choose **Aventro Certified Pre-Owned Cars?**`
   - **Chunk ID**: `doc_pre_owned_certified_cars_chunk_001`
   - **Excerpt**: Each Aventro Certified Pre-Owned vehicle undergoes an exhaustive **380+ Check Points Inspection**...
2. **Rank 2** (Score: `17.9267`):
   - **Source**: `Pre-Owned Certified Cars.md`
   - **Section**: `Pre-Owned Certified Cars`
   - **Chunk ID**: `doc_pre_owned_certified_cars_chunk_000`
3. **Rank 3** (Score: `13.6991`):
   - **Source**: `Pre-Owned Certified Cars.md`
   - **Section**: `**Contact Us**`
   - **Chunk ID**: `doc_pre_owned_certified_cars_chunk_006`
4. **Rank 4** (Score: `11.9919`):
   - **Source**: `Pre-Owned Certified Cars.md`
   - **Section**: `**Buy with Confidence**`
   - **Chunk ID**: `doc_pre_owned_certified_cars_chunk_005`
5. **Rank 5** (Score: `9.2718`):
   - **Source**: `Pre-Owned Certified Cars.md`
   - **Section**: `**Aventro 380+ Checkpoints: The Gold Standard in Assurance**`
   - **Chunk ID**: `doc_pre_owned_certified_cars_chunk_003`

---

## 4. Verification and Scope Compliance

- **Vector Index & Embeddings**: Unmodified; `artifacts/chroma_db/` and `artifacts/embeddings/` untouched.
- **No LLM / External API Calls**: Indexing and retrieval are 100% local and deterministic.
- **Ready for Hybrid Retrieval (Step 8)**: The BM25 retriever is modularized to be combined with the ChromaDB vector retriever via Reciprocal Rank Fusion (RRF) in subsequent Phase 2 steps.
