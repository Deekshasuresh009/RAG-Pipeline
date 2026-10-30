# Aventro Motors RAG — Phase 2, Step 9: Reciprocal Rank Fusion (RRF) Report

## 1. Overview & Objective

The objective of Step 9 is to combine the deterministic keyword retrieval capabilities of the Step 7 BM25 retriever (`src/phase2/bm25_retriever.py`) and the dense semantic retrieval capabilities of the Step 8 ChromaDB vector retriever (`src/phase2/vector_retriever.py`) into a single, unified, deduplicated ranking using **Reciprocal Rank Fusion (RRF)**.

The solution ensures:
- Full citation metadata preservation (`source_document`, `document_title`, `section_heading`, `heading_path`, `content_hash`, etc.).
- Independent configuration of candidate pool sizes (`bm25_top_k`, `vector_top_k`, `final_top_k`, and RRF constant `k`).
- Rank-based fusion rather than direct addition of uncalibrated raw scores / distances.
- Deterministic deduplication by canonical `chunk_id` and predictable tie-breaking.
- Robust input validation and error propagation (with an explicitly supported degraded mode).
- Complete immutability of Phase 1 canonical artifacts and Step 7/8 retriever code.

---

## 2. Mathematical Formulation & Fusion Logic

### Reciprocal Rank Fusion Formula
For any candidate document chunk $d$:
$$RRF\_score(d) = \sum_{m \in M} \frac{1}{k + rank_m(d)}$$
where:
- $M \subseteq \{\text{bm25}, \text{vector}\}$ is the set of retrievers that retrieved chunk $d$.
- $rank_m(d) \in \mathbb{Z}^+$ is the 1-based rank position of chunk $d$ in the output of retriever $m$.
- $k \in \mathbb{Z}^+$ is a configurable constant smoothing factor (default $k = 60$).

### Key Principles Followed
1. **Rank-based, Not Score-based**: Raw BM25 scores (positive unbounded floats based on term frequency and inverted document frequency) and ChromaDB cosine distances (floats bounded in $[0, 2]$ where 0 is identical) have radically different scales and statistical distributions. Direct score addition is avoided entirely; only 1-based ordinal rank positions are summed.
2. **Deterministic Tie-Breaking**: When two candidates produce identical RRF scores (e.g. one candidate ranked #1 by BM25 only: $1/61 \approx 0.0163934$, and another ranked #1 by Vector only: $1/61 \approx 0.0163934$), the tie-breaker is deterministic based on the order of first appearance across the combined candidate streams (BM25 stream evaluated first, followed by Vector stream).
3. **Chunk Deduplication**: If a chunk $d$ appears in both retriever pools, its metadata is merged into a single entry with `contributors = ["bm25", "vector"]`, and its individual ranks (`bm25_rank`, `vector_rank`) and scores (`bm25_score`, `vector_distance`) are preserved.

---

## 3. Implementation Details

### Files Created
- [`src/phase2/rrf_fusion.py`](file:///C:/Users/devi.c/RAG%20Phase1/src/phase2/rrf_fusion.py): Core implementation defining `RRFFusion`, `fuse_rankings`, `compute_rrf_score`, and validation functions.
- [`src/phase2/demo_rrf.py`](file:///C:/Users/devi.c/RAG%20Phase1/src/phase2/demo_rrf.py): CLI demonstration script formatting and displaying hybrid retrieval results.
- [`tests/test_phase2_rrf.py`](file:///C:/Users/devi.c/RAG%20Phase1/tests/test_phase2_rrf.py): 16 comprehensive unit and integration tests covering math, edge cases, degraded mode, and corpus integration.
- [`reports/phase2_step9_rrf_report.md`](file:///C:/Users/devi.c/RAG%20Phase1/reports/phase2_step9_rrf_report.md): This report document.

### Result Schema
Every result dictionary returned by `RRFFusion.fuse()` includes:
```python
{
    "chunk_id": "doc_pre_owned_certified_cars_chunk_001",
    "text": "### Why Choose **Aventro Certified Pre-Owned Cars?** ...",
    "citation_metadata": {
        "document_id": "doc_pre_owned_certified_cars",
        "source_document": "Pre-Owned Certified Cars.md",
        "document_title": "Aventro Motors Pvt. Ltd.",
        "section_heading": "Why Choose **Aventro Certified Pre-Owned Cars?**",
        "heading_path": "Aventro Motors Pvt. Ltd. > Pre-Owned Certified Cars > Why Choose **Aventro Certified Pre-Owned Cars?**",
        "chunk_index": 1,
        "character_count": 820,
        "estimated_token_count": 205,
        "content_hash": "...",
        "is_sub_chunk": False,
        "heading_hierarchy": ["Aventro Motors Pvt. Ltd.", "Pre-Owned Certified Cars", "Why Choose **Aventro Certified Pre-Owned Cars?**"]
    },
    "rrf_score": 0.032266,
    "rrf_rank": 1,
    "contributors": ["bm25", "vector"],
    "bm25_rank": 1,
    "bm25_score": 19.7134,
    "vector_rank": 3,
    "vector_distance": 0.4713
}
```

### Error Handling & Degraded Mode
- **Empty / Whitespace Query**: Safely returns `[]` without triggering retriever errors.
- **Invalid Types / Values**: Non-string queries raise `TypeError`; non-positive or boolean `k` / `top_k` values raise `ValueError`.
- **Retriever Failure**:
  - Default (`allow_degraded_mode=False`): Propagates a clear `RuntimeError` if either BM25 or Vector retriever fails.
  - Degraded Mode (`allow_degraded_mode=True`): Falls back to single retriever if one fails, while raising `RuntimeError` if both fail.
- **Security Guard**: Validates that no chunk metadata contains `query_text` or `expected_answer`, preventing evaluation question leakage.

---

## 4. Real-Corpus Demonstration Output

Command executed:
```powershell
& ".\.venv\Scripts\python.exe" -m src.phase2.demo_rrf "What are the 380 checkpoints for pre-owned cars?"
```

Terminal Output:
```text
================================================================================
 Aventro Motors RAG — Phase 2, Step 9: Reciprocal Rank Fusion (RRF) Demo
================================================================================
Query: 'What are the 380 checkpoints for pre-owned cars?'
Configuration: bm25_top_k=5, vector_top_k=5, final_top_k=10, k=60
--------------------------------------------------------------------------------
BM25 retriever matches      : 5
Vector retriever matches    : 5
Unique fused results count  : 8
================================================================================
Fused Rank #1 | RRF Score: 0.032266 | Contributors: [bm25, vector]
  Contributing Ranks : BM25 rank #1 (score: 19.7134); Vector rank #3 (dist: 0.4713)
  Chunk ID           : doc_pre_owned_certified_cars_chunk_001
  Document Title     : Aventro Motors Pvt. Ltd.
  Source Document    : Pre-Owned Certified Cars.md
  Section Heading    : Why Choose **Aventro Certified Pre-Owned Cars?**
  Heading Path       : Aventro Motors Pvt. Ltd. > Pre-Owned Certified Cars > Why Choose **Aventro Certified Pre-Owned Cars?**
  Text Excerpt       : ### Why Choose **Aventro Certified Pre-Owned Cars?**  ...
--------------------------------------------------------------------------------
Fused Rank #2 | RRF Score: 0.031514 | Contributors: [bm25, vector]
  Contributing Ranks : BM25 rank #5 (score: 9.2718); Vector rank #2 (dist: 0.4359)
  Chunk ID           : doc_pre_owned_certified_cars_chunk_003
  Document Title     : Aventro Motors Pvt. Ltd.
  Source Document    : Pre-Owned Certified Cars.md
  Section Heading    : **Aventro 380+ Checkpoints: The Gold Standard in Assurance**
  Heading Path       : Aventro Motors Pvt. Ltd. > **Aventro 380+ Checkpoints: The Gold Standard in Assurance**
  Text Excerpt       : ## **Aventro 380+ Checkpoints: The Gold Standard in Assurance** (Continued)...
--------------------------------------------------------------------------------
Fused Rank #3 | RRF Score: 0.016393 | Contributors: [vector]
  Contributing Ranks : Vector rank #1 (dist: 0.4197)
  Chunk ID           : doc_pre_owned_certified_cars_chunk_002
  Document Title     : Aventro Motors Pvt. Ltd.
  Source Document    : Pre-Owned Certified Cars.md
  Section Heading    : **Aventro 380+ Checkpoints: The Gold Standard in Assurance**
  Heading Path       : Aventro Motors Pvt. Ltd. > **Aventro 380+ Checkpoints: The Gold Standard in Assurance**
  Text Excerpt       : ## **Aventro 380+ Checkpoints: The Gold Standard in Assurance**  ...
--------------------------------------------------------------------------------
Fused Rank #4 | RRF Score: 0.016129 | Contributors: [bm25]
  Contributing Ranks : BM25 rank #2 (score: 17.9267)
  Chunk ID           : doc_pre_owned_certified_cars_chunk_000
  Document Title     : Aventro Motors Pvt. Ltd.
  Source Document    : Pre-Owned Certified Cars.md
  Section Heading    : Pre-Owned Certified Cars
  Heading Path       : Aventro Motors Pvt. Ltd. > Pre-Owned Certified Cars
  Text Excerpt       : # Aventro Motors Pvt. Ltd.  ...
--------------------------------------------------------------------------------
Fused Rank #5 | RRF Score: 0.015873 | Contributors: [bm25]
  Contributing Ranks : BM25 rank #3 (score: 13.6991)
  Chunk ID           : doc_pre_owned_certified_cars_chunk_006
  Document Title     : Aventro Motors Pvt. Ltd.
  Source Document    : Pre-Owned Certified Cars.md
  Section Heading    : **Contact Us**
  Heading Path       : Aventro Motors Pvt. Ltd. > **Buy with Confidence** > **Contact Us**
  Text Excerpt       : ### **Contact Us**...
--------------------------------------------------------------------------------
Fused Rank #6 | RRF Score: 0.015625 | Contributors: [bm25]
  Contributing Ranks : BM25 rank #4 (score: 11.9919)
  Chunk ID           : doc_pre_owned_certified_cars_chunk_005
  Document Title     : Aventro Motors Pvt. Ltd.
  Source Document    : Pre-Owned Certified Cars.md
  Section Heading    : **Buy with Confidence**
  Heading Path       : Aventro Motors Pvt. Ltd. > **Buy with Confidence**
  Text Excerpt       : ## **Buy with Confidence**  ...
--------------------------------------------------------------------------------
Fused Rank #7 | RRF Score: 0.015625 | Contributors: [vector]
  Contributing Ranks : Vector rank #4 (dist: 0.5565)
  Chunk ID           : doc_pre_owned_certified_cars_chunk_004
  Document Title     : Aventro Motors Pvt. Ltd.
  Source Document    : Pre-Owned Certified Cars.md
  Section Heading    : **All Aventro Certified Cars Come With:**
  Heading Path       : Aventro Motors Pvt. Ltd. > **All Aventro Certified Cars Come With:**
  Text Excerpt       : ## **All Aventro Certified Cars Come With:**  ...
--------------------------------------------------------------------------------
Fused Rank #8 | RRF Score: 0.015385 | Contributors: [vector]
  Contributing Ranks : Vector rank #5 (dist: 0.5615)
  Chunk ID           : doc_test_drive_at_your_doorstep_chunk_001
  Document Title     : Request a Test Drive at Your Doorstep
  Source Document    : Test Drive at Your Doorstep.md
  Section Heading    : How It Works
  Heading Path       : Request a Test Drive at Your Doorstep > How It Works
  Text Excerpt       : ## How It Works  ...
--------------------------------------------------------------------------------
```

**Observation on Fusion Quality**: Chunks `chunk_001` and `chunk_003` which were retrieved by **both** BM25 and Vector retrievers received the highest combined RRF scores ($0.032266$ and $0.031514$), placing them at RRF Rank #1 and #2.

---

## 5. Test Suite Verification & Regression Results

### 1. Step 9 Test Suite
```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests/test_phase2_rrf.py -v
```
**Result**: **16 passed**, 14 warnings in 29.29s.

### 2. Step 7 & Step 8 Retrievers Regression Suite
```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests/test_phase2_bm25.py tests/test_phase2_vector.py -v
```
**Result**: **33 passed** (15 BM25 + 18 Vector), 14 warnings in 28.65s.

### 3. Full Project Test Suite (Steps 3 through 9)
```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests/ -v
```
**Result**: **88 passed**, 14 warnings in 35.24s.
- Step 3 (Markdown extraction & CSV validation): 9 passed
- Step 4 (Document chunking & audit): 14 passed
- Step 5 (Local embedding generation): 7 passed
- Step 6 (Persistent ChromaDB indexing): 9 passed
- Step 7 (BM25 keyword retriever): 15 passed
- Step 8 (ChromaDB vector retriever): 18 passed
- Step 9 (Reciprocal Rank Fusion): 16 passed
- **Total: 88 passing tests, 0 failures, 0 skipped.**

---

## 6. Limitations & Boundaries
- RRF combines rankings based purely on rank positions; it does not evaluate semantic sufficiency or whether the retrieved text actually answers the user's question. Evidence sufficiency filtering and answer generation will be handled in subsequent steps.
- Existing Phase 1 files (`document_chunks.jsonl`, `chunk_embeddings.npy`, `artifacts/chroma_db/`) remain completely unchanged and validated.
