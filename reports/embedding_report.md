# Aventro Motors RAG Phase 1 - Local Embedding Generation Report

## 1. Executive Summary

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Verified Output Dimension**: **384** dimensions
- **Total Chunks Embedded**: **459** chunks
- **Embedding Matrix Shape**: `(459, 384)`
- **Vector Data Type**: `float32`
- **Normalization Setting**: L2 unit-norm (normalize_embeddings=True)
- **Batch Size**: 32
- **All Values Finite**: True
- **Mean L2 Norm**: 1.000000
- **Output Vector Artifact**: `artifacts/embeddings/chunk_embeddings.npy`
- **Output Metadata Artifact**: `artifacts/embeddings/embedding_metadata.json`

---

## 2. Model & Normalization Rationale

### Model Architecture
- **Model**: `all-MiniLM-L6-v2` is a 6-layer BERT-based transformer mapped to 384 dimensions via mean pooling.
- **Local Execution**: Runs entirely on local CPU/GPU through PyTorch and `sentence-transformers`. No external API calls, token credits, or third-party cloud services are used.

### Normalization
- **L2 Unit Normalization**: Embeddings are normalized such that $\|v\|_2 = 1.0$.
- **Retrieval Benefit**: For unit vectors, cosine similarity equals the dot product (`cos(u, v) = u · v`), dramatically accelerating similarity calculations in ChromaDB and enabling equivalent L2 / inner-product indexing.

---

## 3. Data Lineage & Alignment

- **Input Artifact**: `C:/Users/devi.c/RAG Phase1/artifacts/chunks/document_chunks.jsonl`
- **Input SHA-256**: `098c06cded3ea87c7c988e2999e638919b0701da9027ebb49bf12a6e7af37de7`
- **Strict Alignment**: Row index `i` in `chunk_embeddings.npy` maps 1:1 to record `i` in `document_chunks.jsonl`.
- **Evaluation Quarantine**: Evaluation questions and CSV records are strictly excluded from embedding generation.

---

## 4. Cache & Fingerprint Protocol

The embedding generator validates cache metadata before re-executing inference:
1. **Input Hash Match**: Validates input chunk file SHA-256 hash.
2. **Model Match**: Validates model name and dimension.
3. **Config Match**: Validates normalization and chunk count.
4. **Matrix Integrity**: Validates `.npy` file loadability, finite values, and non-empty rows.
If all conditions match, redundant inference is bypassed automatically.
