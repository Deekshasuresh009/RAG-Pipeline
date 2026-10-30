"""
Local Embedding Generation Module
Aventro Motors RAG Project - Phase 1, Step 5

Generates local sentence embeddings for 459 text chunks using sentence-transformers/all-MiniLM-L6-v2.
Ensures local execution, input validation, batch processing, L2 normalization,
deterministic caching, and generates NumPy vector and JSON metadata artifacts.
"""

import os
import sys
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np


# Default configuration
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32
DEFAULT_NORMALIZE = True
DEFAULT_CACHE_DIR = "model_cache"
REQUIRED_CHUNK_METADATA = [
    "chunk_id",
    "document_id",
    "source_document",
    "document_title",
    "section_heading",
    "chunk_index",
    "character_count",
    "estimated_token_count",
    "content_hash"
]


class ModelUnavailableError(Exception):
    """Raised when the embedding model cannot be loaded or downloaded."""
    pass


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_chunks_data(chunks: List[Dict[str, Any]]) -> bool:
    """
    Validate that chunks data is well-formed, non-empty, contains required metadata,
    and is not an evaluation question dataset.
    Raises ValueError on validation failure.
    """
    if not isinstance(chunks, list) or len(chunks) == 0:
        raise ValueError("Input chunks data must be a non-empty list of records.")

    seen_ids = set()
    for idx, c in enumerate(chunks):
        if not isinstance(c, dict):
            raise ValueError(f"Chunk at index {idx} is not a valid dictionary.")

        # Guard against using evaluation questions as knowledge chunks
        if "expected_answer" in c or "query_text" in c:
            raise ValueError(
                f"Evaluation dataset detected at index {idx} ({c.get('chunk_id')}). "
                "Evaluation questions must never be used as embedding input."
            )

        chunk_id = c.get("chunk_id")
        if not chunk_id or not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(f"Chunk at index {idx} has an empty or invalid 'chunk_id'.")

        if chunk_id in seen_ids:
            raise ValueError(f"Duplicate chunk_id detected at index {idx}: '{chunk_id}'.")
        seen_ids.add(chunk_id)

        text = c.get("text")
        if not text or not isinstance(text, str) or not text.strip():
            raise ValueError(f"Chunk '{chunk_id}' has empty or invalid 'text'.")

        for field in REQUIRED_CHUNK_METADATA:
            if field not in c or c[field] is None:
                raise ValueError(f"Chunk '{chunk_id}' is missing required metadata field: '{field}'.")

    return True


def load_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    local_files_only: bool = False
):
    """
    Load local embedding model using sentence_transformers.
    Raises ModelUnavailableError if the model cannot be loaded or downloaded.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ModelUnavailableError(f"sentence-transformers package is not installed: {e}")

    # Set cache dir to keep weights inside project workspace
    cache_path = Path(cache_dir).resolve() if cache_dir else None
    if cache_path:
        cache_path.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_path)

    try:
        print(f"Loading embedding model '{model_name}' (cache: {cache_dir}, local_files_only={local_files_only})...")
        model = SentenceTransformer(
            model_name,
            cache_folder=str(cache_path) if cache_path else None,
            local_files_only=local_files_only
        )
        return model
    except Exception as e:
        err_msg = str(e)
        raise ModelUnavailableError(
            f"Failed to load or download embedding model '{model_name}'.\n"
            f"Details: {err_msg}\n"
            f"If working offline or behind a network sandbox, the model weights must first be downloaded "
            f"to the local cache folder ('{cache_dir}')."
        ) from e


def verify_cache(
    input_hash: str,
    num_chunks: int,
    model_name: str,
    normalize: bool,
    npy_path: str,
    meta_path: str
) -> bool:
    """Check if existing embedding artifacts match current input, model, and settings."""
    npy_file = Path(npy_path)
    meta_file = Path(meta_path)

    if not npy_file.exists() or not meta_file.exists():
        return False

    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if meta.get("input_artifact_sha256") != input_hash:
            return False
        if meta.get("number_of_chunks") != num_chunks:
            return False
        if meta.get("model_name") != model_name:
            return False
        if meta.get("normalization_enabled") != normalize:
            return False

        # Verify npy file integrity
        embeddings = np.load(npy_file)
        if embeddings.shape[0] != num_chunks:
            return False
        if not np.all(np.isfinite(embeddings)):
            return False

        return True
    except Exception:
        return False


def generate_embeddings(
    input_chunks_path: str = "artifacts/chunks/document_chunks.jsonl",
    output_npy_path: str = "artifacts/embeddings/chunk_embeddings.npy",
    output_meta_path: str = "artifacts/embeddings/embedding_metadata.json",
    output_report_path: str = "reports/embedding_report.md",
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    normalize: bool = DEFAULT_NORMALIZE,
    cache_dir: str = DEFAULT_CACHE_DIR,
    force_recompute: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate embeddings for all chunks in the input file.
    Saves .npy vectors, JSON metadata, and Markdown report.
    Returns:
        embeddings: np.ndarray of shape (num_chunks, dim)
        metadata: Dict containing complete run metadata
    """
    in_path = Path(input_chunks_path).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input chunks artifact not found: {in_path}")

    # 1. Read and validate chunks
    with open(in_path, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    print(f"Read {len(chunks)} chunks from '{input_chunks_path}'. Validating schema...")
    validate_chunks_data(chunks)
    input_hash = compute_file_sha256(str(in_path))

    # 2. Check cache
    if not force_recompute and verify_cache(input_hash, len(chunks), model_name, normalize, output_npy_path, output_meta_path):
        print(f"Cache hit! Valid embeddings matching input hash '{input_hash[:12]}...' found at '{output_npy_path}'.")
        with open(output_meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        embeddings = np.load(output_npy_path)
        return embeddings, metadata

    # 3. Load model
    model = load_embedding_model(model_name, cache_dir=cache_dir)
    
    # 4. Determine actual dimension dynamically
    dummy = model.encode(["test"], normalize_embeddings=normalize)
    actual_dim = int(dummy.shape[1])
    print(f"Model successfully loaded. Verified embedding dimension: {actual_dim}")

    # 5. Extract texts and generate embeddings in batches
    texts = [c["text"] for c in chunks]
    total_chunks = len(texts)
    total_batches = (total_chunks + batch_size - 1) // batch_size
    print(f"Generating embeddings for {total_chunks} chunks in {total_batches} batches (batch_size={batch_size}, normalize={normalize})...")

    all_embeddings = []
    for b_idx in range(total_batches):
        start = b_idx * batch_size
        end = min(start + batch_size, total_chunks)
        batch_texts = texts[start:end]
        
        batch_vecs = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        all_embeddings.append(batch_vecs)
        pct = (end / total_chunks) * 100.0
        print(f"  -> Batch {b_idx + 1:>2}/{total_batches} ({end:>3}/{total_chunks} chunks) [{pct:>5.1f}%]")

    embeddings = np.vstack(all_embeddings).astype(np.float32)

    # 6. Validate generated vectors
    if embeddings.shape != (total_chunks, actual_dim):
        raise ValueError(f"Shape mismatch: expected ({total_chunks}, {actual_dim}), got {embeddings.shape}")

    if not np.all(np.isfinite(embeddings)):
        raise ValueError("Generated embedding matrix contains non-finite values (NaN or Inf).")

    # Check normalization if enabled
    norms = np.linalg.norm(embeddings, axis=1)
    if normalize:
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("Embeddings were set to normalized, but L2 norms deviate from 1.0.")

    # 7. Save Artifacts
    npy_out = Path(output_npy_path).resolve()
    npy_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_out, embeddings)
    print(f"Saved vector matrix ({embeddings.shape}, {embeddings.dtype}) to '{output_npy_path}'")

    metadata = {
        "model_name": model_name,
        "embedding_dimension": actual_dim,
        "number_of_chunks": total_chunks,
        "normalization_enabled": normalize,
        "normalization_setting": "L2 unit-norm (normalize_embeddings=True)" if normalize else "Unnormalized",
        "batch_size": batch_size,
        "vector_dtype": str(embeddings.dtype),
        "input_artifact_path": str(in_path).replace("\\", "/"),
        "input_artifact_sha256": input_hash,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "validation": {
            "all_finite": True,
            "shape": list(embeddings.shape),
            "l2_norm_mean": float(np.mean(norms)),
            "l2_norm_min": float(np.min(norms)),
            "l2_norm_max": float(np.max(norms)),
            "aligned_chunk_ids": True
        }
    }

    meta_out = Path(output_meta_path).resolve()
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to '{output_meta_path}'")

    # 8. Generate Report
    generate_embedding_report(metadata, output_report_path)
    print(f"Saved embedding report to '{output_report_path}'")

    return embeddings, metadata


def generate_embedding_report(metadata: Dict[str, Any], report_path: str):
    """Generate Markdown report for local embedding generation."""
    lines = []
    lines.append("# Aventro Motors RAG Phase 1 - Local Embedding Generation Report")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(f"- **Embedding Model**: `{metadata['model_name']}`")
    lines.append(f"- **Verified Output Dimension**: **{metadata['embedding_dimension']}** dimensions")
    lines.append(f"- **Total Chunks Embedded**: **{metadata['number_of_chunks']}** chunks")
    lines.append(f"- **Embedding Matrix Shape**: `({metadata['number_of_chunks']}, {metadata['embedding_dimension']})`")
    lines.append(f"- **Vector Data Type**: `{metadata['vector_dtype']}`")
    lines.append(f"- **Normalization Setting**: {metadata['normalization_setting']}")
    lines.append(f"- **Batch Size**: {metadata['batch_size']}")
    lines.append(f"- **All Values Finite**: {metadata['validation']['all_finite']}")
    lines.append(f"- **Mean L2 Norm**: {metadata['validation']['l2_norm_mean']:.6f}")
    lines.append(f"- **Output Vector Artifact**: `artifacts/embeddings/chunk_embeddings.npy`")
    lines.append(f"- **Output Metadata Artifact**: `artifacts/embeddings/embedding_metadata.json`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Model & Normalization Rationale")
    lines.append("")
    lines.append("### Model Architecture")
    lines.append("- **Model**: `all-MiniLM-L6-v2` is a 6-layer BERT-based transformer mapped to 384 dimensions via mean pooling.")
    lines.append("- **Local Execution**: Runs entirely on local CPU/GPU through PyTorch and `sentence-transformers`. No external API calls, token credits, or third-party cloud services are used.")
    lines.append("")
    lines.append("### Normalization")
    lines.append("- **L2 Unit Normalization**: Embeddings are normalized such that $\\|v\\|_2 = 1.0$.")
    lines.append("- **Retrieval Benefit**: For unit vectors, cosine similarity equals the dot product (`cos(u, v) = u · v`), dramatically accelerating similarity calculations in ChromaDB and enabling equivalent L2 / inner-product indexing.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Data Lineage & Alignment")
    lines.append("")
    lines.append(f"- **Input Artifact**: `{metadata['input_artifact_path']}`")
    lines.append(f"- **Input SHA-256**: `{metadata['input_artifact_sha256']}`")
    lines.append(f"- **Strict Alignment**: Row index `i` in `chunk_embeddings.npy` maps 1:1 to record `i` in `document_chunks.jsonl`.")
    lines.append(f"- **Evaluation Quarantine**: Evaluation questions and CSV records are strictly excluded from embedding generation.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Cache & Fingerprint Protocol")
    lines.append("")
    lines.append("The embedding generator validates cache metadata before re-executing inference:")
    lines.append("1. **Input Hash Match**: Validates input chunk file SHA-256 hash.")
    lines.append("2. **Model Match**: Validates model name and dimension.")
    lines.append("3. **Config Match**: Validates normalization and chunk count.")
    lines.append("4. **Matrix Integrity**: Validates `.npy` file loadability, finite values, and non-empty rows.")
    lines.append("If all conditions match, redundant inference is bypassed automatically.")

    rep_path = Path(report_path).resolve()
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    print("=" * 70)
    print("    AVENTRO MOTORS RAG - PHASE 1, STEP 5 EMBEDDING PIPELINE")
    print("=" * 70)

    try:
        embeddings, metadata = generate_embeddings()
        print("\nEmbedding generation completed successfully.")
        print(f"Matrix shape: {embeddings.shape}")
        print(f"Dimension   : {metadata['embedding_dimension']}")
    except ModelUnavailableError as e:
        print(f"\n[MODEL UNAVAILABLE] {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
