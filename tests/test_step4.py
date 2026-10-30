"""
Pytest Test Suite for Phase 1, Step 4
Tests structure-aware document chunking, metadata preservation, deterministic IDs,
table & list retention, boundary splitting, and artifact generation.
"""

import json
import re
from pathlib import Path
import pytest

from src.chunk_documents import (
    estimate_tokens,
    compute_content_hash,
    extract_table_header,
    chunk_document,
    chunk_all_documents,
)


DOCS_PATH = "artifacts/extracted/documents.jsonl"
CHUNKS_PATH = "artifacts/chunks/document_chunks.jsonl"
REPORT_PATH = "reports/chunking_report.md"


@pytest.fixture(scope="module")
def extracted_docs():
    """Load extracted documents artifact from Step 3."""
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(scope="module")
def chunked_data():
    """Run chunking pipeline and return chunks and stats."""
    chunks, stats = chunk_all_documents(DOCS_PATH, CHUNKS_PATH, REPORT_PATH)
    return chunks, stats


# -------------------------------------------------------------------------
# 1. Tests for Metadata Completeness & Determinism
# -------------------------------------------------------------------------

def test_every_chunk_has_required_metadata(chunked_data):
    """Ensure every chunk contains all 10 required metadata fields."""
    chunks, _ = chunked_data
    required_fields = [
        "chunk_id",
        "document_id",
        "source_document",
        "document_title",
        "section_heading",
        "chunk_index",
        "text",
        "character_count",
        "estimated_token_count",
        "content_hash"
    ]

    for chunk in chunks:
        for field in required_fields:
            assert field in chunk, f"Chunk {chunk.get('chunk_id')} missing field: {field}"
            assert chunk[field] is not None, f"Chunk {chunk.get('chunk_id')} has null for: {field}"

        # Verify specific types and constraints
        assert isinstance(chunk["chunk_index"], int) and chunk["chunk_index"] >= 0
        assert isinstance(chunk["character_count"], int) and chunk["character_count"] > 0
        assert isinstance(chunk["estimated_token_count"], int) and chunk["estimated_token_count"] > 0
        assert len(chunk["content_hash"]) == 64 and re.match(r"^[0-9a-f]{64}$", chunk["content_hash"])
        assert chunk["document_id"].startswith("doc_")
        assert chunk["source_document"].endswith(".md")


def test_chunk_ids_unique_and_deterministic(extracted_docs):
    """Verify chunk IDs are unique within the dataset and remain completely deterministic."""
    doc_sample = extracted_docs[0]
    run1 = chunk_document(doc_sample)
    run2 = chunk_document(doc_sample)

    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2):
        assert c1["chunk_id"] == c2["chunk_id"]
        assert c1["content_hash"] == c2["content_hash"]
        assert c1["text"] == c2["text"]

    # Verify global uniqueness across all generated chunks
    chunks, stats = chunk_all_documents(DOCS_PATH, CHUNKS_PATH, REPORT_PATH)
    chunk_ids = [c["chunk_id"] for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs found in output!"


def test_chunk_text_non_empty(chunked_data):
    """Verify no chunk has empty text or zero character count."""
    chunks, stats = chunked_data
    assert stats["empty_chunks_count"] == 0
    for chunk in chunks:
        assert chunk["text"].strip(), f"Empty chunk detected: {chunk['chunk_id']}"
        assert chunk["character_count"] == len(chunk["text"])


# -------------------------------------------------------------------------
# 2. Tests for Heading Context & Structure Retention
# -------------------------------------------------------------------------

def test_heading_context_retained(chunked_data):
    """Verify heading context and hierarchy are stored and retained in chunks."""
    chunks, _ = chunked_data
    for chunk in chunks:
        assert chunk["section_heading"], f"Chunk {chunk['chunk_id']} has empty section_heading"
        assert chunk["heading_path"], f"Chunk {chunk['chunk_id']} has empty heading_path"
        # The section heading should appear in the heading path
        assert chunk["section_heading"] in chunk["heading_path"]


def test_tables_preserved_and_headers_propagated(extracted_docs):
    """Verify markdown tables, numbers, prices, and continuation headers are preserved."""
    # Find a document known to have large pricing tables
    pricing_doc = next(d for d in extracted_docs if "Aventro Bolt Sedan" in d["document_name"])
    chunks = chunk_document(pricing_doc)

    # Check that pricing values and table syntax are preserved
    table_chunks = [c for c in chunks if "|" in c["text"]]
    assert len(table_chunks) >= 1

    found_price = any("8.2" in c["text"] or "Lakh" in c["text"] for c in table_chunks)
    assert found_price, "Price specifications lost during chunking!"

    # Check continuation chunk has headers propagated if sub-split
    sub_chunks = [c for c in chunks if c.get("is_sub_chunk") and c.get("sub_chunk_index", 0) > 0]
    if sub_chunks:
        for sc in sub_chunks:
            assert "Continued" in sc["text"] or "|" in sc["text"]


def test_lists_and_safety_instructions_preserved(extracted_docs):
    """Verify safety instructions and procedural steps remain intact."""
    safety_doc = next(d for d in extracted_docs if "Advice for Pregnant Women" in d["document_name"])
    chunks = chunk_document(safety_doc)

    combined_text = "\n".join(c["text"] for c in chunks)
    assert "Seat Belt" in combined_text
    assert "under the belly" in combined_text
    assert "Three-Point Seat Belt" in combined_text


# -------------------------------------------------------------------------
# 3. Tests for Oversized Section Splitting & Boundaries
# -------------------------------------------------------------------------

def test_oversized_sections_are_split(extracted_docs):
    """Verify sections exceeding target threshold are properly sub-split."""
    # Aventro Edge Sedan has a >3000 character pricing section
    edge_doc = next(d for d in extracted_docs if "Aventro Edge Sedan" in d["document_name"])
    chunks = chunk_document(edge_doc)

    # Find the sub-split chunks for Variants & Pricing
    pricing_chunks = [c for c in chunks if c["section_heading"] == "Variants & Pricing"]
    assert len(pricing_chunks) >= 2, "Expected oversized Variants & Pricing section to be sub-split"

    for pc in pricing_chunks:
        assert pc["character_count"] <= 2000, f"Sub-chunk exceeded target size: {pc['character_count']}"
        assert pc["is_sub_chunk"] is True


def test_no_meaningful_source_text_silently_lost(extracted_docs):
    """Verify that all alphanumeric tokens from original documents exist in the generated chunks."""
    for doc in extracted_docs:
        chunks = chunk_document(doc)
        orig_words = set(re.findall(r"[a-zA-Z0-9]+", doc["content"]))
        chunks_words = set(re.findall(r"[a-zA-Z0-9]+", " ".join(c["text"] for c in chunks)))

        missing = orig_words - chunks_words
        assert not missing, f"Document '{doc['document_name']}' lost words: {missing}"


# -------------------------------------------------------------------------
# 4. Tests for Output Artifact Integrity
# -------------------------------------------------------------------------

def test_generated_artifacts_exist_and_readable():
    """Verify that document_chunks.jsonl and chunking_report.md can be read back cleanly."""
    chunk_file = Path(CHUNKS_PATH)
    report_file = Path(REPORT_PATH)

    assert chunk_file.exists(), f"Missing {CHUNKS_PATH}"
    assert report_file.exists(), f"Missing {REPORT_PATH}"

    lines = chunk_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 459, f"Expected 459 chunks in {CHUNKS_PATH}, got {len(lines)}"

    # Parse every line as JSON
    parsed_chunks = [json.loads(line) for line in lines]
    assert len(parsed_chunks) == 459

    # Verify report contains key stats
    report_text = report_file.read_text(encoding="utf-8")
    assert "459" in report_text
    assert "Two-Stage" in report_text
    assert "100% Lexical Retention" in report_text


# -------------------------------------------------------------------------
# 5. Advanced Structural Integrity Checks (Beyond Bag-of-Words)
# -------------------------------------------------------------------------

def test_structural_table_row_integrity(extracted_docs, chunked_data):
    """Verify table data rows are 100% retained with matching column counts and zero corrupted rows."""
    chunks, _ = chunked_data

    # 1. Column count consistency in all chunks containing tables
    for c in chunks:
        lines = c["text"].splitlines()
        in_table = False
        expected_cols = None
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                cells = [cell.strip() for cell in stripped.split("|")[1:-1]]
                if not in_table:
                    in_table = True
                    expected_cols = len(cells)
                else:
                    is_divider = set(stripped.replace("|", "").replace("-", "").replace(":", "").strip()) == set()
                    if not is_divider:
                        assert len(cells) == expected_cols, (
                            f"Column mismatch in {c['chunk_id']}: expected {expected_cols}, got {len(cells)} on line: {stripped}"
                        )
            else:
                in_table = False

    # 2. Verify all unique source table rows exist in chunks
    for doc in extracted_docs:
        if "|" not in doc["content"]:
            continue
        doc_rows = [
            line.strip()
            for line in doc["content"].splitlines()
            if line.strip().startswith("|")
            and not line.strip().startswith("| ---")
            and not line.strip().startswith("|---")
        ]
        if len(doc_rows) <= 1:
            continue
        # Exclude table header row from check
        data_rows = doc_rows[1:]

        doc_chunks = [c for c in chunks if c["document_id"] == doc["doc_id"]]
        chunk_lines = set()
        for c in doc_chunks:
            chunk_lines.update(line.strip() for line in c["text"].splitlines())

        for row in data_rows:
            assert row in chunk_lines, f"Table row missing from chunks of {doc['document_name']}: {row}"


def test_contiguous_specifications_and_entities(chunked_data):
    """Verify that multi-word specifications and safety instructions are not fractured across boundaries."""
    chunks, _ = chunked_data
    all_chunks_text = [c["text"] for c in chunks]

    key_phrases = [
        "Ex-Showroom Price (Lakh",
        "Same as Petrol ZX, Diesel engine",
        "ADAS Level 2",
        "11-inch Infotainment",
        "Adaptive Cruise Control (ACC)",
        "3810 x 1680 x 1515 mm",
        "Three-Point Seat Belt",
        "under the belly",
        "+91 124 1234567",
        "1.5L Turbo"
    ]

    for phrase in key_phrases:
        found_as_contiguous_substring = any(phrase in text for text in all_chunks_text)
        assert found_as_contiguous_substring, f"Critical phrase broken or missing: '{phrase}'"


def test_html_tag_closure_and_validity(chunked_data):
    """Verify HTML block tags (<details>, <summary>) are balanced and not left orphaned at boundaries."""
    chunks, _ = chunked_data

    for c in chunks:
        text = c["text"]
        # Check no chunk ends with a dangling open <details> tag
        assert not text.rstrip().endswith("<details>"), f"Dangling <details> at end of {c['chunk_id']}"
        assert not text.rstrip().endswith("<summary>"), f"Dangling <summary> at end of {c['chunk_id']}"

        # If details tag is opened in the chunk, verify count of opening and closing details tags
        open_count = len(re.findall(r"<details>", text))
        close_count = len(re.findall(r"</details>", text))
        assert open_count == close_count, (
            f"Unbalanced <details> tags in {c['chunk_id']}: {open_count} open vs {close_count} close"
        )


def test_sequential_chunk_indices_per_document(chunked_data):
    """Verify that chunk_index is strictly contiguous (0, 1, ... N-1) for every document."""
    chunks, _ = chunked_data
    from collections import defaultdict
    doc_indices = defaultdict(list)

    for c in chunks:
        doc_indices[c["document_id"]].append(c["chunk_index"])

    for doc_id, indices in doc_indices.items():
        expected = list(range(len(indices)))
        assert indices == expected, f"Document {doc_id} has non-sequential chunk indices: {indices}"


def test_token_estimation_consistency_and_labeling(chunked_data):
    """Verify token estimation is labeled as an estimate and applied consistently."""
    chunks, stats = chunked_data

    assert "estimated_token_count" in chunks[0]
    for c in chunks:
        chars = c["character_count"]
        tokens = c["estimated_token_count"]
        assert tokens > 0
        # Ratio of tokens to characters in technical English is typically between 0.15 and 0.35
        ratio = tokens / chars
        assert 0.10 <= ratio <= 0.45, f"Suspicious token/char ratio ({ratio:.2f}) in {c['chunk_id']}"

