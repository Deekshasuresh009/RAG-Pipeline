"""
Pytest Test Suite for Phase 1, Step 3
Tests Markdown extraction, metadata preservation, duplicate detection,
and evaluation CSV validation.
"""

import json
import re
from pathlib import Path
import pytest

from src.markdown_extractor import (
    extract_all_documents,
    clean_markdown_text,
    generate_doc_id,
    compute_sha256
)
from src.csv_eval_parser import parse_evaluation_csv, validate_item, attempt_json_repair


MD_DIR = "Aventro Motors/md"
CSV_PATH = "Aventro Motors/Aventro Motors.csv"


# -------------------------------------------------------------------------
# 1. Tests for Markdown Extraction & Structure Preservation
# -------------------------------------------------------------------------

def test_markdown_extraction_count_and_completeness():
    """Verify that all 50 Markdown documents are found, processed, and non-empty."""
    documents, duplicate_passages, stats = extract_all_documents(MD_DIR)

    assert stats["total_files_scanned"] == 50, f"Expected 50 files, got {stats['total_files_scanned']}"
    assert stats["processed_documents"] == 50, f"Expected 50 processed docs, got {stats['processed_documents']}"
    assert stats["empty_documents"] == 0, f"Expected 0 empty docs, got {stats['empty_documents']}"
    assert len(documents) == 50

    # Ensure every doc has a title and content
    for doc in documents:
        assert doc["title"] and doc["title"] != "Untitled Document", f"Doc {doc['document_name']} missing title"
        assert len(doc["content"]) > 100, f"Doc {doc['document_name']} content suspiciously short"
        assert len(doc["section_headings"]) > 0, f"Doc {doc['document_name']} has no section headings"


def test_markdown_metadata_preservation():
    """Verify stable document ID generation, paths, headings, and hashes."""
    documents, _, _ = extract_all_documents(MD_DIR)

    doc_ids = set()
    for doc in documents:
        # Stable ID format
        assert doc["doc_id"].startswith("doc_")
        assert doc["doc_id"] not in doc_ids, f"Duplicate doc_id detected: {doc['doc_id']}"
        doc_ids.add(doc["doc_id"])

        # Source path check
        assert "Aventro Motors/md" in doc["source_path"]

        # Hashes
        assert len(doc["content_hash_raw"]) == 64
        assert len(doc["content_hash_cleaned"]) == 64
        assert re.match(r"^[0-9a-f]{64}$", doc["content_hash_raw"])
        assert re.match(r"^[0-9a-f]{64}$", doc["content_hash_cleaned"])

        # Heading hierarchy
        for h in doc["heading_hierarchy"]:
            assert "level" in h and 1 <= h["level"] <= 6
            assert "text" in h and len(h["text"]) > 0


def test_whitespace_cleaning_integrity():
    """Verify that cleaning whitespace preserves numbers, dates, specs, and safety keywords."""
    sample_text = (
        "# Aventro Bolt Sedan\n\n\n"
        "Price: \u00a011.2 Lakh \u20b9\r\n"
        "Engine: 1.5L Turbo, 115 kW, 250 Nm.\r\n\r\n\r\n\r\n"
        "Features:\n"
        "- 11-inch Infotainment\n"
        "- ADAS Level 2 with ACC\n\n"
        "Safety Warning: Always fasten your seat belt before driving.   \n"
    )

    cleaned = clean_markdown_text(sample_text)

    # Check that excess blank lines are collapsed
    assert "\n\n\n" not in cleaned
    assert "\r" not in cleaned

    # Check that critical facts, numbers, and strings are preserved intact
    assert "11.2 Lakh" in cleaned
    assert "1.5L Turbo, 115 kW, 250 Nm." in cleaned
    assert "11-inch Infotainment" in cleaned
    assert "ADAS Level 2 with ACC" in cleaned
    assert "Always fasten your seat belt before driving." in cleaned


def test_tables_and_lists_preservation():
    """Verify that markdown tables and lists are detected and preserved."""
    documents, _, stats = extract_all_documents(MD_DIR)

    assert stats["docs_with_tables"] >= 20, "Expected at least 20 documents with tables"
    assert stats["docs_with_lists"] >= 45, "Expected at least 45 documents with lists"

    # Spot-check a document known to contain pricing tables
    pricing_doc = next(d for d in documents if "Variants & Pricing" in d["document_name"])
    assert pricing_doc["has_tables"] is True
    assert "|" in pricing_doc["content"]
    assert "Petrol" in pricing_doc["content"]


# -------------------------------------------------------------------------
# 2. Tests for Duplicate Content Detection
# -------------------------------------------------------------------------

def test_duplicate_content_detection():
    """Verify duplicate passage detection across documents without dropping data."""
    documents, duplicate_passages, stats = extract_all_documents(MD_DIR)

    # Ensure duplicate passages are detected (e.g. corporate contact footers)
    assert stats["duplicate_passage_groups"] >= 1
    assert len(duplicate_passages) == stats["duplicate_passage_groups"]

    for dup in duplicate_passages:
        assert dup["occurrence_count"] >= 2
        assert len(dup["documents"]) == dup["occurrence_count"]
        assert len(dup["sample_text"]) > 10

    # Ensure all 50 documents were kept despite duplicate passages
    assert len(documents) == 50


# -------------------------------------------------------------------------
# 3. Tests for Evaluation CSV Parsing and JSON Validation
# -------------------------------------------------------------------------

def test_evaluation_csv_parsing_and_counts():
    """Verify CSV parsing yields expected row and question counts."""
    valid_q, rec_q, malformed_rec, stats = parse_evaluation_csv(CSV_PATH)

    assert stats["total_csv_rows"] == 50
    assert stats["clean_json_rows"] == 31
    assert stats["recovered_rows"] == 15
    assert stats["empty_rows"] == 2
    assert stats["unrecoverable_truncated_rows"] == 2
    assert stats["total_unrecoverable_rows"] == 4

    assert stats["valid_questions_count"] == 248  # 31 * 8
    assert stats["recovered_questions_count"] == 66
    assert stats["total_usable_questions"] == 314


def test_evaluation_question_schema_integrity():
    """Verify that every accepted evaluation question has valid non-empty fields."""
    valid_q, rec_q, _, _ = parse_evaluation_csv(CSV_PATH)
    all_questions = valid_q + rec_q

    for q in all_questions:
        assert q["question_id"].startswith("eval_q_")
        assert q["source_document"].endswith(".md")
        assert q["query_text"] and len(q["query_text"].strip()) > 5
        assert q["expected_answer"] and len(q["expected_answer"].strip()) > 5
        assert q["facts"] is not None
        # Factual queries must have non-empty supporting facts; feedback queries have empty facts array
        if q["category"] != "Feedback Related Query":
            assert len(q["facts"]) > 0, f"Factual question {q['question_id']} has empty facts"
        assert q["origin_status"] in ["valid_clean_json", "recovered_from_truncated_json"]


def test_no_fabricated_data_in_unrecoverable_rows():
    """Ensure rows 0, 2, 30, and 49 do not produce fake questions."""
    valid_q, rec_q, malformed_rec, _ = parse_evaluation_csv(CSV_PATH)
    all_questions = valid_q + rec_q

    unrecoverable_rows = {0, 2, 30, 49}
    for q in all_questions:
        assert q["source_row"] not in unrecoverable_rows, (
            f"Row {q['source_row']} ({q['source_document']}) should have 0 questions, but found one!"
        )

    # Ensure these 4 rows are explicitly present in malformed_records with recoverable=False
    unrec_logged = [r for r in malformed_rec if not r.get("recoverable")]
    unrec_row_indices = {r["row_index"] for r in unrec_logged}
    assert unrec_row_indices == unrecoverable_rows


# -------------------------------------------------------------------------
# 4. Tests for Output Artifact Generation
# -------------------------------------------------------------------------

def test_generated_artifacts_integrity():
    """Verify generated JSONL files and report files exist and have correct record counts."""
    from src.run_step3 import run_pipeline

    # Run the pipeline
    run_pipeline()

    doc_file = Path("artifacts/extracted/documents.jsonl")
    eval_file = Path("artifacts/extracted/evaluation_questions.jsonl")
    extract_rep = Path("reports/extraction_report.md")
    eval_rep = Path("reports/evaluation_data_quality_report.md")

    assert doc_file.exists()
    assert eval_file.exists()
    assert extract_rep.exists()
    assert eval_rep.exists()

    # Check line counts in JSONL files
    doc_lines = [json.loads(line) for line in doc_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    eval_lines = [json.loads(line) for line in eval_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(doc_lines) == 50
    assert len(eval_lines) == 314

    # Verify report content
    assert "50" in extract_rep.read_text(encoding="utf-8")
    assert "314" in eval_rep.read_text(encoding="utf-8")
