"""
Evaluation CSV Parser and Validation Module
Aventro Motors RAG Project - Phase 1, Step 3

Parses Aventro Motors.csv (columns: document_name, response).
Safely validates JSON, attempts conservative recovery on truncated JSON without fabrication,
and isolates valid questions from malformed/unrecoverable records.
"""

import csv
import json
import re
from pathlib import Path


def validate_item(item: dict) -> bool:
    """Validate that an evaluation item has all required non-empty fields."""
    if not isinstance(item, dict):
        return False
    
    query = item.get("query_text")
    answer = item.get("expected_answer")
    facts = item.get("facts")

    # query_text and expected_answer must be non-empty strings
    if not query or not isinstance(query, str) or not query.strip():
        return False
    if not answer or not isinstance(answer, str) or not answer.strip():
        return False
    
    # facts must exist (can be list or non-empty string; empty list is valid for feedback queries)
    if facts is None:
        return False
    if not isinstance(facts, (list, str)):
        return False
    if isinstance(facts, str) and not facts.strip():
        return False

    return True


def attempt_json_repair(raw_json: str):
    """
    Attempt safe recovery of truncated JSON array of items without fabricating any data.
    Finds the last valid closing item bracket '}' and appends array and object closing ']}'.
    Returns list of validated items, or None if no valid items could be cleanly recovered.
    """
    text = raw_json.strip()
    # Search backwards from the end for candidate item endings
    for pos in range(len(text), 0, -1):
        sub = text[:pos].strip()
        if sub.endswith(","):
            sub = sub[:-1].strip()
        if sub.endswith("}"):
            candidate = sub + "\n]}"
            try:
                data = json.loads(candidate)
                items = data.get("items", [])
                valid_items = [it for it in items if validate_item(it)]
                if valid_items:
                    return valid_items
            except Exception:
                continue
    return None


def parse_evaluation_csv(csv_path: str):
    """
    Parse Aventro Motors.csv.
    Categorizes records into:
        - valid_questions: list of validated evaluation question records
        - recovered_questions: list of recovered evaluation question records from truncated rows
        - malformed_records: list of rows that were unrecoverable or empty
        - stats: summary audit metrics
    """
    path = Path(csv_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    valid_questions = []
    recovered_questions = []
    malformed_records = []
    
    row_count = 0
    clean_json_rows = 0
    recovered_rows = 0
    empty_rows = 0
    unrecoverable_truncated_rows = 0

    with open(path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            row_count += 1
            doc_name = row.get("document_name", "").strip()
            raw_response = row.get("response", "").strip()

            if not raw_response:
                empty_rows += 1
                malformed_records.append({
                    "row_index": idx,
                    "document_name": doc_name,
                    "issue": "Empty response field in CSV",
                    "raw_length": 0,
                    "recoverable": False
                })
                continue

            # 1. Attempt standard clean JSON parsing
            parsed_clean = False
            try:
                data = json.loads(raw_response)
                items = data.get("items", [])
                clean_items = [it for it in items if validate_item(it)]
                if clean_items:
                    clean_json_rows += 1
                    parsed_clean = True
                    for it_idx, it in enumerate(clean_items):
                        q_record = {
                            "question_id": f"eval_q_{row_count:03d}_{it_idx:02d}",
                            "source_document": doc_name,
                            "source_row": idx,
                            "origin_status": "valid_clean_json",
                            "category": it.get("category", "General Query"),
                            "query_text": it["query_text"].strip(),
                            "expected_answer": it["expected_answer"].strip(),
                            "facts": it["facts"]
                        }
                        valid_questions.append(q_record)
            except json.JSONDecodeError:
                parsed_clean = False

            if parsed_clean:
                continue

            # 2. Attempt conservative recovery on truncated JSON
            recovered_items = attempt_json_repair(raw_response)
            if recovered_items:
                recovered_rows += 1
                for it_idx, it in enumerate(recovered_items):
                    q_record = {
                        "question_id": f"eval_q_{row_count:03d}_{it_idx:02d}_rec",
                        "source_document": doc_name,
                        "source_row": idx,
                        "origin_status": "recovered_from_truncated_json",
                        "category": it.get("category", "General Query"),
                        "query_text": it["query_text"].strip(),
                        "expected_answer": it["expected_answer"].strip(),
                        "facts": it["facts"]
                    }
                    recovered_questions.append(q_record)
                
                # Note that trailing partial text was pruned
                malformed_records.append({
                    "row_index": idx,
                    "document_name": doc_name,
                    "issue": "Truncated JSON payload; recovered complete items, pruned trailing incomplete fragment",
                    "raw_length": len(raw_response),
                    "items_recovered": len(recovered_items),
                    "recoverable": True
                })
            else:
                unrecoverable_truncated_rows += 1
                malformed_records.append({
                    "row_index": idx,
                    "document_name": doc_name,
                    "issue": "Truncated before any complete item was serialized; zero complete items recoverable",
                    "raw_length": len(raw_response),
                    "recoverable": False
                })

    stats = {
        "total_csv_rows": row_count,
        "clean_json_rows": clean_json_rows,
        "recovered_rows": recovered_rows,
        "empty_rows": empty_rows,
        "unrecoverable_truncated_rows": unrecoverable_truncated_rows,
        "total_unrecoverable_rows": empty_rows + unrecoverable_truncated_rows,
        "valid_questions_count": len(valid_questions),
        "recovered_questions_count": len(recovered_questions),
        "total_usable_questions": len(valid_questions) + len(recovered_questions)
    }

    return valid_questions, recovered_questions, malformed_records, stats
