"""
Step 3 Pipeline Execution Script
Aventro Motors RAG Project - Phase 1

Coordinates:
1. Extraction, cleaning, and deduplication of 50 Markdown knowledge documents.
2. Parsing, recovery, and validation of evaluation questions from CSV.
3. Generation of JSONL artifacts and detailed Markdown reports.
"""

import json
import os
from pathlib import Path
from src.markdown_extractor import extract_all_documents
from src.csv_eval_parser import parse_evaluation_csv


def write_jsonl(records, filepath):
    """Write list of dictionaries to a JSONL file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def generate_extraction_report(doc_stats, duplicate_passages, documents, report_path):
    """Generate Markdown report for document extraction, cleaning, and deduplication."""
    lines = []
    lines.append("# Aventro Motors RAG Phase 1 - Markdown Extraction & Cleaning Report")
    lines.append("")
    lines.append("## 1. Overview & Summary Metrics")
    lines.append("")
    lines.append(f"- **Source Directory**: `Aventro Motors/md/`")
    lines.append(f"- **Total Markdown Files Scanned**: {doc_stats['total_files_scanned']}")
    lines.append(f"- **Successfully Processed Documents**: **{doc_stats['processed_documents']}**")
    lines.append(f"- **Empty Documents**: **{doc_stats['empty_documents']}**")
    lines.append(f"- **Documents with Tables Preserved**: {doc_stats['docs_with_tables']}")
    lines.append(f"- **Documents with Lists Preserved**: {doc_stats['docs_with_lists']}")
    lines.append(f"- **Total Raw Character Count**: {doc_stats['total_characters_raw']:,} characters")
    lines.append(f"- **Total Cleaned Character Count**: {doc_stats['total_characters_cleaned']:,} characters")
    cleaned_diff = doc_stats['total_characters_raw'] - doc_stats['total_characters_cleaned']
    lines.append(f"- **Whitespace / Redundancy Trimmed**: {cleaned_diff:,} characters")
    lines.append(f"- **Output Artifact**: `artifacts/extracted/documents.jsonl`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Text Cleaning & Formatting Preservation")
    lines.append("")
    lines.append("The text cleaning pipeline applies strictly non-destructive whitespace normalization:")
    lines.append("- **Preserved**: All technical numbers, dates, currency values (e.g., Lakhs, ₹), battery and engine specs, dimensions, tables, lists, and safety procedures.")
    lines.append("- **Cleaned**: Normalized Windows `CRLF` to `LF`, trimmed trailing line whitespace, collapsed 3+ consecutive blank lines to standard single blank lines, and converted non-breaking spaces (`\\u00a0`).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Duplicate Content Detection")
    lines.append("")
    lines.append(f"- **Unique Duplicate Passage Groups**: **{doc_stats['duplicate_passage_groups']}**")
    lines.append("- **Full Document Duplicates**: **0** (all 50 documents have distinct topics and unique hashes).")
    lines.append("")
    lines.append("### Identified Shared Passages Across Documents")
    lines.append("")
    for idx, dup in enumerate(duplicate_passages, 1):
        lines.append(f"#### Group {idx} (Found in {dup['occurrence_count']} documents)")
        lines.append(f"- **Shared in**: `{', '.join(dup['documents'])}`")
        lines.append(f"- **Sample Text**: *\"{dup['sample_text']}...\"*")
        lines.append(f"- **Resolution**: Preserved in each document. These represent standard corporate contact footers; no documents were merged or discarded.")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Document Inventory Table")
    lines.append("")
    lines.append("| Doc ID | Document Title | Section Headings | Tables | Lists | Cleaned Chars |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")
    for doc in documents:
        headings_count = len(doc['section_headings'])
        t_icon = "Yes" if doc['has_tables'] else "No"
        l_icon = "Yes" if doc['has_lists'] else "No"
        lines.append(f"| `{doc['doc_id']}` | {doc['title']} | {headings_count} | {t_icon} | {l_icon} | {doc['cleaned_char_count']:,} |")
    
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def generate_evaluation_report(csv_stats, valid_questions, recovered_questions, malformed_records, report_path):
    """Generate Markdown report for evaluation CSV data quality and recovery."""
    lines = []
    lines.append("# Aventro Motors RAG Phase 1 - Evaluation Data Quality & Validation Report")
    lines.append("")
    lines.append("## 1. Overview & Data Quality Audit")
    lines.append("")
    lines.append(f"- **Source CSV File**: `Aventro Motors/Aventro Motors.csv`")
    lines.append(f"- **Total Rows in CSV**: {csv_stats['total_csv_rows']}")
    lines.append(f"- **Fully Valid JSON Rows**: **{csv_stats['clean_json_rows']}**")
    lines.append(f"- **Safely Recovered Truncated Rows**: **{csv_stats['recovered_rows']}**")
    lines.append(f"- **Empty Response Rows**: **{csv_stats['empty_rows']}**")
    lines.append(f"- **Unrecoverable Truncated Rows**: **{csv_stats['unrecoverable_truncated_rows']}**")
    lines.append(f"- **Total Unusable / Unrecoverable Rows**: **{csv_stats['total_unrecoverable_rows']}**")
    lines.append("")
    lines.append("### Question Counts Breakdown")
    lines.append(f"- **Valid Questions (from clean JSON)**: **{csv_stats['valid_questions_count']}** (8 per valid row)")
    lines.append(f"- **Recovered Questions (from truncated rows)**: **{csv_stats['recovered_questions_count']}**")
    lines.append(f"- **Total Verified Usable Questions**: **{csv_stats['total_usable_questions']}**")
    lines.append(f"- **Output Artifact**: `artifacts/extracted/evaluation_questions.jsonl`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Evaluation Schema & Field Validation")
    lines.append("")
    lines.append("Every validated question in `evaluation_questions.jsonl` meets strict schema requirements:")
    lines.append("- `query_text`: Non-empty string containing the user query.")
    lines.append("- `expected_answer`: Non-empty ground-truth answer string.")
    lines.append("- `facts`: Non-empty array or string of supporting evidence.")
    lines.append("- `category`: Classification label (e.g., Specific Query, Summary Query).")
    lines.append("- `source_document`: Associated Markdown filename.")
    lines.append("- `origin_status`: Tagged as `valid_clean_json` or `recovered_from_truncated_json`.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Detailed Audit of Malformed and Truncated Records")
    lines.append("")
    lines.append("| Row Index | Source Document Reference | Category / Status | Details / Impact |")
    lines.append("| :---: | :--- | :--- | :--- |")
    
    for rec in malformed_records:
        rec_type = "Recovered Partial" if rec.get("recoverable") else "Unrecoverable"
        if rec.get("recoverable"):
            impact = f"Recovered {rec.get('items_recovered')} complete Q&A items; discarded incomplete tail"
        else:
            impact = "Zero complete Q&A pairs recovered; cannot fabricate missing expected answer"
        lines.append(f"| {rec['row_index']} | `{rec['document_name']}` | **{rec_type}** | {impact} |")
    
    lines.append("")
    lines.append("### Specific Root Causes for Unrecoverable Rows (4 rows total):")
    lines.append("1. **Row 0 (`Aventro Edge Sedan.md`)**: The `response` column is completely empty (`len = 0`).")
    lines.append("2. **Row 2 (` Aventro Motors- How to Change a Tyre.md`)**: The `response` column is completely empty (`len = 0`). Also has a leading whitespace in filename reference.")
    lines.append("3. **Row 30 (`Aventro Glory SUV.md`)**: JSON string was truncated during item 0's `facts` table before any `expected_answer` or item closure occurred.")
    lines.append("4. **Row 49 (`Aventro Bolt Sedan.md`)**: JSON string was truncated midway through item 0's `expected_answer` sentence.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Data Separation Notice")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append("> Evaluation questions and expected answers are strictly quarantined from the document knowledge base.")
    lines.append("> Under no circumstances are records from `Aventro Motors.csv` indexed as source documents in ChromaDB.")

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_pipeline():
    print("=" * 70)
    print("       AVENTRO MOTORS RAG - PHASE 1, STEP 3 PIPELINE")
    print("=" * 70)

    # 1. Process Markdown files
    md_dir = "Aventro Motors/md"
    print(f"[1/4] Extracting Markdown knowledge documents from '{md_dir}'...")
    documents, duplicate_passages, doc_stats = extract_all_documents(md_dir)
    print(f"      -> Processed documents : {doc_stats['processed_documents']}")
    print(f"      -> Empty documents     : {doc_stats['empty_documents']}")
    print(f"      -> Duplicate passages  : {doc_stats['duplicate_passage_groups']}")
    print(f"      -> Tables preserved    : {doc_stats['docs_with_tables']} documents")
    print(f"      -> Lists preserved     : {doc_stats['docs_with_lists']} documents")

    # 2. Process Evaluation CSV
    csv_path = "Aventro Motors/Aventro Motors.csv"
    print(f"\n[2/4] Parsing evaluation CSV data from '{csv_path}'...")
    valid_q, rec_q, malformed_rec, csv_stats = parse_evaluation_csv(csv_path)
    all_questions = valid_q + rec_q
    print(f"      -> Total CSV rows      : {csv_stats['total_csv_rows']}")
    print(f"      -> Valid clean rows    : {csv_stats['clean_json_rows']} ({len(valid_q)} questions)")
    print(f"      -> Recovered rows      : {csv_stats['recovered_rows']} ({len(rec_q)} questions)")
    print(f"      -> Unrecoverable rows  : {csv_stats['total_unrecoverable_rows']}")
    print(f"      -> Total usable Q&A    : {len(all_questions)} questions")

    # 3. Write Extracted Artifacts
    print("\n[3/4] Writing extracted JSONL artifacts...")
    doc_out = "artifacts/extracted/documents.jsonl"
    eval_out = "artifacts/extracted/evaluation_questions.jsonl"
    write_jsonl(documents, doc_out)
    write_jsonl(all_questions, eval_out)
    print(f"      -> Saved {len(documents)} document records to '{doc_out}'")
    print(f"      -> Saved {len(all_questions)} evaluation questions to '{eval_out}'")

    # 4. Generate Reports
    print("\n[4/4] Generating markdown audit reports...")
    rep_extract = "reports/extraction_report.md"
    rep_eval = "reports/evaluation_data_quality_report.md"
    generate_extraction_report(doc_stats, duplicate_passages, documents, rep_extract)
    generate_evaluation_report(csv_stats, valid_q, rec_q, malformed_rec, rep_eval)
    print(f"      -> Saved extraction report to '{rep_extract}'")
    print(f"      -> Saved evaluation report to '{rep_eval}'")

    print("\n" + "=" * 70)
    print("STEP 3 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
