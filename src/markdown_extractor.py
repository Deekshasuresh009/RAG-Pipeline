"""
Markdown Extraction and Text Cleaning Module
Aventro Motors RAG Project - Phase 1, Step 3

Extracts titles, headings, tables, lists, and content from all 50 Markdown files.
Performs whitespace normalization without modifying facts, numbers, specs, or safety instructions.
Computes stable IDs, SHA-256 hashes, and detects passage-level duplicate content.
"""

import os
import re
import hashlib
from pathlib import Path
from collections import defaultdict


def generate_doc_id(filename: str) -> str:
    """Generate a stable, sanitized document ID from filename."""
    stem = Path(filename).stem.strip()
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', stem).strip('_').lower()
    return f"doc_{slug}"


def compute_sha256(text: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_markdown_text(raw_text: str) -> str:
    """
    Clean whitespace and redundant line breaks without removing any facts,
    numbers, dates, tables, lists, specifications, or safety instructions.
    """
    # Normalize Windows CRLF to standard LF
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Replace non-breaking spaces with standard space
    text = text.replace("\u00a0", " ")

    # Strip trailing whitespace on each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Collapse 3 or more consecutive newlines into 2 (one blank line)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def extract_markdown_metadata(raw_text: str):
    """
    Extract title, section headings, table presence, and list presence from markdown.
    """
    lines = raw_text.splitlines()
    
    # 1. Document Title: Look for top-level H1 heading first, else first heading found
    title = None
    for line in lines:
        h1_match = re.match(r'^#\s+(.+)$', line.strip())
        if h1_match:
            title = h1_match.group(1).strip()
            break
            
    if not title:
        for line in lines:
            h_match = re.match(r'^#{1,6}\s+(.+)$', line.strip())
            if h_match:
                title = h_match.group(1).strip()
                break

    if not title:
        title = "Untitled Document"

    # 2. Section Headings (H2, H3, etc.)
    section_headings = []
    for line in lines:
        h_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if h_match:
            heading_text = h_match.group(2).strip()
            level = len(h_match.group(1))
            section_headings.append({
                "level": level,
                "text": heading_text
            })

    # 3. Detection of tables and lists
    has_tables = bool(re.search(r'^\s*\|.*?\|.*?\|\s*$', raw_text, re.MULTILINE))
    has_lists = bool(re.search(r'^\s*[-*+]\s+', raw_text, re.MULTILINE) or 
                     re.search(r'^\s*\d+\.\s+', raw_text, re.MULTILINE))

    return {
        "title": title,
        "section_headings": section_headings,
        "heading_names": [h["text"] for h in section_headings],
        "has_tables": has_tables,
        "has_lists": has_lists
    }


def extract_all_documents(md_dir: str):
    """
    Process all Markdown files in the directory.
    Returns:
        documents: list of extracted document dictionaries
        duplicate_passages: list of duplicate passages found across documents
        stats: dictionary of processing statistics
    """
    dir_path = Path(md_dir).resolve()
    if not dir_path.exists():
        raise FileNotFoundError(f"Markdown directory not found: {dir_path}")

    md_files = sorted([f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() == ".md"])

    documents = []
    passage_index = defaultdict(list)
    empty_docs = []

    for file_path in md_files:
        raw_text = file_path.read_text(encoding="utf-8")
        if not raw_text.strip():
            empty_docs.append(file_path.name)
            continue

        raw_hash = compute_sha256(raw_text)
        cleaned_text = clean_markdown_text(raw_text)
        cleaned_hash = compute_sha256(cleaned_text)
        meta = extract_markdown_metadata(cleaned_text)

        # Relative source path from workspace root
        try:
            rel_source_path = str(file_path.relative_to(Path(".").resolve())).replace("\\", "/")
        except ValueError:
            rel_source_path = f"Aventro Motors/md/{file_path.name}"

        doc_record = {
            "doc_id": generate_doc_id(file_path.name),
            "document_name": file_path.name,
            "source_path": rel_source_path,
            "title": meta["title"],
            "section_headings": meta["heading_names"],
            "heading_hierarchy": meta["section_headings"],
            "content_hash_raw": raw_hash,
            "content_hash_cleaned": cleaned_hash,
            "raw_char_count": len(raw_text),
            "cleaned_char_count": len(cleaned_text),
            "raw_line_count": len(raw_text.splitlines()),
            "cleaned_line_count": len(cleaned_text.splitlines()),
            "has_tables": meta["has_tables"],
            "has_lists": meta["has_lists"],
            "content": cleaned_text
        }
        documents.append(doc_record)

        # Index passages for duplicate detection (min length 40 chars)
        paragraphs = [p.strip() for p in cleaned_text.split("\n\n") if len(p.strip()) >= 40]
        for p in paragraphs:
            norm_p = " ".join(p.split())
            p_hash = hashlib.sha256(norm_p.encode("utf-8")).hexdigest()
            passage_index[p_hash].append({
                "doc_name": file_path.name,
                "doc_id": doc_record["doc_id"],
                "passage_sample": norm_p[:120]
            })

    # Find duplicates (passages appearing in 2 or more distinct documents)
    duplicate_passages = []
    for p_hash, occurrences in passage_index.items():
        doc_names = list(dict.fromkeys([occ["doc_name"] for occ in occurrences]))
        if len(doc_names) > 1:
            duplicate_passages.append({
                "hash": p_hash,
                "occurrence_count": len(doc_names),
                "documents": doc_names,
                "sample_text": occurrences[0]["passage_sample"]
            })

    stats = {
        "total_files_scanned": len(md_files),
        "processed_documents": len(documents),
        "empty_documents": len(empty_docs),
        "empty_document_names": empty_docs,
        "duplicate_passage_groups": len(duplicate_passages),
        "total_characters_raw": sum(d["raw_char_count"] for d in documents),
        "total_characters_cleaned": sum(d["cleaned_char_count"] for d in documents),
        "docs_with_tables": sum(1 for d in documents if d["has_tables"]),
        "docs_with_lists": sum(1 for d in documents if d["has_lists"])
    }

    return documents, duplicate_passages, stats
