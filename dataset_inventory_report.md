# Aventro Motors Dataset Inventory & Inspection Report

**Project**: Aventro Motors RAG - Phase 1  
**Task**: Dataset Inspection, Verification, and Inventory  

---

## 1. Executive Summary

- **Workspace Path**: `C:\Users\devi.c\RAG Phase1`
- **Dataset Location**: `Aventro Motors/`
- **ZIP Archive Check**: None detected (dataset is fully extracted)
- **Total Files in Dataset**: 253
- **Core Knowledge Documents**: 50 unique topics (available identically across 5 document formats)
- **Evaluation Dataset**: `Aventro Motors.csv` (contains 50 rows referencing `.md` files, with evaluation Q&A pairs embedded as JSON)

---

## 2. Folder Structure and File Counts by Format

| Directory / Format | File Extension | File Count | Description |
| :--- | :--- | :--- | :--- |
| `.docx` | `.docx` | **50** | Microsoft Word documents (50 distinct topics) |
| `.html` | `.html` | **50** | HTML formatted documents (50 distinct topics) |
| `.md` | `.md` | **50** | Clean Markdown source documents (50 distinct topics) |
| `.pdf` | `.pdf` | **50** | PDF document files (50 distinct topics) |
| `.pptx` | `.pptx` | **50** | PowerPoint presentation files (50 distinct topics) |
| `(no extension)` | `(no extension)` | **2** | macOS system metadata files (`.DS_Store`) |
| `.csv` | `.csv` | **1** | Evaluation ground truth dataset (`Aventro Motors.csv`) |
| **Total** | | **253** | |

### Directory Layout
```
Aventro Motors/
├── Aventro Motors.csv            [Evaluation dataset: 50 rows]
├── .DS_Store                     [OS metadata file]
├── docx/                         [50 .docx files]
├── html/                         [50 .html files]
├── md/                           [50 .md files + 1 .DS_Store file]
├── pdf/                          [50 .pdf files]
└── pptx/                         [50 .pptx files]
```

---

## 3. Deduplication & Cross-Format Redundancy Analysis

### A. Exact SHA-256 Bit-for-Bit Duplicates
- **Result**: **0 exact SHA-256 hash collisions** detected across all 253 files.
- Every individual file has a unique binary SHA-256 checksum.

### B. Cross-Format Redundancy (Identical Topic Duplication)
- **50 Core Topics** exist in the dataset.
- **100% format overlap**: Every single one of the 50 topics is duplicated across all 5 format folders (`docx`, `html`, `md`, `pdf`, `pptx`).
- Total document files: 250 files across 5 formats representing only 50 distinct knowledge sources.

### C. Size Analysis
| Format | Min Size (bytes) | Max Size (bytes) | Avg Size (bytes) | Total Size (bytes) |
| :--- | :--- | :--- | :--- | :--- |
| `.md` | 544 | 6,488 | 3,417 | 170,892 |
| `.html` | 679 | 10,762 | 4,530 | 226,530 |
| `.docx` | 10,091 | 12,974 | 11,956 | 597,810 |
| `.pdf` | 13,700 | 33,867 | 25,929 | 1,296,454 |
| `.pptx` | 27,887 | 48,135 | 35,633 | 1,781,696 |
| `.csv` | 257,424 | 257,424 | 257,424 | 257,424 |

---

## 4. Evaluation Dataset Analysis (`Aventro Motors.csv`)

- **CSV Headers**: `document_name, response`
- **Total Rows**: 50
- **Document Mapping**: All 50/50 rows reference `.md` files matching `Aventro Motors/md/` exactly.

### Identified Evaluation Columns & Fields
| Logical Role | Source Column / JSON Key | Type | Description |
| :--- | :--- | :--- | :--- |
| **Document Reference** | `document_name` (CSV column) | String | Target document filename (e.g. `Adaptive Cruise Control (ACC).md`) |
| **Evaluation Question** | `response` -> `items[].query_text` | String | User question / evaluation prompt |
| **Expected Answer** | `response` -> `items[].expected_answer` | String | Ground-truth reference answer |
| **Supporting Facts** | `response` -> `items[].facts` | Array of strings | Key factual sentences extracted from the document |
| **Query Category** | `response` -> `items[].category` | String | Question taxonomy (e.g., 'Specific Query', 'Summary Query') |

### Evaluation Data Integrity Audit
- **Valid Complete JSON Rows**: 31 rows (yielding **248** cleanly parsed Q&A pairs)
- **Truncated / Malformed JSON Rows**: 17 rows (the JSON payload was truncated before closing braces)
- **Empty Response Rows**: 2 rows (`Aventro Edge Sedan.md`, ` Aventro Motors- How to Change a Tyre.md`)
- **Total Regex Extractable Queries**: **327 questions**

### Sample Q&A Record
```json
{
  "document": "Adaptive Cruise Control (ACC).md",
  "item": {
    "category": "Specific Query",
    "query_text": "What exactly is Adaptive Cruise Control and how does it work in terms of detecting vehicles on the road?",
    "facts": [
      "Adaptive Cruise Control (ACC) is an advanced driver assistance system (ADAS) designed to automatically adjust a vehicle\u2019s cruising speed to maintain a safe following distance from vehicles ahead.",
      "Adaptive Cruise Control: Uses sensors to detect vehicles ahead and can increase, decrease, or maintain speed based on traffic conditions.",
      "Constantly monitors the distance to the vehicle in front and adjusts speed to maintain a safe, pre-set gap."
    ],
    "expected_answer": "Adaptive Cruise Control (ACC) is an advanced driver assistance system (ADAS) that uses sensors to detect vehicles ahead, automatically adjusting the vehicle\u2019s speed to maintain a safe, pre-set gap."
  }
}
```

---

## 5. Smallest Set of Files Needed for Phase 1 RAG Pipeline

Based on the inspection, ingesting all 250 files across all 5 formats would produce **400% content redundancy**, severe vector database clutter, and skewed retrieval results, as each topic is represented 5 times in different file wrappers.

**Proposed Minimal Canonical Dataset (51 files total):**
1. **50 Markdown files (`Aventro Motors/md/*.md`)**: The Markdown documents provide clean, structured, human-readable text with headings and markdown tables, without parsing overhead or encoding artifacts.
2. **1 Evaluation CSV (`Aventro Motors/Aventro Motors.csv`)**: Required for evaluation questions, expected answers, and ground-truth validation.

**Files to Exclude from Vector Indexing:**
- `docx/` (50 files): Redundant binary format of the same 50 markdown files.
- `html/` (50 files): Redundant HTML conversion containing noisy HTML tags and entity artifacts.
- `pdf/` (50 files): Redundant PDF exports.
- `pptx/` (50 files): Redundant presentation exports.
- `.DS_Store` (2 files): macOS system metadata files (should be ignored).