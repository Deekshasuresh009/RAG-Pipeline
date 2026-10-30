# Aventro Motors RAG Phase 1 - Evaluation Data Quality & Validation Report

## 1. Overview & Data Quality Audit

- **Source CSV File**: `Aventro Motors/Aventro Motors.csv`
- **Total Rows in CSV**: 50
- **Fully Valid JSON Rows**: **31**
- **Safely Recovered Truncated Rows**: **15**
- **Empty Response Rows**: **2**
- **Unrecoverable Truncated Rows**: **2**
- **Total Unusable / Unrecoverable Rows**: **4**

### Question Counts Breakdown
- **Valid Questions (from clean JSON)**: **248** (8 per valid row)
- **Recovered Questions (from truncated rows)**: **66**
- **Total Verified Usable Questions**: **314**
- **Output Artifact**: `artifacts/extracted/evaluation_questions.jsonl`

---

## 2. Evaluation Schema & Field Validation

Every validated question in `evaluation_questions.jsonl` meets strict schema requirements:
- `query_text`: Non-empty string containing the user query.
- `expected_answer`: Non-empty ground-truth answer string.
- `facts`: Non-empty array or string of supporting evidence.
- `category`: Classification label (e.g., Specific Query, Summary Query).
- `source_document`: Associated Markdown filename.
- `origin_status`: Tagged as `valid_clean_json` or `recovered_from_truncated_json`.

---

## 3. Detailed Audit of Malformed and Truncated Records

| Row Index | Source Document Reference | Category / Status | Details / Impact |
| :---: | :--- | :--- | :--- |
| 0 | `Aventro Edge Sedan.md` | **Unrecoverable** | Zero complete Q&A pairs recovered; cannot fabricate missing expected answer |
| 2 | `Aventro Motors- How to Change a Tyre.md` | **Unrecoverable** | Zero complete Q&A pairs recovered; cannot fabricate missing expected answer |
| 6 | `Sell Your Car in 3 Easy Steps.md` | **Recovered Partial** | Recovered 7 complete Q&A items; discarded incomplete tail |
| 9 | `How to Handle Engine Wont Start Issue.md` | **Recovered Partial** | Recovered 2 complete Q&A items; discarded incomplete tail |
| 13 | `Aventro  Connectivity.md` | **Recovered Partial** | Recovered 7 complete Q&A items; discarded incomplete tail |
| 15 | `Best Ways to Use Your headlamps.md` | **Recovered Partial** | Recovered 4 complete Q&A items; discarded incomplete tail |
| 19 | `Parta-1.md` | **Recovered Partial** | Recovered 7 complete Q&A items; discarded incomplete tail |
| 23 | `Aventro Motors Variants & Pricing.md` | **Recovered Partial** | Recovered 1 complete Q&A items; discarded incomplete tail |
| 26 | `Aventro Service Center Locations.md` | **Recovered Partial** | Recovered 5 complete Q&A items; discarded incomplete tail |
| 27 | `Aventro Aero.md` | **Recovered Partial** | Recovered 4 complete Q&A items; discarded incomplete tail |
| 30 | `Aventro Glory SUV.md` | **Unrecoverable** | Zero complete Q&A pairs recovered; cannot fabricate missing expected answer |
| 31 | `What is ABS.md` | **Recovered Partial** | Recovered 6 complete Q&A items; discarded incomplete tail |
| 34 | `Pre-Owned Certified Cars.md` | **Recovered Partial** | Recovered 1 complete Q&A items; discarded incomplete tail |
| 36 | `Aventro Swift Sedan.md` | **Recovered Partial** | Recovered 4 complete Q&A items; discarded incomplete tail |
| 38 | `Parts-6.md` | **Recovered Partial** | Recovered 7 complete Q&A items; discarded incomplete tail |
| 39 | `How to Tow Your Aventro Car.md` | **Recovered Partial** | Recovered 2 complete Q&A items; discarded incomplete tail |
| 40 | `How to Start Your Aventro Car.md` | **Recovered Partial** | Recovered 5 complete Q&A items; discarded incomplete tail |
| 42 | `Parts-2.md` | **Recovered Partial** | Recovered 4 complete Q&A items; discarded incomplete tail |
| 49 | `Aventro Bolt Sedan.md` | **Unrecoverable** | Zero complete Q&A pairs recovered; cannot fabricate missing expected answer |

### Specific Root Causes for Unrecoverable Rows (4 rows total):
1. **Row 0 (`Aventro Edge Sedan.md`)**: The `response` column is completely empty (`len = 0`).
2. **Row 2 (` Aventro Motors- How to Change a Tyre.md`)**: The `response` column is completely empty (`len = 0`). Also has a leading whitespace in filename reference.
3. **Row 30 (`Aventro Glory SUV.md`)**: JSON string was truncated during item 0's `facts` table before any `expected_answer` or item closure occurred.
4. **Row 49 (`Aventro Bolt Sedan.md`)**: JSON string was truncated midway through item 0's `expected_answer` sentence.

---

## 4. Data Separation Notice

> [!IMPORTANT]
> Evaluation questions and expected answers are strictly quarantined from the document knowledge base.
> Under no circumstances are records from `Aventro Motors.csv` indexed as source documents in ChromaDB.
