# Aventro Motors RAG Phase 1 - Structure-Aware Chunking Report

## 1. Executive Summary

- **Source Documents Processed**: **50**
- **Total Generated Chunks**: **459**
- **Empty Chunks**: **0** (none detected)
- **Oversized Chunks (>525 tokens)**: **0** (none detected)
- **Average Chunks per Document**: **9.2** (Min: 1, Max: 23)
- **Output Artifact**: `artifacts/chunks/document_chunks.jsonl`

---

## 2. Chunk Size Distribution & Statistics

| Metric | Character Count | Estimated Token Count |
| :--- | :---: | :---: |
| **Minimum** | 24 | 6 |
| **Maximum** | 1984 | 412 |
| **Average (Mean)** | 375.4 | 82.8 |
| **Median** | 274 | 63 |

> [!NOTE]
> Token counts are clearly labeled as **estimated token counts** calculated using a calibrated local heuristic
> `(characters / 4.0 + words * 1.33) / 2.0`. No external tokenizer or paid LLM API was called.

---

## 3. Chunking Strategy & Boundary Handling

### Two-Stage Structure-Aware Chunking Strategy
1. **Stage 1: Markdown Semantic Splitting**
   - Natural boundary detection using `MarkdownHeaderTextSplitter` across heading levels (`#`, `##`, `###`, `####`).
   - When a semantic section fits within the target size (~500 tokens / 2,000 characters), it is kept completely intact.
   - Preserves section headings, lists, tables, vehicle technical specifications, and procedural steps as coherent semantic units.

2. **Stage 2: Recursive Boundary Sub-Splitting for Oversized Sections**
   - For sections exceeding ~500 tokens, recursive character splitting is applied with a target of ~500 tokens and an overlap of ~75 tokens (~300 characters).
   - Separators prioritized: `['\n\n', '\n', '. ', ' ']` to ensure table rows and numbered steps stay intact.
   - **Heading Context Propagation**: Continuation sub-chunks retain the section heading (`## Heading (Continued)`).
   - **Table Header Propagation**: Sub-chunks splitting large markdown tables retain the column headers and divider rows.

---

## 4. Special Handling: Oversized Sections and Small Stems

### Oversized Sections Sub-Split (7 sections total):

- **`Aventro Bolt Sedan.md`** (`Available Variants & Prices`): Sub-chunk `doc_aventro_bolt_sedan_chunk_002` (1904 chars)
- **`Aventro Bolt Sedan.md`** (`Available Variants & Prices`): Sub-chunk `doc_aventro_bolt_sedan_chunk_003` (828 chars)
- **`Aventro Edge Sedan.md`** (`Variants & Pricing`): Sub-chunk `doc_aventro_edge_sedan_chunk_003` (1905 chars)
- **`Aventro Edge Sedan.md`** (`Variants & Pricing`): Sub-chunk `doc_aventro_edge_sedan_chunk_004` (1786 chars)
- **`Aventro Spark Hatchback.md`** (`Variants & Pricing`): Sub-chunk `doc_aventro_spark_hatchback_chunk_004` (1883 chars)
- **`Aventro Spark Hatchback.md`** (`Variants & Pricing`): Sub-chunk `doc_aventro_spark_hatchback_chunk_005` (1054 chars)
- **`Aventro Storm SUV.md`** (`Storm Variants & Pricing`): Sub-chunk `doc_aventro_storm_suv_chunk_003` (1901 chars)
- **`Aventro Storm SUV.md`** (`Storm Variants & Pricing`): Sub-chunk `doc_aventro_storm_suv_chunk_004` (765 chars)
- **`Parts-5.md`** (`Parts List with Industry Standard Prices`): Sub-chunk `doc_parts_5_chunk_001` (1984 chars)
- **`Parts-5.md`** (`Parts List with Industry Standard Prices`): Sub-chunk `doc_parts_5_chunk_002` (1286 chars)
- **`Parts-6.md`** (`Parts Details and Prices (Industry Standard Estimate)`): Sub-chunk `doc_parts_6_chunk_001` (1927 chars)
- **`Parts-6.md`** (`Parts Details and Prices (Industry Standard Estimate)`): Sub-chunk `doc_parts_6_chunk_002` (1346 chars)
- **`Pre-Owned Certified Cars.md`** (`**Aventro 380+ Checkpoints: The Gold Standard in Assurance**`): Sub-chunk `doc_pre_owned_certified_cars_chunk_002` (1930 chars)
- **`Pre-Owned Certified Cars.md`** (`**Aventro 380+ Checkpoints: The Gold Standard in Assurance**`): Sub-chunk `doc_pre_owned_certified_cars_chunk_003` (817 chars)

### Short Semantic Stems Retained (No Content Dropped):
- **`Aventro Motors Variants & Pricing.md`** (`(Other Models Continue Similarly…)`): *"## (Other Models Continue Similarly…)   ---"*
- **`Test Drive at Your Doorstep.md`** (`Example Email Request`): *"## Example Email Request"*

---

## 5. Source Text Retention Evidence

- **100% Lexical Retention**: Across all 50 documents, every alphanumeric word from the extracted source documents is accounted for in the generated chunks.
- Zero documents lost content during heading decomposition and chunking.

---

## 6. Structural Retention & Quality Audit

### Beyond Lexical Retention: Structural Integrity Checks
1. **Markdown Table Integrity**: All data rows across all 23 documents containing tables were verified. Zero rows were truncated, dropped, or split into invalid column counts.
2. **Contiguous Specification Preservation**: Critical technical entities (e.g. `11.2 Lakh`, `ADAS Level 2`, `11-inch Infotainment`, `1.5L Turbo, 115 kW`) exist as unbroken contiguous phrases within single chunks.
3. **HTML & Block Closure**: Block elements such as `<details>` and `<summary>` are kept cohesive with zero dangling open tags.
4. **List & Safety Cohesion**: Step-by-step procedures and safety guidelines retain their numbering and context.

### Limitations of Automated Retention Metrics
- **Bag-of-Words Limitations**: Pure lexical set matching detects missing words, but fails to detect corrupted word order, fragmented phrases, or disconnected table relationships (e.g., if a number is separated from its unit across a chunk boundary).
- **Duplication Blindness**: Lexical set checks do not detect accidental runaway duplication or bloat caused by excessive overlap.
- **Semantic Continuity**: Structural and automated checks cannot evaluate cross-chunk coreference (e.g., resolving pronouns referring to an entity mentioned in a previous chunk).

---

## 7. Chunks Generated per Document (Sample)

| Document Name | Chunks Count |
| :--- | :---: |
| `About Aventro Motors.md` | **6** |
| `Adaptive Cruise Control (ACC).md` | **11** |
| `Adjusting the Seat Positions.md` | **9** |
| `Adjusting the Steering Wheel.md` | **12** |
| `Advice for Pregnant Women While in a Car.md` | **10** |
| `Aventro  Awards & Recognitions.md` | **7** |
| `Aventro  Connectivity.md` | **19** |
| `Aventro Aero.md` | **11** |
| `Aventro Bolt Sedan.md` | **9** |
| `Aventro Edge Sedan.md` | **8** |
| `Aventro Glory SUV.md` | **7** |
| `Aventro Grand SUV.md` | **10** |
| `Aventro Motors Variants & Pricing.md` | **14** |
| `Aventro Motors- How to Change a Tyre.md` | **16** |
| `Aventro Prime Sedan.md` | **9** |
| `Aventro Pulse Hatchback.md` | **10** |
| `Aventro Service Center Locations.md` | **21** |
| `Aventro Spark Hatchback.md` | **9** |
| `Aventro Storm SUV.md` | **10** |
| `Aventro Swift Sedan.md` | **7** |
| *(and 30 other documents)* | ... |

