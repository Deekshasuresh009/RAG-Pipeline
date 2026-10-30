# Aventro Motors RAG — Phase 2, Step 10: Evidence Relevance and Sufficiency Checking Report

## 1. Overview & Objective

The objective of Step 10 is to implement a standalone evidence evaluation component for the Aventro Motors RAG system. Positioned directly downstream of the Step 9 Reciprocal Rank Fusion (RRF) hybrid retriever, the **Evidence Evaluator** determines:
1. Whether individual retrieved chunks are relevant to the user query.
2. Whether the retrieved chunks collectively contain sufficient, non-conflicting information to support a reliable, grounded answer.
3. Explicit machine-readable abstention decisions (`is_sufficient: False` with distinct statuses `partially_sufficient`, `insufficient`, `conflicting`, or `no_evidence`), preventing downstream generation components from hallucinating answers when evidence is lacking.

---

## 2. Architecture & Design Principles

### Core Pipeline Position
```
User Query
    │
    ▼
[BM25 Retriever (Step 7)]   +   [ChromaDB Vector Retriever (Step 8)]
    │                                     │
    └──────────────────┬──────────────────┘
                       ▼
          [RRF Fusion (Step 9)]
                       │ (Ranked Candidates)
                       ▼
       [Evidence Evaluator (Step 10)]
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
[Relevance Filter]            [Sufficiency & Conflict Analysis]
       │                               │
       └───────────────┬───────────────┘
                       ▼
     Structured Assessment Report
     - status ('sufficient' | 'partially_sufficient' | 'insufficient' | 'conflicting' | 'no_evidence')
     - is_sufficient (bool)
     - reason (str)
     - relevant_evidence (list)
     - irrelevant_evidence (list)
     - missing_information (list)
     - conflicts (list)
```

### Key Principles Followed
1. **Never Equate Retrieval with Sufficiency**: A chunk having high semantic similarity or top RRF rank does not prove that the user's specific factual question is answerable.
2. **Collective Sufficiency**: A multi-part question requires all sub-aspects to be covered. Evaluating chunks individually would miss multi-chunk synthesis or fail to recognize that only half the question was answered.
3. **Stop Word Immunity**: Common conversational filler ("what", "can", "you", "tell", "me", "about") is filtered out before computing overlap ratios.
4. **Zero Knowledge Modification**: Original chunk text, source filenames, section headings, and citation metadata are preserved strictly without modification or truncation.
5. **No Evaluation Leakage**: Strict security guard verifies that no chunk contains evaluation metadata (`query_text` or `expected_answer`).

---

## 3. Heuristic Relevance and Sufficiency Rules

### A. Salient Term Extraction & Relevance Heuristic
- **Normalization**: Text is converted to lowercase; numbers with decimals (`380`, `42.5`) and alphanumeric words are tokenized via regex.
- **Stopword Filtering**: 150+ common English stopwords and conversational prompt words are ignored.
- **Chunk Evaluation Rules**:
  - **Short Queries (1–2 salient terms)**: At least 1 salient term must be matched.
  - **Standard Queries (3+ salient terms)**:
    - Overlap ratio $\ge 35\%$ of unique salient query terms; OR
    - Phrase match (consecutive 2-grams or 3-grams of salient query terms found in chunk); OR
    - Sub-aspect coverage ($\ge 2$ salient terms from any individual clause of a multi-part query); OR
    - Number match (if query contains specific numeric identifiers like "380" and context terms).

### B. Collective Sufficiency Rules & Status Definitions
- `sufficient` (`is_sufficient = True`): All query sub-aspects are covered, overall salient query concept coverage $\ge 60\%$, and no conflicting statements are detected.
- `partially_sufficient` (`is_sufficient = False`): For multi-part queries, at least one sub-aspect is covered while others are missing. The missing sub-aspects are explicitly listed in `missing_information`.
- `insufficient` (`is_sufficient = False`): Evidence has minor topic overlap ($< 40\%$), but core factual predicates or required entity attributes are not answered.
- `conflicting` (`is_sufficient = False`): Contradictory numeric specifications (e.g. 1-year warranty vs 3-year warranty) or opposing polarity statements (feature standard vs optional/not included) are detected across relevant chunks.
- `no_evidence` (`is_sufficient = False`): The retrieval pool was empty, or none of the retrieved chunks met relevance criteria.

### C. Limitations of Heuristic Relevance and Conflict Detection
- **No Deep Semantic Reasoning**: The heuristic relies on lexical overlap, n-gram phrases, and rule-based attribute extraction. Complex paraphrases or metaphorical statements that share zero lexical root words may not be recognized without an embedding cross-encoder or LLM.
- **Open-Domain Conflict Detection**: Rule-based conflict detection reliably catches explicit attribute clashes (warranties, checkpoint counts, standard vs optional status), but nuanced semantic contradictions across open prose require a natural language inference (NLI) model or LLM.

---

## 4. Files Created

1. [`src/phase2/evidence_evaluator.py`](file:///C:/Users/devi.c/RAG%20Phase1/src/phase2/evidence_evaluator.py)
   - Contains `EvidenceEvaluator`, `tokenize_text`, `extract_salient_terms`, `split_query_sub_aspects`, and `detect_conflicts`.
2. [`src/phase2/demo_evidence.py`](file:///C:/Users/devi.c/RAG%20Phase1/src/phase2/demo_evidence.py)
   - CLI demo script that performs hybrid RRF retrieval, evaluates evidence relevance and sufficiency, and prints a structured diagnostic report.
3. [`tests/test_phase2_evidence.py`](file:///C:/Users/devi.c/RAG%20Phase1/tests/test_phase2_evidence.py)
   - 24 unit and integration tests covering input validation, relevance assessment, collective sufficiency, conflict detection, real-corpus integration, and Phase 1 artifact immutability.
4. [`reports/phase2_step10_evidence_report.md`](file:///C:/Users/devi.c/RAG%20Phase1/reports/phase2_step10_evidence_report.md)
   - This comprehensive documentation report.

---

## 5. Demonstration Outputs on Actual Corpus

### Scenario 1: Answerable Query (380 Checkpoints for Pre-Owned Cars)
Command:
```powershell
& ".\.venv\Scripts\python.exe" -m src.phase2.demo_evidence "What are the 380 checkpoints for pre-owned cars?"
```
Output:
```text
================================================================================
 Aventro Motors RAG — Phase 2, Step 10: Evidence Relevance & Sufficiency Demo
================================================================================
Query: 'What are the 380 checkpoints for pre-owned cars?'
Retrieval Pool: top 5 from BM25 and Vector fused with RRF (k=60)
--------------------------------------------------------------------------------
Retrieved candidate chunks : 5
Evidence Status            : SUFFICIENT
Is Evidence Sufficient     : True
Decision Reason            : The retrieved evidence addresses the requested information with 100% query concept coverage across 5 relevant chunk(s).
Relevant Chunks Count      : 5 (IDs: ['doc_pre_owned_certified_cars_chunk_001', 'doc_pre_owned_certified_cars_chunk_003', 'doc_pre_owned_certified_cars_chunk_002', 'doc_pre_owned_certified_cars_chunk_000', 'doc_pre_owned_certified_cars_chunk_006'])
Irrelevant Chunks Count    : 0 (IDs: [])
================================================================================
 EVALUATED CHUNKS BREAKDOWN
================================================================================
[1] RELEVANT | doc_pre_owned_certified_cars_chunk_001 | RRF Rank #1 Score: 0.03227
    Source Document : Pre-Owned Certified Cars.md
    Section Heading : Why Choose **Aventro Certified Pre-Owned Cars?**
    Heading Path    : Aventro Motors Pvt. Ltd. > Pre-Owned Certified Cars > Why Choose **Aventro Certified Pre-Owned Cars?**
    Assessment Note : High keyword overlap: matched 4/5 terms (80%): ['380', 'cars', 'owned', 'pre'].
    Text Excerpt    : ### Why Choose **Aventro Certified Pre-Owned Cars?**  ...
[2] RELEVANT | doc_pre_owned_certified_cars_chunk_003 | RRF Rank #2 Score: 0.03151
    Source Document : Pre-Owned Certified Cars.md
    Section Heading : **Aventro 380+ Checkpoints: The Gold Standard in Assurance**
    Heading Path    : Aventro Motors Pvt. Ltd. > **Aventro 380+ Checkpoints: The Gold Standard in Assurance**
    Assessment Note : High keyword overlap: matched 3/5 terms (60%): ['380', 'checkpoints', 'pre'].
    Text Excerpt    : ## **Aventro 380+ Checkpoints: The Gold Standard in Assurance** (Continued)...
[3] RELEVANT | doc_pre_owned_certified_cars_chunk_002 | RRF Rank #3 Score: 0.01639
    Source Document : Pre-Owned Certified Cars.md
    Section Heading : **Aventro 380+ Checkpoints: The Gold Standard in Assurance**
    Heading Path    : Aventro Motors Pvt. Ltd. > **Aventro 380+ Checkpoints: The Gold Standard in Assurance**
    Assessment Note : High keyword overlap: matched 5/5 terms (100%): ['380', 'cars', 'checkpoints', 'owned', 'pre'].
    Text Excerpt    : ## **Aventro 380+ Checkpoints: The Gold Standard in Assurance**  ...
...
```

### Scenario 2: Genuinely Absent Topic (Abstention Decision)
Command:
```powershell
& ".\.venv\Scripts\python.exe" -m src.phase2.demo_evidence "What are the lunar orbit propulsion specifications of Aventro rocket engines?"
```
Output:
```text
================================================================================
 Aventro Motors RAG — Phase 2, Step 10: Evidence Relevance & Sufficiency Demo
================================================================================
Query: 'What are the lunar orbit propulsion specifications of Aventro rocket engines?'
Retrieval Pool: top 5 from BM25 and Vector fused with RRF (k=60)
--------------------------------------------------------------------------------
Retrieved candidate chunks : 5
Evidence Status            : NO_EVIDENCE
Is Evidence Sufficient     : False
Decision Reason            : Retrieved chunks do not contain meaningful relevance to the query.
Relevant Chunks Count      : 0 (IDs: [])
Irrelevant Chunks Count    : 5 (IDs: ['doc_aventro_spark_hatchback_chunk_000', 'doc_traction_control_chunk_000', 'doc_parts_4_chunk_000', 'doc_parts_2_chunk_004', 'doc_aventro_edge_sedan_chunk_007'])
Missing Information        : ["All requested information for: 'What are the lunar orbit propulsion specifications of Aventro rocket engines?'"]
================================================================================
 EVALUATED CHUNKS BREAKDOWN
================================================================================
[1] IRRELEVANT | doc_aventro_spark_hatchback_chunk_000 | RRF Rank #1 Score: 0.03101
    Source Document : Aventro Spark Hatchback.md
    Section Heading : Aventro Spark Hatchback
    Heading Path    : Aventro Spark Hatchback
    Assessment Note : Insufficient term overlap: matched only 2/7 terms (29%): ['aventro', 'specifications'].
...
```

### Scenario 3: Multi-Part Query with Partial Coverage
Command:
```powershell
& ".\.venv\Scripts\python.exe" -m src.phase2.demo_evidence "What are the 380 checkpoints for pre-owned cars and what is the return policy for moon rovers?"
```
Output:
```text
================================================================================
 Aventro Motors RAG — Phase 2, Step 10: Evidence Relevance & Sufficiency Demo
================================================================================
Query: 'What are the 380 checkpoints for pre-owned cars and what is the return policy for moon rovers?'
Retrieval Pool: top 5 from BM25 and Vector fused with RRF (k=60)
--------------------------------------------------------------------------------
Retrieved candidate chunks : 5
Evidence Status            : PARTIALLY_SUFFICIENT
Is Evidence Sufficient     : False
Decision Reason            : Evidence partially covers the query: 1/2 sub-aspects covered. Missing information for: ['what is the return policy for moon rovers'].
Relevant Chunks Count      : 5 (IDs: ['doc_pre_owned_certified_cars_chunk_001', 'doc_pre_owned_certified_cars_chunk_002', 'doc_pre_owned_certified_cars_chunk_000', 'doc_pre_owned_certified_cars_chunk_003', 'doc_pre_owned_certified_cars_chunk_006'])
Irrelevant Chunks Count    : 0 (IDs: [])
Missing Information        : ['what is the return policy for moon rovers']
```

---

## 6. Exact Test Commands & Regression Results

### 1. Step 10 Evidence Evaluator Tests
Command:
```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests/test_phase2_evidence.py -v
```
**Outcome**: **24 passed**, 14 warnings in 28.40s.

### 2. Phase 2 Retrievers Regression Suite (Steps 7, 8, 9)
Command:
```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests/test_phase2_bm25.py tests/test_phase2_vector.py tests/test_phase2_rrf.py -v
```
**Outcome**: **49 passed** (15 BM25 + 18 Vector + 16 RRF), 14 warnings in 28.34s.

### 3. Full Project Test Suite (Steps 3 through 10)
Command:
```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests/ -v
```
**Outcome**: **112 passed**, 14 warnings in 43.95s.
- Step 3 (Markdown extraction & CSV validation): 9 passed
- Step 4 (Document chunking & audit): 14 passed
- Step 5 (Local embedding generation): 7 passed
- Step 6 (Persistent ChromaDB indexing): 9 passed
- Step 7 (BM25 keyword retriever): 15 passed
- Step 8 (ChromaDB vector retriever): 18 passed
- Step 9 (Reciprocal Rank Fusion): 16 passed
- Step 10 (Evidence relevance & sufficiency): 24 passed
- **Total: 112 passing tests, 0 failures, 0 skipped.**

---

## 7. Artifact Integrity Confirmation
- `artifacts/chunks/document_chunks.jsonl`: Unchanged (459 records verified).
- `artifacts/embeddings/chunk_embeddings.npy`: Unchanged (shape `(459, 384)` verified).
- `artifacts/chroma_db/`: Unchanged (459 indexed records in `aventro_motors_knowledge`).
- Step 7 (`bm25_retriever.py`), Step 8 (`vector_retriever.py`), and Step 9 (`rrf_fusion.py`): Unchanged and verified.
