# Aventro Motors RAG — Phase 2, Step 11: LangGraph Orchestration Report

## 1. Overview & Objective

The objective of Step 11 is to build a structured, observable, and resilient **LangGraph orchestration pipeline** for the Aventro Motors RAG system. The pipeline integrates:
1. Input query validation and normalization.
2. Step 9 hybrid retrieval (`RRFFusion`) combining BM25 keyword and ChromaDB vector retrieval.
3. Step 10 evidence evaluation (`EvidenceEvaluator`) assessing relevance and collective sufficiency.
4. Explicit conditional edge routing based on evidence assessment status (`sufficient`, `partially_sufficient`, `insufficient`, `no_evidence`, `conflicting`), directing execution toward answer generation readiness or machine-readable abstention / further retrieval.

---

## 2. Dependency Configuration

- **Package Installed**: `langgraph>=0.2.0` (version `1.2.18` installed cleanly alongside `langgraph-checkpoint-4.3.1`, `langgraph-prebuilt-1.1.1`, and `langgraph-sdk-0.4.10`).
- **Compatibility**: Compatible with the project virtual environment (Python 3.14.7, `langchain-core 1.7.1`).
- **Requirements Update**: `requirements.txt` updated under section `7. Phase 2 LangGraph Orchestration`.

---

## 3. Shared Graph State (`RAGState`)

Defined in [`src/phase2/rag_state.py`](file:///C:/Users/devi.c/RAG%20Phase1/src/phase2/rag_state.py):
```python
class RAGState(TypedDict, total=False):
    query: str
    retrieved_results: List[Dict[str, Any]]
    evidence_assessment: Optional[Dict[str, Any]]
    workflow_status: str
    error: Optional[str]
    trace: List[str]
```

### Workflow Status Constants
- `WORKFLOW_READY_FOR_GENERATION = "ready_for_generation"`: Triggered when evidence status is `sufficient`. Indicates pipeline is ready for downstream answer generation.
- `WORKFLOW_REQUIRES_ABSTENTION = "requires_abstention"`: Triggered when evidence status is `insufficient` or `no_evidence`. Enforces clean abstention.
- `WORKFLOW_REQUIRES_FURTHER_RETRIEVAL = "requires_further_retrieval"`: Triggered when evidence is `partially_sufficient` (e.g. multi-part queries missing specific aspects).
- `WORKFLOW_REQUIRES_CONFLICT_RESOLUTION = "requires_conflict_resolution"`: Triggered when evidence contains contradictory specifications across chunks.
- `WORKFLOW_VALIDATION_FAILED = "validation_failed"`: Triggered when input query is empty, whitespace, or non-string.
- `WORKFLOW_ERROR = "workflow_error"`: Triggered on component-level exceptions during retrieval or evidence assessment.

---

## 4. Graph Architecture, Nodes & Routing Rules

```
               ┌────────────────────────┐
               │         START          │
               └───────────┬────────────┘
                           ▼
               ┌────────────────────────┐
               │     validate_query     │
               └───────────┬────────────┘
                           │
       [Validation OK?]    │    [Invalid / Empty Query]
       ┌───────────────────┴───────────────────┐
       ▼                                       ▼
┌────────────────────────┐                ┌─────────┐
│    retrieve_hybrid     │                │   END   │
└──────────┬─────────────┘                │ (FAILED)│
           │                              └─────────┘
   [Retrieval OK?] │ [Retrieval Exception]
   ┌───────────────┴───────────────┐
   ▼                               ▼
┌────────────────────────┐    ┌─────────┐
│   evaluate_evidence    │    │   END   │
└──────────┬─────────────┘    │ (ERROR) │
           │                  └─────────┘
           │ Evidence Status Conditional Routing
           ├────────────────────────┬────────────────────────┬────────────────────────┐
           ▼                        ▼                        ▼                        ▼
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│  handle_sufficient   │ │handle_partially_suff.│ │ handle_insufficient_ │ │  handle_conflicting  │
│                      │ │                      │ │    or_no_evidence    │ │                      │
│(ready_for_generation)│ │(further_retrieval)   │ │ (requires_abstention)│ │ (conflict_resolution)│
└──────────┬───────────┘ └──────────┬───────────┘ └──────────┬───────────┘ └──────────┬───────────┘
           │                        │                        │                        │
           └────────────────────────┴───────────┬────────────┴────────────────────────┘
                                                ▼
                                           ┌─────────┐
                                           │   END   │
                                           └─────────┘
```

### Nodes Defined
1. `validate_query_node`: Normalizes query string and rejects non-strings and empty input with clear diagnostics.
2. `retrieve_hybrid_node`: Invokes `RRFFusion.fuse()` with dependency injection and error handling.
3. `evaluate_evidence_node`: Invokes `EvidenceEvaluator.evaluate()` to assess evidence relevance, collective sufficiency, missing information, and conflicts.
4. Decision handler nodes: `handle_sufficient_node`, `handle_partially_sufficient_node`, `handle_insufficient_or_no_evidence_node`, and `handle_conflicting_node`.

### Conditional Edges
- `after_validation_router`: Routes to `retrieve_hybrid` if query is valid; terminates early at `END` with status `validation_failed` otherwise.
- `after_retrieval_router`: Routes to `evaluate_evidence` if retrieval succeeded; terminates early at `END` with status `workflow_error` on component failure.
- `evidence_sufficiency_router`: Routes to the matching status handler node based on `evidence_assessment["status"]`.

---

## 5. Demonstration Outputs on Actual Corpus

### Scenario 1: Answerable Query (380 Checkpoints)
Command:
```powershell
& ".\.venv\Scripts\python.exe" -m src.phase2.demo_rag_graph "What are the 380 checkpoints for pre-owned cars?"
```
Output:
```text
================================================================================
 Aventro Motors RAG — Phase 2, Step 11: LangGraph Pipeline Demo
================================================================================
User Query : 'What are the 380 checkpoints for pre-owned cars?'
Top K      : 5
--------------------------------------------------------------------------------
Final Workflow Status    : READY_FOR_GENERATION
Evidence Assessment Status: SUFFICIENT
Retrieved Candidates     : 5
Relevant Evidence Count  : 5
Irrelevant Evidence Count: 0

>>> Pipeline Decision: Evidence is SUFFICIENT. Ready for grounded answer generation.

================================================================================
 RELEVANT EVIDENCE EXCERPTS
================================================================================
[1] Chunk ID        : doc_pre_owned_certified_cars_chunk_001
    Source Document : Pre-Owned Certified Cars.md
    Section Heading : Why Choose **Aventro Certified Pre-Owned Cars?**
    Heading Path    : Aventro Motors Pvt. Ltd. > Pre-Owned Certified Cars > Why Choose **Aventro Certified Pre-Owned Cars?**
    Excerpt         : ### Why Choose **Aventro Certified Pre-Owned Cars?**  ...
[2] Chunk ID        : doc_pre_owned_certified_cars_chunk_003
    Source Document : Pre-Owned Certified Cars.md
    Section Heading : **Aventro 380+ Checkpoints: The Gold Standard in Assurance**
    Heading Path    : Aventro Motors Pvt. Ltd. > **Aventro 380+ Checkpoints: The Gold Standard in Assurance**
    Excerpt         : ## **Aventro 380+ Checkpoints: The Gold Standard in Assurance** (Continued)...
[3] Chunk ID        : doc_pre_owned_certified_cars_chunk_002
    Source Document : Pre-Owned Certified Cars.md
    Section Heading : **Aventro 380+ Checkpoints: The Gold Standard in Assurance**
    Heading Path    : Aventro Motors Pvt. Ltd. > **Aventro 380+ Checkpoints: The Gold Standard in Assurance**
    Excerpt         : ## **Aventro 380+ Checkpoints: The Gold Standard in Assurance**  ...
...
================================================================================
 EXECUTION TRACE
================================================================================
  • validate_query: Validating input query.
  • validate_query: SUCCESS - Normalized query: 'What are the 380 checkpoints for pre-owned cars?'
  • retrieve_hybrid: Querying BM25 and Vector with top_k=5.
  • retrieve_hybrid: SUCCESS - Retrieved 5 candidate chunk(s).
  • evaluate_evidence: Assessing 5 chunk(s) for relevance & sufficiency.
  • evaluate_evidence: SUCCESS - Assessment status: SUFFICIENT.
  • route: Evidence is sufficient -> ready_for_generation.
================================================================================
```

### Scenario 2: Absent/Unrelated Topic (Clean Abstention)
Command:
```powershell
& ".\.venv\Scripts\python.exe" -m src.phase2.demo_rag_graph "What are the lunar orbit propulsion specifications of Aventro rocket engines?"
```
Output:
```text
================================================================================
 Aventro Motors RAG — Phase 2, Step 11: LangGraph Pipeline Demo
================================================================================
User Query : 'What are the lunar orbit propulsion specifications of Aventro rocket engines?'
Top K      : 5
--------------------------------------------------------------------------------
Final Workflow Status    : REQUIRES_ABSTENTION
Evidence Assessment Status: NO_EVIDENCE
Retrieved Candidates     : 5
Relevant Evidence Count  : 0
Irrelevant Evidence Count: 5

>>> Pipeline Decision: ABSTAIN FROM GENERATING ANSWER.
    Explanation: The retrieved documents do not provide sufficient factual support.
    Missing Information: ["All requested information for: 'What are the lunar orbit propulsion specifications of Aventro rocket engines?'"]

================================================================================
 EXECUTION TRACE
================================================================================
  • validate_query: Validating input query.
  • validate_query: SUCCESS - Normalized query: 'What are the lunar orbit propulsion specifications of Aventro rocket engines?'
  • retrieve_hybrid: Querying BM25 and Vector with top_k=5.
  • retrieve_hybrid: SUCCESS - Retrieved 5 candidate chunk(s).
  • evaluate_evidence: Assessing 5 chunk(s) for relevance & sufficiency.
  • evaluate_evidence: SUCCESS - Assessment status: NO_EVIDENCE.
  • route: Evidence status is 'no_evidence' -> requires_abstention.
================================================================================
```

### Scenario 3: Multi-Part Query with Partial Coverage
Command:
```powershell
& ".\.venv\Scripts\python.exe" -m src.phase2.demo_rag_graph "What are the 380 checkpoints for pre-owned cars and what is the return policy for moon rovers?"
```
Output:
```text
================================================================================
 Aventro Motors RAG — Phase 2, Step 11: LangGraph Pipeline Demo
================================================================================
User Query : 'What are the 380 checkpoints for pre-owned cars and what is the return policy for moon rovers?'
Top K      : 5
--------------------------------------------------------------------------------
Final Workflow Status    : REQUIRES_FURTHER_RETRIEVAL
Evidence Assessment Status: PARTIALLY_SUFFICIENT
Retrieved Candidates     : 5
Relevant Evidence Count  : 5
Irrelevant Evidence Count: 0

>>> Pipeline Decision: PARTIAL EVIDENCE DETECTED.
    Explanation: The query has multiple parts and some requested facts are missing.
    Missing Information: ['what is the return policy for moon rovers']

================================================================================
 EXECUTION TRACE
================================================================================
  • validate_query: Validating input query.
  • validate_query: SUCCESS - Normalized query: 'What are the 380 checkpoints for pre-owned cars and what is the return policy for moon rovers?'
  • retrieve_hybrid: Querying BM25 and Vector with top_k=5.
  • retrieve_hybrid: SUCCESS - Retrieved 5 candidate chunk(s).
  • evaluate_evidence: Assessing 5 chunk(s) for relevance & sufficiency.
  • evaluate_evidence: SUCCESS - Assessment status: PARTIALLY_SUFFICIENT.
  • route: Evidence is partially sufficient -> requires_further_retrieval.
================================================================================
```

---

## 6. Exact Test Commands & Results

### 1. Step 11 Test Suite
Command:
```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests/test_phase2_rag_graph.py -v
```
**Outcome**: **14 passed**, 14 warnings in 28.56s.

### 2. Full Project Test Suite (Steps 3 through 11)
Command:
```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests/ -v
```
**Outcome**: **126 passed**, 14 warnings in 34.71s (0 failures, 0 skipped).
- Step 3 (Markdown extraction & CSV validation): 9 passed
- Step 4 (Document chunking & structural audit): 14 passed
- Step 5 (Local embedding generation): 7 passed
- Step 6 (Persistent ChromaDB indexing): 9 passed
- Step 7 (BM25 keyword retriever): 15 passed
- Step 8 (ChromaDB vector retriever): 18 passed
- Step 9 (Reciprocal Rank Fusion): 16 passed
- Step 10 (Evidence relevance & sufficiency): 24 passed
- Step 11 (LangGraph orchestration): 14 passed
- **Total: 126 passing tests.**

---

## 7. Known Limitations & Strict Boundaries
- **No LLM Answer Generation in this Step**: The graph orchestrates query validation, hybrid retrieval, and evidence assessment, and sets machine-readable statuses (`ready_for_generation`, `requires_abstention`, etc.). Actual grounded answer generation will be added in Step 12.
- **Phase 1 Artifacts and Prior Retrievers**: All 459 canonical chunks, vector database files, and Step 7–10 retriever/evaluator code remain unmodified.
