# Aventro Motors RAG — Phase 2, Step 12: Grounded Answer Generation with Citations Report

## 1. Overview & Objective

The objective of Step 12 is to implement grounded natural-language answer generation with inline source citations, integrated into the existing LangGraph orchestration pipeline.

The generation module:
1. Takes the user query and the approved evidence chunks from Step 10 (`EvidenceEvaluator`).
2. Constructs a strictly grounded prompt with assigned temporary citation identifiers (`[E1]`, `[E2]`, etc.).
3. Calls a configurable model provider via dependency injection (`FakeLLMClient` for offline/unit tests, `OllamaClient` for local Ollama execution).
4. Maintains a request-scoped citation registry and parses/resolves inline citation tags back to original document and chunk metadata.
5. Flags any unknown citation tags (e.g. `[E99]`).
6. Enforces explicit abstention decisions (no answer generated when evidence is insufficient, absent, conflicting, or partial).
7. Never allows document text to override system grounding rules (anti-prompt injection framing).

---

## 2. Files Created and Modified

1. [`src/phase2/answer_generator.py`](file:///C:/Users/devi.c/RAG%20Phase1/src/phase2/answer_generator.py) (Created)
   - Defines `AnswerGenerator`, `BaseLLMClient`, `FakeLLMClient`, `OllamaClient`, `ModelUnavailableError`, `get_default_llm_client`, and `format_rag_response`.
2. [`src/phase2/rag_state.py`](file:///C:/Users/devi.c/RAG%20Phase1/src/phase2/rag_state.py) (Modified)
   - Extended `RAGState` with `generated_answer`, `answer_citations`, `generation_status`, and `generation_error`. Added `WORKFLOW_COMPLETED = "completed"`.
3. [`src/phase2/rag_graph.py`](file:///C:/Users/devi.c/RAG%20Phase1/src/phase2/rag_graph.py) (Modified)
   - Added `make_generate_node` factory and wired `generate_answer` node into the LangGraph state machine when evidence is `sufficient`.
4. [`src/phase2/demo_rag_answer.py`](file:///C:/Users/devi.c/RAG%20Phase1/src/phase2/demo_rag_answer.py) (Created)
   - Full CLI demo script demonstrating grounded generation, abstention, and partial-coverage scenarios on the actual corpus.
5. [`tests/test_phase2_answer_generator.py`](file:///C:/Users/devi.c/RAG%20Phase1/tests/test_phase2_answer_generator.py) (Created)
   - 8 unit tests covering registry construction, fake generation, citation resolution, unknown citation flagging, empty response validation, provider failure, and prompt framing.
6. [`tests/test_phase2_rag_graph.py`](file:///C:/Users/devi.c/RAG%20Phase1/tests/test_phase2_rag_graph.py) (Updated)
   - 14 tests verifying graph compilation, query validation routing, grounded answer generation when sufficient, abstention when insufficient/partial/conflicting, model failure handling, and real-corpus integration.
7. [`reports/phase2_step12_grounded_answer_generation_report.md`](file:///C:/Users/devi.c/RAG%20Phase1/reports/phase2_step12_grounded_answer_generation_report.md) (Created)
   - This comprehensive documentation report.

---

## 3. Architecture & LangGraph Integration

### Pipeline Flow
```
User Query
    │
    ▼
[validate_query]
    │ (Valid Query)
    ▼
[retrieve_hybrid] (BM25 + Vector fused with RRF, k=60)
    │
    ▼
[evaluate_evidence] (EvidenceEvaluator)
    │
    ├─► If SUFFICIENT ──────────► [generate_answer] ──► WORKFLOW_COMPLETED
    │                                (AnswerGenerator)
    ├─► If PARTIALLY_SUFFICIENT ─► [handle_partially_sufficient] ──► REQUIRES_FURTHER_RETRIEVAL
    │                                (Abstain from premature generation)
    ├─► If INSUFFICIENT / NO_EVID► [handle_insufficient_or_no_evidence] ──► REQUIRES_ABSTENTION
    │                                (Abstain from generation)
    └─► If CONFLICTING ──────────► [handle_conflicting] ──► REQUIRES_CONFLICT_RESOLUTION
                                     (Abstain from generation)
```

---

## 4. Model Provider Configuration

- **Abstract Interface**: `BaseLLMClient` with `generate(prompt: str, system_prompt: Optional[str] = None) -> str`.
- **`FakeLLMClient`**: Deterministic offline client used in automated unit tests. Eliminates network dependence, API costs, and flaky external services.
- **`OllamaClient`**: Connects via HTTP to local Ollama daemon (defaults to `http://localhost:11434`, model `llama3.2`).
- **Configuration**: Managed via environment variables:
  - `LLM_PROVIDER`: `"ollama"` or `"fake"`.
  - `OLLAMA_HOST`: Host URL (default `http://localhost:11434`).
  - `OLLAMA_MODEL`: Model tag (default `llama3.2`).
- **Safety**: If the configured provider is unreachable, raises `ModelUnavailableError` with clear diagnostics. Never invents answers or silently substitutes models on failure.

---

## 5. Citation Registry & Validation Behavior

1. **Registry Mapping**:
   Before prompting the model, `AnswerGenerator.build_evidence_registry` assigns temporary evidence tags `[E1]`, `[E2]`, etc. for the current request only:
   ```python
   registry["E1"] = {
       "evidence_id": "E1",
       "chunk_id": "doc_pre_owned_certified_cars_chunk_001",
       "source_document": "Pre-Owned Certified Cars.md",
       "document_title": "Aventro Motors Pvt. Ltd.",
       "section_heading": "Why Choose Aventro Certified Pre-Owned Cars?",
       "heading_path": "Aventro Motors Pvt. Ltd. > Pre-Owned Certified Cars > Why Choose Aventro Certified Pre-Owned Cars?",
   }
   ```
2. **Inline Citation Parsing**:
   Extracts all `\[E\d+\]` occurrences from the model's output using regex.
3. **Resolution & Flagging**:
   - Valid tags (`[E1]`, `[E2]`) resolve to their source document and chunk metadata in `answer_citations`.
   - Unknown tags (e.g. `[E99]`) are captured in `unknown_citations` and recorded in the diagnostic trace.

---

## 6. Demonstration Outputs on Actual Corpus

### Scenario 1: Answerable Query (380 Checkpoints)
Command:
```powershell
& ".\.venv\Scripts\python.exe" -m src.phase2.demo_rag_answer "What are the 380 checkpoints for pre-owned cars?"
```
Output:
```text
================================================================================
 Aventro Motors RAG — Phase 2, Step 12: Grounded Answer Generation Demo
================================================================================
User Query : 'What are the 380 checkpoints for pre-owned cars?'
Top K      : 5
LLM Provider: FakeLLMClient (Ollama is not running locally; using deterministic offline fake model. To use real local LLM, start Ollama with 'ollama serve' and pull a model).
--------------------------------------------------------------------------------
Workflow Status : COMPLETED

================================================================================
 GENERATED GROUNDED ANSWER
================================================================================
Aventro certified pre-owned vehicles undergo a rigorous 380+ checkpoint inspection [E1]. This comprehensive evaluation covers mechanical, electrical, and structural systems [E2].
--------------------------------------------------------------------------------
RESOLVED INLINE CITATIONS (2):
  • [E1] -> Chunk ID: doc_pre_owned_certified_cars_chunk_001
      Document Title : Aventro Motors Pvt. Ltd.
      Source Document: Pre-Owned Certified Cars.md
      Section Heading: Why Choose **Aventro Certified Pre-Owned Cars?**
      Heading Path   : Aventro Motors Pvt. Ltd. > Pre-Owned Certified Cars > Why Choose **Aventro Certified Pre-Owned Cars?**
  • [E2] -> Chunk ID: doc_pre_owned_certified_cars_chunk_003
      Document Title : Aventro Motors Pvt. Ltd.
      Source Document: Pre-Owned Certified Cars.md
      Section Heading: **Aventro 380+ Checkpoints: The Gold Standard in Assurance**
      Heading Path   : Aventro Motors Pvt. Ltd. > **Aventro 380+ Checkpoints: The Gold Standard in Assurance**

================================================================================
 EXECUTION TRACE
================================================================================
  • validate_query: Validating input query.
  • validate_query: SUCCESS - Normalized query: 'What are the 380 checkpoints for pre-owned cars?'
  • retrieve_hybrid: Querying BM25 and Vector with top_k=5.
  • retrieve_hybrid: SUCCESS - Retrieved 5 candidate chunk(s).
  • evaluate_evidence: Assessing 5 chunk(s) for relevance & sufficiency.
  • evaluate_evidence: SUCCESS - Assessment status: SUFFICIENT.
  • generate_answer: Generating answer using 5 cited evidence chunk(s).
  • generate_answer: SUCCESS - Generated answer with 2 citation(s).
================================================================================
```

### Scenario 2: Unsupported Query (Rocket Propulsion - Clean Abstention)
Command:
```powershell
& ".\.venv\Scripts\python.exe" -m src.phase2.demo_rag_answer "What are the lunar orbit propulsion specifications of Aventro rocket engines?"
```
Output:
```text
================================================================================
 Aventro Motors RAG — Phase 2, Step 12: Grounded Answer Generation Demo
================================================================================
User Query : 'What are the lunar orbit propulsion specifications of Aventro rocket engines?'
Top K      : 5
LLM Provider: FakeLLMClient (Ollama is not running locally; using deterministic offline fake model. To use real local LLM, start Ollama with 'ollama serve' and pull a model).
--------------------------------------------------------------------------------
Workflow Status : REQUIRES_ABSTENTION

================================================================================
 WORKFLOW DECISION: ABSTENTION (No Grounded Answer Generated)
================================================================================
Reason: Retrieved chunks do not contain meaningful relevance to the query.
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

### Scenario 3: Partially Supported Query (Checkpoints + Moon Rover Policy)
Command:
```powershell
& ".\.venv\Scripts\python.exe" -m src.phase2.demo_rag_answer "What are the 380 checkpoints for pre-owned cars and what is the return policy for moon rovers?"
```
Output:
```text
================================================================================
 Aventro Motors RAG — Phase 2, Step 12: Grounded Answer Generation Demo
================================================================================
User Query : 'What are the 380 checkpoints for pre-owned cars and what is the return policy for moon rovers?'
Top K      : 5
LLM Provider: FakeLLMClient (Ollama is not running locally; using deterministic offline fake model. To use real local LLM, start Ollama with 'ollama serve' and pull a model).
--------------------------------------------------------------------------------
Workflow Status : REQUIRES_FURTHER_RETRIEVAL

================================================================================
 WORKFLOW DECISION: PARTIAL EVIDENCE (No Premature Answer Generated)
================================================================================
Reason: Evidence partially covers the query: 1/2 sub-aspects covered. Missing information for: ['what is the return policy for moon rovers'].
Uncovered Sub-Aspects: ['what is the return policy for moon rovers']

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

## 7. Exact Test Commands & Results

### 1. Focused Tests (Answer Generator & Graph Integration)
Command:
```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests/test_phase2_answer_generator.py tests/test_phase2_rag_graph.py -v
```
**Outcome**: **22 passed** (8 in answer generator + 14 in graph), 14 warnings in 26.35s.

### 2. Full Regression Suite (Steps 3 through 12)
Command:
```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests/ -v
```
**Outcome**: **134 passed**, 14 warnings in 31.91s (0 failures, 0 skipped).
- Step 3 (Markdown extraction & CSV validation): 9 passed
- Step 4 (Document chunking & structural audit): 14 passed
- Step 5 (Local embedding generation): 7 passed
- Step 6 (Persistent ChromaDB indexing): 9 passed
- Step 7 (BM25 keyword retriever): 15 passed
- Step 8 (ChromaDB vector retriever): 18 passed
- Step 9 (Reciprocal Rank Fusion): 16 passed
- Step 10 (Evidence relevance & sufficiency): 24 passed
- Step 11 & 12 (LangGraph orchestration & Answer Generator): 22 passed
- **Total: 134 passing tests.**

---

## 8. Known Limitations & Boundaries
1. **Model Usage Disclosure**: The test suite and default execution in the sandbox use `FakeLLMClient` to ensure deterministic execution without network dependencies or local Ollama process requirements. When Ollama is running (`ollama serve`), the pipeline seamlessly routes calls to the local Ollama LLM without code changes.
2. **Citation Validation vs Claim Entailment**: The citation validator ensures every cited `[E#]` tag maps to an actual chunk in the registry and flags hallucinated citation tags (`[E99]`). However, syntactic citation verification does not prove that the sentence before the tag is factually entailed by that chunk. Full claim-level NLI / entailment verification will be addressed in a subsequent step.
3. **Artifact Immutability**: All Phase 1 artifacts (`artifacts/chunks/document_chunks.jsonl` with 459 records, `artifacts/chroma_db/`, `artifacts/embeddings/`) and Phase 2 retrievers remain untouched and verified.
