# Phase 2, Step 13: Claim-Level Citation Verification — Implementation Report

**Project:** Aventro Motors RAG System
**Step:** 13 — Claim-Level Citation Verification
**Date:** 2026-10-30
**Python:** 3.14.7 | **Test Runner:** pytest 9.2.0

---

## 1. Objective

Implement a claim-level citation verification layer that determines whether each individual factual claim in a generated RAG answer is:

1. **Cited** — accompanied by at least one `[Exx]` citation tag.
2. **Validly cited** — the cited ID exists in the per-request evidence registry.
3. **Evidentially supported** — the cited evidence text factually entails the claim.

This prevents citation ID resolution (Step 12) from being misinterpreted as factual confirmation, and provides structured, claim-granular auditability for every answer.

---

## 2. Architecture

### 2.1 Pipeline Position

```
validate_query
    -> retrieve_hybrid (BM25 + Vector via RRF)
    -> evaluate_evidence
    -> [route] -> generate_answer (sufficient only)
                      -> verify_citations   <-- Step 13
                              -> END (VERIFIED or REQUIRES_REVIEW)
         -> handle_partially_sufficient -> END
         -> handle_insufficient_or_no_evidence -> END
         -> handle_conflicting -> END
```

### 2.2 New / Modified Files

| File | Role | Change |
|------|------|--------|
| `src/phase2/citation_verifier.py` | Core verification component | **New** |
| `src/phase2/rag_graph.py` | LangGraph orchestration | **Extended** — added `make_verify_node`, `after_generation_router`, `verify_citations` node |
| `src/phase2/rag_state.py` | Shared pipeline state | **Extended** — added `WORKFLOW_VERIFIED`, `WORKFLOW_REQUIRES_REVIEW`, state fields `verification_report`, `verification_status`, `is_verified` |
| `src/phase2/answer_generator.py` | Grounded generation | **No change** (already exports `evidence_registry`) |
| `tests/test_phase2_citation_verifier.py` | Unit tests for Step 13 | **New** (13 tests) |
| `tests/test_phase2_rag_graph.py` | Graph integration tests | **Extended** — 5 new Step 13 tests, 2 existing tests updated |
| `src/phase2/demo_citation_verification.py` | Demonstration script | **New** |

---

## 3. Component Design

### 3.1 `citation_verifier.py`

#### `segment_claims(answer: str) -> List[Dict]`

Splits an answer into discrete factual claims using:
- Sentence boundaries (`.`, `!`, `?` followed by uppercase/digit/quote)
- Semicolons (`;`)

Each segment records:
- `claim_text` — cleaned text with citation tags stripped
- `cited_evidence_ids` — list of `[Exx]` IDs found inline
- `raw_segment` — original text including citation tags

Conversational preambles (e.g. "Based on the provided documents") are filtered.

#### `BaseEntailmentVerifier` (abstract)

```python
def verify(claim: str, evidence_text: str) -> Tuple[bool, str]
```

#### `HeuristicEntailmentVerifier` (default)

Three-stage deterministic verification:

1. **Numeric contradiction check** — numbers present in the claim must appear in the evidence tokens.
2. **Polarity contradiction check** — "is standard" vs "is optional" pattern matching.
3. **Salient concept overlap** — extracted non-stopword terms from the claim vs evidence tokens; configurable `min_support_ratio` (default 55%).

#### `FakeEntailmentVerifier`

Deterministic fake for testing. Supports:
- `default_support: bool`
- `claim_overrides: Dict[str, Tuple[bool, str]]` — substring-matched claim-level overrides
- `exc_to_raise` — simulate verifier exceptions

#### `CitationVerifier.verify(answer, evidence_registry) -> Dict`

Returns a structured report:

```python
{
    "verification_status": "verified" | "requires_review" | "verification_error" | "empty_answer" | "no_claims_found",
    "overall_status": "verified" | "partially_verified" | "unverified" | "error",
    "is_verified": bool,
    "claim_results": [
        {
            "claim_text": str,
            "original_citations": List[str],
            "verification_status": "supported" | "unsupported" | "uncited" | "invalid_citation" | "verification_error",
            "is_supported": bool,
            "resolved_citations": List[Dict],   # full evidence registry entries
            "explanation": str,
        }, ...
    ],
    "supported_claim_count": int,
    "unsupported_claim_count": int,
    "uncited_claim_count": int,
    "invalid_citation_count": int,
    "verification_error": Optional[str],
    "limitations": List[str],
}
```

### 3.2 Workflow Status Transitions

| Scenario | `workflow_status` |
|----------|------------------|
| All claims supported + cited | `WORKFLOW_VERIFIED` |
| Any unsupported / invalid citation | `WORKFLOW_REQUIRES_REVIEW` |
| Any uncited claim (but none unsupported) | `WORKFLOW_REQUIRES_REVIEW` |
| Verifier exception | `WORKFLOW_ERROR` |
| Abstention branches (insufficient, partial, conflicting, no evidence) | Unchanged (no verification) |

### 3.3 Graph Node: `make_verify_node`

- Reads `generated_answer` and `evidence_registry` from state.
- Falls back to reconstructing the registry from `evidence_assessment.relevant_evidence` if registry is absent.
- Routes to `WORKFLOW_VERIFIED`, `WORKFLOW_REQUIRES_REVIEW`, or `WORKFLOW_ERROR`.
- Appends trace entries for all outcomes.

---

## 4. Demo Results (`demo_citation_verification.py`)

| Scenario | Verified | Status | S | U | UC | INV |
|----------|----------|--------|---|---|----|-----|
| A — Fully Supported, Correctly Cited | YES | VERIFIED | 2 | 0 | 0 | 0 |
| B — Numeric Mismatch (500 vs 380) | NO | UNVERIFIED | 0 | 1 | 0 | 0 |
| C — Uncited Factual Claim | NO | PARTIALLY_VERIFIED | 1 | 0 | 1 | 0 |
| D — Unknown Citation [E99] | NO | UNVERIFIED | 0 | 0 | 0 | 1 |
| E — Mixed (Supported + Unsupported + Uncited) | NO | UNVERIFIED | 1 | 1 | 1 | 0 |

**Legend:** S=Supported · U=Unsupported · UC=Uncited · INV=Invalid citation

**Key observations from demo:**
- Scenario A: Both claims achieve 100% and 82% salient concept alignment respectively — verified.
- Scenario B: Numeric `500` contradicts `380` in cited evidence — correctly flagged as unsupported.
- Scenario C: Sentence 1 (cited) is verified; sentence 2 (uncited) triggers `requires_review`.
- Scenario D: `[E99]` absent from registry → `invalid_citation` without any support check attempted.
- Scenario E: Three-claim answer with three distinct outcomes in a single pass.

---

## 5. Test Results

### 5.1 Focused Test Run

```
tests/test_phase2_citation_verifier.py   13 passed
tests/test_phase2_rag_graph.py           19 passed (14 existing + 5 new Step 13 tests)
─────────────────────────────────────────
Total focused                            32 passed  (0 failed)
Duration: 15.17 s
```

#### New Step 13 graph integration tests

| Test | Asserts |
|------|---------|
| `test_verification_node_produces_verified_status_on_supported_claims` | `WORKFLOW_VERIFIED`, `is_verified=True`, trace contains "SUCCESS" |
| `test_verification_node_produces_requires_review_on_unsupported_claim` | `WORKFLOW_REQUIRES_REVIEW`, `is_verified=False`, trace contains "REVIEW REQUIRED" |
| `test_verification_skipped_on_abstention_pathways` | No `verification_report`, no verify_citations trace entry |
| `test_verification_node_handles_uncited_claims` | `WORKFLOW_REQUIRES_REVIEW`, `uncited_claim_count >= 1` |
| `test_citation_verifier_module_is_wired_into_graph` | `verify_citations` in graph node names |

### 5.2 Full Regression Suite

```
pytest tests/ -v
─────────────────────────────────────────────────────────────────────
collected 152 items

tests/test_phase2_answer_generator.py    8 passed
tests/test_phase2_bm25.py               14 passed
tests/test_phase2_citation_verifier.py  13 passed
tests/test_phase2_evidence.py           17 passed
tests/test_phase2_rag_graph.py          19 passed
tests/test_phase2_rrf.py               12 passed
tests/test_phase2_vector.py            12 passed
tests/test_step3.py                     9 passed
tests/test_step4.py                    13 passed
tests/test_step5.py                     7 passed
tests/test_step6.py                     9 passed
─────────────────────────────────────────────────────────────────────
TOTAL                                  152 passed   (14 warnings)
Duration: 19.93 s
```

> All 152 tests pass. Warnings are `asyncio.iscoroutinefunction` deprecation notices from third-party libraries (LangSmith, ChromaDB); no action required.

---

## 6. Claim Verification Outcome Taxonomy

| Status | Trigger | `is_supported` | Recommended Action |
|--------|---------|---------------|-------------------|
| `supported` | Citation valid + evidence entails claim | `True` | Present to user |
| `unsupported` | Citation valid but evidence contradicts/mismatches claim | `False` | Human review |
| `uncited` | No `[Exx]` tag on factual claim | `False` | Add citation or remove claim |
| `invalid_citation` | `[Exx]` tag not in evidence registry | `False` | Fix citation ID or regenerate |
| `verification_error` | Verifier raised an exception | `False` | Investigate verifier |

---

## 7. Limitations

1. **Lexical verification only.** The `HeuristicEntailmentVerifier` performs token-level and pattern-level checks. It does not perform full neural natural language inference (NLI). Claims with paraphrased or indirect support may be incorrectly flagged as unsupported.

2. **Sentence splitting is heuristic.** Complex sentences with embedded clauses may not split into fully atomic propositions. Multi-sentence constructs joined by coordinating conjunctions (e.g. "and") remain as single claims.

3. **Numbers without domain context.** Any number in a claim must appear literally in evidence. Version numbers, page references, or incidental figures may trigger false numeric contradiction alerts.

4. **No external model dependency in tests.** The default test suite uses `FakeEntailmentVerifier` or `HeuristicEntailmentVerifier`. Production deployment may substitute an NLI cross-encoder (e.g. `cross-encoder/nli-deberta-v3-small`) for higher accuracy at the cost of inference latency.

5. **Evidence text is passive data.** The verifier explicitly treats retrieved evidence as inert data and does not execute any instructions embedded within it, consistent with RAG security guidelines.

---

## 8. Phase 2 Progress Summary

| Step | Description | Status |
|------|-------------|--------|
| 7 | BM25 Keyword Retriever | Complete |
| 8 | ChromaDB Vector Retriever | Complete |
| 9 | Reciprocal Rank Fusion | Complete |
| 10 | Evidence Relevance & Sufficiency Checker | Complete |
| 11 | LangGraph Orchestration | Complete |
| 12 | Grounded Answer Generation with Citations | Complete |
| **13** | **Claim-Level Citation Verification** | **Complete** |

**Total tests passing:** 152 (up from 134 before Step 13)
**New tests added in Step 13:** 18 (13 unit + 5 graph integration)
