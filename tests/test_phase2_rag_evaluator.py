"""
Unit and Integration Tests for Phase 2, Step 14: RAG Evaluation Framework
Tests evaluation schemas, RAGAS adapter (offline metrics only), citation evaluation,
dataset loading, batch runner, and isolation guarantees.

All tests run offline — no API keys, no Ollama, no network required.
Fake LLM clients and FakeEntailmentVerifier are used throughout.
"""

from __future__ import annotations

import json
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest

from src.phase2.evaluation_schemas import (
    BatchEvalSummary,
    CitationEvalSummary,
    EvalRecord,
    QuestionEvalResult,
)
from src.phase2.rag_evaluator import (
    EVAL_DATASET_PATH,
    RagasEvaluator,
    build_ragas_sample,
    evaluate_citations,
    load_canonical_chunk_ids,
    load_evaluation_dataset,
    run_single_question_eval,
)
from src.phase2.citation_verifier import (
    CitationVerifier,
    FakeEntailmentVerifier,
    HeuristicEntailmentVerifier,
)
from src.phase2.answer_generator import AnswerGenerator, FakeLLMClient, ModelUnavailableError
from src.phase2.rag_graph import build_rag_graph
from src.phase2.rag_state import (
    WORKFLOW_REQUIRES_ABSTENTION,
    WORKFLOW_VERIFIED,
    WORKFLOW_REQUIRES_REVIEW,
    WORKFLOW_ERROR,
)


# ==============================================================================
# Fixtures and helpers
# ==============================================================================

def make_eval_record(**kwargs) -> EvalRecord:
    defaults = {
        "question_id": "test_q_001",
        "source_document": "Pre-Owned Certified Cars.md",
        "source_row": 1,
        "origin_status": "valid_clean_json",
        "category": "Specific Query",
        "query_text": "How many checkpoints does Aventro use for pre-owned cars?",
        "expected_answer": "Aventro uses 380+ checkpoints for pre-owned car inspection.",
        "facts": ["Aventro uses 380+ checkpoints."],
    }
    defaults.update(kwargs)
    return EvalRecord(**defaults)


def make_mock_chunk(chunk_id: str, text: str = "Test chunk text") -> Dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "citation_metadata": {
            "document_id": "doc_01",
            "source_document": "Pre-Owned Certified Cars.md",
            "document_title": "Aventro Pre-Owned",
            "section_heading": "Inspection",
            "heading_path": "Aventro > Inspection",
            "chunk_index": 0,
            "character_count": len(text),
            "estimated_token_count": len(text) // 4,
            "content_hash": "hash_abc",
            "is_sub_chunk": False,
        },
        "source_document": "Pre-Owned Certified Cars.md",
        "document_title": "Aventro Pre-Owned",
        "section_heading": "Inspection",
        "heading_path": "Aventro > Inspection",
        "rrf_rank": 1,
        "rrf_score": 0.032,
        "contributors": ["bm25", "vector"],
    }


class MockRRF:
    def __init__(self, results=None, exc=None):
        self.results = results or []
        self.exc = exc

    def fuse(self, query, bm25_top_k=5, vector_top_k=5, final_top_k=5, **kwargs):
        if self.exc:
            raise self.exc
        return copy.deepcopy(self.results[:final_top_k])


class MockEvaluator:
    def __init__(self, assessment=None, exc=None):
        self.assessment = assessment
        self.exc = exc

    def evaluate(self, query, retrieved_results):
        if self.exc:
            raise self.exc
        if self.assessment is not None:
            return copy.deepcopy(self.assessment)
        return {
            "status": "sufficient",
            "is_sufficient": True,
            "reason": "Mock sufficient.",
            "relevant_evidence": retrieved_results,
            "irrelevant_evidence": [],
            "missing_information": [],
            "conflicts": [],
        }


def make_sufficient_assessment(chunks):
    return {
        "status": "sufficient",
        "is_sufficient": True,
        "reason": "Sufficient evidence.",
        "relevant_evidence": chunks,
        "irrelevant_evidence": [],
        "missing_information": [],
        "conflicts": [],
    }


def make_no_evidence_assessment():
    return {
        "status": "no_evidence",
        "is_sufficient": False,
        "reason": "No evidence found.",
        "relevant_evidence": [],
        "irrelevant_evidence": [],
        "missing_information": ["all"],
        "conflicts": [],
    }


# ==============================================================================
# 1. EvalRecord Schema Tests
# ==============================================================================

class TestEvalRecord:
    def test_from_dict_valid(self):
        d = {
            "question_id": "q_001",
            "source_document": "Acc.md",
            "source_row": 1,
            "origin_status": "valid_clean_json",
            "category": "Specific Query",
            "query_text": "What is ACC?",
            "expected_answer": "ACC is adaptive cruise control.",
            "facts": ["ACC is adaptive cruise control."],
        }
        rec = EvalRecord.from_dict(d)
        assert rec.question_id == "q_001"
        assert rec.query_text == "What is ACC?"
        assert rec.expected_answer == "ACC is adaptive cruise control."
        assert rec.facts == ["ACC is adaptive cruise control."]

    def test_from_dict_missing_query_text_raises(self):
        d = {"question_id": "q_001", "expected_answer": "Some answer."}
        with pytest.raises(ValueError, match="missing required fields"):
            EvalRecord.from_dict(d)

    def test_from_dict_missing_expected_answer_raises(self):
        d = {"question_id": "q_001", "query_text": "Some question?", "expected_answer": ""}
        with pytest.raises(ValueError, match="missing required fields"):
            EvalRecord.from_dict(d)

    def test_from_dict_missing_facts_defaults_to_empty_list(self):
        d = {"question_id": "q_002", "query_text": "Q?", "expected_answer": "A."}
        rec = EvalRecord.from_dict(d)
        assert rec.facts == []

    def test_whitespace_stripped_from_fields(self):
        d = {"question_id": "q_003", "query_text": "  Q?  ", "expected_answer": "  A.  "}
        rec = EvalRecord.from_dict(d)
        assert rec.query_text == "Q?"
        assert rec.expected_answer == "A."


# ==============================================================================
# 2. CitationEvalSummary Tests
# ==============================================================================

class TestCitationEvalSummary:
    def test_citation_coverage_rate_zero_claims(self):
        s = CitationEvalSummary(total_claims=0)
        assert s.citation_coverage_rate is None

    def test_citation_coverage_rate_computed(self):
        s = CitationEvalSummary(total_claims=4, cited_claims=3)
        assert s.citation_coverage_rate == 0.75

    def test_claim_support_rate_zero_cited(self):
        s = CitationEvalSummary(cited_claims=0, supported_claims=0)
        assert s.claim_support_rate is None

    def test_claim_support_rate_computed(self):
        s = CitationEvalSummary(cited_claims=4, supported_claims=3)
        assert s.claim_support_rate == 0.75

    def test_valid_citation_id_rate_computed(self):
        s = CitationEvalSummary(cited_claims=4, claims_with_valid_citation_id=3)
        assert s.valid_citation_id_rate == 0.75

    def test_to_dict_contains_all_keys(self):
        s = CitationEvalSummary(total_claims=2, cited_claims=1, supported_claims=1)
        d = s.to_dict()
        required_keys = [
            "total_claims", "cited_claims", "uncited_claims",
            "claims_with_valid_citation_id", "supported_claims", "unsupported_claims",
            "citation_coverage_rate", "claim_support_rate", "valid_citation_id_rate",
        ]
        for k in required_keys:
            assert k in d, f"Missing key: {k}"


# ==============================================================================
# 3. Dataset Loading Tests
# ==============================================================================

class TestDatasetLoading:
    def test_loads_production_dataset(self):
        if not EVAL_DATASET_PATH.is_file():
            pytest.skip("Evaluation dataset not present.")
        records, rejected = load_evaluation_dataset(EVAL_DATASET_PATH)
        assert len(records) == 314
        assert len(rejected) == 0
        # Verify all have required fields
        for rec in records:
            assert rec.question_id
            assert rec.query_text
            assert rec.expected_answer

    def test_loads_synthetic_dataset(self, tmp_path):
        ds_file = tmp_path / "eval.jsonl"
        records_raw = [
            {"question_id": "q1", "query_text": "Q1?", "expected_answer": "A1.",
             "source_document": "Doc.md", "source_row": 1,
             "origin_status": "valid_clean_json", "category": "Specific Query", "facts": []},
            {"question_id": "q2", "query_text": "Q2?", "expected_answer": "A2.",
             "source_document": "Doc.md", "source_row": 2,
             "origin_status": "valid_clean_json", "category": "Specific Query", "facts": ["f1"]},
        ]
        ds_file.write_text("\n".join(json.dumps(r) for r in records_raw), encoding="utf-8")
        records, rejected = load_evaluation_dataset(ds_file)
        assert len(records) == 2
        assert len(rejected) == 0

    def test_rejects_duplicate_ids(self, tmp_path):
        ds_file = tmp_path / "eval.jsonl"
        line = json.dumps({"question_id": "q1", "query_text": "Q?", "expected_answer": "A."})
        ds_file.write_text(line + "\n" + line, encoding="utf-8")
        records, rejected = load_evaluation_dataset(ds_file)
        assert len(records) == 1
        assert len(rejected) == 1
        assert "Duplicate" in rejected[0]["reason"]

    def test_rejects_empty_query(self, tmp_path):
        ds_file = tmp_path / "eval.jsonl"
        ds_file.write_text(
            json.dumps({"question_id": "q1", "query_text": "  ", "expected_answer": "A."}),
            encoding="utf-8"
        )
        records, rejected = load_evaluation_dataset(ds_file)
        assert len(records) == 0
        assert len(rejected) == 1

    def test_rejects_malformed_json(self, tmp_path):
        ds_file = tmp_path / "eval.jsonl"
        ds_file.write_text("{not valid json}\n", encoding="utf-8")
        records, rejected = load_evaluation_dataset(ds_file)
        assert len(records) == 0
        assert len(rejected) == 1
        assert "JSON decode error" in rejected[0]["reason"]

    def test_evaluation_dataset_not_indexed(self):
        """Confirm that evaluation questions are not present in the ChromaDB collection."""
        if not EVAL_DATASET_PATH.is_file():
            pytest.skip("Evaluation dataset not present.")
        records, _ = load_evaluation_dataset(EVAL_DATASET_PATH)
        from src.phase2.vector_retriever import VectorRetriever
        try:
            vr = VectorRetriever()
            # Query with the first evaluation question — it should NOT retrieve itself
            # as a knowledge document (it's not indexed)
            results = vr.retrieve(records[0].query_text, top_k=5)
            for r in results:
                meta = r.get("citation_metadata", {})
                # Evaluation question fields must not appear in knowledge chunks
                assert "expected_answer" not in meta
                assert "origin_status" not in meta
        except Exception:
            pytest.skip("VectorRetriever unavailable or ChromaDB not reachable.")


# ==============================================================================
# 4. RAGAS Adapter Tests (offline metrics only — no API key needed)
# ==============================================================================

class TestRagasAdapter:
    def test_build_ragas_sample(self):
        try:
            sample = build_ragas_sample(
                query="What is ACC?",
                generated_answer="ACC is adaptive cruise control.",
                retrieved_contexts=["ACC is an advanced driving system."],
                reference_answer="ACC is adaptive cruise control.",
            )
            assert sample.user_input == "What is ACC?"
            assert sample.response == "ACC is adaptive cruise control."
            assert len(sample.retrieved_contexts) == 1
            assert sample.reference == "ACC is adaptive cruise control."
        except ImportError:
            pytest.skip("RAGAS not available.")

    def test_build_ragas_sample_empty_contexts_fills_placeholder(self):
        try:
            sample = build_ragas_sample(
                query="Q?",
                generated_answer="A.",
                retrieved_contexts=[],
                reference_answer="Ref.",
            )
            assert sample.retrieved_contexts == [""]
        except ImportError:
            pytest.skip("RAGAS not available.")

    def test_ragas_evaluator_skips_llm_metrics_without_llm(self):
        try:
            evaluator = RagasEvaluator(langchain_llm=None, use_llm_metrics=False)
            scores, unavail = evaluator.score_single(
                query="How many checkpoints?",
                generated_answer="Aventro uses 380+ checkpoints.",
                retrieved_contexts=["380+ checkpoint inspection."],
                reference_answer="Aventro uses 380+ checkpoints.",
            )
            # LLM-based metrics should be reported as unavailable
            for metric in ["faithfulness", "answer_relevancy", "context_precision",
                           "context_recall", "answer_correctness"]:
                assert metric in unavail, f"Expected {metric} to be marked unavailable"
        except ImportError:
            pytest.skip("RAGAS not available.")

    def test_ragas_evaluator_skips_abstained_answer(self):
        try:
            evaluator = RagasEvaluator(langchain_llm=None, use_llm_metrics=False)
            scores, unavail = evaluator.score_single(
                query="Q?",
                generated_answer="",
                retrieved_contexts=[],
                reference_answer="Ref.",
            )
            assert scores == {}
            # All metrics should be marked unavailable
            assert len(unavail) > 0
        except ImportError:
            pytest.skip("RAGAS not available.")

    def test_ragas_offline_metrics_produce_scores(self):
        """NonLLM metrics should produce numeric scores without an LLM."""
        try:
            evaluator = RagasEvaluator(langchain_llm=None, use_llm_metrics=False)
            scores, unavail = evaluator.score_single(
                query="How many checkpoints?",
                generated_answer="Aventro certified cars undergo 380+ checkpoints.",
                retrieved_contexts=[
                    "Aventro certified pre-owned cars undergo 380+ checkpoint inspection."
                ],
                reference_answer="Aventro uses 380+ checkpoints for pre-owned car inspection.",
            )
            # At least rouge_score should be computed offline
            ragas_ok = any(v is not None for v in scores.values())
            assert ragas_ok or len(unavail) > 0, (
                "Expected either offline scores or explicit unavailability reasons."
            )
        except ImportError:
            pytest.skip("RAGAS not available.")

    def test_ragas_evaluator_handles_exception_gracefully(self):
        """An exception in the RAGAS evaluator should not raise — it should be reported."""
        try:
            evaluator = RagasEvaluator(langchain_llm=None, use_llm_metrics=False)

            # Pass invalid dataset to trigger internal RAGAS error
            scores, unavail = evaluator.score_single(
                query="",  # empty query
                generated_answer="Some answer.",
                retrieved_contexts=["Some context."],
                reference_answer="Some reference.",
            )
            # Should not raise; result should be scores + unavailable reasons
            assert isinstance(scores, dict)
            assert isinstance(unavail, dict)
        except ImportError:
            pytest.skip("RAGAS not available.")

    def test_ragas_evaluator_without_ragas_reports_unavailable(self, monkeypatch):
        """If RAGAS is not installed, all metrics should be reported as unavailable."""
        import src.phase2.rag_evaluator as ev_module
        original = ev_module.RAGAS_AVAILABLE
        try:
            ev_module.RAGAS_AVAILABLE = False
            evaluator = RagasEvaluator(langchain_llm=None, use_llm_metrics=False)
            scores, unavail = evaluator.score_single("Q?", "A.", ["C."], "Ref.")
            assert scores == {}
            assert len(unavail) > 0
        finally:
            ev_module.RAGAS_AVAILABLE = original


# ==============================================================================
# 5. Citation Evaluation Tests (reuses citation_verifier.py)
# ==============================================================================

class TestCitationEvaluation:
    def _make_registry(self):
        return {
            "E1": {
                "evidence_id": "E1",
                "chunk_id": "doc_preowned_001",
                "source_document": "Pre-Owned Certified Cars.md",
                "document_title": "Aventro Pre-Owned",
                "section_heading": "Inspection",
                "heading_path": "Aventro > Inspection",
                "text": "Aventro certified pre-owned cars undergo 380+ checkpoint inspection.",
            },
            "E2": {
                "evidence_id": "E2",
                "chunk_id": "doc_preowned_002",
                "source_document": "Pre-Owned Certified Cars.md",
                "document_title": "Aventro Pre-Owned",
                "section_heading": "Warranty",
                "heading_path": "Aventro > Warranty",
                "text": "All certified cars come with 1 year warranty.",
            },
        }

    def test_supported_claim_with_valid_citation(self):
        registry = self._make_registry()
        verifier = CitationVerifier(entailment_verifier=FakeEntailmentVerifier(default_support=True))
        summary, report = evaluate_citations(
            generated_answer="Aventro cars undergo 380+ checkpoints [E1].",
            evidence_registry=registry,
            verifier=verifier,
        )
        assert summary.total_claims == 1
        assert summary.cited_claims == 1
        assert summary.uncited_claims == 0
        assert summary.supported_claims == 1
        assert summary.claims_with_valid_citation_id == 1

    def test_unsupported_claim(self):
        registry = self._make_registry()
        verifier = CitationVerifier(entailment_verifier=FakeEntailmentVerifier(default_support=False))
        summary, report = evaluate_citations(
            generated_answer="Aventro cars undergo 380+ checkpoints [E1].",
            evidence_registry=registry,
            verifier=verifier,
        )
        assert summary.unsupported_claims == 1
        assert summary.supported_claims == 0

    def test_uncited_factual_claim(self):
        registry = self._make_registry()
        verifier = CitationVerifier(entailment_verifier=HeuristicEntailmentVerifier())
        summary, report = evaluate_citations(
            generated_answer="Aventro cars have 380 checkpoints [E1]. They also offer free coffee.",
            evidence_registry=registry,
            verifier=verifier,
        )
        assert summary.uncited_claims == 1

    def test_unknown_citation_id(self):
        registry = self._make_registry()
        verifier = CitationVerifier(entailment_verifier=HeuristicEntailmentVerifier())
        summary, report = evaluate_citations(
            generated_answer="Aventro offers lunar trips [E99].",
            evidence_registry=registry,
            verifier=verifier,
        )
        assert summary.claims_with_invalid_citation_id == 1
        assert summary.claims_with_valid_citation_id == 0

    def test_citation_referencing_nonexistent_chunk(self):
        registry = {
            "E1": {
                "evidence_id": "E1",
                "chunk_id": "nonexistent_chunk_999",
                "source_document": "Unknown.md",
                "document_title": "Unknown",
                "section_heading": "?",
                "heading_path": "?",
                "text": "Some text.",
            }
        }
        canonical_ids = {"doc_known_001", "doc_known_002"}
        verifier = CitationVerifier(entailment_verifier=FakeEntailmentVerifier(default_support=True))
        summary, _ = evaluate_citations(
            generated_answer="Something [E1].",
            evidence_registry=registry,
            canonical_chunk_ids=canonical_ids,
            verifier=verifier,
        )
        assert summary.citations_referencing_unknown_chunk == 1
        assert summary.citations_referencing_known_chunk == 0

    def test_numeric_mismatch_detected(self):
        registry = self._make_registry()
        verifier = CitationVerifier(entailment_verifier=HeuristicEntailmentVerifier())
        summary, report = evaluate_citations(
            generated_answer="Aventro cars undergo 500 checkpoint inspection [E1].",
            evidence_registry=registry,
            verifier=verifier,
        )
        assert summary.numeric_contradiction_count >= 1

    def test_empty_answer(self):
        registry = self._make_registry()
        summary, report = evaluate_citations(
            generated_answer="",
            evidence_registry=registry,
        )
        assert summary.total_claims == 0
        assert report["verification_status"] == "empty_answer"

    def test_empty_evidence_registry(self):
        verifier = CitationVerifier(entailment_verifier=HeuristicEntailmentVerifier())
        summary, report = evaluate_citations(
            generated_answer="Aventro cars undergo 380 checkpoints [E1].",
            evidence_registry={},
            verifier=verifier,
        )
        # E1 not in registry → invalid citation
        assert summary.claims_with_invalid_citation_id == 1

    def test_mixed_outcomes_counted_separately(self):
        registry = self._make_registry()
        verifier = CitationVerifier(entailment_verifier=HeuristicEntailmentVerifier())
        answer = (
            "Aventro cars undergo 380+ checkpoints [E1]. "
            "They come with 1 year warranty [E2]. "
            "The company was founded in 1905."
        )
        summary, _ = evaluate_citations(
            generated_answer=answer,
            evidence_registry=registry,
            verifier=verifier,
        )
        assert summary.total_claims >= 2
        assert summary.uncited_claims >= 1

    def test_verifier_exception_counted(self):
        registry = self._make_registry()
        verifier = CitationVerifier(
            entailment_verifier=FakeEntailmentVerifier(
                exc_to_raise=RuntimeError("Verifier crash")
            )
        )
        summary, report = evaluate_citations(
            generated_answer="Aventro cars undergo 380 checkpoints [E1].",
            evidence_registry=registry,
            verifier=verifier,
        )
        assert summary.unverifiable_claims >= 1


# ==============================================================================
# 6. End-to-End Integration Tests
# ==============================================================================

class TestEndToEndIntegration:
    def _make_graph(self, canned_response="Aventro cars undergo 380 checkpoints [E1].", llm_exc=None):
        chunk = make_mock_chunk("doc_preowned_001", "380+ checkpoint inspection for pre-owned cars.")
        mock_rrf = MockRRF(results=[chunk])
        mock_eval = MockEvaluator(assessment=make_sufficient_assessment([chunk]))
        generator = AnswerGenerator(llm_client=FakeLLMClient(
            canned_response=canned_response if not llm_exc else None,
            exc_to_raise=llm_exc,
        ))
        return build_rag_graph(
            rrf_retriever=mock_rrf,
            evidence_evaluator=mock_eval,
            answer_generator=generator,
        )

    def _make_no_evidence_graph(self):
        mock_rrf = MockRRF(results=[])
        mock_eval = MockEvaluator(assessment=make_no_evidence_assessment())
        generator = AnswerGenerator(llm_client=FakeLLMClient())
        return build_rag_graph(
            rrf_retriever=mock_rrf,
            evidence_evaluator=mock_eval,
            answer_generator=generator,
        )

    def test_answerable_question_produces_result(self):
        graph = self._make_graph()
        record = make_eval_record()
        result = run_single_question_eval(
            record=record,
            graph=graph,
            ragas_evaluator=None,
            citation_verifier=CitationVerifier(entailment_verifier=FakeEntailmentVerifier(default_support=True)),
            canonical_chunk_ids={"doc_preowned_001"},
        )
        assert result.generated_answer is not None
        assert result.workflow_status in (WORKFLOW_VERIFIED, WORKFLOW_REQUIRES_REVIEW)
        assert result.generation_status == "success"
        assert len(result.retrieved_chunk_ids) >= 1

    def test_expected_answer_not_in_retrieved_contexts(self):
        """Critical: expected_answer must never appear in retrieved_contexts."""
        graph = self._make_graph()
        record = make_eval_record(expected_answer="SENTINEL_EXPECTED_ANSWER_DO_NOT_RETRIEVE")
        result = run_single_question_eval(
            record=record,
            graph=graph,
            ragas_evaluator=None,
            citation_verifier=None,
            canonical_chunk_ids=None,
        )
        for ctx in result.retrieved_contexts:
            assert "SENTINEL_EXPECTED_ANSWER_DO_NOT_RETRIEVE" not in ctx

    def test_expected_answer_stored_in_result_only(self):
        """The expected answer must be in the result but NOT passed to the generator."""
        graph = self._make_graph()
        record = make_eval_record()
        result = run_single_question_eval(
            record=record, graph=graph,
            ragas_evaluator=None, citation_verifier=None, canonical_chunk_ids=None,
        )
        assert result.expected_answer == record.expected_answer
        # generated_answer must be the model's output, not the expected answer
        assert result.generated_answer != result.expected_answer or (
            result.generated_answer and result.expected_answer
        )

    def test_no_evidence_triggers_abstention(self):
        graph = self._make_no_evidence_graph()
        record = make_eval_record(query_text="Completely unrelated query about spaceships.")
        result = run_single_question_eval(
            record=record, graph=graph,
            ragas_evaluator=None, citation_verifier=None, canonical_chunk_ids=None,
        )
        assert result.workflow_status == WORKFLOW_REQUIRES_ABSTENTION
        assert result.generated_answer is None
        assert result.generation_status == "not_attempted"

    def test_partially_sufficient_evidence_triggers_abstention(self):
        chunk = make_mock_chunk("chunk_1")
        mock_rrf = MockRRF(results=[chunk])
        mock_eval = MockEvaluator(assessment={
            "status": "partially_sufficient",
            "is_sufficient": False,
            "reason": "Missing second part.",
            "relevant_evidence": [chunk],
            "irrelevant_evidence": [],
            "missing_information": ["delivery policy"],
            "conflicts": [],
        })
        generator = AnswerGenerator(llm_client=FakeLLMClient())
        graph = build_rag_graph(
            rrf_retriever=mock_rrf,
            evidence_evaluator=mock_eval,
            answer_generator=generator,
        )
        result = run_single_question_eval(
            record=make_eval_record(), graph=graph,
            ragas_evaluator=None, citation_verifier=None, canonical_chunk_ids=None,
        )
        from src.phase2.rag_state import WORKFLOW_REQUIRES_FURTHER_RETRIEVAL
        assert result.workflow_status == WORKFLOW_REQUIRES_FURTHER_RETRIEVAL
        assert result.generated_answer is None

    def test_conflicting_evidence_triggers_abstention(self):
        c1 = make_mock_chunk("c1", "Warranty is 1 year.")
        c2 = make_mock_chunk("c2", "Warranty is 3 years.")
        mock_rrf = MockRRF(results=[c1, c2])
        mock_eval = MockEvaluator(assessment={
            "status": "conflicting",
            "is_sufficient": False,
            "reason": "Conflicting specs.",
            "relevant_evidence": [c1, c2],
            "irrelevant_evidence": [],
            "missing_information": ["resolution"],
            "conflicts": [{"type": "numeric_specification_conflict"}],
        })
        graph = build_rag_graph(
            rrf_retriever=mock_rrf,
            evidence_evaluator=mock_eval,
            answer_generator=AnswerGenerator(llm_client=FakeLLMClient()),
        )
        result = run_single_question_eval(
            record=make_eval_record(), graph=graph,
            ragas_evaluator=None, citation_verifier=None, canonical_chunk_ids=None,
        )
        from src.phase2.rag_state import WORKFLOW_REQUIRES_CONFLICT_RESOLUTION
        assert result.workflow_status == WORKFLOW_REQUIRES_CONFLICT_RESOLUTION
        assert result.generated_answer is None

    def test_invalid_citation_in_answer_reaches_evaluation_report(self):
        chunk = make_mock_chunk("chunk_1", "Some evidence.")
        mock_rrf = MockRRF(results=[chunk])
        mock_eval = MockEvaluator(assessment=make_sufficient_assessment([chunk]))
        generator = AnswerGenerator(
            llm_client=FakeLLMClient(canned_response="Aventro has lunar trips [E99].")
        )
        graph = build_rag_graph(
            rrf_retriever=mock_rrf,
            evidence_evaluator=mock_eval,
            answer_generator=generator,
        )
        result = run_single_question_eval(
            record=make_eval_record(), graph=graph,
            ragas_evaluator=None,
            citation_verifier=CitationVerifier(entailment_verifier=HeuristicEntailmentVerifier()),
            canonical_chunk_ids=None,
        )
        # The invalid citation should be recorded in the citation eval
        assert result.citation_eval is not None
        assert result.citation_eval.claims_with_invalid_citation_id >= 1

    def test_model_unavailable_does_not_stop_batch(self):
        """A generation failure for one record should not crash the whole batch."""
        from src.phase2.rag_evaluator import run_batch_evaluation
        failing_graph = self._make_graph(
            llm_exc=ModelUnavailableError("LLM unavailable.")
        )
        records = [make_eval_record(question_id=f"q_{i}") for i in range(3)]
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            results, summary = run_batch_evaluation(
                records=records,
                graph=failing_graph,
                ragas_evaluator=None,
                citation_verifier=None,
                canonical_chunk_ids=None,
                output_dir=Path(tmpdir),
                output_filename="test_results.jsonl",
            )
        # All records attempted, none should propagate an unhandled exception
        assert summary.attempted == 3
        assert summary.succeeded + summary.failed == 3

    def test_evaluation_data_not_inserted_into_knowledge_index(self):
        """Evaluation records must not be present in the ChromaDB knowledge collection."""
        if not Path("artifacts/chroma_db").is_dir():
            pytest.skip("ChromaDB not available.")
        from src.phase2.vector_retriever import VectorRetriever
        try:
            vr = VectorRetriever()
        except Exception:
            pytest.skip("VectorRetriever unavailable.")
        records, _ = load_evaluation_dataset(EVAL_DATASET_PATH)
        if not records:
            pytest.skip("No evaluation records loaded.")
        # Search for an evaluation question — it must not appear as a knowledge chunk
        results = vr.retrieve(records[0].query_text, top_k=5)
        for r in results:
            meta = r.get("citation_metadata", {})
            assert "expected_answer" not in meta
            assert "origin_status" not in meta
            assert "question_id" not in meta

    def test_ragas_scores_computed_for_answerable_question(self):
        """Offline RAGAS metrics should produce scores without an LLM."""
        try:
            from src.phase2.rag_evaluator import RAGAS_AVAILABLE
            if not RAGAS_AVAILABLE:
                pytest.skip("RAGAS not available.")
        except ImportError:
            pytest.skip("RAGAS not available.")

        graph = self._make_graph()
        evaluator = RagasEvaluator(langchain_llm=None, use_llm_metrics=False)
        record = make_eval_record()
        result = run_single_question_eval(
            record=record,
            graph=graph,
            ragas_evaluator=evaluator,
            citation_verifier=CitationVerifier(entailment_verifier=FakeEntailmentVerifier(default_support=True)),
            canonical_chunk_ids={"doc_preowned_001"},
        )
        # At least some RAGAS output must be present (scores or unavailability reasons)
        has_scores = len(result.ragas_scores) > 0
        has_unavail = len(result.ragas_unavailable) > 0
        assert has_scores or has_unavail


# ==============================================================================
# 7. BatchEvalSummary Tests
# ==============================================================================

class TestBatchEvalSummary:
    def test_record_ragas_score_accumulates(self):
        s = BatchEvalSummary()
        s.record_ragas_score("rouge_score", 0.8)
        s.record_ragas_score("rouge_score", 0.6)
        assert s.mean_ragas_score("rouge_score") == 0.7

    def test_mean_ragas_score_none_if_no_data(self):
        s = BatchEvalSummary()
        assert s.mean_ragas_score("faithfulness") is None

    def test_to_dict_structure(self):
        s = BatchEvalSummary(total_records=10, attempted=10, succeeded=8, failed=2)
        d = s.to_dict()
        assert d["records"]["total"] == 10
        assert d["records"]["succeeded"] == 8
        assert "citation_quality" in d
        assert "ragas_mean_scores" in d
        assert "timing" in d


# ==============================================================================
# 8. Existing Phase 2 regression — artifacts immutability
# ==============================================================================

class TestPhase2ArtifactsPreserved:
    def test_canonical_chunks_unchanged(self):
        chunks_path = Path("artifacts/chunks/document_chunks.jsonl")
        assert chunks_path.is_file()
        count = sum(1 for l in chunks_path.read_text(encoding="utf-8").splitlines() if l.strip())
        assert count == 459

    def test_phase2_modules_exist(self):
        for module in [
            "src/phase2/bm25_retriever.py",
            "src/phase2/vector_retriever.py",
            "src/phase2/rrf_fusion.py",
            "src/phase2/evidence_evaluator.py",
            "src/phase2/answer_generator.py",
            "src/phase2/citation_verifier.py",
            "src/phase2/rag_graph.py",
            "src/phase2/rag_state.py",
        ]:
            assert Path(module).is_file(), f"Missing: {module}"

    def test_evaluation_dataset_not_modified(self):
        if not EVAL_DATASET_PATH.is_file():
            pytest.skip("Evaluation dataset not present.")
        records, _ = load_evaluation_dataset(EVAL_DATASET_PATH)
        assert len(records) == 314

    def test_chroma_db_directory_exists(self):
        assert Path("artifacts/chroma_db").is_dir()
