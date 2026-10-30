"""
Phase 2, Step 14: RAG Evaluation Framework
Implements RAGAS metric scoring, extended citation validation, and
batch evaluation using the existing LangGraph RAG pipeline.

Key design decisions:
- RAGAS version: 0.2.15 (EvaluationDataset / SingleTurnSample API)
- LLM-based RAGAS metrics require a LangChain-compatible LLM provider.
  Without one, they are reported as UNAVAILABLE with an explicit reason.
- NonLLM-based metrics (NonLLMContextPrecisionWithReference, NonLLMContextRecall,
  RougeScore, ExactMatch) run fully offline without any API key.
- Citation validation reuses src.phase2.citation_verifier — no duplicate implementation.
- The expected_answer is stored in evaluation results ONLY; it is never passed
  to the AnswerGenerator or included in the generation prompt.
- Evaluation questions are NEVER inserted into the ChromaDB knowledge index.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RAGAS imports — all guarded so the module is importable without RAGAS
# ---------------------------------------------------------------------------
try:
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import (
        NonLLMContextPrecisionWithReference,
        NonLLMContextRecall,
        RougeScore,
    )
    from ragas.metrics._string import ExactMatch, DistanceMeasure
    from ragas import evaluate as ragas_evaluate
    from ragas.run_config import RunConfig

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("RAGAS is not installed. RAGAS metrics will be unavailable.")

from src.phase2.evaluation_schemas import (
    BatchEvalSummary,
    CitationEvalSummary,
    EvalRecord,
    QuestionEvalResult,
)
from src.phase2.citation_verifier import CitationVerifier, HeuristicEntailmentVerifier
from src.phase2.rag_graph import build_rag_graph
from src.phase2.answer_generator import AnswerGenerator, FakeLLMClient
from src.phase2.rag_state import (
    WORKFLOW_VERIFIED,
    WORKFLOW_REQUIRES_REVIEW,
    WORKFLOW_COMPLETED,
    WORKFLOW_REQUIRES_ABSTENTION,
    WORKFLOW_REQUIRES_FURTHER_RETRIEVAL,
    WORKFLOW_REQUIRES_CONFLICT_RESOLUTION,
    WORKFLOW_VALIDATION_FAILED,
    WORKFLOW_ERROR,
)

ABSTENTION_STATUSES = {
    WORKFLOW_REQUIRES_ABSTENTION,
    WORKFLOW_REQUIRES_FURTHER_RETRIEVAL,
    WORKFLOW_REQUIRES_CONFLICT_RESOLUTION,
    WORKFLOW_VALIDATION_FAILED,
}

EVAL_DATASET_PATH = Path("artifacts/extracted/evaluation_questions.jsonl")
EVAL_OUTPUT_DIR = Path("artifacts/evaluation")

# ---------------------------------------------------------------------------
# Dataset loading and validation
# ---------------------------------------------------------------------------

def load_evaluation_dataset(
    path: Path = EVAL_DATASET_PATH,
) -> Tuple[List[EvalRecord], List[Dict[str, Any]]]:
    """
    Load and validate evaluation_questions.jsonl.
    Returns (valid_records, rejected_records).
    Does NOT insert records into the knowledge index.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found at {path}")

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    valid: List[EvalRecord] = []
    rejected: List[Dict[str, Any]] = []

    seen_ids = set()
    for line_no, line in enumerate(raw_lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            rejected.append({"line": line_no, "reason": f"JSON decode error: {e}", "raw": line[:120]})
            continue

        qid = raw.get("question_id", f"__line_{line_no}")

        # Duplicate check
        if qid in seen_ids:
            rejected.append({"line": line_no, "question_id": qid, "reason": "Duplicate question_id"})
            continue
        seen_ids.add(qid)

        # Required field check
        if not raw.get("query_text", "").strip():
            rejected.append({"line": line_no, "question_id": qid, "reason": "Empty query_text"})
            continue
        if not raw.get("expected_answer", "").strip():
            rejected.append({"line": line_no, "question_id": qid, "reason": "Empty expected_answer"})
            continue

        try:
            rec = EvalRecord.from_dict(raw)
            valid.append(rec)
        except (ValueError, KeyError) as e:
            rejected.append({"line": line_no, "question_id": qid, "reason": str(e)})

    logger.info(
        "Evaluation dataset loaded: %d valid, %d rejected from %s",
        len(valid), len(rejected), path,
    )
    return valid, rejected


# ---------------------------------------------------------------------------
# Citation evaluation — reuses citation_verifier.py
# ---------------------------------------------------------------------------

def evaluate_citations(
    generated_answer: str,
    evidence_registry: Dict[str, Dict[str, Any]],
    canonical_chunk_ids: Optional[set] = None,
    verifier: Optional[CitationVerifier] = None,
) -> Tuple[CitationEvalSummary, Dict[str, Any]]:
    """
    Run claim-level citation evaluation using the existing CitationVerifier.
    Returns (CitationEvalSummary, raw_verification_report).

    canonical_chunk_ids: optional set of known chunk IDs for existence checks.
    """
    if verifier is None:
        verifier = CitationVerifier(entailment_verifier=HeuristicEntailmentVerifier())

    report = verifier.verify(answer=generated_answer, evidence_registry=evidence_registry)
    summary = CitationEvalSummary()

    claim_results = report.get("claim_results", [])
    summary.total_claims = len(claim_results)
    summary.cited_claims = report.get("supported_claim_count", 0) + report.get("unsupported_claim_count", 0) + report.get("invalid_citation_count", 0)
    summary.uncited_claims = report.get("uncited_claim_count", 0)
    summary.claims_with_invalid_citation_id = report.get("invalid_citation_count", 0)
    summary.supported_claims = report.get("supported_claim_count", 0)
    summary.unsupported_claims = report.get("unsupported_claim_count", 0)
    summary.claims_with_valid_citation_id = summary.cited_claims - summary.claims_with_invalid_citation_id
    summary.unverifiable_claims = sum(
        1 for c in claim_results if c.get("verification_status") == "verification_error"
    )

    # Numeric contradiction count from explanations
    summary.numeric_contradiction_count = sum(
        1 for c in claim_results
        if "Numeric contradiction" in c.get("explanation", "")
    )

    # Chunk existence checks (if canonical_chunk_ids provided)
    if canonical_chunk_ids is not None:
        for c in claim_results:
            for rc in c.get("resolved_citations", []):
                cid = rc.get("chunk_id", "")
                if cid:
                    if cid in canonical_chunk_ids:
                        summary.citations_referencing_known_chunk += 1
                    else:
                        summary.citations_referencing_unknown_chunk += 1

    return summary, report


# ---------------------------------------------------------------------------
# RAGAS adapter — maps pipeline output to RAGAS SingleTurnSample
# ---------------------------------------------------------------------------

def build_ragas_sample(
    query: str,
    generated_answer: str,
    retrieved_contexts: List[str],
    reference_answer: str,
) -> "SingleTurnSample":
    """
    Map RAG pipeline outputs to a RAGAS SingleTurnSample.

    Fields:
    - user_input: the user's original query
    - retrieved_contexts: list of retrieved chunk texts (NOT the reference answer)
    - response: the generated answer
    - reference: the expected answer (ground truth for metrics that need it)

    The reference_answer is used ONLY as a RAGAS evaluation input;
    it is never passed to the AnswerGenerator.
    """
    if not RAGAS_AVAILABLE:
        raise ImportError("RAGAS is not installed.")

    return SingleTurnSample(
        user_input=query,
        retrieved_contexts=retrieved_contexts if retrieved_contexts else [""],
        response=generated_answer,
        reference=reference_answer,
    )


class RagasEvaluator:
    """
    Adapter that runs RAGAS metrics on pipeline outputs.
    Supports:
    - NonLLM metrics: run fully offline (context precision w/ reference, context recall, rouge).
    - LLM metrics: require a configured LangChain-compatible LLM (faithfulness, answer_relevancy, etc.)

    If an LLM is not configured, LLM-based metrics are skipped and reported as UNAVAILABLE.
    Unit tests use non-LLM metrics only — no API keys required.
    """

    # Non-LLM metrics that run fully offline
    OFFLINE_METRICS_NAMES = [
        "non_llm_context_precision",
        "non_llm_context_recall",
        "rouge_score",
    ]

    # LLM-based metrics — need external model
    LLM_METRICS_NAMES = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
    ]

    def __init__(
        self,
        langchain_llm: Optional[Any] = None,
        langchain_embeddings: Optional[Any] = None,
        use_llm_metrics: bool = False,
        run_config: Optional[Any] = None,
    ) -> None:
        self.langchain_llm = langchain_llm
        self.langchain_embeddings = langchain_embeddings
        self.use_llm_metrics = use_llm_metrics and langchain_llm is not None
        self._run_config = run_config

        if not RAGAS_AVAILABLE:
            logger.warning("RAGAS unavailable — all RAGAS metrics will be skipped.")

    def _build_offline_metrics(self) -> List[Any]:
        """Return a list of non-LLM RAGAS metric instances."""
        if not RAGAS_AVAILABLE:
            return []
        return [
            NonLLMContextPrecisionWithReference(),
            NonLLMContextRecall(),
            RougeScore(),
        ]

    def _build_llm_metrics(self) -> List[Any]:
        """Return LLM-based metrics if configured; empty list otherwise."""
        if not RAGAS_AVAILABLE or not self.use_llm_metrics:
            return []
        from ragas.metrics import faithfulness, answer_relevancy, answer_correctness
        return [faithfulness, answer_relevancy, answer_correctness]

    def score_single(
        self,
        query: str,
        generated_answer: str,
        retrieved_contexts: List[str],
        reference_answer: str,
    ) -> Tuple[Dict[str, Optional[float]], Dict[str, str]]:
        """
        Compute RAGAS metrics for a single evaluation sample.
        Returns:
          - scores: {metric_name: float_or_None}
          - unavailable: {metric_name: reason_string}
        """
        if not RAGAS_AVAILABLE:
            unavail = {m: "RAGAS package not installed." for m in self.OFFLINE_METRICS_NAMES + self.LLM_METRICS_NAMES}
            return {}, unavail

        scores: Dict[str, Optional[float]] = {}
        unavailable: Dict[str, str] = {}

        # Mark LLM metrics as unavailable if no LLM is configured
        if not self.use_llm_metrics:
            for m in self.LLM_METRICS_NAMES:
                unavailable[m] = (
                    "LLM-based metric requires a configured LangChain LLM provider. "
                    "Set use_llm_metrics=True and supply langchain_llm to RagasEvaluator."
                )

        # Build metrics to actually run
        active_metrics = self._build_offline_metrics()
        if self.use_llm_metrics:
            active_metrics.extend(self._build_llm_metrics())

        if not active_metrics:
            unavailable["all_metrics"] = "No metrics could be initialized."
            return scores, unavailable

        # Skip RAGAS scoring if answer was not generated (abstention)
        if not generated_answer or not generated_answer.strip():
            for m in active_metrics:
                unavailable[getattr(m, "name", str(m))] = "Answer was not generated (abstention)."
            return scores, unavailable

        # Skip non-LLM context metrics if no contexts available
        if not retrieved_contexts or all(not c.strip() for c in retrieved_contexts):
            for m in ["non_llm_context_precision", "non_llm_context_recall"]:
                unavailable[m] = "No retrieved contexts available for context metric computation."
            active_metrics = [
                m for m in active_metrics
                if getattr(m, "name", "") not in ("non_llm_context_precision", "non_llm_context_recall")
            ]

        try:
            sample = build_ragas_sample(
                query=query,
                generated_answer=generated_answer,
                retrieved_contexts=retrieved_contexts,
                reference_answer=reference_answer,
            )
            dataset = EvaluationDataset(samples=[sample])

            kwargs: Dict[str, Any] = {
                "dataset": dataset,
                "metrics": active_metrics,
                "show_progress": False,
                "raise_exceptions": False,
            }
            if self.langchain_llm is not None:
                kwargs["llm"] = self.langchain_llm
            if self.langchain_embeddings is not None:
                kwargs["embeddings"] = self.langchain_embeddings
            if self._run_config is not None:
                kwargs["run_config"] = self._run_config

            result = ragas_evaluate(**kwargs)
            result_df = result.to_pandas()
            if len(result_df) > 0:
                row = result_df.iloc[0].to_dict()
                for m in active_metrics:
                    mname = getattr(m, "name", str(m))
                    val = row.get(mname)
                    if val is None or (isinstance(val, float) and val != val):  # NaN check
                        unavailable[mname] = "Metric returned None or NaN for this sample."
                    else:
                        scores[mname] = float(val)

        except Exception as exc:
            logger.warning("RAGAS evaluation failed for sample: %s", exc)
            for m in active_metrics:
                mname = getattr(m, "name", str(m))
                if mname not in scores:
                    unavailable[mname] = f"Evaluator exception: {exc}"

        return scores, unavailable


# ---------------------------------------------------------------------------
# Single-question evaluator
# ---------------------------------------------------------------------------

def run_single_question_eval(
    record: EvalRecord,
    graph: Any,
    ragas_evaluator: Optional[RagasEvaluator],
    citation_verifier: Optional[CitationVerifier],
    canonical_chunk_ids: Optional[set],
    top_k: int = 5,
) -> QuestionEvalResult:
    """
    Run the full pipeline for one evaluation question and compute evaluation metrics.

    IMPORTANT: record.expected_answer is NEVER passed to the graph or AnswerGenerator.
    It is only used in RAGAS metric computation after generation.
    """
    result = QuestionEvalResult(
        question_id=record.question_id,
        query_text=record.query_text,
        expected_answer=record.expected_answer,
        source_document=record.source_document,
        category=record.category,
        origin_status=record.origin_status,
    )

    t0 = time.monotonic()

    try:
        # ----------------------------------------------------------------
        # 1. Run the RAG pipeline (expected_answer is NOT passed here)
        # ----------------------------------------------------------------
        state = graph.invoke({"query": record.query_text, "trace": []})

        # 2. Extract retrieval metadata
        retrieved = state.get("retrieved_results", [])
        result.retrieved_chunk_ids = [r.get("chunk_id", "") for r in retrieved]
        result.retrieved_sources = [
            r.get("citation_metadata", {}).get("source_document", "") or r.get("source_document", "")
            for r in retrieved
        ]
        result.retrieval_ranks = [r.get("rrf_rank", i + 1) for i, r in enumerate(retrieved)]
        result.retrieval_rrf_scores = [float(r.get("rrf_score", 0.0)) for r in retrieved]
        result.retrieved_contexts = [r.get("text", "") for r in retrieved]

        # 3. Evidence and workflow status
        assessment = state.get("evidence_assessment") or {}
        result.evidence_status = assessment.get("status")
        result.is_sufficient = assessment.get("is_sufficient")
        result.workflow_status = state.get("workflow_status")
        result.error = state.get("error") or state.get("generation_error")

        # 4. Generation outcome
        result.generated_answer = state.get("generated_answer")
        result.generation_status = state.get("generation_status")
        result.answer_citations = state.get("answer_citations", [])
        result.is_verified = state.get("is_verified")
        result.verification_status = state.get("verification_status")
        result.verification_report = state.get("verification_report")

        # Abstention reason
        if result.workflow_status in ABSTENTION_STATUSES:
            result.abstention_reason = assessment.get("reason") or result.error

        # ----------------------------------------------------------------
        # 5. Citation evaluation (if answer was generated)
        # ----------------------------------------------------------------
        if result.generated_answer:
            evidence_registry = state.get("evidence_registry") or {}
            try:
                cit_summary, cit_report = evaluate_citations(
                    generated_answer=result.generated_answer,
                    evidence_registry=evidence_registry,
                    canonical_chunk_ids=canonical_chunk_ids,
                    verifier=citation_verifier,
                )
                result.citation_eval = cit_summary
                # If pipeline didn't run verify_citations, store the standalone report
                if result.verification_report is None:
                    result.verification_report = cit_report
            except Exception as exc:
                logger.warning("Citation evaluation failed for %s: %s", record.question_id, exc)
                result.eval_error = f"Citation eval error: {exc}"

        # ----------------------------------------------------------------
        # 6. RAGAS metrics
        # ----------------------------------------------------------------
        if ragas_evaluator is not None and result.generated_answer:
            try:
                ragas_scores, ragas_unavail = ragas_evaluator.score_single(
                    query=record.query_text,
                    generated_answer=result.generated_answer,
                    retrieved_contexts=result.retrieved_contexts,
                    reference_answer=record.expected_answer,  # ground truth for RAGAS only
                )
                result.ragas_scores = ragas_scores
                result.ragas_unavailable = ragas_unavail
            except Exception as exc:
                logger.warning("RAGAS scoring failed for %s: %s", record.question_id, exc)
                result.ragas_unavailable["all"] = f"RAGAS scoring exception: {exc}"
        elif ragas_evaluator is not None and not result.generated_answer:
            result.ragas_unavailable["all"] = "RAGAS skipped: answer not generated (abstention or error)."

    except Exception as exc:
        logger.error("Pipeline execution failed for %s: %s", record.question_id, exc)
        result.eval_error = f"Pipeline error: {exc}"
        result.workflow_status = WORKFLOW_ERROR

    result.duration_seconds = round(time.monotonic() - t0, 3)
    return result


# ---------------------------------------------------------------------------
# Batch evaluation runner
# ---------------------------------------------------------------------------

def run_batch_evaluation(
    records: List[EvalRecord],
    graph: Optional[Any] = None,
    ragas_evaluator: Optional[RagasEvaluator] = None,
    citation_verifier: Optional[CitationVerifier] = None,
    canonical_chunk_ids: Optional[set] = None,
    top_k: int = 5,
    max_records: Optional[int] = None,
    output_dir: Path = EVAL_OUTPUT_DIR,
    output_filename: str = "evaluation_results.jsonl",
    smoke_label: str = "",
) -> Tuple[List[QuestionEvalResult], BatchEvalSummary]:
    """
    Run evaluation across a list of EvalRecords.
    Saves per-question JSONL results to output_dir.
    Returns (results, summary).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename

    if max_records is not None:
        records = records[:max_records]

    summary = BatchEvalSummary(total_records=len(records))
    results: List[QuestionEvalResult] = []

    with open(output_path, "w", encoding="utf-8") as fout:
        for i, record in enumerate(records):
            summary.attempted += 1
            logger.info(
                "[%d/%d] Evaluating: %s — %s",
                i + 1, len(records), record.question_id, record.query_text[:60]
            )

            qr = run_single_question_eval(
                record=record,
                graph=graph,
                ragas_evaluator=ragas_evaluator,
                citation_verifier=citation_verifier,
                canonical_chunk_ids=canonical_chunk_ids,
                top_k=top_k,
            )
            results.append(qr)

            # Update summary counts
            ws = qr.workflow_status or ""
            summary.workflow_status_counts[ws] = summary.workflow_status_counts.get(ws, 0) + 1

            es = qr.evidence_status or "unknown"
            summary.evidence_status_counts[es] = summary.evidence_status_counts.get(es, 0) + 1

            if qr.eval_error and not qr.generated_answer:
                summary.failed += 1
            else:
                summary.succeeded += 1

            if qr.generated_answer:
                summary.generated_answers += 1
            elif ws in ABSTENTION_STATUSES:
                summary.abstentions += 1
            elif ws == WORKFLOW_ERROR:
                summary.generation_errors += 1

            if qr.is_verified:
                summary.verified_answers += 1
            elif qr.verification_status == "requires_review":
                summary.requires_review_answers += 1

            # Aggregate RAGAS scores
            for metric, score in qr.ragas_scores.items():
                summary.record_ragas_score(metric, score)
            for metric, reason in qr.ragas_unavailable.items():
                summary.ragas_unavailable_reasons.setdefault(metric, reason)

            # Aggregate citation stats
            if qr.citation_eval:
                ce = qr.citation_eval
                summary.total_claims_evaluated += ce.total_claims
                summary.total_cited_claims += ce.cited_claims
                summary.total_uncited_claims += ce.uncited_claims
                summary.total_supported_claims += ce.supported_claims
                summary.total_unsupported_claims += ce.unsupported_claims
                summary.total_invalid_citation_ids += ce.claims_with_invalid_citation_id
                summary.total_unverifiable_claims += ce.unverifiable_claims

            summary.total_duration_seconds += qr.duration_seconds or 0.0

            # Save result to JSONL
            fout.write(json.dumps(qr.to_dict(), ensure_ascii=False) + "\n")
            fout.flush()

    # Write summary JSON
    summary_path = output_dir / output_filename.replace(".jsonl", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(
        "Batch evaluation complete: %d/%d succeeded, %d failed. Results at %s",
        summary.succeeded, summary.attempted, summary.failed, output_path,
    )
    return results, summary


# ---------------------------------------------------------------------------
# Canonical chunk ID loader (for existence checks in citation validation)
# ---------------------------------------------------------------------------

def load_canonical_chunk_ids(chunks_path: Path = Path("artifacts/chunks/document_chunks.jsonl")) -> set:
    """Load chunk IDs from the canonical knowledge base."""
    if not chunks_path.is_file():
        return set()
    ids = set()
    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            cid = obj.get("chunk_id")
            if cid:
                ids.add(cid)
        except json.JSONDecodeError:
            pass
    return ids
