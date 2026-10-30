"""
Phase 2, Step 14: Evaluation Schemas
Defines typed data structures used across the RAG evaluation framework:
evaluation records, citation validation summaries, RAGAS sample mappings,
and batch evaluation results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Evaluation Record — one entry from the evaluation dataset
# ---------------------------------------------------------------------------

@dataclass
class EvalRecord:
    """
    A single entry from artifacts/extracted/evaluation_questions.jsonl.
    Fields map directly to the validated schema.
    """
    question_id: str
    source_document: str
    source_row: int
    origin_status: str
    category: str
    query_text: str
    expected_answer: str
    facts: List[str]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvalRecord":
        missing = [k for k in ("question_id", "query_text", "expected_answer") if not d.get(k)]
        if missing:
            raise ValueError(f"EvalRecord missing required fields: {missing} in record {d.get('question_id', '?')!r}")
        return cls(
            question_id=d["question_id"],
            source_document=d.get("source_document", ""),
            source_row=int(d.get("source_row", 0)),
            origin_status=d.get("origin_status", "unknown"),
            category=d.get("category", ""),
            query_text=d["query_text"].strip(),
            expected_answer=d["expected_answer"].strip(),
            facts=d.get("facts") or [],
        )


# ---------------------------------------------------------------------------
# Per-Question Evaluation Result
# ---------------------------------------------------------------------------

@dataclass
class QuestionEvalResult:
    """
    Stores the full pipeline output and evaluation metrics for one evaluation question.
    The expected_answer is stored only here and is NEVER passed to the generation model.
    """
    question_id: str
    query_text: str
    expected_answer: str          # ground truth only — not sent to generation
    source_document: str
    category: str
    origin_status: str

    # --- Retrieval ---
    retrieved_chunk_ids: List[str] = field(default_factory=list)
    retrieved_sources: List[str] = field(default_factory=list)
    retrieval_ranks: List[int] = field(default_factory=list)
    retrieval_rrf_scores: List[float] = field(default_factory=list)
    retrieved_contexts: List[str] = field(default_factory=list)   # raw chunk texts for RAGAS

    # --- Evidence Assessment ---
    evidence_status: Optional[str] = None   # sufficient/insufficient/no_evidence/...
    is_sufficient: Optional[bool] = None

    # --- Generation ---
    workflow_status: Optional[str] = None
    generated_answer: Optional[str] = None
    generation_status: Optional[str] = None
    answer_citations: List[Dict[str, Any]] = field(default_factory=list)

    # --- Citation Verification (Step 13) ---
    verification_report: Optional[Dict[str, Any]] = None
    is_verified: Optional[bool] = None
    verification_status: Optional[str] = None

    # --- Citation Evaluation Counts ---
    citation_eval: Optional["CitationEvalSummary"] = None

    # --- RAGAS Metrics ---
    ragas_scores: Dict[str, Optional[float]] = field(default_factory=dict)
    ragas_unavailable: Dict[str, str] = field(default_factory=dict)

    # --- Execution metadata ---
    abstention_reason: Optional[str] = None
    error: Optional[str] = None
    eval_error: Optional[str] = None
    duration_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "question_id": self.question_id,
            "query_text": self.query_text,
            "expected_answer": self.expected_answer,
            "source_document": self.source_document,
            "category": self.category,
            "origin_status": self.origin_status,
            "retrieval": {
                "chunk_ids": self.retrieved_chunk_ids,
                "sources": self.retrieved_sources,
                "ranks": self.retrieval_ranks,
                "rrf_scores": self.retrieval_rrf_scores,
            },
            "evidence_status": self.evidence_status,
            "is_sufficient": self.is_sufficient,
            "workflow_status": self.workflow_status,
            "generated_answer": self.generated_answer,
            "generation_status": self.generation_status,
            "answer_citations": self.answer_citations,
            "citation_verification": {
                "is_verified": self.is_verified,
                "verification_status": self.verification_status,
                "report_summary": _summarise_verification_report(self.verification_report),
            },
            "citation_eval": self.citation_eval.to_dict() if self.citation_eval else None,
            "ragas_scores": self.ragas_scores,
            "ragas_unavailable": self.ragas_unavailable,
            "abstention_reason": self.abstention_reason,
            "error": self.error,
            "eval_error": self.eval_error,
            "duration_seconds": self.duration_seconds,
        }
        return d


def _summarise_verification_report(report: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not report:
        return None
    return {
        "overall_status": report.get("overall_status"),
        "supported_claim_count": report.get("supported_claim_count"),
        "unsupported_claim_count": report.get("unsupported_claim_count"),
        "uncited_claim_count": report.get("uncited_claim_count"),
        "invalid_citation_count": report.get("invalid_citation_count"),
    }


# ---------------------------------------------------------------------------
# Citation Evaluation Summary — extended per-answer citation stats
# ---------------------------------------------------------------------------

@dataclass
class CitationEvalSummary:
    """
    Comprehensive citation quality summary for a single generated answer.
    Counts and rates are computed against clearly documented denominators.
    """
    total_claims: int = 0
    cited_claims: int = 0
    uncited_claims: int = 0
    claims_with_valid_citation_id: int = 0
    claims_with_invalid_citation_id: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    unverifiable_claims: int = 0   # verification_error outcomes
    # chunk existence checks
    citations_referencing_known_chunk: int = 0
    citations_referencing_unknown_chunk: int = 0
    # numeric contradiction detection
    numeric_contradiction_count: int = 0

    @property
    def citation_coverage_rate(self) -> Optional[float]:
        """Fraction of claims that have at least one citation. Denominator = total_claims."""
        if self.total_claims == 0:
            return None
        return round(self.cited_claims / self.total_claims, 4)

    @property
    def claim_support_rate(self) -> Optional[float]:
        """Fraction of cited claims that are factually supported. Denominator = cited_claims."""
        if self.cited_claims == 0:
            return None
        return round(self.supported_claims / self.cited_claims, 4)

    @property
    def valid_citation_id_rate(self) -> Optional[float]:
        """Fraction of cited claims whose IDs exist in the registry. Denominator = cited_claims."""
        if self.cited_claims == 0:
            return None
        return round(self.claims_with_valid_citation_id / self.cited_claims, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "cited_claims": self.cited_claims,
            "uncited_claims": self.uncited_claims,
            "claims_with_valid_citation_id": self.claims_with_valid_citation_id,
            "claims_with_invalid_citation_id": self.claims_with_invalid_citation_id,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "unverifiable_claims": self.unverifiable_claims,
            "citations_referencing_known_chunk": self.citations_referencing_known_chunk,
            "citations_referencing_unknown_chunk": self.citations_referencing_unknown_chunk,
            "numeric_contradiction_count": self.numeric_contradiction_count,
            "citation_coverage_rate": self.citation_coverage_rate,
            "claim_support_rate": self.claim_support_rate,
            "valid_citation_id_rate": self.valid_citation_id_rate,
        }


# ---------------------------------------------------------------------------
# Batch Evaluation Summary
# ---------------------------------------------------------------------------

@dataclass
class BatchEvalSummary:
    """
    Aggregate statistics across a full evaluation batch.
    """
    total_records: int = 0
    attempted: int = 0
    succeeded: int = 0
    skipped: int = 0
    failed: int = 0

    # Generation outcomes
    generated_answers: int = 0
    abstentions: int = 0
    generation_errors: int = 0

    # Evidence status breakdown
    evidence_status_counts: Dict[str, int] = field(default_factory=dict)

    # Workflow status breakdown
    workflow_status_counts: Dict[str, int] = field(default_factory=dict)

    # Verification outcomes
    verified_answers: int = 0
    requires_review_answers: int = 0

    # Aggregate RAGAS scores
    ragas_metric_sums: Dict[str, float] = field(default_factory=dict)
    ragas_metric_counts: Dict[str, int] = field(default_factory=dict)
    ragas_unavailable_reasons: Dict[str, str] = field(default_factory=dict)

    # Aggregate citation stats
    total_claims_evaluated: int = 0
    total_cited_claims: int = 0
    total_uncited_claims: int = 0
    total_supported_claims: int = 0
    total_unsupported_claims: int = 0
    total_invalid_citation_ids: int = 0
    total_unverifiable_claims: int = 0

    # Timing
    total_duration_seconds: float = 0.0

    def record_ragas_score(self, metric_name: str, score: Optional[float]) -> None:
        if score is not None:
            self.ragas_metric_sums[metric_name] = (
                self.ragas_metric_sums.get(metric_name, 0.0) + score
            )
            self.ragas_metric_counts[metric_name] = (
                self.ragas_metric_counts.get(metric_name, 0) + 1
            )

    def mean_ragas_score(self, metric_name: str) -> Optional[float]:
        count = self.ragas_metric_counts.get(metric_name, 0)
        if count == 0:
            return None
        return round(self.ragas_metric_sums[metric_name] / count, 4)

    def to_dict(self) -> Dict[str, Any]:
        ragas_means = {
            m: self.mean_ragas_score(m)
            for m in self.ragas_metric_sums
        }
        citation_summary = {
            "total_claims": self.total_claims_evaluated,
            "cited_claims": self.total_cited_claims,
            "uncited_claims": self.total_uncited_claims,
            "supported_claims": self.total_supported_claims,
            "unsupported_claims": self.total_unsupported_claims,
            "invalid_citation_id_claims": self.total_invalid_citation_ids,
            "unverifiable_claims": self.total_unverifiable_claims,
            "citation_coverage_rate": (
                round(self.total_cited_claims / self.total_claims_evaluated, 4)
                if self.total_claims_evaluated > 0 else None
            ),
            "claim_support_rate": (
                round(self.total_supported_claims / self.total_cited_claims, 4)
                if self.total_cited_claims > 0 else None
            ),
        }
        return {
            "records": {
                "total": self.total_records,
                "attempted": self.attempted,
                "succeeded": self.succeeded,
                "skipped": self.skipped,
                "failed": self.failed,
            },
            "generation": {
                "generated": self.generated_answers,
                "abstentions": self.abstentions,
                "errors": self.generation_errors,
            },
            "evidence_status": self.evidence_status_counts,
            "workflow_status": self.workflow_status_counts,
            "verification": {
                "verified": self.verified_answers,
                "requires_review": self.requires_review_answers,
            },
            "ragas_mean_scores": ragas_means,
            "ragas_valid_sample_counts": self.ragas_metric_counts,
            "ragas_unavailable": self.ragas_unavailable_reasons,
            "citation_quality": citation_summary,
            "timing": {
                "total_duration_seconds": round(self.total_duration_seconds, 2),
                "mean_per_question_seconds": (
                    round(self.total_duration_seconds / self.attempted, 2)
                    if self.attempted > 0 else None
                ),
            },
        }
