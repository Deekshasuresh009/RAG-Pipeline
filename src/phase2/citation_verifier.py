"""
Phase 2, Step 13: Claim-Level Citation Verification
Segments generated answers into factual claims, resolves cited evidence IDs against the active
request registry, assesses evidence support via deterministic heuristics or model-based verifiers,
and produces structured verification reports with clear abstention/review decisions.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.phase2.evidence_evaluator import (
    STOP_WORDS,
    tokenize_text,
    extract_salient_terms,
)


class BaseEntailmentVerifier:
    """Abstract base class for claim-level evidence support verification."""
    def verify(self, claim: str, evidence_text: str) -> Tuple[bool, str]:
        raise NotImplementedError


class FakeEntailmentVerifier(BaseEntailmentVerifier):
    """
    Deterministic fake verifier for testing and controlled demonstrations.
    Supports fixed outcomes, claim-specific overrides, or custom predicates.
    """
    def __init__(
        self,
        default_support: bool = True,
        default_reason: str = "Claim is verified by mock verifier.",
        claim_overrides: Optional[Dict[str, Tuple[bool, str]]] = None,
        exc_to_raise: Optional[Exception] = None,
    ) -> None:
        self.default_support = default_support
        self.default_reason = default_reason
        self.claim_overrides = claim_overrides or {}
        self.exc_to_raise = exc_to_raise
        self.calls: List[Tuple[str, str]] = []

    def verify(self, claim: str, evidence_text: str) -> Tuple[bool, str]:
        self.calls.append((claim, evidence_text))
        if self.exc_to_raise is not None:
            raise self.exc_to_raise

        for pattern, (supp, reason) in self.claim_overrides.items():
            if pattern.lower() in claim.lower():
                return supp, reason

        return self.default_support, self.default_reason


class HeuristicEntailmentVerifier(BaseEntailmentVerifier):
    """
    Deterministic rule-based verifier.
    Performs:
    1. Numeric contradiction detection (e.g. 500 checkpoints in claim vs 380 in evidence).
    2. Polarity / availability contradiction checks (standard vs optional / not included).
    3. Salient term overlap and entity alignment.
    """
    def __init__(self, min_support_ratio: float = 0.55) -> None:
        self.min_support_ratio = min_support_ratio

    def verify(self, claim: str, evidence_text: str) -> Tuple[bool, str]:
        if not claim.strip():
            return False, "Empty claim text."
        if not evidence_text.strip():
            return False, "Cited evidence text is empty."

        clean_claim = claim.strip()
        evidence_lower = evidence_text.lower()
        evidence_tokens = set(tokenize_text(evidence_lower))

        # 1. Numeric contradiction check
        # Extract numbers (integers or decimals) from the claim
        claim_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", clean_claim)
        for num in claim_numbers:
            # If the number is a key specification (not a citation ID), it should be in the evidence
            if num not in evidence_tokens:
                # Find numbers in evidence for context
                evidence_nums = [t for t in evidence_tokens if re.match(r"^\d+(?:\.\d+)?$", t)]
                return (
                    False,
                    f"Numeric contradiction: Claim states '{num}', but cited evidence does not contain this number (evidence numbers: {evidence_nums[:5]})."
                )

        # 2. Polarity contradiction check (standard vs optional)
        claim_has_standard = bool(re.search(r"\b(is|are)\s+(standard|included|complimentary)\b", clean_claim, re.IGNORECASE))
        claim_has_optional = bool(re.search(r"\b(is|are)\s+(optional|not included|not available)\b", clean_claim, re.IGNORECASE))

        evidence_has_standard = bool(re.search(r"\b(is|are)\s+(standard|included|complimentary)\b", evidence_lower))
        evidence_has_optional = bool(re.search(r"\b(is|are)\s+(optional|not included|not available)\b", evidence_lower))

        if claim_has_standard and evidence_has_optional and not evidence_has_standard:
            return (
                False,
                "Polarity contradiction: Claim asserts feature is standard, but evidence indicates it is optional/not included."
            )
        if claim_has_optional and evidence_has_standard and not evidence_has_optional:
            return (
                False,
                "Polarity contradiction: Claim asserts feature is optional, but evidence indicates it is standard/included."
            )

        # 3. Salient concept coverage check
        claim_terms = extract_salient_terms(clean_claim)
        if not claim_terms:
            return True, "Claim contains no specific non-stopword factual terms."

        matched = [t for t in set(claim_terms) if t in evidence_tokens]
        coverage = len(matched) / len(set(claim_terms))

        if coverage >= self.min_support_ratio:
            return True, f"Claim is supported by cited evidence ({coverage:.0%} salient concept alignment: {matched})."
        else:
            unmatched = [t for t in set(claim_terms) if t not in evidence_tokens]
            return False, f"Insufficient evidence support: only {coverage:.0%} of claim concepts matched (missing: {unmatched})."


def segment_claims(answer: str) -> List[Dict[str, Any]]:
    """
    Segment generated answer text into discrete factual claims.
    Associates each claim with its inline citation tags ([E1], [E2]...).
    Preserves original claim boundaries and text position.
    """
    if not answer or not answer.strip():
        return []

    # First split into sentences/clauses on sentence terminators (. ! ?) or semicolons
    # Avoid splitting inside numbers like 3.2 or 380+
    raw_segments = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\[])|;\s*", answer.strip())
    claims: List[Dict[str, Any]] = []

    for seg in raw_segments:
        clean_seg = seg.strip()
        if not clean_seg:
            continue

        # Extract citation tags like [E1], [E2]
        citation_ids = re.findall(r"\[(E\d+)\]", clean_seg)

        # Remove citation brackets from clean claim text
        clean_claim_text = re.sub(r"\s*\[E\d+\]", "", clean_seg).strip()

        # Skip conversational transitions or headings that lack factual predicates
        if not clean_claim_text or clean_claim_text.endswith(":") and len(clean_claim_text.split()) <= 4:
            continue

        # If sentence is purely polite framing, skip
        lower_claim = clean_claim_text.lower()
        if lower_claim in {
            "here is the information",
            "based on the provided documents",
            "in summary",
            "please note",
        }:
            continue

        claims.append({
            "claim_text": clean_claim_text,
            "raw_segment": clean_seg,
            "cited_evidence_ids": citation_ids,
        })

    return claims


class CitationVerifier:
    """
    Claim-Level Citation Verification component.
    Validates that each factual claim in an answer is accompanied by valid citations
    and that the cited evidence semantically/factually supports the claim.
    """
    def __init__(self, entailment_verifier: Optional[BaseEntailmentVerifier] = None) -> None:
        self.verifier = entailment_verifier or HeuristicEntailmentVerifier()

    def verify(
        self,
        answer: str,
        evidence_registry: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Verify an answer against the active evidence registry.
        Returns a structured verification report.
        """
        limitations = [
            "Lexical and rule-based verification verifies term presence, numbers, and polarity, but does not guarantee full linguistic semantic entailment without human review or an NLI cross-encoder.",
            "Sentence and clause splitting relies on heuristic delimiters; complex nested sentences may not separate into atomic propositions.",
            "Retrieved evidence text is treated strictly as passive data and not as instruction directives.",
        ]

        if not answer or not answer.strip():
            return {
                "verification_status": "empty_answer",
                "overall_status": "unverified",
                "is_verified": False,
                "claim_results": [],
                "supported_claim_count": 0,
                "unsupported_claim_count": 0,
                "uncited_claim_count": 0,
                "invalid_citation_count": 0,
                "verification_error": "Answer is empty or whitespace-only.",
                "limitations": limitations,
            }

        # 1. Segment answer into claims
        segmented = segment_claims(answer)
        if not segmented:
            return {
                "verification_status": "no_claims_found",
                "overall_status": "unverified",
                "is_verified": False,
                "claim_results": [],
                "supported_claim_count": 0,
                "unsupported_claim_count": 0,
                "uncited_claim_count": 0,
                "invalid_citation_count": 0,
                "verification_error": "No verifiable factual claims detected in the answer.",
                "limitations": limitations,
            }

        # 2. Evaluate each claim
        claim_results: List[Dict[str, Any]] = []

        for item in segmented:
            claim_text = item["claim_text"]
            cited_ids = item["cited_evidence_ids"]

            # Outcome A: Uncited claim
            if not cited_ids:
                claim_results.append({
                    "claim_text": claim_text,
                    "original_citations": [],
                    "verification_status": "uncited",
                    "is_supported": False,
                    "resolved_citations": [],
                    "explanation": "Factual claim has no inline citations.",
                })
                continue

            # Outcome B: Check for unknown citation IDs
            unknown_ids = [eid for eid in cited_ids if eid not in evidence_registry]
            if unknown_ids:
                resolved = [
                    evidence_registry[eid] for eid in cited_ids if eid in evidence_registry
                ]
                claim_results.append({
                    "claim_text": claim_text,
                    "original_citations": cited_ids,
                    "verification_status": "invalid_citation",
                    "is_supported": False,
                    "resolved_citations": resolved,
                    "explanation": f"Claim cites unknown evidence ID(s): {unknown_ids}.",
                })
                continue

            # Outcome C: Citations are valid in registry. Check evidence support.
            resolved = [evidence_registry[eid] for eid in cited_ids]
            combined_evidence = "\n".join([r.get("text", "") for r in resolved])

            if not combined_evidence.strip():
                claim_results.append({
                    "claim_text": claim_text,
                    "original_citations": cited_ids,
                    "verification_status": "unsupported",
                    "is_supported": False,
                    "resolved_citations": resolved,
                    "explanation": "Cited evidence passages contain empty text.",
                })
                continue

            try:
                is_supported, explanation = self.verifier.verify(claim_text, combined_evidence)
                claim_results.append({
                    "claim_text": claim_text,
                    "original_citations": cited_ids,
                    "verification_status": "supported" if is_supported else "unsupported",
                    "is_supported": is_supported,
                    "resolved_citations": resolved,
                    "explanation": explanation,
                })
            except Exception as exc:
                claim_results.append({
                    "claim_text": claim_text,
                    "original_citations": cited_ids,
                    "verification_status": "verification_error",
                    "is_supported": False,
                    "resolved_citations": resolved,
                    "explanation": f"Verifier exception during assessment: {exc}",
                })

        # 3. Calculate tallies
        supported_count = sum(1 for c in claim_results if c["verification_status"] == "supported")
        unsupported_count = sum(1 for c in claim_results if c["verification_status"] == "unsupported")
        uncited_count = sum(1 for c in claim_results if c["verification_status"] == "uncited")
        invalid_count = sum(1 for c in claim_results if c["verification_status"] == "invalid_citation")
        error_count = sum(1 for c in claim_results if c["verification_status"] == "verification_error")

        # 4. Overall status calculation
        if error_count > 0:
            overall_status = "error"
            verification_status = "verification_error"
            is_verified = False
            verification_error = f"{error_count} claim(s) experienced verifier execution errors."
        elif supported_count == len(claim_results) and len(claim_results) > 0:
            overall_status = "verified"
            verification_status = "verified"
            is_verified = True
            verification_error = None
        elif unsupported_count > 0 or invalid_count > 0:
            overall_status = "unverified"
            verification_status = "requires_review"
            is_verified = False
            verification_error = f"{unsupported_count} unsupported and {invalid_count} invalid-citation claim(s) detected."
        elif uncited_count > 0:
            overall_status = "partially_verified"
            verification_status = "requires_review"
            is_verified = False
            verification_error = f"{uncited_count} uncited claim(s) detected in answer."
        else:
            overall_status = "unverified"
            verification_status = "requires_review"
            is_verified = False
            verification_error = None

        return {
            "verification_status": verification_status,
            "overall_status": overall_status,
            "is_verified": is_verified,
            "claim_results": claim_results,
            "supported_claim_count": supported_count,
            "unsupported_claim_count": unsupported_count,
            "uncited_claim_count": uncited_count,
            "invalid_citation_count": invalid_count,
            "verification_error": verification_error,
            "limitations": limitations,
        }
