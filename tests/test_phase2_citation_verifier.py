"""
Unit Tests for Phase 2, Step 13: Claim-Level Citation Verification
Tests claim segmentation, citation resolution, support assessment, error handling, and reporting.
"""

from __future__ import annotations

from typing import Any, Dict, List
import pytest

from src.phase2.citation_verifier import (
    CitationVerifier,
    HeuristicEntailmentVerifier,
    FakeEntailmentVerifier,
    segment_claims,
)


def make_test_registry() -> Dict[str, Dict[str, Any]]:
    return {
        "E1": {
            "evidence_id": "E1",
            "chunk_id": "doc_pre_owned_chunk_001",
            "source_document": "Pre-Owned Certified Cars.md",
            "document_title": "Aventro Motors Pvt. Ltd.",
            "section_heading": "Inspection Checkpoints",
            "heading_path": "Aventro Motors > Pre-Owned > Checkpoints",
            "text": "Aventro certified pre-owned cars undergo a rigorous 380+ checkpoint quality inspection covering engine, brakes, and electricals.",
        },
        "E2": {
            "evidence_id": "E2",
            "chunk_id": "doc_pre_owned_chunk_002",
            "source_document": "Pre-Owned Certified Cars.md",
            "document_title": "Aventro Motors Pvt. Ltd.",
            "section_heading": "Warranty and Roadside",
            "heading_path": "Aventro Motors > Pre-Owned > Warranty",
            "text": "Every certified pre-owned vehicle comes with 1 year complimentary warranty and 24/7 roadside assistance.",
        },
        "E3": {
            "evidence_id": "E3",
            "chunk_id": "doc_battery_chunk_001",
            "source_document": "Battery Care.md",
            "document_title": "Aventro Battery Guide",
            "section_heading": "Battery Specifications",
            "heading_path": "Battery Guide > Specifications",
            "text": "The Aventro Spark Electric features a 42 kWh lithium-ion battery pack with fast DC charging.",
        },
    }


# ==============================================================================
# 1. Claim Segmentation Tests
# ==============================================================================

def test_segment_claims_basic():
    answer = (
        "Aventro vehicles undergo 380+ checkpoints [E1]. "
        "They include a 1-year complimentary warranty [E2]."
    )
    segmented = segment_claims(answer)

    assert len(segmented) == 2
    assert segmented[0]["cited_evidence_ids"] == ["E1"]
    assert "Aventro vehicles undergo 380+ checkpoints." in segmented[0]["claim_text"]
    assert segmented[1]["cited_evidence_ids"] == ["E2"]
    assert "They include a 1-year complimentary warranty." in segmented[1]["claim_text"]


def test_segment_claims_multiple_citations_on_one_claim():
    answer = "Vehicles receive 380 checkpoints and warranty coverage [E1][E2]."
    segmented = segment_claims(answer)

    assert len(segmented) == 1
    assert segmented[0]["cited_evidence_ids"] == ["E1", "E2"]
    assert "[E1]" not in segmented[0]["claim_text"]
    assert "[E2]" not in segmented[0]["claim_text"]


def test_segment_claims_semicolon_clause_splitting():
    answer = "Engine systems are checked [E1]; roadside assistance is included 24/7 [E2]."
    segmented = segment_claims(answer)

    assert len(segmented) == 2
    assert segmented[0]["cited_evidence_ids"] == ["E1"]
    assert segmented[1]["cited_evidence_ids"] == ["E2"]


def test_segment_claims_uncited_claim_detection():
    answer = "Aventro cars undergo 380 checkpoints [E1]. Some models also feature sunroofs."
    segmented = segment_claims(answer)

    assert len(segmented) == 2
    assert segmented[0]["cited_evidence_ids"] == ["E1"]
    assert segmented[1]["cited_evidence_ids"] == []
    assert "Some models also feature sunroofs." in segmented[1]["claim_text"]


# ==============================================================================
# 2. Citation Verification Scenarios
# ==============================================================================

def test_verify_supported_claim():
    registry = make_test_registry()
    verifier = CitationVerifier(entailment_verifier=HeuristicEntailmentVerifier())

    answer = "Aventro certified pre-owned cars undergo a rigorous 380+ checkpoint inspection [E1]."
    report = verifier.verify(answer, registry)

    assert report["is_verified"] is True
    assert report["verification_status"] == "verified"
    assert report["overall_status"] == "verified"
    assert report["supported_claim_count"] == 1
    assert report["unsupported_claim_count"] == 0

    c0 = report["claim_results"][0]
    assert c0["verification_status"] == "supported"
    assert c0["is_supported"] is True
    assert c0["resolved_citations"][0]["chunk_id"] == "doc_pre_owned_chunk_001"


def test_verify_numeric_contradiction_unsupported():
    registry = make_test_registry()
    verifier = CitationVerifier(entailment_verifier=HeuristicEntailmentVerifier())

    # Claim states 500 checkpoints, but E1 states 380
    answer = "Aventro certified pre-owned cars undergo 500 inspection checkpoints [E1]."
    report = verifier.verify(answer, registry)

    assert report["is_verified"] is False
    assert report["verification_status"] == "requires_review"
    assert report["unsupported_claim_count"] == 1
    c0 = report["claim_results"][0]
    assert c0["verification_status"] == "unsupported"
    assert "Numeric contradiction" in c0["explanation"]


def test_verify_uncited_factual_claim():
    registry = make_test_registry()
    verifier = CitationVerifier(entailment_verifier=HeuristicEntailmentVerifier())

    answer = "Aventro cars undergo 380 checkpoints [E1]. The company offers special holiday discounts."
    report = verifier.verify(answer, registry)

    assert report["is_verified"] is False
    assert report["verification_status"] == "requires_review"
    assert report["supported_claim_count"] == 1
    assert report["uncited_claim_count"] == 1

    c1 = report["claim_results"][1]
    assert c1["verification_status"] == "uncited"
    assert c1["is_supported"] is False
    assert "no inline citations" in c1["explanation"]


def test_verify_unknown_citation_id():
    registry = make_test_registry()
    verifier = CitationVerifier(entailment_verifier=HeuristicEntailmentVerifier())

    # Cites [E99] which is not in registry
    answer = "Aventro certified cars include lunar rover return privileges [E99]."
    report = verifier.verify(answer, registry)

    assert report["is_verified"] is False
    assert report["verification_status"] == "requires_review"
    assert report["invalid_citation_count"] == 1

    c0 = report["claim_results"][0]
    assert c0["verification_status"] == "invalid_citation"
    assert c0["is_supported"] is False
    assert "unknown evidence ID" in c0["explanation"]


def test_verify_valid_citation_pointing_to_irrelevant_evidence():
    registry = make_test_registry()
    verifier = CitationVerifier(entailment_verifier=HeuristicEntailmentVerifier())

    # Cites E3 (battery care) for a claim about brake inspection
    answer = "Pre-owned brake rotors and calipers are inspected thoroughly [E3]."
    report = verifier.verify(answer, registry)

    assert report["is_verified"] is False
    assert report["unsupported_claim_count"] == 1
    c0 = report["claim_results"][0]
    assert c0["verification_status"] == "unsupported"
    assert "Insufficient evidence support" in c0["explanation"]


def test_verify_mixed_answer_outcomes():
    registry = make_test_registry()
    verifier = CitationVerifier(entailment_verifier=HeuristicEntailmentVerifier())

    answer = (
        "Vehicles undergo 380+ quality checkpoints [E1]. "
        "They include free unlimited lunar space flights [E2]. "
        "Delivery is guaranteed within 10 minutes."
    )
    report = verifier.verify(answer, registry)

    assert report["is_verified"] is False
    assert report["supported_claim_count"] == 1
    assert report["unsupported_claim_count"] == 1
    assert report["uncited_claim_count"] == 1
    assert report["verification_status"] == "requires_review"


def test_verify_multiple_citations_supporting_one_claim():
    registry = make_test_registry()
    verifier = CitationVerifier(entailment_verifier=HeuristicEntailmentVerifier())

    # Combined text from E1 and E2 supports this synthesis
    answer = "Pre-owned certified cars receive 380+ checkpoints inspection and 1 year complimentary warranty [E1][E2]."
    report = verifier.verify(answer, registry)

    assert report["is_verified"] is True
    assert report["supported_claim_count"] == 1
    assert len(report["claim_results"][0]["resolved_citations"]) == 2


def test_verify_empty_answer():
    registry = make_test_registry()
    verifier = CitationVerifier()

    report = verifier.verify("", registry)
    assert report["is_verified"] is False
    assert report["verification_status"] == "empty_answer"
    assert report["claim_results"] == []


def test_verifier_exception_handling():
    registry = make_test_registry()
    fake_entailment = FakeEntailmentVerifier(exc_to_raise=RuntimeError("NLI model inference error"))
    verifier = CitationVerifier(entailment_verifier=fake_entailment)

    answer = "Vehicles undergo 380 checkpoints [E1]."
    report = verifier.verify(answer, registry)

    assert report["is_verified"] is False
    assert report["overall_status"] == "error"
    assert report["verification_status"] == "verification_error"
    assert report["claim_results"][0]["verification_status"] == "verification_error"
