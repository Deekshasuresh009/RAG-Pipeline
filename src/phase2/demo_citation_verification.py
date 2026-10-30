"""
Phase 2, Step 13: Demo — Claim-Level Citation Verification
Demonstrates five citation verification scenarios using the Aventro Motors RAG system:

  A. Fully supported, correctly cited answer  (380+ checkpoints)
  B. Numeric mismatch — unsupported claim     (500 checkpoints vs 380)
  C. Uncited factual claim                    (claim has no [Exx] tag)
  D. Unknown citation ID                      ([E99] not in registry)
  E. Mixed answer — supported + unsupported + uncited claims

Run:
    python -m src.phase2.demo_citation_verification
"""

from __future__ import annotations

import json
import textwrap
from typing import Any, Dict

from src.phase2.citation_verifier import CitationVerifier, HeuristicEntailmentVerifier

# ---------------------------------------------------------------------------
# Shared test registry (mirrors realistic Aventro Motors evidence chunks)
# ---------------------------------------------------------------------------
DEMO_REGISTRY: Dict[str, Dict[str, Any]] = {
    "E1": {
        "evidence_id": "E1",
        "chunk_id": "doc_preowned_001",
        "source_document": "Pre-Owned Certified Cars.md",
        "document_title": "Aventro Motors Pvt. Ltd.",
        "section_heading": "Inspection Checkpoints",
        "heading_path": "Aventro Motors > Pre-Owned > Checkpoints",
        "text": (
            "Aventro certified pre-owned cars undergo a rigorous 380+ checkpoint quality "
            "inspection covering engine, transmission, brakes, suspension, and electricals."
        ),
    },
    "E2": {
        "evidence_id": "E2",
        "chunk_id": "doc_preowned_002",
        "source_document": "Pre-Owned Certified Cars.md",
        "document_title": "Aventro Motors Pvt. Ltd.",
        "section_heading": "Warranty and Roadside Assistance",
        "heading_path": "Aventro Motors > Pre-Owned > Warranty",
        "text": (
            "Every certified pre-owned vehicle comes with a 1 year complimentary warranty "
            "and 24/7 roadside assistance at no additional cost."
        ),
    },
    "E3": {
        "evidence_id": "E3",
        "chunk_id": "doc_battery_001",
        "source_document": "Battery Care.md",
        "document_title": "Aventro Battery Guide",
        "section_heading": "Battery Specifications",
        "heading_path": "Battery Guide > Specifications",
        "text": (
            "The Aventro Spark Electric features a 42 kWh lithium-ion battery pack "
            "supporting DC fast charging up to 100 kW."
        ),
    },
}


def print_divider(title: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_report(report: Dict[str, Any]) -> None:
    print(f"\n  Overall Status  : {report['overall_status'].upper()}")
    print(f"  Verified?       : {report['is_verified']}")
    print(f"  Supported       : {report['supported_claim_count']}")
    print(f"  Unsupported     : {report['unsupported_claim_count']}")
    print(f"  Uncited         : {report['uncited_claim_count']}")
    print(f"  Invalid Cites   : {report['invalid_citation_count']}")
    if report.get("verification_error"):
        print(f"  Error Detail    : {report['verification_error']}")

    for i, c in enumerate(report.get("claim_results", []), start=1):
        status_label = c["verification_status"].upper()
        claim_short = textwrap.shorten(c["claim_text"], width=65, placeholder="…")
        print(f"\n  Claim {i}: [{status_label}]")
        print(f"    Text       : {claim_short}")
        print(f"    Cited IDs  : {c['original_citations'] or '(none)'}")
        print(f"    Explanation: {textwrap.shorten(c['explanation'], width=65, placeholder='…')}")
        if c.get("resolved_citations"):
            for rc in c["resolved_citations"]:
                src = rc.get("source_document", "?")
                sec = rc.get("section_heading", "?")
                print(f"    -> Resolved : {src} | {sec}")


def main() -> None:
    verifier = CitationVerifier(entailment_verifier=HeuristicEntailmentVerifier())

    # ------------------------------------------------------------------
    # Scenario A: Fully supported and correctly cited answer
    # ------------------------------------------------------------------
    print_divider("Scenario A — Fully Supported, Correctly Cited Answer")
    answer_a = (
        "Aventro certified pre-owned cars undergo a rigorous 380+ checkpoint inspection "
        "covering engine, brakes, and electricals [E1]. "
        "Each vehicle also includes a complimentary 1 year warranty and 24/7 roadside assistance [E2]."
    )
    print(f"\n  Answer: {textwrap.shorten(answer_a, width=70, placeholder='…')}")
    report_a = verifier.verify(answer_a, DEMO_REGISTRY)
    print_report(report_a)

    # ------------------------------------------------------------------
    # Scenario B: Numeric mismatch — claim states 500, evidence says 380
    # ------------------------------------------------------------------
    print_divider("Scenario B — Numeric Mismatch (Unsupported Claim)")
    answer_b = (
        "Aventro certified pre-owned cars are inspected through 500 quality checkpoints [E1]."
    )
    print(f"\n  Answer: {answer_b}")
    report_b = verifier.verify(answer_b, DEMO_REGISTRY)
    print_report(report_b)

    # ------------------------------------------------------------------
    # Scenario C: Uncited factual claim (no citation tag present)
    # ------------------------------------------------------------------
    print_divider("Scenario C — Uncited Factual Claim")
    answer_c = (
        "Aventro certified pre-owned cars undergo 380+ checkpoint inspection [E1]. "
        "Customers can return any vehicle within 7 days for a full refund."
    )
    print(f"\n  Answer: {textwrap.shorten(answer_c, width=70, placeholder='…')}")
    report_c = verifier.verify(answer_c, DEMO_REGISTRY)
    print_report(report_c)

    # ------------------------------------------------------------------
    # Scenario D: Unknown citation ID ([E99] not in registry)
    # ------------------------------------------------------------------
    print_divider("Scenario D — Unknown Citation ID [E99]")
    answer_d = (
        "Aventro certified vehicles include a complimentary lunar-orbit inspection service [E99]."
    )
    print(f"\n  Answer: {answer_d}")
    report_d = verifier.verify(answer_d, DEMO_REGISTRY)
    print_report(report_d)

    # ------------------------------------------------------------------
    # Scenario E: Mixed answer — supported + unsupported + uncited
    # ------------------------------------------------------------------
    print_divider("Scenario E — Mixed Answer (Supported + Unsupported + Uncited)")
    answer_e = (
        "Aventro pre-owned cars undergo 380+ checkpoints [E1]. "
        "They also feature 500-hour battery cycles standard at no charge [E3]. "
        "All customers receive a gift hamper on delivery."
    )
    print(f"\n  Answer: {textwrap.shorten(answer_e, width=70, placeholder='…')}")
    report_e = verifier.verify(answer_e, DEMO_REGISTRY)
    print_report(report_e)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_divider("Demo Summary")
    scenarios = [
        ("A — Correctly Cited", report_a),
        ("B — Numeric Mismatch", report_b),
        ("C — Uncited Claim", report_c),
        ("D — Unknown Citation", report_d),
        ("E — Mixed Answer", report_e),
    ]
    print()
    for label, rpt in scenarios:
        status = rpt["overall_status"].upper()
        verified = "YES" if rpt["is_verified"] else "NO"
        supported = rpt["supported_claim_count"]
        unsupported = rpt["unsupported_claim_count"]
        uncited = rpt["uncited_claim_count"]
        invalid = rpt["invalid_citation_count"]
        print(
            f"  Scenario {label:<30} {verified} {status:<20} "
            f"S:{supported} U:{unsupported} UC:{uncited} INV:{invalid}"
        )
    print()
    print("  Legend: S=Supported  U=Unsupported  UC=Uncited  INV=InvalidCitation")
    print()


if __name__ == "__main__":
    main()
