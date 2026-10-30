"""
Phase 2, Step 10: Evidence Relevance and Sufficiency Checking
Evaluates whether retrieved chunks from hybrid RRF retrieval are relevant to the user's query
and collectively sufficient to support a reliable, grounded answer.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple


# Standard set of common English stop words to prevent false relevance from syntax words
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "should", "shouldn't", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "were", "weren't", "what",
    "when", "where", "which", "while", "who", "whom", "why", "with", "won't", "would", "wouldn't",
    "you", "your", "yours", "yourself", "yourselves", "tell", "give", "provide", "please"
}


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric terms and decimal numbers."""
    if not text or not isinstance(text, str):
        return []
    return re.findall(r"\d+\.\d+|[a-zA-Z0-9_]+", text.lower())


def extract_salient_terms(text: str) -> List[str]:
    """Extract non-stopword meaningful tokens of length >= 2 or numbers."""
    tokens = tokenize_text(text)
    salient: List[str] = []
    for t in tokens:
        if t not in STOP_WORDS and (len(t) >= 2 or t.isdigit()):
            salient.append(t)
    return salient


def split_query_sub_aspects(query: str) -> List[str]:
    """
    Split a query into distinct sub-aspects/clauses for multi-part sufficiency checking.
    Splits on conjunctions ('and', 'as well as', 'also') or punctuation ('?', ';').
    """
    if not query:
        return []

    # First split on sentence terminators or clauses
    parts = re.split(r"\?+|;|(?:\s+and\s+)|\b(?:as well as)\b|\b(?:also)\b", query, flags=re.IGNORECASE)
    cleaned_aspects: List[str] = []
    for p in parts:
        p_clean = p.strip()
        # Only retain aspects with meaningful content
        if extract_salient_terms(p_clean):
            cleaned_aspects.append(p_clean)

    # If splitting resulted in no aspects or just one, fallback to the entire query
    return cleaned_aspects if cleaned_aspects else [query.strip()]


def detect_conflicts(relevant_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deterministic rule-based conflict detector.
    Detects contradictory numbers or opposing polarity statements on shared attributes
    across retrieved relevant chunks.

    LIMITATION: This heuristic is rule-based and limited to numeric and polarity patterns
    on shared attributes. General semantic contradiction detection across open-domain prose
    requires an LLM.
    """
    if len(relevant_chunks) < 2:
        return []

    conflicts: List[Dict[str, Any]] = []

    # 1. Attribute pattern: Warranty duration
    warranty_pattern = re.compile(r"\b(?:warranty|guarantee)\b[^\.\n;]{0,50}?\b(\d+)\s*(years?|months?|days?)", re.IGNORECASE)
    warranty_claims: Dict[str, List[Tuple[str, str]]] = {}  # duration_key -> [(chunk_id, matched_text)]

    # 2. Attribute pattern: Pre-owned checkpoints count
    checkpoints_pattern = re.compile(r"\b(\d+)\+?\s*(?:quality\s*)?checkpoints?\b", re.IGNORECASE)
    checkpoint_claims: Dict[str, List[Tuple[str, str]]] = {}

    # 3. Attribute pattern: Explicit feature availability polarity (e.g. "standard / included" vs "optional / not included")
    # Search for statements like "<feature> is standard" vs "<feature> is optional / not available"
    polarity_pattern = re.compile(r"\b([a-zA-Z\s]{3,25})\b\s+(?:is|are)\s+(standard|included|complimentary|optional|not included|not available)\b", re.IGNORECASE)
    polarity_claims: Dict[str, Dict[str, str]] = {}  # feature -> {polarity_type: chunk_id}

    for c in relevant_chunks:
        cid = c["chunk_id"]
        text = c["text"]

        # Check warranty
        for match in warranty_pattern.finditer(text):
            duration = f"{match.group(1)} {match.group(2).lower()}"
            warranty_claims.setdefault(duration, []).append((cid, match.group(0)))

        # Check checkpoints
        for match in checkpoints_pattern.finditer(text):
            count = match.group(1)
            checkpoint_claims.setdefault(count, []).append((cid, match.group(0)))

        # Check polarity
        for match in polarity_pattern.finditer(text):
            feature = match.group(1).strip().lower()
            pol = match.group(2).strip().lower()
            is_positive = pol in {"standard", "included", "complimentary"}
            pol_group = "positive" if is_positive else "negative"
            if feature not in polarity_claims:
                polarity_claims[feature] = {}
            polarity_claims[feature][pol_group] = cid

    # Detect warranty discrepancies
    if len(warranty_claims) > 1:
        distinct_vals = list(warranty_claims.keys())
        chunks_involved = list({item[0] for items in warranty_claims.values() for item in items})
        conflicts.append({
            "type": "numeric_specification_conflict",
            "attribute": "warranty_duration",
            "values": distinct_vals,
            "chunks_involved": chunks_involved,
            "description": f"Conflicting warranty duration statements detected: {', '.join(distinct_vals)} across chunks {chunks_involved}."
        })

    # Detect checkpoint count discrepancies
    if len(checkpoint_claims) > 1:
        distinct_counts = list(checkpoint_claims.keys())
        chunks_involved = list({item[0] for items in checkpoint_claims.values() for item in items})
        conflicts.append({
            "type": "numeric_specification_conflict",
            "attribute": "inspection_checkpoints",
            "values": distinct_counts,
            "chunks_involved": chunks_involved,
            "description": f"Conflicting inspection checkpoint counts detected: {', '.join(distinct_counts)} across chunks {chunks_involved}."
        })

    # Detect polarity discrepancies on same feature
    for feature, pol_dict in polarity_claims.items():
        if "positive" in pol_dict and "negative" in pol_dict:
            pos_chunk = pol_dict["positive"]
            neg_chunk = pol_dict["negative"]
            if pos_chunk != neg_chunk:
                conflicts.append({
                    "type": "feature_availability_conflict",
                    "attribute": feature,
                    "chunks_involved": [pos_chunk, neg_chunk],
                    "description": f"Conflicting availability statements for '{feature}': affirmative in chunk '{pos_chunk}' vs negative/optional in chunk '{neg_chunk}'."
                })

    return conflicts


class EvidenceEvaluator:
    """
    Evaluates relevance and sufficiency of retrieved knowledge chunks against a query.
    Provides transparent, deterministic assessment and machine-readable abstention decisions.
    """

    def __init__(self, min_relevance_threshold: float = 0.35) -> None:
        """
        Initialize the EvidenceEvaluator.
        :param min_relevance_threshold: Minimum salient term overlap ratio required for a chunk
                                         to be considered relevant when query has >= 3 salient terms.
        """
        self.min_relevance_threshold = min_relevance_threshold

    def assess_chunk_relevance(
        self,
        query: str,
        chunk: Dict[str, Any],
        salient_query_terms: List[str]
    ) -> Tuple[bool, str, List[str]]:
        """
        Assess whether a single retrieved chunk is relevant to the query.
        Returns: (is_relevant, relevance_reason, matched_terms)
        """
        if not salient_query_terms:
            return False, "Query contains no meaningful non-stopword terms.", []

        # Combine text, section heading, and heading path for matching
        text = chunk.get("text", "")
        citation_metadata = chunk.get("citation_metadata", {})
        section_heading = citation_metadata.get("section_heading", "") or chunk.get("section_heading", "")
        heading_path = citation_metadata.get("heading_path", "") or chunk.get("heading_path", "")
        doc_title = citation_metadata.get("document_title", "") or chunk.get("document_title", "")

        combined_text = f"{doc_title} {heading_path} {section_heading} {text}".lower()
        chunk_tokens_set = set(tokenize_text(combined_text))

        matched = [t for t in salient_query_terms if t in chunk_tokens_set]
        unique_salient = sorted(set(salient_query_terms))
        unique_matched = sorted(set(matched))

        coverage_ratio = len(unique_matched) / len(unique_salient) if unique_salient else 0.0

        # Check for phrase match (2-grams or 3-grams of salient query terms)
        has_phrase_match = False
        if len(salient_query_terms) >= 2:
            for n in (3, 2):
                for i in range(len(salient_query_terms) - n + 1):
                    ngram = " ".join(salient_query_terms[i:i+n]).lower()
                    if ngram in combined_text:
                        has_phrase_match = True
                        break
                if has_phrase_match:
                    break

        # Check for numeric matches if query contains numbers (e.g. "380")
        query_numbers = [t for t in salient_query_terms if t.isdigit() or re.match(r"^\d+\.\d+$", t)]
        matched_numbers = [num for num in query_numbers if num in chunk_tokens_set]

        # Check for sub-aspect match if compound query
        sub_aspects = split_query_sub_aspects(query)
        matched_sub_aspect = False
        if len(sub_aspects) > 1:
            for aspect in sub_aspects:
                aspect_terms = extract_salient_terms(aspect)
                if aspect_terms:
                    matched_aspect_terms = [t for t in set(aspect_terms) if t in chunk_tokens_set]
                    if len(matched_aspect_terms) >= 2 or (len(set(aspect_terms)) <= 2 and len(matched_aspect_terms) >= 1):
                        matched_sub_aspect = True
                        break

        # Decision rules for chunk relevance:
        is_relevant = False
        reason = ""

        if len(unique_salient) <= 2:
            # Short query: at least 1 salient term must match
            if len(unique_matched) >= 1:
                is_relevant = True
                reason = f"Matched {len(unique_matched)}/{len(unique_salient)} salient terms: {unique_matched}."
            else:
                is_relevant = False
                reason = f"None of the {len(unique_salient)} salient query terms matched."
        else:
            # Query with 3+ salient terms:
            # Relevant if coverage >= threshold, OR phrase match found, OR key number matched with some overlap, OR matched a sub-aspect
            if coverage_ratio >= self.min_relevance_threshold:
                is_relevant = True
                reason = f"High keyword overlap: matched {len(unique_matched)}/{len(unique_salient)} terms ({coverage_ratio:.0%}): {unique_matched}."
            elif has_phrase_match:
                is_relevant = True
                reason = f"Matched query phrase and terms: {unique_matched}."
            elif matched_sub_aspect:
                is_relevant = True
                reason = f"Relevant to sub-aspect: matched terms {unique_matched}."
            elif query_numbers and len(matched_numbers) == len(query_numbers) and len(unique_matched) >= 2:
                is_relevant = True
                reason = f"Matched key numeric terms {matched_numbers} and context terms: {unique_matched}."
            else:
                is_relevant = False
                reason = f"Insufficient term overlap: matched only {len(unique_matched)}/{len(unique_salient)} terms ({coverage_ratio:.0%}): {unique_matched}."

        return is_relevant, reason, unique_matched

    def evaluate(
        self,
        query: str,
        retrieved_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate retrieved results against the query.
        Returns a structured evaluation report including:
        - status: 'sufficient' | 'partially_sufficient' | 'insufficient' | 'conflicting' | 'no_evidence'
        - is_sufficient: bool (True only when status == 'sufficient')
        - reason: str
        - relevant_evidence: List[Dict]
        - irrelevant_evidence: List[Dict]
        - missing_information: List[str]
        - conflicts: List[Dict]
        """
        # 1. Input Validation
        if not isinstance(query, str):
            raise TypeError(f"Query must be a string, got {type(query).__name__}.")

        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query must not be empty or whitespace-only.")

        if not isinstance(retrieved_results, list):
            raise TypeError(f"retrieved_results must be a list, got {type(retrieved_results).__name__}.")

        # Handle empty retrieved_results immediately
        if not retrieved_results:
            return {
                "status": "no_evidence",
                "is_sufficient": False,
                "reason": "No evidence was retrieved for the query.",
                "relevant_evidence": [],
                "irrelevant_evidence": [],
                "missing_information": [f"All requested information for: '{clean_query}'"],
                "conflicts": [],
            }

        # Validate each item in retrieved_results and safely deduplicate by chunk_id
        validated_results: List[Dict[str, Any]] = []
        seen_chunk_ids: Set[str] = set()

        for idx, item in enumerate(retrieved_results):
            if not isinstance(item, dict):
                raise ValueError(f"Retrieved result at index {idx} is not a dictionary ({item!r}).")

            chunk_id = item.get("chunk_id")
            if not chunk_id or not isinstance(chunk_id, str):
                raise ValueError(f"Retrieved result at index {idx} is missing or has invalid 'chunk_id'.")

            text = item.get("text")
            if text is None or not isinstance(text, str):
                raise ValueError(f"Retrieved result at index {idx} ({chunk_id}) is missing or has invalid 'text'.")

            citation_metadata = item.get("citation_metadata")
            if not isinstance(citation_metadata, dict):
                raise ValueError(f"Retrieved result at index {idx} ({chunk_id}) is missing 'citation_metadata'.")

            # Security guard against evaluation question leakage
            if (
                "query_text" in citation_metadata
                or "expected_answer" in citation_metadata
                or "query_text" in item
                or "expected_answer" in item
            ):
                raise ValueError(f"Evaluation question detected in retrieved chunk '{chunk_id}'.")

            # Handle duplicate chunk IDs safely by keeping the first occurrence
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            validated_results.append(item)

        # 2. Assess Individual Evidence Relevance
        salient_query_terms = extract_salient_terms(clean_query)
        relevant_evidence: List[Dict[str, Any]] = []
        irrelevant_evidence: List[Dict[str, Any]] = []

        for item in validated_results:
            is_rel, rel_reason, matched_terms = self.assess_chunk_relevance(
                query=clean_query,
                chunk=item,
                salient_query_terms=salient_query_terms
            )

            # Build standardized evidence record preserving all original text and citation metadata
            meta = item.get("citation_metadata", {})
            evidence_record: Dict[str, Any] = {
                "chunk_id": item["chunk_id"],
                "text": item["text"],
                "citation_metadata": dict(meta),
                "source_document": meta.get("source_document") or item.get("source_document", ""),
                "document_title": meta.get("document_title") or item.get("document_title", ""),
                "section_heading": meta.get("section_heading") or item.get("section_heading", ""),
                "heading_path": meta.get("heading_path") or item.get("heading_path", ""),
                "rrf_rank": item.get("rrf_rank"),
                "rrf_score": item.get("rrf_score"),
                "is_relevant": is_rel,
                "relevance_reason": rel_reason,
                "matched_terms": matched_terms,
            }
            if "bm25_rank" in item:
                evidence_record["bm25_rank"] = item["bm25_rank"]
            if "vector_rank" in item:
                evidence_record["vector_rank"] = item["vector_rank"]
            if "bm25_score" in item:
                evidence_record["bm25_score"] = item["bm25_score"]
            if "vector_distance" in item:
                evidence_record["vector_distance"] = item["vector_distance"]

            if is_rel:
                relevant_evidence.append(evidence_record)
            else:
                irrelevant_evidence.append(evidence_record)

        # 3. Assess Collective Evidence Sufficiency
        # Case A: No relevant evidence found
        if not relevant_evidence:
            return {
                "status": "no_evidence",
                "is_sufficient": False,
                "reason": "Retrieved chunks do not contain meaningful relevance to the query.",
                "relevant_evidence": [],
                "irrelevant_evidence": irrelevant_evidence,
                "missing_information": [f"All requested information for: '{clean_query}'"],
                "conflicts": [],
            }

        # Case B: Check for conflicting evidence among relevant chunks
        conflicts = detect_conflicts(relevant_evidence)
        if conflicts:
            return {
                "status": "conflicting",
                "is_sufficient": False,
                "reason": f"Conflicting evidence detected across retrieved chunks: {conflicts[0]['description']}",
                "relevant_evidence": relevant_evidence,
                "irrelevant_evidence": irrelevant_evidence,
                "missing_information": ["Resolution of conflicting claims among source documents."],
                "conflicts": conflicts,
            }

        # Case C: Multi-part query analysis
        sub_aspects = split_query_sub_aspects(clean_query)
        collective_relevant_text = " ".join([
            f"{c['document_title']} {c['heading_path']} {c['section_heading']} {c['text']}"
            for c in relevant_evidence
        ]).lower()
        collective_tokens = set(tokenize_text(collective_relevant_text))

        uncovered_aspects: List[str] = []
        covered_aspects: List[str] = []

        for aspect in sub_aspects:
            aspect_terms = extract_salient_terms(aspect)
            if not aspect_terms:
                continue
            matched_aspect_terms = [t for t in aspect_terms if t in collective_tokens]
            aspect_coverage = len(matched_aspect_terms) / len(set(aspect_terms)) if aspect_terms else 0.0

            # Sub-aspect is covered if at least 50% of its salient terms match, or >= 1 term for short aspects
            threshold = 0.5 if len(set(aspect_terms)) >= 3 else 0.4
            if aspect_coverage >= threshold:
                covered_aspects.append(aspect)
            else:
                uncovered_aspects.append(aspect)

        # Case D: Decision based on aspect coverage & overall keyword support
        total_unique_salient = set(salient_query_terms)
        collective_matched_query_terms = [t for t in total_unique_salient if t in collective_tokens]
        overall_coverage = len(collective_matched_query_terms) / len(total_unique_salient) if total_unique_salient else 0.0

        if len(sub_aspects) > 1 and uncovered_aspects:
            # Multi-part question with some covered and some missing aspects
            if covered_aspects:
                return {
                    "status": "partially_sufficient",
                    "is_sufficient": False,
                    "reason": f"Evidence partially covers the query: {len(covered_aspects)}/{len(sub_aspects)} sub-aspects covered. Missing information for: {uncovered_aspects}.",
                    "relevant_evidence": relevant_evidence,
                    "irrelevant_evidence": irrelevant_evidence,
                    "missing_information": uncovered_aspects,
                    "conflicts": [],
                }
            else:
                return {
                    "status": "insufficient",
                    "is_sufficient": False,
                    "reason": f"Evidence is insufficient: none of the sub-aspects were substantively answered.",
                    "relevant_evidence": relevant_evidence,
                    "irrelevant_evidence": irrelevant_evidence,
                    "missing_information": uncovered_aspects,
                    "conflicts": [],
                }

        # Single aspect or multi-aspect with all sub-aspects covered
        all_aspects_covered = len(sub_aspects) > 1 and len(covered_aspects) == len(sub_aspects) and not uncovered_aspects

        if all_aspects_covered or overall_coverage >= 0.60 or (len(total_unique_salient) <= 2 and len(collective_matched_query_terms) >= 1):
            return {
                "status": "sufficient",
                "is_sufficient": True,
                "reason": f"The retrieved evidence addresses the requested information with {overall_coverage:.0%} query concept coverage across {len(relevant_evidence)} relevant chunk(s).",
                "relevant_evidence": relevant_evidence,
                "irrelevant_evidence": irrelevant_evidence,
                "missing_information": [],
                "conflicts": [],
            }
        elif overall_coverage >= 0.40:
            missing_terms = [t for t in total_unique_salient if t not in collective_tokens]
            return {
                "status": "partially_sufficient",
                "is_sufficient": False,
                "reason": f"Evidence is partially sufficient ({overall_coverage:.0%} coverage), but key terms are missing: {missing_terms}.",
                "relevant_evidence": relevant_evidence,
                "irrelevant_evidence": irrelevant_evidence,
                "missing_information": [f"Details regarding: {', '.join(missing_terms)}"],
                "conflicts": [],
            }
        else:
            missing_terms = [t for t in total_unique_salient if t not in collective_tokens]
            return {
                "status": "insufficient",
                "is_sufficient": False,
                "reason": f"Evidence is insufficient: only {overall_coverage:.0%} of query concepts are mentioned.",
                "relevant_evidence": relevant_evidence,
                "irrelevant_evidence": irrelevant_evidence,
                "missing_information": [f"Substantive facts for: '{clean_query}'"],
                "conflicts": [],
            }
