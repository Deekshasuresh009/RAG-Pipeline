"""
Phase 2, Step 14: Evaluation Runner Script
Runs the full evaluation pipeline on the Aventro Motors evaluation dataset.

Usage:
    python -m src.phase2.run_evaluation [--smoke N] [--no-ragas] [--llm-provider fake]

Options:
    --smoke N         Run only the first N evaluation questions (smoke test).
    --no-ragas        Skip all RAGAS metric computation.
    --llm-provider    LLM provider for answer generation: 'fake' or 'ollama' (default: fake).
    --top-k N         Number of chunks to retrieve per question (default: 5).
    --output-dir DIR  Directory for evaluation output files (default: artifacts/evaluation/).
    --label LABEL     Label for output files (default: 'eval').

Examples:
    # Smoke test with 5 questions, fake LLM, no external metrics
    python -m src.phase2.run_evaluation --smoke 5 --llm-provider fake

    # Full evaluation with Ollama LLM (must have ollama running locally)
    python -m src.phase2.run_evaluation --llm-provider ollama --smoke 20

IMPORTANT:
    - The evaluation dataset is loaded from artifacts/extracted/evaluation_questions.jsonl.
    - Evaluation questions are NEVER indexed into ChromaDB.
    - Expected answers are NEVER passed to the generation model.
    - Results are saved to artifacts/evaluation/ as JSONL + summary JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_evaluation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aventro Motors RAG Evaluation Runner")
    p.add_argument("--smoke", type=int, default=None, metavar="N",
                   help="Run only the first N questions (smoke test).")
    p.add_argument("--no-ragas", action="store_true", help="Disable RAGAS metric computation.")
    p.add_argument("--llm-provider", default="fake", choices=["fake", "ollama"],
                   help="LLM provider for answer generation.")
    p.add_argument("--top-k", type=int, default=5, metavar="K",
                   help="Top-K chunks to retrieve (default 5).")
    p.add_argument("--output-dir", default="artifacts/evaluation", metavar="DIR",
                   help="Output directory for results.")
    p.add_argument("--label", default="eval", metavar="LABEL",
                   help="Filename label for output files.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from src.phase2.rag_evaluator import (
        EVAL_DATASET_PATH,
        RagasEvaluator,
        load_canonical_chunk_ids,
        load_evaluation_dataset,
        run_batch_evaluation,
    )
    from src.phase2.citation_verifier import CitationVerifier, HeuristicEntailmentVerifier
    from src.phase2.answer_generator import AnswerGenerator, FakeLLMClient, OllamaClient
    from src.phase2.rag_graph import build_rag_graph

    output_dir = Path(args.output_dir)
    smoke_suffix = f"_smoke{args.smoke}" if args.smoke else ""
    output_filename = f"{args.label}{smoke_suffix}_results.jsonl"

    # -----------------------------------------------------------------------
    # 1. Load and validate evaluation dataset
    # -----------------------------------------------------------------------
    logger.info("Loading evaluation dataset from %s", EVAL_DATASET_PATH)
    records, rejected = load_evaluation_dataset(EVAL_DATASET_PATH)
    logger.info("Valid records: %d | Rejected: %d", len(records), len(rejected))
    if rejected:
        logger.warning("Rejected records: %s", json.dumps(rejected[:5], indent=2))

    # -----------------------------------------------------------------------
    # 2. Load canonical chunk IDs (for citation existence checks)
    # -----------------------------------------------------------------------
    canonical_chunk_ids = load_canonical_chunk_ids()
    logger.info("Loaded %d canonical chunk IDs.", len(canonical_chunk_ids))

    # -----------------------------------------------------------------------
    # 3. Build LLM client for generation (NEVER passes expected answers to it)
    # -----------------------------------------------------------------------
    if args.llm_provider == "ollama":
        llm_client = OllamaClient()
        logger.info("Using Ollama LLM client for generation.")
    else:
        llm_client = FakeLLMClient()
        logger.info("Using FakeLLMClient for generation (offline mode).")

    generator = AnswerGenerator(llm_client=llm_client)

    # -----------------------------------------------------------------------
    # 4. Build the LangGraph RAG pipeline
    # -----------------------------------------------------------------------
    graph = build_rag_graph(top_k=args.top_k, answer_generator=generator)
    logger.info("RAG graph built with top_k=%d.", args.top_k)

    # -----------------------------------------------------------------------
    # 5. Configure RAGAS evaluator
    # -----------------------------------------------------------------------
    ragas_evaluator = None
    if not args.no_ragas:
        ragas_evaluator = RagasEvaluator(
            langchain_llm=None,          # No external LLM — only offline metrics run
            use_llm_metrics=False,       # LLM metrics require explicit configuration
        )
        logger.info(
            "RAGAS evaluator configured. Only offline metrics (non_llm_context_precision, "
            "non_llm_context_recall, rouge_score) will be computed. "
            "LLM-based metrics (faithfulness, answer_relevancy, etc.) are UNAVAILABLE "
            "without a configured LangChain LLM provider."
        )
    else:
        logger.info("RAGAS metrics disabled by --no-ragas flag.")

    # -----------------------------------------------------------------------
    # 6. Configure citation verifier
    # -----------------------------------------------------------------------
    citation_verifier = CitationVerifier(entailment_verifier=HeuristicEntailmentVerifier())

    # -----------------------------------------------------------------------
    # 7. Run evaluation
    # -----------------------------------------------------------------------
    smoke_label = f"SMOKE ({args.smoke} records)" if args.smoke else "FULL"
    logger.info(
        "Starting %s evaluation: %d records, llm=%s",
        smoke_label, min(len(records), args.smoke or len(records)), args.llm_provider
    )

    results, summary = run_batch_evaluation(
        records=records,
        graph=graph,
        ragas_evaluator=ragas_evaluator,
        citation_verifier=citation_verifier,
        canonical_chunk_ids=canonical_chunk_ids,
        top_k=args.top_k,
        max_records=args.smoke,
        output_dir=output_dir,
        output_filename=output_filename,
        smoke_label=smoke_label,
    )

    # -----------------------------------------------------------------------
    # 8. Print summary
    # -----------------------------------------------------------------------
    summary_dict = summary.to_dict()
    logger.info("\n=== EVALUATION SUMMARY ===")
    logger.info("Records: %s", summary_dict["records"])
    logger.info("Generation: %s", summary_dict["generation"])
    logger.info("Evidence status: %s", summary_dict["evidence_status"])
    logger.info("Workflow status: %s", summary_dict["workflow_status"])
    logger.info("RAGAS mean scores: %s", summary_dict["ragas_mean_scores"])
    logger.info("RAGAS unavailable: %s", list(summary_dict["ragas_unavailable"].keys()))
    logger.info("Citation quality: %s", summary_dict["citation_quality"])

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print(f"  Records attempted : {summary_dict['records']['attempted']}")
    print(f"  Succeeded         : {summary_dict['records']['succeeded']}")
    print(f"  Failed            : {summary_dict['records']['failed']}")
    print(f"  Generated answers : {summary_dict['generation']['generated']}")
    print(f"  Abstentions       : {summary_dict['generation']['abstentions']}")
    print(f"  Results saved to  : {output_dir / output_filename}")


if __name__ == "__main__":
    main()
