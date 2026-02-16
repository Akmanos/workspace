from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
import os
from statistics import mean
import argparse


# RAGAS imports
try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas import SingleTurnSample
    from ragas.metrics import (
        ContextRelevance,
        BleuScore,
        NonLLMContextPrecisionWithReference,
        ResponseRelevancy,
        Faithfulness,
        RougeScore,
    )
    from ragas import evaluate
    from ragas import EvaluationDataset

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

_REFERENCE_CACHE: Dict[str, str] = {}


def _load_reference_dataset(path: str = "test_questions.json") -> Dict[str, str]:
    """
    Load evaluation dataset containing reference answers.

    Expected format:
    [
        {
            "question": "...",
            "reference": "..."
        }
    ]
    """
    global _REFERENCE_CACHE

    dataset_path = Path(path)

    if not dataset_path.exists():
        return {}

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Map question -> reference
        _REFERENCE_CACHE = {
            item["question"].strip(): item["reference"]
            for item in data
            if "question" in item and "reference" in item
        }

        return _REFERENCE_CACHE
    except Exception:
        return {}


def evaluate_response_quality(
    question: str, answer: str, contexts: List[str]
) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    try:
        # Create evaluator LLM with model gpt-3.5-turbo
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                base_url="https://openai.vocareum.com/v1",
            )
        )

        # Create evaluator_embeddings with model test-embedding-3-small
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-small",
                base_url="https://openai.vocareum.com/v1",
            )
        )

        # Define an instance for each metric to evaluate
        metrics = [
            BleuScore(),
            RougeScore(),
            # This metric requires a ground-truth reference answer in the sample.
            # chat.py does not provide a reference, so we conditionally remove it below.
            NonLLMContextPrecisionWithReference(),
            ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            Faithfulness(llm=evaluator_llm),
            ContextRelevance(llm=evaluator_llm),
        ]

        # Attempt to load references once (cached)
        if not _REFERENCE_CACHE:
            _load_reference_dataset()
        reference_answer = _REFERENCE_CACHE.get(question.strip())

        # Evaluate the response using the metrics
        sample_kwargs = {
            "question": question,
            "user_input": question,  # keep for compatibility
            "response": answer,
            "retrieved_contexts": contexts or [],
        }

        if reference_answer:
            sample_kwargs["reference"] = reference_answer
        else:
            # No reference available: remove metrics that require it
            metrics = [
                m
                for m in metrics
                if not isinstance(
                    m, (NonLLMContextPrecisionWithReference, BleuScore, RougeScore)
                )
            ]

        # sample = SingleTurnSample(**sample_kwargs)
        # dataset_obj = EvaluationDataset.from_list([sample])
        dataset_obj = EvaluationDataset.from_list([sample_kwargs])

        result = evaluate(
            dataset=dataset_obj,
            metrics=metrics,
        )

        # Return the evaluation results
        df = result.to_pandas()
        row = df.iloc[0]

        # Return ONLY metrics that were run
        drop_cols = {
            "question",
            "user_input",
            "response",
            "retrieved_contexts",
            "reference",
        }
        out: Dict[str, float] = {}
        for k, v in row.items():
            if k in drop_cols:
                continue
            try:
                out[k] = float(v)
            except Exception:
                pass

        return out
    except Exception as e:
        return {"error": f"Evaluation failed: {e}"}


def run_batch_evaluation(
    openai_key: str,
    chroma_dir: str,
    collection_name: str,
    questions_path: str = "test_questions.json",
    n_results: int = 3,
    mission_filter: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
) -> Dict[str, float]:
    """
    Batch evaluation runner.

    Loads test questions from JSON, runs retrieval + LLM generation + RAGAS metrics,
    prints per-question results, and returns aggregate means per metric.
    """
    # Local imports to avoid import cycles and keep module import lightweight.
    import rag_client
    import llm_client

    os.environ["OPENAI_API_KEY"] = openai_key  # used by rag_client retrieval embedding

    # Load questions
    path = Path(questions_path)
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Cache references for reference-based metrics (if present)
    global _REFERENCE_CACHE
    _REFERENCE_CACHE = {}
    for item in data:
        q = (item.get("question") or "").strip()
        r = item.get("reference")
        if q and isinstance(r, str) and r.strip():
            _REFERENCE_CACHE[q] = r.strip()

    # Init RAG
    collection, ok, err = rag_client.initialize_rag_system(chroma_dir, collection_name)
    if not ok:
        raise RuntimeError(f"Failed to init RAG system: {err}")

    per_metric_values: Dict[str, List[float]] = {}
    per_q_results: List[Dict[str, Any]] = []

    for idx, item in enumerate(data, start=1):
        question = (item.get("question") or "").strip()
        if not question:
            continue

        # Retrieve
        docs_result = rag_client.retrieve_documents(
            collection=collection,
            query=question,
            n_results=n_results,
            mission_filter=mission_filter,
        )

        contexts_list: List[str] = []
        context_str = ""
        if docs_result and docs_result.get("documents"):
            contexts_list = docs_result["documents"][0] or []
            metadatas = (docs_result.get("metadatas") or [[]])[0]
            context_str = rag_client.format_context(contexts_list, metadatas)

        # Generate
        answer = llm_client.generate_response(
            openai_key=openai_key,
            user_message=question,
            context=context_str,
            conversation_history=[],
            model=model,
        )

        # Evaluate
        scores = evaluate_response_quality(question, answer, contexts_list)

        per_q_results.append(
            {
                "i": idx,
                "question": question,
                "answer": answer,
                "scores": scores,
            }
        )

        # Print per-question summary
        print("=" * 60)
        print(f"Q{idx}: {question}")
        print(f"A{idx}: {answer}")
        if "error" in scores:
            print(f"Scores: ERROR - {scores['error']}")
        else:
            print("Scores:")
            for k, v in scores.items():
                print(f"  {k}: {v:.4f}")
                per_metric_values.setdefault(k, []).append(v)

    # Aggregate means
    aggregates: Dict[str, float] = {}
    for k, vals in per_metric_values.items():
        if vals:
            aggregates[k] = float(mean(vals))

    print("\n" + "#" * 60)
    print("AGGREGATE METRICS (MEAN)")
    for k, v in aggregates.items():
        print(f"{k}: {v:.4f}")
    print("#" * 60 + "\n")

    return aggregates


def main():
    parser = argparse.ArgumentParser(
        description="Run batch RAGAS evaluation on a test set"
    )
    parser.add_argument("--openai-key", required=True, help="OpenAI API key")
    parser.add_argument(
        "--chroma-dir", default="./chroma_db_openai", help="ChromaDB persist directory"
    )
    parser.add_argument(
        "--collection-name",
        default="nasa_space_missions_text",
        help="ChromaDB collection name",
    )
    parser.add_argument(
        "--test-questions-path",
        default="test_questions.json",
        help="Path to test questions JSON file",
    )
    parser.add_argument(
        "--topk", type=int, default=3, help="Top-k docs to retrieve per question"
    )
    parser.add_argument(
        "--mission-filter",
        default=None,
        help="Mission filter (e.g., apollo_13). Use 'all' for no filter.",
    )
    parser.add_argument(
        "--model", default="gpt-3.5-turbo", help="LLM model to use for answers"
    )
    args = parser.parse_args()

    mission_filter = args.mission_filter
    if mission_filter and mission_filter.strip().lower() in {"all", "*", "any"}:
        mission_filter = None

    aggregates = run_batch_evaluation(
        openai_key=args.openai_key,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        questions_path=args.test_questions_path,
        n_results=args.topk,
        mission_filter=mission_filter,
        model=args.model,
    )

    # Return code-friendly summary line at end
    print("DONE. Aggregate metrics:", aggregates)


if __name__ == "__main__":
    main()
