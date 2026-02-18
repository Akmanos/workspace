from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from typing import Dict, List, Optional, Any, Tuple
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

# Configuration constants
EVALUATOR_LLM_MODEL = "gpt-3.5-turbo"
EVALUATOR_EMBEDDINGS_MODEL = "text-embedding-3-small"
OPENAI_BASE_URL = "https://openai.vocareum.com/v1"
DEFAULT_CHROMA_DIR = "./chroma_db_openai"
DEFAULT_COLLECTION_NAME = "nasa_space_missions_text"

# Module-level cached instances (created once per module load)
_REFERENCE_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHED_EVALUATOR_LLM: Optional[LangchainLLMWrapper] = None
_CACHED_EVALUATOR_EMBEDDINGS: Optional[LangchainEmbeddingsWrapper] = None


def _get_evaluator_instances() -> Tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    """Get or create cached evaluator LLM and embeddings instances."""
    global _CACHED_EVALUATOR_LLM, _CACHED_EVALUATOR_EMBEDDINGS
    
    if _CACHED_EVALUATOR_LLM is None:
        _CACHED_EVALUATOR_LLM = LangchainLLMWrapper(
            ChatOpenAI(
                model=EVALUATOR_LLM_MODEL,
                temperature=0,
                base_url=OPENAI_BASE_URL,
            )
        )
    
    if _CACHED_EVALUATOR_EMBEDDINGS is None:
        _CACHED_EVALUATOR_EMBEDDINGS = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=EVALUATOR_EMBEDDINGS_MODEL,
                base_url=OPENAI_BASE_URL,
            )
        )
    
    return _CACHED_EVALUATOR_LLM, _CACHED_EVALUATOR_EMBEDDINGS


def _filter_metrics(
    has_reference: bool, has_reference_contexts: bool
) -> List[Any]:
    """Determine which metrics to run based on available reference data."""
    evaluator_llm, evaluator_embeddings = _get_evaluator_instances()
    
    metrics = [
        BleuScore(),
        RougeScore(),
        NonLLMContextPrecisionWithReference(),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        Faithfulness(llm=evaluator_llm),
        ContextRelevance(llm=evaluator_llm),
    ]
    
    # Remove metrics that require reference answer if not available
    if not has_reference:
        metrics = [
            metric for metric in metrics
            if not isinstance(metric, (BleuScore, RougeScore, NonLLMContextPrecisionWithReference))
        ]
    
    # Remove metric that specifically requires reference contexts if not available
    if not has_reference_contexts:
        metrics = [
            metric for metric in metrics
            if not isinstance(metric, NonLLMContextPrecisionWithReference)
        ]
    return metrics


def evaluate_response_quality(
    question: str, answer: str, contexts: List[str]
) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    try:
        evaluator_llm, evaluator_embeddings = _get_evaluator_instances()

        # Get reference data for this question (cached)
        ref_entry = _REFERENCE_CACHE.get(question.strip()) or {}
        reference_answer = ref_entry.get("reference")
        reference_contexts = ref_entry.get("reference_contexts")
        has_reference_contexts = isinstance(reference_contexts, list) and len(reference_contexts) > 0

        # Determine which metrics to use based on available references
        metrics = _filter_metrics(bool(reference_answer), has_reference_contexts)

        # Build sample data
        sample_kwargs = {
            "question": question,
            "user_input": question,  # keep for compatibility
            "response": answer,
            "retrieved_contexts": contexts or [],
        }

        if reference_answer:
            sample_kwargs["reference"] = reference_answer
        if has_reference_contexts:
            sample_kwargs["reference_contexts"] = reference_contexts

        # Evaluate
        dataset_obj = EvaluationDataset.from_list([sample_kwargs])
        result = evaluate(dataset=dataset_obj, metrics=metrics)

        # Extract numeric results
        df = result.to_pandas()
        row = df.iloc[0]

        # Return only computed metric scores (exclude metadata columns)
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

    # Cache references for reference-based metrics (load from questions file)
    global _REFERENCE_CACHE
    _REFERENCE_CACHE = {}
    for item in data:
        question = (item.get("question") or "").strip()
        reference = item.get("reference")
        reference_contexts = item.get("reference_contexts")
        
        if not question or not isinstance(reference, str) or not reference.strip():
            continue
        
        entry: Dict[str, Any] = {"reference": reference.strip()}
        if isinstance(reference_contexts, list) and all(isinstance(x, str) for x in reference_contexts):
            entry["reference_contexts"] = [x.strip() for x in reference_contexts if x.strip()]
        
        _REFERENCE_CACHE[question] = entry

    # Initialize RAG system
    collection, success, error = rag_client.initialize_rag_system(chroma_dir, collection_name)
    if not success:
        raise RuntimeError(f"Failed to init RAG system: {error}")

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
        "--chroma-dir", default=DEFAULT_CHROMA_DIR, help="ChromaDB persist directory"
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
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
