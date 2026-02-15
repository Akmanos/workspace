from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    # Create evaluator LLM with model gpt-3.5-turbo
    # (LangChain -> RAGAS wrapper)
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    )

    # Create evaluator_embeddings with model text-embedding-3-small
    evaluator_embeddings = LangchainEmbeddingsWrapper(
         OpenAIEmbeddings(model="text-embedding-3-small")
         )

     # Define an instance for each metric to evaluate
     metrics = [
          BleuScore(),
          RougeScore(),
          # requires reference/ground truth in sample
          NonLLMContextPrecisionWithReference(),
          ResponseRelevancy(llm=evaluator_llm,
                             embeddings=evaluator_embeddings),
          Faithfulness(llm=evaluator_llm),
          ]

      # Evaluate the response using the metrics
      sample_kwargs = {
           "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts or [],
           }

       if "reference" not in sample_kwargs:
            metrics = [m for m in metrics if not isinstance(
                m, NonLLMContextPrecisionWithReference)]

        sample = SingleTurnSample(**sample_kwargs)

        result = evaluate(
            dataset=[sample],
            metrics=metrics,
        )

        # Return the evaluation results
        out: Dict[str, float] = {}
        try:
            # RAGAS returns a pandas-like table; easiest path is to_dict() then flatten.
            d = result.to_pandas().to_dict(orient="records")[0]
            for k, v in d.items():
                if k in {"user_input", "response", "retrieved_contexts", "reference"}:
                    continue
                try:
                    out[k] = float(v)
                except Exception:
                    # keep non-floatable values out
                    pass
        except Exception:
            # fallback: result may already behave like a dict in some versions
            try:
                for k, v in dict(result).items():
                    try:
                        out[k] = float(v)
                    except Exception:
                        pass
            except Exception as e:
                return {"error": f"Failed to parse RAGAS result: {e}"}
        return out
