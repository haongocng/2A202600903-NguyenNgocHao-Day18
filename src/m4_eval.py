from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os
import sys
import json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from langchain_cohere import CohereEmbeddings
        from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, COHERE_API_KEY
        
        # Instantiate custom LLM and Embeddings to use
        # Ragas will use these instead of default OpenAI models
        evaluator_llm = ChatOpenAI(
            model=OPENAI_MODEL,
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_BASE_URL if OPENAI_BASE_URL else None
        )
        
        evaluator_embeddings = CohereEmbeddings(
            cohere_api_key=COHERE_API_KEY,
            model="embed-multilingual-v3.0"
        )
        
        dataset = Dataset.from_dict({
            "user_input": questions,
            "response": answers,
            "retrieved_contexts": contexts,
            "reference": ground_truths,
        })
        
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )
        
        df = result.to_pandas()
        per_question = []
        for _, row in df.iterrows():
            per_question.append(EvalResult(
                question=row["user_input"],
                answer=row["response"],
                contexts=row["retrieved_contexts"],
                ground_truth=row["reference"],
                faithfulness=float(row.get("faithfulness") or 0.0) if not row.get("faithfulness") is None else 0.0,
                answer_relevancy=float(row.get("answer_relevancy") or 0.0) if not row.get("answer_relevancy") is None else 0.0,
                context_precision=float(row.get("context_precision") or 0.0) if not row.get("context_precision") is None else 0.0,
                context_recall=float(row.get("context_recall") or 0.0) if not row.get("context_recall") is None else 0.0
            ))
            
        scores_dict = getattr(result, "_repr_dict", result)
        return {
            "faithfulness": float(scores_dict.get("faithfulness") or 0.0),
            "answer_relevancy": float(scores_dict.get("answer_relevancy") or 0.0),
            "context_precision": float(scores_dict.get("context_precision") or 0.0),
            "context_recall": float(scores_dict.get("context_recall") or 0.0),
            "per_question": per_question
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}", flush=True)
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": []
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }
    
    analyzed = []
    for r in eval_results:
        metrics = {
            "faithfulness": r.faithfulness,
            "context_recall": r.context_recall,
            "context_precision": r.context_precision,
            "answer_relevancy": r.answer_relevancy
        }
        
        # Calculate average score of all 4 metrics
        avg_score = sum(metrics.values()) / 4.0
        
        # Find the worst metric (lowest score)
        worst_metric = min(metrics, key=metrics.get)
        worst_score = metrics[worst_metric]
        
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        
        analyzed.append({
            "avg_score": avg_score,
            "question": r.question,
            "worst_metric": worst_metric,
            "score": float(worst_score),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix
        })
        
    # Sort by avg_score ascending (worst first)
    analyzed_sorted = sorted(analyzed, key=lambda x: x["avg_score"])[:bottom_n]
    
    # Strip avg_score
    for item in analyzed_sorted:
        item.pop("avg_score", None)
        
    return analyzed_sorted


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
