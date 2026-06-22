from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "jina-reranker-v2-base-multilingual"):
        self.model_name = model_name

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents using Jina AI API: top-20 → top-k."""
        if not documents:
            return []
            
        import requests
        from config import JINA_API_KEY
        
        headers = {
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json"
        }
        
        doc_texts = [doc["text"] for doc in documents]
        data = {
            "model": self.model_name,
            "query": query,
            "documents": doc_texts,
            "top_n": top_k
        }
        
        try:
            res = requests.post("https://api.jina.ai/v1/rerank", headers=headers, json=data)
            res.raise_for_status()
            results = res.json()["results"]
            
            reranked_results = []
            for r in results:
                idx = r["index"]
                score = r["relevance_score"]
                original_doc = documents[idx]
                
                reranked_results.append(RerankResult(
                    text=original_doc["text"],
                    original_score=float(original_doc.get("score", 0.0)),
                    rerank_score=float(score),
                    metadata=original_doc.get("metadata", {}),
                    rank=len(reranked_results)
                ))
            return reranked_results
            
        except Exception as e:
            # Fallback if API fails: return documents sorted by original score
            print(f"  ⚠️  Jina Reranker API failed: {e}. Falling back to original scoring.", flush=True)
            scored = sorted(documents, key=lambda x: x.get("score", 0.0), reverse=True)[:top_k]
            return [
                RerankResult(
                    text=doc["text"],
                    original_score=float(doc.get("score", 0.0)),
                    rerank_score=float(doc.get("score", 0.0)),
                    metadata=doc.get("metadata", {}),
                    rank=i
                )
                for i, doc in enumerate(scored)
            ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        pass

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        return []


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
