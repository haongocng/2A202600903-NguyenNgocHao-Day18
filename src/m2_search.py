from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os
import sys
import json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K,
                    COHERE_API_KEY, WEAVIATE_URL, WEAVIATE_API_KEY)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    from underthesea import word_tokenize
    try:
        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception as e:
        # Fallback if underthesea has any issue
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = []
        for chunk in chunks:
            segmented_text = segment_vietnamese(chunk["text"])
            tokens = segmented_text.lower().split()
            self.corpus_tokens.append(tokens)
        
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or not self.documents:
            return []
            
        tokenized_query = segment_vietnamese(query).lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort indices descending by score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            score = scores[idx]
            if score > 0:
                doc = self.documents[idx]
                results.append(SearchResult(
                    text=doc["text"],
                    score=float(score),
                    metadata=doc.get("metadata", {}),
                    method="bm25"
                ))
        return results


class DenseSearch:
    def __init__(self):
        self.client = None

    def _get_client(self):
        if self.client is None:
            import weaviate
            url = WEAVIATE_URL
            if not url.startswith("http"):
                url = f"https://{url}"
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=url,
                auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY)
            )
        return self.client

    def _get_weaviate_class_name(self, name: str) -> str:
        parts = name.split("_")
        return "".join(p.capitalize() for p in parts)

    def _encode_texts(self, texts: list[str], input_type: str = "search_document") -> list[list[float]]:
        import cohere
        co = cohere.Client(api_key=COHERE_API_KEY)
        all_embeddings = []
        batch_size = 90
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            res = co.embed(
                texts=batch,
                model="embed-multilingual-v3.0",
                input_type=input_type
            )
            all_embeddings.extend(res.embeddings)
        return all_embeddings

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Weaviate Cloud using Cohere Embeddings."""
        client = self._get_client()
        class_name = self._get_weaviate_class_name(collection)
        
        # Recreate collection
        if client.collections.exists(class_name):
            client.collections.delete(class_name)
            
        collection_obj = client.collections.create(
            name=class_name,
            vectorizer_config=None
        )
        
        if not chunks:
            return
            
        texts = [c["text"] for c in chunks]
        vectors = self._encode_texts(texts, input_type="search_document")
        
        with collection_obj.batch.dynamic() as batch:
            for i, chunk in enumerate(chunks):
                properties = {
                    "text": chunk["text"],
                    "source": chunk.get("metadata", {}).get("source", ""),
                    "chunk_type": chunk.get("metadata", {}).get("chunk_type", ""),
                    "parent_id": chunk.get("metadata", {}).get("parent_id", "") or "",
                    "section": chunk.get("metadata", {}).get("section", "") or "",
                    "strategy": chunk.get("metadata", {}).get("strategy", "") or "",
                    "metadata_json": json.dumps(chunk.get("metadata", {}))
                }
                batch.add_object(
                    properties=properties,
                    vector=vectors[i]
                )

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        client = self._get_client()
        class_name = self._get_weaviate_class_name(collection)
        
        if not client.collections.exists(class_name):
            return []
            
        collection_obj = client.collections.get(class_name)
        
        # Encode query
        query_vector = self._encode_texts([query], input_type="search_query")[0]
        
        import weaviate
        response = collection_obj.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=weaviate.classes.query.MetadataQuery(distance=True, score=True)
        )
        
        results = []
        for obj in response.objects:
            meta = {}
            if "metadata_json" in obj.properties:
                try:
                    meta = json.loads(obj.properties["metadata_json"])
                except:
                    pass
            
            # Convert cosine distance to similarity score
            distance = obj.metadata.distance if obj.metadata.distance is not None else 0.0
            similarity = 1.0 - distance
            
            results.append(SearchResult(
                text=obj.properties.get("text", ""),
                score=float(similarity),
                metadata=meta,
                method="dense"
            ))
            
        return results

    def close(self):
        if self.client:
            try:
                self.client.close()
            except:
                pass
            self.client = None

    def __del__(self):
        try:
            self.close()
        except:
            pass


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores = {}  # text -> {"score": float, "result": SearchResult}
    for res_list in results_list:
        for rank, result in enumerate(res_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {"score": 0.0, "result": result}
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)
            
    sorted_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=item["result"].metadata,
            method="hybrid"
        )
        for item in sorted_docs
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF."""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
