"""Hybrid RAG: reciprocal rank fusion of vector (dense) and BM25 (lexical)
result lists — the classic "best of both worlds" retrieval strategy."""
from src.retrievers.base import Retriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.vector_retriever import VectorRetriever


class HybridRetriever(Retriever):
    name = "hybrid"

    def __init__(self, k_rrf: int = 60):
        self.vector = VectorRetriever()
        self.bm25 = BM25Retriever()
        self.k_rrf = k_rrf

    def retrieve(self, vertical: str, query: str, top_k: int = 5) -> list[dict]:
        vec_results = self.vector.retrieve(vertical, query, top_k=top_k * 2)
        bm25_results = self.bm25.retrieve(vertical, query, top_k=top_k * 2)

        fused_scores: dict[str, float] = {}
        texts: dict[str, str] = {}
        for rank_list in (vec_results, bm25_results):
            for rank, r in enumerate(rank_list):
                fused_scores[r["doc_id"]] = fused_scores.get(r["doc_id"], 0.0) + 1.0 / (
                    self.k_rrf + rank + 1
                )
                texts[r["doc_id"]] = r["text"]

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"doc_id": doc_id, "text": texts[doc_id], "score": score} for doc_id, score in ranked]
