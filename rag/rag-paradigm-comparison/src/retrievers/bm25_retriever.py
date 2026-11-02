"""Vectorless RAG: pure lexical BM25 search, no embeddings/vector DB."""
import pickle
from functools import lru_cache

from src.common.config import INDEX_DIR
from src.ingestion.bm25_ingest import tokenize
from src.retrievers.base import Retriever


@lru_cache(maxsize=8)
def _load(vertical: str):
    with (INDEX_DIR / f"{vertical}_bm25.pkl").open("rb") as f:
        return pickle.load(f)


class BM25Retriever(Retriever):
    name = "bm25"

    def retrieve(self, vertical: str, query: str, top_k: int = 5) -> list[dict]:
        store = _load(vertical)
        bm25, docs = store["bm25"], store["docs"]
        scores = bm25.get_scores(tokenize(query))
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"doc_id": d["doc_id"], "text": d["text"], "score": float(s)} for d, s in ranked]
