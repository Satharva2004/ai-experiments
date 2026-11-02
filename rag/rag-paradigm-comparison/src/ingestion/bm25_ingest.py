"""Vectorless RAG ingestion: build a local BM25 index per vertical (no
embeddings, no vector DB — pure lexical search)."""
import pickle

from rank_bm25 import BM25Okapi

from src.common.config import INDEX_DIR, VERTICALS
from src.ingestion.build_corpus import load_corpus

INDEX_DIR.mkdir(parents=True, exist_ok=True)


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_index(vertical: str):
    docs = load_corpus(vertical)
    tokenized = [tokenize(d["text"]) for d in docs]
    bm25 = BM25Okapi(tokenized)
    out_path = INDEX_DIR / f"{vertical}_bm25.pkl"
    with out_path.open("wb") as f:
        pickle.dump({"bm25": bm25, "docs": docs}, f)
    print(f"{vertical}: BM25 index built over {len(docs)} docs -> {out_path}")


if __name__ == "__main__":
    for v in VERTICALS:
        build_index(v)
