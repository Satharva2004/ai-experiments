"""Nomic embed wrapper (local, via sentence-transformers)."""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.common.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _model():
    return SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)


def embed_documents(texts: list[str]) -> list[list[float]]:
    prefixed = [f"search_document: {t}" for t in texts]
    return _model().encode(prefixed, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    return _model().encode(f"search_query: {text}", normalize_embeddings=True).tolist()
