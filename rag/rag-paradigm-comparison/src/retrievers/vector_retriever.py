"""Vector RAG: dense semantic search over Pinecone."""
from pinecone import Pinecone

from src.common.config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from src.common.embeddings import embed_query
from src.retrievers.base import Retriever

pc = Pinecone(api_key=PINECONE_API_KEY)


class VectorRetriever(Retriever):
    name = "vector"

    def __init__(self):
        self.index = pc.Index(PINECONE_INDEX_NAME)

    def retrieve(self, vertical: str, query: str, top_k: int = 5) -> list[dict]:
        vec = embed_query(query)
        res = self.index.query(
            vector=vec, top_k=top_k, namespace=vertical, include_metadata=True
        )
        return [
            {"doc_id": m.id, "text": m.metadata.get("text", ""), "score": m.score}
            for m in res.matches
        ]
