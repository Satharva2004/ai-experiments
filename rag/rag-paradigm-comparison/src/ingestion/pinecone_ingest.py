"""Vector RAG ingestion: embed each vertical's corpus and upsert to Pinecone,
one namespace per vertical inside a single shared index.
"""
from pinecone import Pinecone, ServerlessSpec

from src.common.config import EMBEDDING_DIM, PINECONE_API_KEY, PINECONE_INDEX_NAME, VERTICALS
from src.common.embeddings import embed_documents
from src.ingestion.build_corpus import load_corpus

pc = Pinecone(api_key=PINECONE_API_KEY)


def ensure_index():
    if PINECONE_INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(PINECONE_INDEX_NAME)


def ingest_vertical(index, vertical: str, batch_size: int = 64):
    docs = load_corpus(vertical)
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        vectors = embed_documents([d["text"] for d in batch])
        payload = [
            {
                "id": d["doc_id"],
                "values": vec,
                "metadata": {"text": d["text"], **{k: v for k, v in d["meta"].items() if v}},
            }
            for d, vec in zip(batch, vectors)
        ]
        index.upsert(vectors=payload, namespace=vertical)
    print(f"{vertical}: upserted {len(docs)} vectors")


if __name__ == "__main__":
    idx = ensure_index()
    for v in VERTICALS:
        ingest_vertical(idx, v)
