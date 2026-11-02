"""Turn each vertical's raw jsonl into a flat corpus of {doc_id, text, meta}
documents that all three retrievers (vector/BM25/graph) index from the same
source, so comparisons are apples-to-apples.
"""
import json

from src.common.config import CORPUS_SIZE, RAW_DIR

EXTRACTORS = {
    "finance": lambda i, r: {
        "doc_id": f"finance-{r.get('ticker','unk')}-{r.get('filing','unk')}-{i}",
        "text": r["context"],
        "meta": {"ticker": r.get("ticker"), "filing": r.get("filing")},
    },
    "marketing": lambda i, r: {
        "doc_id": f"marketing-{i}",
        "text": r["response"],
        "meta": {"instruction": r["instruction"]},
    },
    "product": lambda i, r: {
        "doc_id": f"product-{i}",
        "text": r["answer"],
        "meta": {"query": r.get("query")},
    },
    "service": lambda i, r: {
        "doc_id": f"service-{r.get('category','unk')}-{r.get('intent','unk')}-{i}",
        "text": r["response"],
        "meta": {"category": r.get("category"), "intent": r.get("intent")},
    },
}


def load_corpus(vertical: str) -> list[dict]:
    path = RAW_DIR / f"{vertical}.jsonl"
    extractor = EXTRACTORS[vertical]
    docs = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= CORPUS_SIZE:
                break
            r = json.loads(line)
            doc = extractor(i, r)
            if doc["text"] and doc["text"].strip():
                docs.append(doc)
    return docs


if __name__ == "__main__":
    for v in EXTRACTORS:
        docs = load_corpus(v)
        print(f"{v}: {len(docs)} docs, e.g. {docs[0]}")
