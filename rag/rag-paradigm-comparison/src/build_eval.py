"""Build normalized eval sets (question/answer/context/id) per vertical from
the raw samples in data/raw/. Same schema across verticals so retrieval and
generation metrics are comparable across RAG approaches.

Schema per row:
  id            unique id, "<vertical>-<n>"
  vertical      finance | marketing | product | service
  question      the query a user would ask
  gold_answer   the expected answer (used for answer-quality scoring)
  gold_context  ground-truth supporting text (used for retrieval scoring,
                i.e. did the retriever surface a chunk containing this)
  doc_id        id of source document, for graph RAG entity linking
"""
import json
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

N_EVAL = 20  # held-out eval questions per vertical (trimmed for faster eval runs given Groq's 8000 TPM limit)


def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_eval(vertical, rows):
    out_path = EVAL_DIR / f"{vertical}_eval.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"wrote {len(rows)} eval rows -> {out_path}")


def build_finance():
    raw = read_jsonl(RAW_DIR / "finance.jsonl")[:N_EVAL]
    rows = []
    for i, r in enumerate(raw):
        rows.append({
            "id": f"finance-{i}",
            "vertical": "finance",
            "question": r["question"],
            "gold_answer": r["answer"],
            "gold_context": r["context"],
            "doc_id": f"{r.get('ticker', 'unk')}_{r.get('filing', 'unk')}",
        })
    write_eval("finance", rows)


def build_marketing():
    raw = read_jsonl(RAW_DIR / "marketing.jsonl")[:N_EVAL]
    rows = []
    for i, r in enumerate(raw):
        question = r["instruction"]
        if r.get("input"):
            question = f"{r['instruction']}\nContext: {r['input']}"
        rows.append({
            "id": f"marketing-{i}",
            "vertical": "marketing",
            "question": question,
            "gold_answer": r["response"],
            "gold_context": r["response"],
            "doc_id": f"marketing-doc-{i}",
        })
    write_eval("marketing", rows)


def build_product():
    raw = read_jsonl(RAW_DIR / "product.jsonl")[:N_EVAL]
    rows = []
    for i, r in enumerate(raw):
        rows.append({
            "id": f"product-{i}",
            "vertical": "product",
            "question": r["query"],
            "gold_answer": r["answer"],
            "gold_context": r["answer"],
            "doc_id": f"product-doc-{i}",
        })
    write_eval("product", rows)


def build_service():
    raw = read_jsonl(RAW_DIR / "service.jsonl")[:N_EVAL]
    rows = []
    for i, r in enumerate(raw):
        rows.append({
            "id": f"service-{i}",
            "vertical": "service",
            "question": r["instruction"],
            "gold_answer": r["response"],
            "gold_context": r["response"],
            "doc_id": f"{r.get('category', 'unk')}_{r.get('intent', 'unk')}_{i}",
        })
    write_eval("service", rows)


if __name__ == "__main__":
    build_finance()
    build_marketing()
    build_product()
    build_service()
