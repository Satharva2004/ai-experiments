"""Download small, reproducible samples of the four vertical datasets.

Sources (verified public, see MEMORY / README for license notes):
- finance:   virattt/financial-qa-10K        (has native Q&A pairs)
- marketing: RafaM97/marketing_social_media  (instruction/response pairs)
- product:   sentence-transformers/amazon-qa (has native Q&A pairs)
- service:   bitext/Bitext-customer-support-llm-chatbot-training-dataset
"""
import json
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 500


def dump(rows, name):
    out_path = DATA_DIR / f"{name}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {out_path}")


def load_finance():
    ds = load_dataset("virattt/financial-qa-10K", split="train")
    rows = [dict(r) for r in ds.select(range(min(N_SAMPLES, len(ds))))]
    dump(rows, "finance")


def load_marketing():
    ds = load_dataset("RafaM97/marketing_social_media", split="train")
    rows = [dict(r) for r in ds.select(range(min(N_SAMPLES, len(ds))))]
    dump(rows, "marketing")


def load_product():
    ds = load_dataset("sentence-transformers/amazon-qa", split="train", streaming=True)
    rows = []
    for r in ds:
        rows.append(dict(r))
        if len(rows) >= N_SAMPLES:
            break
    dump(rows, "product")


def load_service():
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train"
    )
    rows = [dict(r) for r in ds.select(range(min(N_SAMPLES, len(ds))))]
    dump(rows, "service")


if __name__ == "__main__":
    for fn in (load_finance, load_marketing, load_product, load_service):
        try:
            fn()
        except Exception as e:
            print(f"FAILED {fn.__name__}: {e}")
