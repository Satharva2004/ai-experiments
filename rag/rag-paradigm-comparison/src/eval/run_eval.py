"""Run every retriever approach against every vertical's eval set, generate
an answer with Groq, score it with LLM-as-judge, and save all raw + summary
results under results/.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from src.common.config import APPROACHES, EVAL_DIR, RESULTS_DIR, VERTICALS
from src.common.llm import generate_answer, judge_answer
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.graph_retriever import GraphRetriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.retrievers.vector_retriever import VectorRetriever

RETRIEVERS = {
    "vector": VectorRetriever,
    "bm25": BM25Retriever,
    "hybrid": HybridRetriever,
    "graph": GraphRetriever,
}

RAW_RESULTS_DIR = RESULTS_DIR / "raw"
RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def retrieval_hit(gold_context: str, retrieved: list[dict]) -> bool:
    """Did any retrieved chunk actually contain (or closely match) the gold context?"""
    gold = gold_context.strip().lower()
    for r in retrieved:
        text = r["text"].strip().lower()
        if gold in text or text in gold:
            return True
    return False


RETRIEVE_TIMEOUT_S = 30  # a stale connection (esp. Neo4j Aura) can hang with no
                          # exception raised at all - bound every call so one bad
                          # question can't stall an entire (vertical, approach) run


def run_one(approach: str, vertical: str, eval_rows: list[dict]) -> list[dict]:
    retriever = RETRIEVERS[approach]()
    rows_out = []
    executor = ThreadPoolExecutor(max_workers=1)
    for qi, row in enumerate(eval_rows):
        print(f"  [{qi+1}/{len(eval_rows)}] {row['id']}", flush=True)
        t0 = time.time()
        try:
            fut = executor.submit(retriever.retrieve, vertical, row["question"], top_k=3)
            retrieved = fut.result(timeout=RETRIEVE_TIMEOUT_S)
        except FutureTimeoutError:
            print(f"    retrieve() timed out after {RETRIEVE_TIMEOUT_S}s, skipping question", flush=True)
            # stale connection: rebuild the executor/retriever so the next
            # question gets a fresh one instead of queuing behind the hang
            executor.shutdown(wait=False, cancel_futures=True)
            executor = ThreadPoolExecutor(max_workers=1)
            retriever = RETRIEVERS[approach]()
            rows_out.append({
                "id": row["id"], "question": row["question"], "gold_answer": row["gold_answer"],
                "generated_answer": "", "retrieved_doc_ids": [], "retrieval_hit": False,
                "judge_score": 0, "judge_reasoning": "retrieval_timeout",
                "retrieval_latency_s": RETRIEVE_TIMEOUT_S, "generation_latency_s": 0,
            })
            continue
        retrieval_latency = time.time() - t0

        t1 = time.time()
        answer = generate_answer(row["question"], [r["text"] for r in retrieved])
        generation_latency = time.time() - t1

        judged = judge_answer(row["question"], row["gold_answer"], answer)

        rows_out.append({
            "id": row["id"],
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "generated_answer": answer,
            "retrieved_doc_ids": [r["doc_id"] for r in retrieved],
            "retrieval_hit": retrieval_hit(row["gold_context"], retrieved),
            "judge_score": judged.get("score", 0),
            "judge_reasoning": judged.get("reasoning", ""),
            "retrieval_latency_s": round(retrieval_latency, 3),
            "generation_latency_s": round(generation_latency, 3),
        })
    executor.shutdown(wait=False, cancel_futures=True)
    return rows_out


def summarize_existing():
    """Recompute results/summary.json from whatever raw/*.json files exist so
    far - lets the report be built from partial results if a run is interrupted."""
    summary = []
    for path in sorted(RAW_RESULTS_DIR.glob("*.json")):
        vertical, approach = path.stem.rsplit("_", 1)
        rows_out = json.loads(path.read_text(encoding="utf-8"))
        n = len(rows_out)
        summary.append({
            "vertical": vertical, "approach": approach, "n": n,
            "retrieval_hit_rate": round(sum(r["retrieval_hit"] for r in rows_out) / n, 3),
            "avg_judge_score": round(sum(r["judge_score"] for r in rows_out) / n, 3),
            "avg_retrieval_latency_s": round(sum(r["retrieval_latency_s"] for r in rows_out) / n, 3),
            "avg_generation_latency_s": round(sum(r["generation_latency_s"] for r in rows_out) / n, 3),
        })
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote results/summary.json ({len(summary)}/{len(VERTICALS) * len(APPROACHES)} combos)")
    return summary


def main():
    summary = []
    for vertical in VERTICALS:
        eval_rows = json.loads((EVAL_DIR / f"{vertical}_eval.json").read_text(encoding="utf-8"))
        for approach in APPROACHES:
            out_path = RAW_RESULTS_DIR / f"{vertical}_{approach}.json"
            if out_path.exists():
                print(f"=== {vertical} / {approach} === (already done, skipping)")
                rows_out = json.loads(out_path.read_text(encoding="utf-8"))
            else:
                print(f"=== {vertical} / {approach} ===")
                try:
                    rows_out = run_one(approach, vertical, eval_rows)
                except Exception as e:
                    print(f"FAILED {vertical}/{approach}: {e}")
                    continue
                out_path.write_text(json.dumps(rows_out, indent=2, ensure_ascii=False), encoding="utf-8")

            n = len(rows_out)
            hit_rate = sum(r["retrieval_hit"] for r in rows_out) / n
            avg_judge = sum(r["judge_score"] for r in rows_out) / n
            avg_ret_lat = sum(r["retrieval_latency_s"] for r in rows_out) / n
            avg_gen_lat = sum(r["generation_latency_s"] for r in rows_out) / n
            summary.append({
                "vertical": vertical, "approach": approach, "n": n,
                "retrieval_hit_rate": round(hit_rate, 3),
                "avg_judge_score": round(avg_judge, 3),
                "avg_retrieval_latency_s": round(avg_ret_lat, 3),
                "avg_generation_latency_s": round(avg_gen_lat, 3),
            })
            print(summary[-1])

    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("wrote results/summary.json")


if __name__ == "__main__":
    main()
