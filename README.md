# Spider Text-to-SQL — QLoRA Fine-Tuning: Qwen3-4B vs. Gemma 3 4B

Fine-tunes **Qwen3-4B** and **Gemma 3 4B-IT** on the [Spider 1.0](https://yale-lily.github.io/spider) text-to-SQL benchmark using **QLoRA** (via [Unsloth](https://github.com/unslothai/unsloth)), then benchmarks base vs. fine-tuned on **exact match**, **valid-SQL rate**, and **execution accuracy** — verified by actually running the generated SQL against the real Spider SQLite databases.

<img width="1408" height="768" alt="1000156319" src="https://github.com/user-attachments/assets/013b4e5d-d652-4b34-8893-7dced4ee5bd2" />


📊 **[View the interactive benchmark report](./spider_sql_benchmark_chart.html)** — open directly in any browser, no server needed.

---

## Results (n = 30, execution-verified sample)

| Model | Exact Match | Valid SQL | **Execution Accuracy** |
|---|---|---|---|
| Qwen3-4B (base) | 3.3% | 3.3% | 3.3% |
| **Qwen3-4B + LoRA** | 56.7% | 93.3% | **80.0%** ✅ best |
| Gemma3-4B (base) | 16.7% | 96.7% | 66.7% |
| Gemma3-4B + LoRA | 43.3% | 86.7% | 60.0% |

**Takeaway:** QLoRA fine-tuning took Qwen3-4B from nearly unusable (3.3%) to the best model in the comparison (80.0% execution accuracy) — a +76.7 point gain. Gemma's base model was already a surprisingly strong zero-shot baseline (66.7%), and its LoRA run improved exact-match phrasing but slightly regressed on actual execution accuracy (66.7% → 60.0%), suggesting some overfitting to gold-query wording rather than logic.

Execution accuracy — not exact match — is the metric that matters here: it checks whether the generated SQL actually returns the correct result when run against the real database, rather than penalizing correct SQL that's just phrased differently.

---

## Repo contents

| File | Description |
|---|---|
| `spider_qwen_gemma_qlora_final.ipynb` | End-to-end Kaggle/Colab notebook: environment detection, dataset download, QLoRA training for both models, adapter upload to Hugging Face, and full evaluation pipeline |
| `spider_sql_benchmark_chart.html` | Interactive, self-contained benchmark report (open in any browser) |
| `README.md` | This file |

---

## Models & dataset

| Resource | Link |
|---|---|
| 🤗 Spider 1.0 dataset | [huggingface.co/datasets/lianghsun/spider-text2sql-bench](https://huggingface.co/datasets/lianghsun/spider-text2sql-bench) |
| 🗄 Spider SQLite databases (execution accuracy) | [kaggle.com/datasets/jeromeblanchet/yale-universitys-spider-10-nlp-dataset](https://www.kaggle.com/datasets/jeromeblanchet/yale-universitys-spider-10-nlp-dataset) |
| 🤗 Qwen3-4B base model | [huggingface.co/unsloth/Qwen3-4B-unsloth-bnb-4bit](https://huggingface.co/unsloth/Qwen3-4B-unsloth-bnb-4bit) |
| 🤗 Gemma 3 4B-IT base model | [huggingface.co/unsloth/gemma-3-4b-it-unsloth-bnb-4bit](https://huggingface.co/unsloth/gemma-3-4b-it-unsloth-bnb-4bit) |
| 🤗 **Qwen3-4B LoRA adapter (this project)** | [huggingface.co/atharvasawant99/qwen3-4b-spider-sql-lora](https://huggingface.co/atharvasawant99/qwen3-4b-spider-sql-lora) |
| 🤗 **Gemma3-4B LoRA adapter (this project)** | [huggingface.co/atharvasawant99/gemma3-4b-spider-sql-lora](https://huggingface.co/atharvasawant99/gemma3-4b-spider-sql-lora) |

---

## How it was trained

- **Method:** QLoRA (4-bit quantized base + LoRA adapters), `r=16`, `alpha=32`, target modules: `q/k/v/o_proj`, `gate/up/down_proj`
- **Framework:** [Unsloth](https://github.com/unslothai/unsloth) + 🤗 TRL `SFTTrainer`
- **Hardware:** 2× NVIDIA Tesla T4 (Kaggle), DDP across both GPUs
- **Data:** Spider 1.0 — 7,000 training examples / 1,034 validation examples
- **Epochs:** 1, batch size 1 with gradient accumulation (effective batch size 8)

## How it was evaluated

For each of the 4 systems (Qwen base / Qwen+LoRA / Gemma base / Gemma+LoRA):
1. Generate SQL greedily for each validation question, given the schema as a `CREATE TABLE` prompt.
2. **Exact match** — normalized string comparison against the gold SQL.
3. **Valid SQL** — the query parses/plans successfully.
4. **Execution accuracy** — the query is actually run against the real Spider SQLite database (auto-resolved by matching the schema's table names against the downloaded database files) and its result set is compared to the gold query's result set.

Qwen3 is a reasoning model and emits `<think>...</think>` before its answer — this is stripped before evaluating the SQL.

---

## Reproducing

1. Open `spider_qwen_gemma_qlora_final.ipynb` in Kaggle (GPU: T4 x2) or Colab.
2. Add a Hugging Face **write** token as a Kaggle Secret named `HF_TOKEN` (or set the `HF_TOKEN` environment variable elsewhere).
3. Run all cells top-to-bottom. The notebook:
   - Auto-detects environment (Kaggle / Colab / local) and GPU count
   - Downloads Spider 1.0 and splits train/validation automatically
   - Trains Qwen3-4B, uploads its adapter, then trains Gemma 3 4B, uploads its adapter
   - Is idempotent — re-running skips any model that already has a saved adapter
4. Open `spider_sql_benchmark_chart.html` in a browser to view the results.

---

## Notes & limitations

- Benchmark numbers above are on a **30-example sample** for fast iteration; the notebook supports scaling `EVAL_SAMPLE_SIZE` up to the full 1,034-example validation set for a more statistically robust comparison.
- "Difficulty" bucketing used during analysis is a heuristic (join/subquery/aggregation count), not Spider's official difficulty labels.
