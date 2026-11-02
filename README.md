# AI Experiments

My LLM fine-tuning and RAG evaluation projects, collected in one repo. Each project lives in its own folder with its own README, requirements, and results. The original repos' commit history is preserved.

## Fine-tuning

| Project | What it does | Headline result |
|---|---|---|
| [`finetuning/smolvlm-pokemon-cards`](finetuning/smolvlm-pokemon-cards) | Fine-tunes **SmolVLM2-500M** to read Pokémon TCG cards and return structured JSON (name, HP, set). Trained on Modal, 1x H100. | **86%** exact match vs. 0% for the base model |
| [`finetuning/qwen-gemma-text-to-sql`](finetuning/qwen-gemma-text-to-sql) | QLoRA (Unsloth) fine-tuning of **Qwen3-4B** and **Gemma 3 4B-IT** on Spider 1.0 text-to-SQL, scored by running the SQL against the real Spider databases. | Qwen3-4B: **3.3% → 80.0%** execution accuracy |

## RAG evaluation

| Project | What it does | Headline result |
|---|---|---|
| [`rag/rag-paradigm-comparison`](rag/rag-paradigm-comparison) | Compares **Vector, Hybrid (BM25 + dense, RRF), Vectorless, and Graph RAG** on a synthetic enterprise corpus (finance, marketing, product, service), with LLM-judge scores and retrieval hit rate. | Hybrid wins on structured/financial docs (**4.90/5**); Vector wins on paraphrased and unstructured queries |

## Layout

```
finetuning/
  smolvlm-pokemon-cards/     # VLM fine-tune: train.py, modal_train.py, notebook, plots
  qwen-gemma-text-to-sql/    # QLoRA text-to-SQL notebook + benchmark
rag/
  rag-paradigm-comparison/   # src/, eval sets, results/, paper
```

Each project is self-contained. `cd` into its folder and follow that README's setup steps.

## Origin

These projects started as separate repos, which are now consolidated here:
`finetuned-vllm`, `qwen-gemma4b-finetuned`, `rag-comparison`.
