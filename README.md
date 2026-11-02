# AI Experiments

My LLM fine-tuning, RAG evaluation, and agent projects, collected in one repo. Each project lives in its own folder with its own README, requirements, and results. The original repos' commit history is preserved.

## Fine-tuning

| Project | What it does | Headline result |
|---|---|---|
| [`finetuning/smolvlm-pokemon-cards`](finetuning/smolvlm-pokemon-cards) | Fine-tunes **SmolVLM2-500M** to read Pokémon TCG cards and return structured JSON (name, HP, set). Trained on Modal, 1x H100. | **86%** exact match vs. 0% for the base model |
| [`finetuning/qwen-gemma-text-to-sql`](finetuning/qwen-gemma-text-to-sql) | QLoRA (Unsloth) fine-tuning of **Qwen3-4B** and **Gemma 3 4B-IT** on Spider 1.0 text-to-SQL, scored by running the SQL against the real Spider databases. | Qwen3-4B: **3.3% → 80.0%** execution accuracy |

## RAG evaluation

| Project | What it does | Headline result |
|---|---|---|
| [`rag/rag-paradigm-comparison`](rag/rag-paradigm-comparison) | Compares **Vector, Hybrid (BM25 + dense, RRF), Vectorless, and Graph RAG** on a synthetic enterprise corpus (finance, marketing, product, service), with LLM-judge scores and retrieval hit rate. | Hybrid wins on structured/financial docs (**4.90/5**); Vector wins on paraphrased and unstructured queries |

## Agents & inference

| Project | What it does | Stack |
|---|---|---|
| [`agents/nemotron-computer-use`](agents/nemotron-computer-use) | A voice + vision agent that controls macOS. Each step sends a screenshot to the model, gets back exactly one JSON action (click / type / key / scroll / open_app / done), with coordinates normalized to a 0–1000 grid, runs it, and repeats. | **NVIDIA Nemotron 3.5 Super VL (120B-A12B MoE)** served with vLLM on Modal, 4x H100 TP=4, MTP speculative decoding, OpenAI-compatible API. Local Whisper-large-v3-turbo (MLX) for speech-to-text |

## Layout

```
finetuning/
  smolvlm-pokemon-cards/     # VLM fine-tune: train.py, modal_train.py, notebook, plots
  qwen-gemma-text-to-sql/    # QLoRA text-to-SQL notebook + benchmark
agents/
  nemotron-computer-use/     # serve.py (Modal + vLLM), agent.py loop, computer.py, voice.py
rag/
  rag-paradigm-comparison/   # src/, eval sets, results/, paper
```

Each project is self-contained. `cd` into its folder and follow that README's setup steps.

## Origin

These projects started as separate repos, which are now consolidated here:
`finetuned-vllm`, `qwen-gemma4b-finetuned`, `rag-comparison`.
