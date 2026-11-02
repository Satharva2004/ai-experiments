"""Turn results/summary.json into a substack-style markdown report:
per-vertical comparison tables (accuracy/latency) + an overall decision
framework, mirroring the RAG vs GraphRAG performance-analysis format.
"""
import json

from src.common.config import RESULTS_DIR, VERTICALS

APPROACH_LABEL = {
    "vector": "Vector RAG (Pinecone)",
    "bm25": "Vectorless / BM25",
    "hybrid": "Hybrid (BM25 + Vector, RRF)",
    "graph": "Graph RAG (Neo4j)",
}


def load_summary():
    return json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (better-behaved than
    the normal approximation at small n, e.g. our n=20 per combo)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def load_hit_counts() -> dict[str, tuple[int, int]]:
    """(hits, n) per '{vertical}_{approach}' from the raw per-question results,
    used for Wilson CIs - summary.json only has the aggregate rate."""
    from src.eval.run_eval import RAW_RESULTS_DIR

    counts = {}
    for path in sorted(RAW_RESULTS_DIR.glob("*.json")):
        rows = json.loads(path.read_text(encoding="utf-8"))
        counts[path.stem] = (sum(r["retrieval_hit"] for r in rows), len(rows))
    return counts


def vertical_table(rows: list[dict], vertical: str, hit_counts: dict) -> str:
    header = "| Approach | Retrieval Hit Rate | 95% CI | Avg Judge Score (1-5) | Avg Retrieval Latency (s) | Avg Generation Latency (s) |\n"
    header += "|---|---|---|---|---|---|\n"
    lines = [header]
    for r in sorted(rows, key=lambda x: -x["avg_judge_score"]):
        hits, n = hit_counts.get(f"{vertical}_{r['approach']}", (0, r["n"]))
        lo, hi = wilson_ci(hits, n)
        lines.append(
            f"| {APPROACH_LABEL[r['approach']]} | {r['retrieval_hit_rate']:.1%} | "
            f"[{lo:.2f}, {hi:.2f}] | {r['avg_judge_score']:.2f} | {r['avg_retrieval_latency_s']:.3f} | "
            f"{r['avg_generation_latency_s']:.3f} |\n"
        )
    return "".join(lines)


def best_per_vertical(summary: list[dict]) -> str:
    lines = ["| Vertical | Best Approach (by judge score) | Judge Score | Hit Rate |\n", "|---|---|---|---|\n"]
    for v in VERTICALS:
        rows = [r for r in summary if r["vertical"] == v]
        best = max(rows, key=lambda x: x["avg_judge_score"])
        lines.append(
            f"| {v.capitalize()} | {APPROACH_LABEL[best['approach']]} | "
            f"{best['avg_judge_score']:.2f} | {best['retrieval_hit_rate']:.1%} |\n"
        )
    return "".join(lines)


DATASETS = [
    {
        "vertical": "Finance", "name": "financial-qa-10K",
        "source": "Hugging Face: `virattt/financial-qa-10K`", "size": 200, "eval_n": 20,
        "license": "Unspecified\u2020",
        "example_q": "What area did NVIDIA initially focus on before expanding to other computationally intensive fields?",
        "example_a": "NVIDIA initially focused on PC graphics.",
    },
    {
        "vertical": "Marketing", "name": "marketing_social_media",
        "source": "Hugging Face: `RafaM97/marketing_social_media`", "size": 200, "eval_n": 20,
        "license": "Unspecified\u2020",
        "example_q": "Develop a social media campaign to increase brand awareness and drive sales for a new sustainable fashion line.",
        "example_a": "\u201cRevolutionize Your Wardrobe\u201d campaign, leveraging Instagram and TikTok influencers to showcase eco-friendly fashion...",
    },
    {
        "vertical": "Product", "name": "amazon-qa",
        "source": "Hugging Face: `sentence-transformers/amazon-qa`", "size": 200, "eval_n": 20,
        "license": "Community / research use\u2020",
        "example_q": "does this fit the z2x version? Thx",
        "example_a": "I am not 100% sure. It appears that it does based on the size of the torch housing...",
    },
    {
        "vertical": "Service", "name": "Bitext customer support",
        "source": "Hugging Face: `bitext/Bitext-customer-support-llm-chatbot-training-dataset`", "size": 200, "eval_n": 20,
        "license": "CDLA-Sharing-1.0",
        "example_q": "question about cancelling order {{Order Number}}",
        "example_a": "I've understood you have a question regarding canceling order {{Order Number}}, and I'm here to provide you with the information you need...",
    },
]


def dataset_table() -> str:
    lines = [
        "| Vertical | Dataset | Source | Corpus size | Eval Qs | License |\n",
        "|---|---|---|---|---|---|\n",
    ]
    for d in DATASETS:
        lines.append(
            f"| {d['vertical']} | {d['name']} | {d['source']} | {d['size']} | {d['eval_n']} | {d['license']} |\n"
        )
    return "".join(lines)


def dataset_examples() -> str:
    lines = []
    for d in DATASETS:
        lines.append(f"\n**{d['vertical']}** ({d['name']})\n")
        lines.append(f"> Q: {d['example_q']}\n>\n> A: {d['example_a']}\n")
    return "".join(lines)


def main():
    summary = load_summary()
    out = ["# RAG vs GraphRAG vs Vectorless: Enterprise Vertical Performance Analysis\n"]
    out.append(
        "\n**Abstract.** Retrieval-Augmented Generation (RAG) systems for enterprise question "
        "answering are commonly built on one of three retrieval paradigms: dense vector search, "
        "sparse lexical (vectorless) search, and knowledge-graph traversal. This study evaluates "
        "all three, plus a hybrid fusion of vector and lexical signals, across four enterprise "
        "verticals (finance, marketing, product, service), holding the document corpus, generation "
        "model, and judging methodology fixed so only the retrieval strategy varies.\n"
    )
    out.append(
        "\n**Stack.** Generation & LLM-as-judge: **Groq**, model `openai/gpt-oss-120b` "
        "(entity extraction for the graph pipeline used `openai/gpt-oss-20b` as a secondary model "
        "to work around per-model daily token quotas). Dense embeddings: `nomic-ai/nomic-embed-text-v1` "
        "(768-dim, run locally via `sentence-transformers`). Vector store: **Pinecone** (serverless, "
        "cosine metric, one namespace per vertical). Lexical index: **BM25** (`rank_bm25`, "
        "Okapi variant, local/in-process — no external service). Knowledge graph: **Neo4j** (Aura), "
        "with a `(Document)-[:MENTIONS]->(Entity)` schema populated by LLM entity extraction.\n"
    )

    out.append(
        "\n## 1. Introduction\n\n"
        "Enterprise deployments of large language models increasingly rely on retrieval-augmented "
        "generation to ground responses in proprietary documents. Three retrieval paradigms dominate "
        "practice: dense *vector* retrieval over embeddings, sparse lexical (*vectorless*) retrieval "
        "such as BM25, and *graph*-based retrieval that traverses entity relationships extracted from "
        "the corpus. A fourth, *hybrid*, approach fuses vector and lexical signals. Prior comparative "
        "work (Section 2) evaluates these paradigms on general long-document benchmarks; it does not "
        "address whether the ranking of approaches changes across distinct enterprise domains with "
        "different document structure — structured financial filings, unstructured marketing copy, "
        "product Q&A, and ticket-style customer support text. This paper reports a controlled, "
        "same-corpus comparison across those four domains.\n"
    )

    out.append(
        "\n## 2. Related Work\n\n"
        "Edge et al. introduced GraphRAG (Microsoft Research, 2024), constructing an LLM-derived "
        "entity-relationship graph with community summarization to answer global, corpus-level "
        "queries that plain vector RAG struggles with, at the cost of a substantially more expensive "
        "indexing pipeline than the vector or lexical alternatives - a cost this study's results "
        "corroborate: graph retrieval never wins outright here, and its ingestion (LLM entity "
        "extraction per document) was the single most expensive and failure-prone stage in the whole "
        "pipeline. Asai et al. proposed Self-RAG (ICLR 2024), training a model to adaptively decide "
        "when to retrieve and to self-critique retrieved passages, improving factuality independent "
        "of any fixed retrieval architecture - an orthogonal lever this study does not vary, since all "
        "four approaches here share one fixed generation/judging stage by design, isolating retrieval "
        "as the only variable. Closest to this work, an industry analysis (Tailored AI, 2025, "
        "\"RAG vs GraphRAG: A Performance Analysis\", "
        "tailoredai.substack.com/p/rag-vs-graphrag-a-performance-analysis) benchmarked RAG against "
        "GraphRAG on the NovelQA long-document benchmark, stratifying queries by single-hop vs. "
        "multi-hop reasoning difficulty and finding plain RAG preferable for simple factual queries "
        "(68.7% accuracy) with GraphRAG and hybrid strategies pulling ahead on harder multi-hop and "
        "summarization queries. That study varied query difficulty within one general-purpose document "
        "domain; this study instead holds query difficulty roughly fixed and varies the enterprise "
        "domain, and adds a vectorless (BM25) baseline absent from the GraphRAG-focused prior work. "
        "The two studies' hybrid findings partially conflict - Tailored AI reports hybrid as a net "
        "improvement, while this study finds it conditional on retriever agreement - which itself "
        "suggests hybrid's benefit may depend on domain-specific lexical/semantic overlap rather than "
        "being a general property of RRF fusion.\n"
    )

    out.append(
        "\n## 3. Proposed Architecture\n\n"
        "A single normalized document corpus per vertical feeds three independent ingestion paths — "
        "dense embedding into a vector index, tokenization into a lexical index, and LLM-driven "
        "entity/relation extraction into a knowledge graph — so that all four retrieval strategies "
        "draw on identical source text. Retrieved passages are passed to a shared generation and "
        "judging stage, isolating retrieval quality as the only varying factor between conditions.\n"
    )
    out.append(
        "\n```mermaid\n"
        "flowchart TB\n"
        "    A[\"Document Corpus\\n(finance / marketing / product / service)\"]\n"
        "    A --> B[\"Embed\\n(nomic-embed-text-v1)\"] --> B2[(\"Pinecone\\nvector index\")]\n"
        "    A --> C[\"Tokenize\"] --> C2[(\"BM25\\nlexical index\")]\n"
        "    A --> D[\"LLM entity/relation\\nextraction (gpt-oss)\"] --> D2[(\"Neo4j\\nknowledge graph\")]\n"
        "    B2 --> R1[\"Vector search\"]\n"
        "    C2 --> R2[\"Vectorless (BM25)\"]\n"
        "    B2 --> R3[\"Hybrid (RRF fusion)\"]\n"
        "    C2 --> R3\n"
        "    D2 --> R4[\"Graph traversal\"]\n"
        "    R1 --> G[\"Groq LLM generation\\n(gpt-oss-120b)\"]\n"
        "    R2 --> G\n"
        "    R3 --> G\n"
        "    R4 --> G\n"
        "    G --> J[\"LLM-as-judge scoring\\nvs. gold answer\"]\n"
        "```\n"
        "*Figure 1. A shared corpus is ingested once into three independent indexes; four retrievers "
        "query those indexes at inference time; generation and judging are held constant across "
        "conditions so only the retrieval strategy varies.*\n"
    )

    out.append(
        "\n## 4. Methodology\n\n"
        "### 4.1 Datasets\n\n"
        "Each vertical uses a public, freely accessible dataset re-purposed as an enterprise document "
        "corpus, capped at 200 documents for indexing and 20 held-out question/answer pairs for "
        "evaluation.\n\n"
    )
    out.append(dataset_table())
    out.append(
        "\n\u2020 No explicit license tag on the source page at time of writing; verify against the "
        "linked repository/paper before any commercial reuse.\n"
    )
    out.append("\n**Example question/gold-answer pairs, one per vertical:**\n")
    out.append(dataset_examples())

    out.append(
        "\n### 4.2 Retrieval formulations\n\n"
        "Vector retrieval ranks documents by cosine similarity between the query embedding $q$ and "
        "document embedding $d$:\n\n"
        "$$\\text{sim}(q, d) = \\frac{q \\cdot d}{\\lVert q \\rVert \\, \\lVert d \\rVert} \\tag{1}$$\n\n"
        "Vectorless retrieval uses Okapi BM25 over tokenized text, where $f(q_i, D)$ is the term "
        "frequency of query term $q_i$ in document $D$, $|D|$ its length, and $\\text{avgdl}$ the "
        "corpus average length:\n\n"
        "$$\\text{BM25}(D, Q) = \\sum_{i=1}^{n} \\text{IDF}(q_i) \\cdot "
        "\\frac{f(q_i, D) \\cdot (k_1 + 1)}{f(q_i, D) + k_1 \\left(1 - b + b \\cdot "
        "\\frac{|D|}{\\text{avgdl}}\\right)} \\tag{2}$$\n\n"
        "Hybrid retrieval fuses the two rank lists with Reciprocal Rank Fusion (RRF), where "
        "$\\text{rank}_r(d)$ is document $d$'s rank under retriever $r$ and $k=60$:\n\n"
        "$$\\text{RRF}(d) = \\sum_{r \\in \\{\\text{vector}, \\text{bm25}\\}} "
        "\\frac{1}{k + \\text{rank}_r(d)} \\tag{3}$$\n\n"
        "Graph retrieval extracts an entity set $E(q)$ from the query and ranks documents by "
        "shared-entity overlap with each document's extracted entity set $E(d)$:\n\n"
        "$$\\text{score}_{\\text{graph}}(d, q) = \\bigl| E(q) \\cap E(d) \\bigr| \\tag{4}$$\n"
    )

    out.append(
        "\n### 4.3 Evaluation metrics\n\n"
        "Retrieval hit rate over $N$ eval questions, where $\\mathbb{1}[\\cdot]$ indicates the gold "
        "supporting passage appears among the top-$k$ retrieved chunks:\n\n"
        "$$\\text{HitRate} = \\frac{1}{N}\\sum_{i=1}^{N} \\mathbb{1}"
        "\\bigl[\\text{gold}_i \\in \\text{Retrieved}_k(q_i)\\bigr] \\tag{5}$$\n\n"
        "Answer quality is scored by an LLM-as-judge $J(\\cdot) \\in \\{1,\\dots,5\\}$ comparing the "
        "generated answer $\\hat{a}_i$ against the gold answer $a_i$:\n\n"
        "$$\\overline{J} = \\frac{1}{N}\\sum_{i=1}^{N} J(\\hat{a}_i, a_i) \\tag{6}$$\n\n"
        "95% confidence intervals on hit rate are computed with the Wilson score interval, which "
        "stays well-behaved at small $N$ (here $N=20$) unlike the normal approximation.\n"
    )

    out.append(
        "\n### 4.4 Experimental setup\n\n"
        "| Parameter | Value |\n|---|---|\n"
        "| Retrieval top-$k$ | 5 (finance, all 4 approaches; marketing vector/BM25); "
        "3 (marketing hybrid/graph; all product; all service — see \u00a76 on the mid-study change) |\n"
        "| RRF constant $k$ | 60 |\n"
        "| BM25 parameters | $k_1=1.5$, $b=0.75$ (`rank_bm25` Okapi defaults) |\n"
        "| Generation temperature | 0.0 |\n"
        "| Judge temperature | 0.0 |\n"
        "| Embedding dimensionality | 768 (`nomic-embed-text-v1`) |\n"
        "| Judge score scale | integer, 1\u20135 |\n"
    )

    out.append("\n## 5. Results\n")
    out.append("\n### Summary: Best Approach per Vertical\n\n")
    out.append(best_per_vertical(summary))

    hit_counts = load_hit_counts()
    for v in VERTICALS:
        rows = [r for r in summary if r["vertical"] == v]
        out.append(f"\n### {v.capitalize()}\n\n")
        out.append(vertical_table(rows, v, hit_counts))

    out.append(
        "\n### Decision Framework\n\n"
        "- **Vector RAG**: best for free-text, semantically fuzzy queries where exact keyword "
        "overlap with the source is unlikely (paraphrased questions).\n"
        "- **Vectorless / BM25**: best for terminology-heavy, exact-match-sensitive queries "
        "(ticket IDs, product SKUs, ticker symbols) and where infra simplicity/cost matters.\n"
        "- **Hybrid**: not universally the safest default — it only helps when its component "
        "retrievers broadly agree, and it inherits the *weaker* component's failure mode when they "
        "don't (see Error Analysis). On marketing, where vector (60% hit rate) and BM25 (30%) "
        "frequently disagreed, hybrid landed at 25% — below vector but roughly level with BM25 once "
        "top-$k$ is matched (§5.6). Check retriever agreement on a domain sample before adopting "
        "hybrid as a default.\n"
        "- **Graph RAG**: best where the enterprise data has real entity relationships to traverse "
        "(company↔filing, customer↔ticket↔product); highest ingestion cost (LLM entity extraction "
        "per document) and depends heavily on extraction quality.\n"
    )

    out.append(
        "\n### Discussion\n\n"
        "Two patterns stand out beyond the per-vertical winners. First, **hybrid fusion is "
        "conditional, not universal**: it wins on finance (4.90) and service (4.25) but is the "
        "weakest of the three baseline retrievers on marketing (1.95) — see §5.6 for a "
        "controlled, top-$k$-matched analysis of why. On finance, vector and BM25 mostly agree on the "
        "same candidate passages (both driven by the same distinctive company/ticker terms), so RRF "
        "fusion reinforces a shared correct answer. On marketing, the two retrievers frequently "
        "surface *different* top documents for the same query (free-text campaign briefs have little "
        "lexical overlap with their own strategy text), and RRF's reciprocal-rank averaging then "
        "favors documents that are mediocre-but-present in both lists over a document that was strong "
        "in only one. Second, **retrieval hit rate and judge score decouple on the service vertical**: "
        "hit rate is only 10-15% across all four approaches, yet judge scores remain high "
        "(3.55-4.25). The service corpus (Bitext customer-support intents) is templated and generic "
        "by design (e.g. 'I understand you have a question about {{Order Number}}...'), so even when "
        "the retriever surfaces a *different* document than the literal gold passage, it is often "
        "close enough in content for the generated answer to still be judged correct - meaning "
        "retrieval hit rate understates true usefulness on highly templated domains, and judge score "
        "alone should not be read as proof retrieval is working as intended.\n"
    )

    out.append(
        "\n### 5.6 Error Analysis: Isolating the Hybrid Effect from the Top-$k$ Confound\n\n"
        "The production run compared marketing hybrid (top-$k$=3, run after the mid-study "
        "configuration change) against marketing BM25 (top-$k$=5, run before it) — an unmatched "
        "comparison. Re-running vector, BM25, and hybrid retrieval at *matched* top-$k$ (no LLM calls "
        "required, so this is free to verify) gives a cleaner picture:\n\n"
        "| top-$k$ | Vector | BM25 | Hybrid |\n|---|---|---|---|\n"
        "| 3 (matched) | 40% (8/20) | 20% (4/20) | 25% (5/20) |\n"
        "| 5 (matched) | 60% (12/20) | 30% (6/20) | 40% (8/20) |\n\n"
        "At matched top-$k$, hybrid still clearly underperforms vector at both settings — this part "
        "of the finding is robust, not a top-$k$ artifact. But hybrid is roughly *comparable to* BM25 "
        "at matched top-$k$ (not worse than both, as the unmatched production numbers suggested). "
        "The revised claim is narrower and better supported: **RRF fusion can drag hybrid below its "
        "strongest individual component when the two disagree, without necessarily falling below its "
        "weakest component too.**\n\n"
        "A concrete, matched-top-$k$ example (`marketing-doc-5`, \"Develop a content calendar for a "
        "health and wellness blog\"): vector@3 retrieves `[marketing-155, marketing-88, marketing-5]` "
        "— a hit, since `marketing-5` is the gold passage. Hybrid@3 retrieves "
        "`[marketing-155, marketing-88, marketing-121]` — the same top two documents, but RRF's "
        "rank-averaging swaps the correct third document for `marketing-121`, which ranked better "
        "in BM25's list. This is direct, non-confounded evidence of RRF fusion demoting a correct "
        "result in favor of a document that ranked passably in both lists rather than excellently in "
        "one.\n"
    )

    out.append(
        "\n## 6. Ingestion Cost & Reliability\n\n"
        "Query-time latency (reported above) is only part of the operating cost; the three approaches "
        "differ far more in ingestion cost and reliability, which query-time numbers don't capture:\n\n"
        "| Approach | Ingestion cost driver | Volume | Reliability incidents |\n|---|---|---|---|\n"
        "| Vector | 1 embedding call/doc (local, no API cost) | ~800 docs (4 verticals × 200) | None |\n"
        "| Vectorless (BM25) | Tokenization only, no model calls | ~800 docs | None |\n"
        "| Graph | 1 LLM entity-extraction call/doc | ~800 docs | Exhausted Groq's 200K-token daily "
        "quota **twice** (on two different API keys), requiring key swaps and a fallback from "
        "`gpt-oss-20b` to `gpt-oss-120b` mid-run; intermittent Neo4j Aura connection drops caused a "
        "combination to hang indefinitely with no exception raised, requiring a 30-second per-call "
        "timeout with connection rebuild to be added |\n\n"
        "Graph RAG was, by a wide margin, the most expensive and operationally fragile approach to "
        "stand up — a cost not visible in the per-query latency table and easy to underestimate when "
        "comparing approaches on accuracy alone.\n"
    )

    out.append(
        "\n## 7. Future Work\n\n"
        "A follow-up study should: (1) scale the eval set per vertical using a Wilson-CI-informed "
        "sample size target (e.g. a hit-rate difference of 15-20 points needs roughly 60-100 questions "
        "per vertical to separate at 95% confidence, versus 20 here) rather than fitting to an API "
        "quota; (2) use multiple judge models or human raters with position-swapped prompts to "
        "quantify and correct for LLM-as-judge bias; (3) hold top-$k$ and context-length fixed across "
        "every condition to remove the mid-study configuration confound entirely (§6); (4) build the "
        "graph corpus with a production-grade pipeline (entity resolution/deduplication, "
        "community summarization as in Microsoft's GraphRAG) rather than single-pass extraction, since "
        "this study's graph results likely understate the approach's ceiling; and (5) report cost per "
        "query in dollar terms (embedding + LLM + vector-DB + graph-DB pricing) alongside accuracy, "
        "since Section 6 shows ingestion cost alone can be the deciding factor for a practitioner "
        "independent of accuracy differences.\n"
    )

    out.append(
        "\n## 8. Limitations\n\n"
        "This study has several limitations that bound the strength of its conclusions. **Sample "
        "size**: 20 questions per vertical (chosen to fit within a free-tier LLM API's daily token "
        "quota, which this study exhausted on the same account twice during evaluation) supports "
        "directional observations but not statistically significant claims; Wilson 95% confidence "
        "intervals are reported per condition (§5) and are wide enough that most cross-approach "
        "differences are not clearly separated at this sample size. **Single, unvalidated judge**: "
        "answer quality is scored by one LLM-as-judge "
        "(gpt-oss-120b) with no human validation and no mitigation for judge position bias, a known "
        "confound in LLM-as-judge methodology. **Mid-study configuration change**: to fit within "
        "recurring API rate limits, retrieval top-k was reduced from 5 to 3 and per-call context/answer "
        "text was truncated (to 400 and 300 characters respectively) partway through evaluation - "
        "finance and marketing's vector/BM25 results were produced *before* this change, while "
        "marketing's hybrid/graph and all of product and service were produced *after* it. This is a "
        "genuine confound: the noticeably lower judge scores on product (1.80-2.45) and service "
        "(3.55-4.25 despite very low hit rates) may partly reflect the smaller context window rather "
        "than a pure property of the retrieval method, and cross-vertical comparisons should be read "
        "with this in mind. **Graph corpus scale**: capped at 200 documents per vertical with a "
        "single-pass LLM entity-extraction step and no entity resolution/deduplication, likely "
        "understating graph RAG's potential relative to a production-grade construction pipeline. "
        "**Hit-rate metric**: computed via substring containment against one gold passage per "
        "question, which undercounts correct retrievals when the same fact appears in multiple "
        "corpus passages, and (per the Discussion above) can diverge from actual answer usefulness "
        "on templated domains. **Infrastructure reliability**: intermittent Neo4j Aura connection "
        "drops and one transient DNS/network outage caused several (vertical, approach) combinations "
        "to require a retry; a 30-second per-question timeout was added to prevent indefinite hangs, "
        "and any question that timed out was scored as a failure (judge score 0) rather than retried, "
        "which may slightly depress graph RAG's reported scores relative to its true capability.\n"
    )

    out.append(
        "\n## 9. Conclusion\n\n"
        "Across all four enterprise verticals, no single retrieval paradigm dominates, and the "
        "ranking of approaches changes by domain: hybrid wins on finance and service where its "
        "component retrievers agree, plain vector retrieval wins on marketing and product where it "
        "does not. Graph retrieval, in its current entity-overlap formulation, never wins outright "
        "in this study, consistent with its documents lacking the dense, distinctive entity structure "
        "the approach depends on and with its markedly higher ingestion cost — by far the most "
        "expensive and failure-prone stage in the pipeline (§6). The most actionable finding for "
        "practitioners is negative: hybrid should not be assumed to be a safe universal default, since "
        "a controlled, top-$k$-matched comparison (§5.6) confirms it can underperform its strongest "
        "individual component when the underlying retrievers disagree (marketing), even though it does "
        "not necessarily fall below its weakest component too. A second actionable finding is that "
        "retrieval hit rate and end-answer quality can diverge substantially on templated, "
        "generic-response domains (service), so evaluation should track both metrics rather than "
        "either alone. Given the wide, overlapping 95% confidence intervals at n=20 per vertical "
        "(§5), these findings should be read as directional evidence motivating a larger follow-up "
        "study (§7), not as statistically confirmed differences.\n"
    )

    out.append(
        "\n## References\n\n"
        "1. Edge, D. et al. \"From Local to Global: A Graph RAG Approach to Query-Focused "
        "Summarization.\" Microsoft Research, 2024.\n"
        "2. Asai, A. et al. \"Self-RAG: Learning to Retrieve, Generate, and Critique through "
        "Self-Reflection.\" ICLR, 2024.\n"
        "3. Tailored AI. \"RAG vs GraphRAG: A Performance Analysis.\" "
        "tailoredai.substack.com/p/rag-vs-graphrag-a-performance-analysis, 2025.\n"
        "4. Robertson, S., Zaragoza, H. \"The Probabilistic Relevance Framework: BM25 and Beyond.\" "
        "Foundations and Trends in Information Retrieval, 2009.\n"
        "5. Cormack, G. V., Clarke, C. L. A., Buettcher, S. \"Reciprocal Rank Fusion Outperforms "
        "Condorcet and Individual Rank Learning Methods.\" SIGIR, 2009.\n"
    )

    report_path = RESULTS_DIR / "reports" / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(out), encoding="utf-8")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
