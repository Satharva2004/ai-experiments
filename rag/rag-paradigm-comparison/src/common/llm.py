"""Groq LLM wrapper used for both answer generation and LLM-as-judge scoring."""
import re
import time

from groq import APITimeoutError, Groq, RateLimitError

from src.common.config import GROQ_API_KEY, GROQ_MODEL

_client = Groq(api_key=GROQ_API_KEY, timeout=30.0, max_retries=0)


_WAIT_RE = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s")
MAX_WAIT_S = 70  # never block a single call longer than this; let the caller skip/retry later


def _parse_wait_seconds(message: str) -> float:
    match = _WAIT_RE.search(message)
    if not match:
        return 5.0
    minutes = float(match.group(1)) if match.group(1) else 0.0
    seconds = float(match.group(2))
    return min(minutes * 60 + seconds + 0.5, MAX_WAIT_S)


def chat(system: str, user: str, temperature: float = 0.0, model: str | None = None) -> str:
    max_retries = 6
    for attempt in range(max_retries):
        try:
            resp = _client.chat.completions.create(
                model=model or GROQ_MODEL,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(_parse_wait_seconds(str(e)))
        except APITimeoutError:
            if attempt == max_retries - 1:
                raise
            time.sleep(2)


MAX_CONTEXT_CHARS = 400  # keep per-call token cost low given free-tier 200k TPD caps
MAX_ANSWER_CHARS = 300


def generate_answer(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(f"[{i+1}] {c[:MAX_CONTEXT_CHARS]}" for i, c in enumerate(contexts))
    system = (
        "You are a helpful enterprise assistant. Answer the question using ONLY the "
        "provided context. If the context does not contain the answer, say so. Be concise."
    )
    user = f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"
    return chat(system, user)


def judge_answer(question: str, gold_answer: str, generated_answer: str) -> dict:
    """LLM-as-judge: score 1-5 for correctness/faithfulness vs. the gold answer."""
    system = (
        "Score a generated answer against a reference answer, 1 (wrong) to 5 (fully "
        'correct/equivalent). Respond ONLY with strict JSON: {"score": <int 1-5>, "reasoning": "<5 words>"}'
    )
    user = (
        f"Q: {question}\nGold: {gold_answer[:MAX_ANSWER_CHARS]}\n"
        f"Generated: {generated_answer[:MAX_ANSWER_CHARS]}\nJSON:"
    )
    raw = chat(system, user)
    import json
    import re

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else {"score": 0, "reasoning": "parse_failed"}
    except json.JSONDecodeError:
        return {"score": 0, "reasoning": "parse_failed"}
