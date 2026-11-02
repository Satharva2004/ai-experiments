import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "rag-comparison")

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1")
EMBEDDING_DIM = 768

VERTICALS = ["finance", "marketing", "product", "service"]
APPROACHES = ["vector", "bm25", "hybrid", "graph"]

# capped so all three retriever types index the SAME corpus size per vertical
# (graph entity/relation extraction is LLM-call-expensive, so this keeps the
# comparison fair and the ingestion time bounded)
CORPUS_SIZE = 200

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EVAL_DIR = ROOT / "eval"
RESULTS_DIR = ROOT / "results"
INDEX_DIR = ROOT / "data" / "indexes"
