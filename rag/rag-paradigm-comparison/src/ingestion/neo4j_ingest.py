"""Graph RAG ingestion: LLM-extract entities per document, write a
(Document)-[:MENTIONS]->(Entity) graph per vertical into Neo4j.
Retrieval later traverses shared entities between the query and documents.
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from neo4j import GraphDatabase

from src.common.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME, VERTICALS
from src.common.llm import chat
from src.ingestion.build_corpus import load_corpus

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

EXTRACT_SYSTEM = (
    "Extract the key named entities (people, companies, products, monetary amounts, "
    "categories, dates, ticket/order types — anything a knowledge graph would link on) "
    'from the text. Respond ONLY with strict JSON: {"entities": ["...", "..."]}. '
    "Return at most 8 entities, short noun phrases only."
)


def extract_entities(text: str) -> list[str]:
    # NOTE: gpt-oss-20b has a separate, low daily token quota (200k TPD on
    # free tier) that gets exhausted fast across ~600+ extraction calls;
    # gpt-oss-120b has no daily cap hit so far, only an 8000 TPM limit our
    # chat() retry/backoff already handles - use it here instead.
    try:
        raw = chat(EXTRACT_SYSTEM, text[:2000], model="openai/gpt-oss-120b")
    except Exception as e:
        print(f"extract_entities failed, skipping doc: {e}")
        return []
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return []
    try:
        return [e.strip() for e in json.loads(match.group(0)).get("entities", []) if e.strip()]
    except json.JSONDecodeError:
        return []


def write_doc(tx, vertical: str, doc: dict, entities: list[str]):
    tx.run(
        """
        MERGE (d:Document {id: $doc_id})
        SET d.text = $text, d.vertical = $vertical
        WITH d
        UNWIND $entities AS ent_name
        MERGE (e:Entity {name: toLower(ent_name), vertical: $vertical})
        MERGE (d)-[:MENTIONS]->(e)
        """,
        doc_id=doc["doc_id"], text=doc["text"], vertical=vertical, entities=entities,
    )


def already_done(vertical: str, expected: int) -> bool:
    with driver.session() as session:
        count = session.run(
            "MATCH (d:Document {vertical: $vertical}) RETURN count(d) AS n", vertical=vertical
        ).single()["n"]
    return count >= expected


def ingest_vertical(vertical: str, max_workers: int = 1):
    docs = load_corpus(vertical)
    if already_done(vertical, len(docs)):
        print(f"{vertical}: already fully graphed ({len(docs)} docs), skipping")
        return
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(extract_entities, d["text"]): d for d in docs}
        with driver.session() as session:
            done = 0
            for fut in as_completed(futures):
                doc = futures[fut]
                entities = fut.result()
                session.execute_write(write_doc, vertical, doc, entities)
                done += 1
                if done % 25 == 0:
                    print(f"{vertical}: {done}/{len(docs)} docs graphed")
    print(f"{vertical}: done, {len(docs)} docs")


if __name__ == "__main__":
    with driver.session() as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.vertical) IS NODE KEY")
    for v in VERTICALS:
        ingest_vertical(v)
    driver.close()
