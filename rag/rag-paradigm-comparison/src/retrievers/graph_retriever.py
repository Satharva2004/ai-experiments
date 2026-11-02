"""Graph RAG: extract entities from the query, traverse the Neo4j
(Document)-[:MENTIONS]->(Entity) graph, rank documents by shared-entity count."""
from neo4j import GraphDatabase

from src.common.config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME
from src.ingestion.neo4j_ingest import extract_entities
from src.retrievers.base import Retriever


QUERY = """
    UNWIND $entities AS q_ent
    MATCH (e:Entity {vertical: $vertical})
    WHERE e.name CONTAINS q_ent OR q_ent CONTAINS e.name
    MATCH (d:Document)-[:MENTIONS]->(e)
    WITH d, count(DISTINCT e) AS shared
    RETURN d.id AS doc_id, d.text AS text, shared
    ORDER BY shared DESC
    LIMIT $top_k
"""


def _run_query(tx, vertical: str, entities: list[str], top_k: int):
    result = tx.run(QUERY, entities=entities, vertical=vertical, top_k=top_k)
    return [{"doc_id": r["doc_id"], "text": r["text"], "score": float(r["shared"])} for r in result]


class GraphRetriever(Retriever):
    name = "graph"

    def __init__(self):
        # a fresh connection pool per instance avoids reusing a connection
        # gone stale from Aura network drops during long-running ingestion
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    def retrieve(self, vertical: str, query: str, top_k: int = 5) -> list[dict]:
        query_entities = [e.lower() for e in extract_entities(query)] or [query.lower()]
        with self.driver.session() as session:
            # execute_read retries automatically on transient errors
            # (SessionExpired, ServiceUnavailable) - plain session.run() does not
            return session.execute_read(_run_query, vertical, query_entities, top_k)
