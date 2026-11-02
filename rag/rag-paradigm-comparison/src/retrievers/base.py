from abc import ABC, abstractmethod


class Retriever(ABC):
    name: str

    @abstractmethod
    def retrieve(self, vertical: str, query: str, top_k: int = 5) -> list[dict]:
        """Return a list of {doc_id, text, score} ranked by relevance."""
