from __future__ import annotations

import time
from dataclasses import dataclass

from utils.models import Chunk, RetrievalResult
from vectorstore.faiss_store import FAISSVectorStore


@dataclass
class RetrievalResponse:
    results: list[RetrievalResult]
    latency_ms: float


class Retriever:
    def __init__(self, vector_store: FAISSVectorStore | None = None) -> None:
        self.vector_store = vector_store or FAISSVectorStore()

    def index(self, chunks: list[Chunk]) -> None:
        self.vector_store.build(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResponse:
        started = time.perf_counter()
        results = self.vector_store.search(query, top_k=top_k)
        latency_ms = (time.perf_counter() - started) * 1000
        return RetrievalResponse(results=results, latency_ms=latency_ms)
