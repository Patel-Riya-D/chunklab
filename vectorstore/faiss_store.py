from __future__ import annotations

from dataclasses import dataclass, field
import math

from utils.models import Chunk, RetrievalResult
from vectorstore.embeddings import Embedder, get_default_embedder
from vectorstore.memory import InMemoryVectorStore, searchable_text


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


@dataclass
class FAISSVectorStore:
    embedder: Embedder = field(default_factory=get_default_embedder)
    chunks: list[Chunk] = field(default_factory=list)
    index: object | None = None
    fallback: InMemoryVectorStore = field(default_factory=InMemoryVectorStore)

    def build(self, chunks: list[Chunk]) -> None:
        try:
            self.fallback.embedder = self.embedder
            self.fallback.build(chunks)
            import faiss
            import numpy as np

            self.chunks = [chunk for chunk in chunks if chunk.text.strip()]
            vectors = self.embedder.embed([searchable_text(chunk) for chunk in self.chunks]) if self.chunks else []
            vectors = [_normalize_vector(vector) for vector in vectors]
            matrix = np.array(vectors, dtype="float32")
            self.index = faiss.IndexFlatIP(matrix.shape[1]) if len(matrix) else None
            if self.index is not None:
                self.index.add(matrix)
        except Exception:
            self.fallback.embedder = self.embedder
            self.fallback.build(chunks)
            self.index = None

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if self.index is None:
            return self.fallback.search(query, top_k)
        try:
            candidate_k = min(len(self.chunks), max(top_k * 5, top_k))
            vector_results = self._vector_search(query, candidate_k)
            hybrid_results = self.fallback.search(query, candidate_k)
            return self._merge_results(vector_results, hybrid_results, top_k)
        except Exception:
            return self.fallback.search(query, top_k)

    def _vector_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self.index is None:
            return []
        import numpy as np

        query_vector = _normalize_vector(self.embedder.embed([query])[0])
        query_matrix = np.array([query_vector], dtype="float32")
        scores, indexes = self.index.search(query_matrix, min(top_k, len(self.chunks)))
        results: list[RetrievalResult] = []
        for rank, (score, chunk_index) in enumerate(zip(scores[0], indexes[0]), start=1):
            if chunk_index >= 0:
                results.append(
                    RetrievalResult(
                        chunk=self.chunks[int(chunk_index)],
                        score=float(score),
                        rank=rank,
                        score_details={
                            "faiss_vector": round(float(score), 4),
                            "faiss_rank": str(rank),
                            "embedder": self.embedder.name,
                        },
                    )
                )
        return results

    def _merge_results(
        self,
        vector_results: list[RetrievalResult],
        hybrid_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        vector_by_id = {result.chunk.id: result for result in vector_results}
        merged: list[RetrievalResult] = []
        for result in hybrid_results:
            vector_result = vector_by_id.get(result.chunk.id)
            if vector_result:
                result.score_details["faiss_vector"] = round(vector_result.score, 4)
                result.score_details["faiss_rank"] = str(vector_result.rank)
                result.score += 0.05 * (1 / max(1, vector_result.rank))
            result.score_details["index"] = "faiss+semantic-hybrid"
            merged.append(result)

        hybrid_ids = {result.chunk.id for result in hybrid_results}
        for result in vector_results:
            if result.chunk.id not in hybrid_ids:
                result.score_details["index"] = "faiss-vector"
                merged.append(result)

        merged.sort(key=lambda result: result.score, reverse=True)
        top = merged[:top_k]
        for rank, result in enumerate(top, start=1):
            result.rank = rank
        return top
