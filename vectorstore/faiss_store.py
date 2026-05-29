from __future__ import annotations

from dataclasses import dataclass, field

from utils.models import Chunk, RetrievalResult
from vectorstore.embeddings import Embedder, get_default_embedder
from vectorstore.memory import InMemoryVectorStore, searchable_text


@dataclass
class FAISSVectorStore:
    embedder: Embedder = field(default_factory=get_default_embedder)
    chunks: list[Chunk] = field(default_factory=list)
    index: object | None = None
    fallback: InMemoryVectorStore = field(default_factory=InMemoryVectorStore)

    def build(self, chunks: list[Chunk]) -> None:
        try:
            self.fallback.build(chunks)
            import faiss
            import numpy as np

            self.chunks = [chunk for chunk in chunks if chunk.text.strip()]
            vectors = self.embedder.embed([searchable_text(chunk) for chunk in self.chunks]) if self.chunks else []
            matrix = np.array(vectors, dtype="float32")
            self.index = faiss.IndexFlatIP(matrix.shape[1]) if len(matrix) else None
            if self.index is not None:
                self.index.add(matrix)
        except Exception:
            self.fallback.build(chunks)
            self.index = None

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        hybrid_results = self.fallback.search(query, top_k)
        if hybrid_results:
            return hybrid_results
        if self.index is None:
            return self.fallback.search(query, top_k)
        try:
            import numpy as np

            query_vector = np.array(self.embedder.embed([query]), dtype="float32")
            scores, indexes = self.index.search(query_vector, min(top_k, len(self.chunks)))
            results: list[RetrievalResult] = []
            for rank, (score, chunk_index) in enumerate(zip(scores[0], indexes[0]), start=1):
                if chunk_index >= 0:
                    results.append(RetrievalResult(chunk=self.chunks[int(chunk_index)], score=float(score), rank=rank))
            return results
        except Exception:
            return self.fallback.search(query, top_k)
