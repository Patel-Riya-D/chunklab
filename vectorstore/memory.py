from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from utils.models import Chunk, RetrievalResult
from vectorstore.embeddings import Embedder, HashingEmbedder, get_default_embedder


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / denom if denom else 0.0


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "who",
    "why",
    "how",
}


def tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9_]+", text.lower()) if token not in STOPWORDS]


def lexical_similarity(query: str, text: str) -> float:
    query_terms = tokenize(query)
    text_terms = tokenize(text)
    if not query_terms or not text_terms:
        return 0.0

    query_counts = Counter(query_terms)
    text_counts = Counter(text_terms)
    overlap = sum(min(query_counts[term], text_counts.get(term, 0)) for term in query_counts)
    coverage = overlap / max(1, sum(query_counts.values()))

    query_phrase = " ".join(query_terms)
    text_normalized = " ".join(text_terms)
    phrase_boost = 0.35 if query_phrase and query_phrase in text_normalized else 0.0
    title_boost = 0.15 if query_terms and text_normalized.startswith(query_terms[0]) else 0.0
    return min(1.0, coverage + phrase_boost + title_boost)


def searchable_text(chunk: Chunk) -> str:
    metadata_text = " ".join(
        str(value)
        for key, value in chunk.metadata.items()
        if key in {"title", "section", "source_unit", "page"} and value is not None
    )
    return f"{metadata_text}\n{chunk.text}".strip()


@dataclass
class InMemoryVectorStore:
    embedder: Embedder = field(default_factory=get_default_embedder)
    chunks: list[Chunk] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)

    def build(self, chunks: list[Chunk]) -> None:
        retrievable = [chunk for chunk in chunks if chunk.text.strip()]
        self.chunks = retrievable
        texts = [searchable_text(chunk) for chunk in retrievable]
        try:
            self.vectors = self.embedder.embed(texts) if texts else []
        except Exception:
            self.embedder = HashingEmbedder()
            self.vectors = self.embedder.embed(texts) if texts else []

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self.chunks:
            return []
        query_vector = self.embedder.embed([query])[0]
        scored = []
        for idx, (chunk, vector) in enumerate(zip(self.chunks, self.vectors), start=1):
            vector_score = cosine_similarity(query_vector, vector)
            lexical_score = lexical_similarity(query, searchable_text(chunk))
            score = (0.75 * lexical_score) + (0.25 * vector_score)
            scored.append(RetrievalResult(chunk=chunk, score=score, rank=idx))
        scored.sort(key=lambda result: result.score, reverse=True)
        top = scored[:top_k]
        for idx, result in enumerate(top, start=1):
            result.rank = idx
        return top
