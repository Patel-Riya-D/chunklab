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


def bm25_similarity(
    query_terms: list[str],
    text_terms: list[str],
    document_frequency: Counter[str],
    document_count: int,
    avg_doc_len: float,
) -> float:
    if not query_terms or not text_terms:
        return 0.0

    term_counts = Counter(text_terms)
    doc_len = len(text_terms)
    k1 = 1.5
    b = 0.75
    score = 0.0
    for term in Counter(query_terms):
        tf = term_counts.get(term, 0)
        if not tf:
            continue
        df = document_frequency.get(term, 0)
        idf = math.log(1 + ((document_count - df + 0.5) / (df + 0.5)))
        denom = tf + k1 * (1 - b + b * (doc_len / max(1.0, avg_doc_len)))
        score += idf * ((tf * (k1 + 1)) / denom)
    return score / (score + 3.0) if score > 0 else 0.0


def query_intent_boost(query: str, text: str) -> float:
    boost = 0.0
    boost += acronym_expansion_boost(query, text)
    boost += definition_boost(query, text)
    boost += proximity_boost(query, text)
    return min(0.35, boost)


def acronym_expansion_boost(query: str, text: str) -> float:
    if not re.search(r"\b(full\s*form|stands?\s+for|meaning\s+of)\b", query, re.IGNORECASE):
        return 0.0
    acronyms = re.findall(r"\b[A-Z]{2,}\b", query)
    if not acronyms:
        return 0.0
    for acronym in acronyms:
        if has_acronym_expansion(text, acronym):
            return 2.0
    if any(re.search(rf"\b{re.escape(acronym)}\b", text) for acronym in acronyms):
        return 0.05
    return 0.0


def has_acronym_expansion(text: str, acronym: str) -> bool:
    before = re.search(rf"([A-Z][A-Za-z&.,'’/ -]{{3,120}}?)\s*\(\s*{re.escape(acronym)}\s*\)", text)
    after = re.search(
        rf"\b{re.escape(acronym)}\b\s*(?:means|stands for|[:=-])\s*([A-Z][A-Za-z&.,'’/ -]{{3,120}})",
        text,
        re.IGNORECASE,
    )
    return bool(before or after)


def definition_boost(query: str, text: str) -> float:
    match = re.search(r"\b(?:what\s+is|define|definition\s+of|explain)\s+(.+)", query, re.IGNORECASE)
    if not match:
        return 0.0
    subject_terms = tokenize(match.group(1))
    if not subject_terms:
        return 0.0
    text_terms = tokenize(text)
    subject = " ".join(subject_terms)
    normalized_text = " ".join(text_terms)
    if subject and subject in normalized_text:
        return 0.5
    coverage = len(set(subject_terms) & set(text_terms)) / max(1, len(set(subject_terms)))
    return 0.25 if coverage >= 0.75 else 0.0


def proximity_boost(query: str, text: str) -> float:
    query_terms = tokenize(query)
    text_terms = tokenize(text)
    if len(query_terms) < 2 or not text_terms:
        return 0.0

    positions_by_term: dict[str, list[int]] = {}
    for index, term in enumerate(text_terms):
        positions_by_term.setdefault(term, []).append(index)

    present_terms = [term for term in dict.fromkeys(query_terms) if term in positions_by_term]
    coverage = len(present_terms) / max(1, len(set(query_terms)))
    if coverage < 0.5:
        return 0.0

    positions = [positions_by_term[term][0] for term in present_terms]
    window = max(positions) - min(positions) + 1 if positions else len(text_terms)
    compactness = len(present_terms) / max(1, window)
    return min(0.4, coverage * compactness)


def searchable_text(chunk: Chunk) -> str:
    metadata_text = " ".join(
        str(value)
        for key, value in chunk.metadata.items()
        if key in {"title", "section", "source_unit", "page"} and value is not None
    )
    row_text = ""
    rows = chunk.metadata.get("rows")
    if isinstance(rows, list):
        row_text = " ".join(
            " ".join(str(cell) for cell in row)
            for row in rows
            if isinstance(row, list)
        )
    return f"{metadata_text}\n{chunk.text}\n{row_text}".strip()


@dataclass
class InMemoryVectorStore:
    embedder: Embedder = field(default_factory=get_default_embedder)
    chunks: list[Chunk] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)
    tokenized_texts: list[list[str]] = field(default_factory=list)
    document_frequency: Counter[str] = field(default_factory=Counter)
    avg_doc_len: float = 0.0
    semantic_weight: float = 0.65
    bm25_weight: float = 0.20
    lexical_weight: float = 0.10
    intent_weight: float = 0.05

    def build(self, chunks: list[Chunk]) -> None:
        retrievable = [chunk for chunk in chunks if chunk.text.strip()]
        self.chunks = retrievable
        texts = [searchable_text(chunk) for chunk in retrievable]
        self.tokenized_texts = [tokenize(text) for text in texts]
        self.document_frequency = Counter()
        for tokens in self.tokenized_texts:
            self.document_frequency.update(set(tokens))
        self.avg_doc_len = sum(len(tokens) for tokens in self.tokenized_texts) / max(1, len(self.tokenized_texts))
        try:
            self.vectors = self.embedder.embed(texts) if texts else []
        except Exception:
            self.embedder = HashingEmbedder()
            self.vectors = self.embedder.embed(texts) if texts else []

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self.chunks:
            return []
        query_vector = self.embedder.embed([query])[0]
        query_terms = tokenize(query)
        scored = []
        for idx, (chunk, vector, tokens) in enumerate(zip(self.chunks, self.vectors, self.tokenized_texts), start=1):
            text = searchable_text(chunk)
            vector_score = cosine_similarity(query_vector, vector)
            lexical_score = lexical_similarity(query, text)
            bm25 = bm25_similarity(query_terms, tokens, self.document_frequency, len(self.chunks), self.avg_doc_len)
            intent_boost = query_intent_boost(query, text)
            score = (
                (self.semantic_weight * vector_score)
                + (self.bm25_weight * bm25)
                + (self.lexical_weight * lexical_score)
                + (self.intent_weight * intent_boost)
            )
            scored.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    rank=idx,
                    score_details={
                        "bm25": round(bm25, 4),
                        "lexical": round(lexical_score, 4),
                        "vector": round(vector_score, 4),
                        "intent": round(intent_boost, 4),
                        "ranking": "semantic-first hybrid",
                        "embedder": self.embedder.name,
                    },
                )
            )
        scored.sort(key=lambda result: result.score, reverse=True)
        top = scored[:top_k]
        for idx, result in enumerate(top, start=1):
            result.rank = idx
        return top
