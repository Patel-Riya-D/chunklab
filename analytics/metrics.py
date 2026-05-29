from __future__ import annotations

from statistics import mean

from utils.models import ChatTurn, Chunk
from utils.text import estimate_tokens


def chunk_metrics(chunks: list[Chunk]) -> dict[str, object]:
    char_sizes = [len(chunk.text) for chunk in chunks]
    token_sizes = [estimate_tokens(chunk.text) for chunk in chunks]
    return {
        "chunk_count": len(chunks),
        "avg_chars": round(mean(char_sizes), 1) if char_sizes else 0,
        "min_chars": min(char_sizes) if char_sizes else 0,
        "max_chars": max(char_sizes) if char_sizes else 0,
        "avg_tokens_est": round(mean(token_sizes), 1) if token_sizes else 0,
        "parent_chunks": sum(1 for chunk in chunks if chunk.children),
        "child_chunks": sum(1 for chunk in chunks if chunk.parent_id),
        "size_distribution": char_sizes,
    }


def retrieval_metrics(history: list[ChatTurn]) -> dict[str, object]:
    latencies = [turn.latency_ms for turn in history]
    top_chunks = []
    for turn in history:
        top_chunks.extend(result.chunk.id for result in turn.retrieved[:3])
    return {
        "queries": len(history),
        "avg_latency_ms": round(mean(latencies), 1) if latencies else 0,
        "last_latency_ms": round(latencies[-1], 1) if latencies else 0,
        "top_retrieved_chunks": top_chunks[-10:],
    }

