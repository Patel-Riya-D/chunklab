from __future__ import annotations

import re

from chunking.base import BaseChunker
from chunking.core import RecursiveChunker
from chunking.structure import LayoutAwareChunker
from utils.models import Chunk, ExtractionResult
from utils.text import split_sentences


class SemanticChunker(BaseChunker):
    strategy_id = "semantic"
    name = "Semantic Chunking"
    description = "Groups adjacent sentences into semantic windows."
    default_config = {"sentences_per_chunk": 5}

    def chunk(self, extraction: ExtractionResult, config: dict | None = None) -> list[Chunk]:
        cfg = self.cfg(config)
        span = int(cfg["sentences_per_chunk"])
        sentences = split_sentences(extraction.text)
        chunks = []
        for start in range(0, len(sentences), span):
            text = " ".join(sentences[start : start + span])
            if text:
                chunks.append(self.make_chunk(len(chunks) + 1, text, {"sentence_start": start + 1, "sentence_end": start + span}))
        return chunks or RecursiveChunker().chunk(extraction, config)


class AdaptiveChunker(BaseChunker):
    strategy_id = "adaptive"
    name = "Adaptive Chunking"
    description = "Chooses chunk granularity from the extracted structure density."
    default_config = {"target_chars": 900}

    def chunk(self, extraction: ExtractionResult, config: dict | None = None) -> list[Chunk]:
        cfg = self.cfg(config)
        target = int(cfg["target_chars"])
        if extraction.units and len(extraction.units) > 2:
            return LayoutAwareChunker().chunk(extraction, config)
        return RecursiveChunker().chunk(extraction, {"chunk_size": target, "overlap": int(target * 0.1)})


class ClauseAwareChunker(BaseChunker):
    strategy_id = "clause_aware"
    name = "Clause-aware Chunking"
    description = "Creates chunks around clauses and enumerated terms."

    def chunk(self, extraction: ExtractionResult, config: dict | None = None) -> list[Chunk]:
        clauses = [c.strip() for c in re.split(r"(?:(?<=;)|\n\s*(?:\([a-z0-9]+\)|[a-z0-9]+\.)\s+)", extraction.text) if c.strip()]
        return [self.make_chunk(idx, clause, {"clause": idx}) for idx, clause in enumerate(clauses, start=1)]

