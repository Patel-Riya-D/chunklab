from __future__ import annotations

from chunking.base import BaseChunker
from utils.models import Chunk, ExtractionResult


class FixedSizeChunker(BaseChunker):
    strategy_id = "fixed_size"
    name = "Fixed-size Chunking"
    description = "Splits text into fixed character windows."
    default_config = {"chunk_size": 1000, "overlap": 100}

    def chunk(self, extraction: ExtractionResult, config: dict | None = None) -> list[Chunk]:
        cfg = self.cfg(config)
        size = int(cfg["chunk_size"])
        overlap = min(int(cfg["overlap"]), size - 1)
        chunks: list[Chunk] = []
        step = max(1, size - overlap)
        for start in range(0, len(extraction.text), step):
            part = extraction.text[start : start + size]
            if part.strip():
                chunks.append(
                    self.make_chunk(
                        len(chunks) + 1,
                        part,
                        {"start": start, "end": start + len(part)},
                        overlap=overlap if chunks else 0,
                    )
                )
        return chunks


class RecursiveChunker(BaseChunker):
    strategy_id = "recursive"
    name = "Recursive Chunking"
    description = "Recursively splits with paragraph, sentence, and whitespace separators."
    default_config = {"chunk_size": 1200, "overlap": 150}

    def chunk(self, extraction: ExtractionResult, config: dict | None = None) -> list[Chunk]:
        cfg = self.cfg(config)
        size = int(cfg["chunk_size"])
        overlap = 0 if extraction.metadata.get("unit_type") in {"table", "code_symbol", "code_context"} else int(cfg["overlap"])
        separators = ["\n\n", "\n", ". ", " "]
        pieces = self._split(extraction.text, size, separators)
        chunks: list[Chunk] = []
        previous_tail = ""
        for piece in pieces:
            text = f"{previous_tail}{piece}".strip() if previous_tail else piece.strip()
            if text:
                chunks.append(
                    self.make_chunk(
                        len(chunks) + 1,
                        text,
                        {"splitter": "recursive"},
                        overlap=overlap if chunks else 0,
                    )
                )
            previous_tail = text[-overlap:] if overlap > 0 else ""
        return chunks

    def _split(self, text: str, size: int, separators: list[str]) -> list[str]:
        if len(text) <= size:
            return [text]
        if not separators:
            return [text[i : i + size] for i in range(0, len(text), size)]
        separator = separators[0]
        parts = text.split(separator)
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = f"{current}{separator}{part}" if current else part
            if len(candidate) <= size:
                current = candidate
            else:
                if current:
                    chunks.extend(self._split(current, size, separators[1:]))
                current = part
        if current:
            chunks.extend(self._split(current, size, separators[1:]))
        return chunks
