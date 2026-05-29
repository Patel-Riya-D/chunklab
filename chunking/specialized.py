from __future__ import annotations

import re

from chunking.base import BaseChunker
from chunking.core import RecursiveChunker
from chunking.structure import LayoutAwareChunker, _base_metadata, _contextual_text, _dedupe_chunks
from utils.models import Chunk, ExtractionResult


class OCRAwareChunker(BaseChunker):
    strategy_id = "ocr_aware"
    name = "OCR-aware Chunking"
    description = "Chunks OCR text while preserving page or OCR layer metadata."
    default_config = {"chunk_size": 800, "overlap": 80}

    def chunk(self, extraction: ExtractionResult, config: dict | None = None) -> list[Chunk]:
        chunks = RecursiveChunker().chunk(extraction, self.cfg(config))
        for chunk in chunks:
            chunk.strategy_id = self.strategy_id
            chunk.metadata["ocr_sensitive"] = True
        return chunks


class TableAwareChunker(BaseChunker):
    strategy_id = "table_aware"
    name = "Table-aware Chunking"
    description = "Keeps detected table-like regions intact."

    def chunk(self, extraction: ExtractionResult, config: dict | None = None) -> list[Chunk]:
        chunks: list[Chunk] = []
        has_explicit_tables = any(unit.unit_type == "table" for unit in extraction.units)
        for unit in extraction.units:
            if unit.unit_type == "table" or (not has_explicit_tables and "|" in unit.text):
                chunks.append(self.make_chunk(len(chunks) + 1, _contextual_text(unit), {**_base_metadata(unit), "table": True}))
            elif unit.text.strip():
                chunks.append(self.make_chunk(len(chunks) + 1, _contextual_text(unit), {"table": False, **_base_metadata(unit)}))
        return _dedupe_chunks(chunks) or LayoutAwareChunker().chunk(extraction, config)


class CodeFunctionChunker(BaseChunker):
    strategy_id = "code_function"
    name = "Code Function-aware Chunking"
    description = "Uses function/class boundaries detected by code-aware extraction."

    def chunk(self, extraction: ExtractionResult, config: dict | None = None) -> list[Chunk]:
        chunks: list[Chunk] = []
        for unit in extraction.units:
            if unit.unit_type in {"code_symbol", "code_context"}:
                chunks.append(self.make_chunk(len(chunks) + 1, unit.text, {"symbol": unit.metadata.get("symbol"), **unit.metadata}))
        if chunks:
            return chunks
        pattern = re.compile(r"^\s*(?:async\s+)?(?:def|class|function)\s+([A-Za-z_][\w]*)", re.MULTILINE)
        matches = list(pattern.finditer(extraction.text))
        for idx, match in enumerate(matches, start=1):
            end = matches[idx].start() if idx < len(matches) else len(extraction.text)
            chunks.append(self.make_chunk(idx, extraction.text[match.start() : end], {"symbol": match.group(1)}))
        return chunks or RecursiveChunker().chunk(extraction, config)


class MultiModalChunker(BaseChunker):
    strategy_id = "multi_modal"
    name = "Multi-modal Chunking"
    description = "Placeholder-ready chunker for text, tables, images, and layout regions."

    def chunk(self, extraction: ExtractionResult, config: dict | None = None) -> list[Chunk]:
        chunks = LayoutAwareChunker().chunk(extraction, config)
        for chunk in chunks:
            chunk.metadata["modality"] = "text"
            chunk.metadata["multi_modal_ready"] = True
        return chunks
