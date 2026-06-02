from __future__ import annotations

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    RecursiveCharacterTextSplitter = None

from chunking.base import BaseChunker
from utils.models import Chunk, ExtractedUnit, ExtractionResult


def _flatten_units(units: list[ExtractedUnit]) -> list[ExtractedUnit]:
    flat: list[ExtractedUnit] = []
    for unit in units:
        flat.append(unit)
        flat.extend(_flatten_units(unit.children))
    return flat


def _chunkable_units(extraction: ExtractionResult) -> list[ExtractedUnit]:
    units = _flatten_units(extraction.units)
    return units if any(unit.unit_type == "table" for unit in units) else []


def _unit_metadata(unit: ExtractedUnit) -> dict[str, object]:
    return {"source_unit": unit.id, "unit_type": unit.unit_type, **unit.metadata}


def _recursive_split_text(text: str, size: int, overlap: int) -> list[str]:
    separators = ["\n\n", "\n", ". ", " "]
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=separators,
            length_function=len,
        )
        return splitter.split_text(text)
    pieces = _split_by_separators(text, size, separators)
    if overlap <= 0:
        return pieces
    overlapped: list[str] = []
    previous_tail = ""
    for piece in pieces:
        chunk_text = f"{previous_tail}{piece}".strip() if previous_tail else piece.strip()
        if chunk_text:
            overlapped.append(chunk_text)
            previous_tail = chunk_text[-overlap:]
    return overlapped


def _split_by_separators(text: str, size: int, separators: list[str]) -> list[str]:
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
                chunks.extend(_split_by_separators(current, size, separators[1:]))
            current = part
    if current:
        chunks.extend(_split_by_separators(current, size, separators[1:]))
    return chunks


class FixedSizeChunker(BaseChunker):
    strategy_id = "fixed_size"
    name = "Fixed-size Chunking"
    description = "Splits text into fixed character windows."
    default_config = {"chunk_size": 1000, "overlap": 100}

    def chunk(self, extraction: ExtractionResult, config: dict | None = None) -> list[Chunk]:
        cfg = self.cfg(config)
        size = int(cfg["chunk_size"])
        configured_overlap = min(int(cfg["overlap"]), size - 1)
        chunks: list[Chunk] = []
        source_units = _chunkable_units(extraction)

        def add_text_chunks(text: str, metadata: dict[str, object], overlap: int) -> None:
            step = max(1, size - overlap)
            for start in range(0, len(text), step):
                part = text[start : start + size]
                if part.strip():
                    chunks.append(
                        self.make_chunk(
                            len(chunks) + 1,
                            part,
                            {"start": start, "end": start + len(part), **metadata},
                            overlap=overlap if chunks else 0,
                        )
                    )

        for unit in source_units:
            if unit.unit_type == "table":
                chunks.append(
                    self.make_chunk(
                        len(chunks) + 1,
                        unit.text,
                        _unit_metadata(unit),
                    )
                )
            elif unit.text.strip():
                add_text_chunks(unit.text, _unit_metadata(unit), configured_overlap)
        if not source_units:
            add_text_chunks(extraction.text, {}, configured_overlap)
        return chunks


class RecursiveChunker(BaseChunker):
    strategy_id = "recursive"
    name = "Recursive Chunking"
    description = "Recursively splits with paragraph, sentence, and whitespace separators."
    default_config = {"chunk_size": 1200, "overlap": 150}

    def chunk(self, extraction: ExtractionResult, config: dict | None = None) -> list[Chunk]:
        cfg = self.cfg(config)
        size = int(cfg["chunk_size"])
        configured_overlap = min(int(cfg["overlap"]), size - 1)
        chunks: list[Chunk] = []
        source_units = _chunkable_units(extraction)

        def add_text_chunks(text: str, metadata: dict[str, object], overlap: int) -> None:
            for piece in _recursive_split_text(text, size, overlap):
                chunk_text = piece.strip()
                if chunk_text:
                    chunks.append(
                        self.make_chunk(
                            len(chunks) + 1,
                            chunk_text,
                            {"splitter": "recursive_text_splitter", **metadata},
                            overlap=overlap if chunks else 0,
                        )
                    )

        for unit in source_units:
            metadata = _unit_metadata(unit)
            if unit.unit_type == "table":
                chunks.append(
                    self.make_chunk(
                        len(chunks) + 1,
                        unit.text,
                        {"splitter": "recursive_text_splitter", **metadata},
                    )
                )
            elif unit.text.strip():
                overlap = 0 if unit.unit_type in {"code_symbol", "code_context"} else configured_overlap
                add_text_chunks(unit.text, metadata, overlap)
        if not source_units:
            overlap = 0 if extraction.metadata.get("unit_type") in {"table", "code_symbol", "code_context"} else configured_overlap
            add_text_chunks(extraction.text, {}, overlap)
        return chunks
