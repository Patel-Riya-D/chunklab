from __future__ import annotations

import re

from chunking.base import BaseChunker
from chunking.core import RecursiveChunker
from utils.models import Chunk, ExtractedUnit, ExtractionResult


def _flatten_units(units: list[ExtractedUnit]) -> list[ExtractedUnit]:
    flat: list[ExtractedUnit] = []
    for unit in units:
        flat.append(unit)
        flat.extend(_flatten_units(unit.children))
    return flat


def _contextual_text(unit: ExtractedUnit, include_content: bool = True) -> str:
    context_parts = []
    section_path = unit.metadata.get("section_path")
    if section_path:
        context_parts.append(f"Section path: {section_path}")
    elif unit.metadata.get("title"):
        context_parts.append(f"Section: {unit.metadata['title']}")
    if unit.metadata.get("page"):
        context_parts.append(f"Page: {unit.metadata['page']}")
    if unit.unit_type == "table":
        rows = unit.metadata.get("rows")
        if isinstance(rows, list) and rows:
            context_parts.extend(_table_context(rows))
    if not include_content:
        return "\n".join(context_parts)
    content = _content_without_repeated_title(unit)
    return "\n\n".join(part for part in [*context_parts, content] if part)


def _content_without_repeated_title(unit: ExtractedUnit) -> str:
    content = unit.text.strip()
    title = str(unit.metadata.get("title", "")).strip()
    if not title:
        return content
    lines = content.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].strip().casefold() == title.casefold():
        return "\n".join(lines[1:]).strip()
    return content


def _table_row_summary(rows: object, limit: int = 4) -> str:
    if not isinstance(rows, list) or len(rows) < 2 or not isinstance(rows[0], list):
        return ""
    headers = [str(cell).strip() for cell in rows[0]]
    summaries = []
    for row in rows[1 : limit + 1]:
        if not isinstance(row, list):
            continue
        pairs = [
            f"{header}: {str(cell).strip()}"
            for header, cell in zip(headers, row)
            if header and str(cell).strip()
        ]
        if pairs:
            summaries.append("; ".join(pairs))
    return "Table sample rows: " + " | ".join(summaries) if summaries else ""


def _table_context(rows: list[object]) -> list[str]:
    if _looks_like_key_value_table(rows):
        pairs = []
        for row in rows[:6]:
            if isinstance(row, list) and len(row) >= 2 and str(row[0]).strip() and str(row[1]).strip():
                pairs.append(f"{str(row[0]).strip()} = {str(row[1]).strip()}")
        return ["Key-value table: " + " | ".join(pairs)] if pairs else []
    headers = [str(cell) for cell in rows[0] if str(cell).strip()] if isinstance(rows[0], list) else []
    parts = ["Table columns: " + ", ".join(headers)] if headers else []
    summary = _table_row_summary(rows)
    if summary:
        parts.append(summary)
    return parts


def _looks_like_key_value_table(rows: list[object]) -> bool:
    if len(rows) < 2:
        return False
    two_col_rows = [row for row in rows if isinstance(row, list) and len(row) == 2 and any(str(cell).strip() for cell in row)]
    if len(two_col_rows) != len(rows):
        return False
    first_column = [str(row[0]).strip() for row in two_col_rows]
    second_column = [str(row[1]).strip() for row in two_col_rows]
    return all(first_column) and sum(1 for value in second_column if value) >= max(1, len(rows) - 1)


def _base_metadata(unit: ExtractedUnit) -> dict[str, object]:
    return {"source_unit": unit.id, "unit_type": unit.unit_type, **unit.metadata}


def _dedupe_chunks(chunks: list[Chunk]) -> list[Chunk]:
    seen: set[str] = set()
    unique: list[Chunk] = []
    for chunk in chunks:
        key = re.sub(r"\W+", " ", chunk.text.lower()).strip()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        unique.append(chunk)
    return unique


class SectionAwareChunker(BaseChunker):
    strategy_id = "section_aware"
    name = "Section-aware Chunking"
    description = "Keeps each detected section as a coherent chunk."
    default_config = {}

    def chunk(self, extraction: ExtractionResult, config: dict | None = None) -> list[Chunk]:
        units = _flatten_units(extraction.units)
        chunks: list[Chunk] = []
        for unit in units:
            if unit.unit_type not in {"section", "table"}:
                continue
            text = _contextual_text(unit)
            if text:
                chunks.append(self.make_chunk(len(chunks) + 1, text, _base_metadata(unit)))
        return _dedupe_chunks(chunks) or [self.make_chunk(1, extraction.text, {"fallback": True})]


class HeaderAwareChunker(SectionAwareChunker):
    strategy_id = "header_aware"
    name = "Header-aware Chunking"
    description = "Uses detected heading levels as chunk metadata."


class LayoutAwareChunker(BaseChunker):
    strategy_id = "layout_aware"
    name = "Layout-aware Chunking"
    description = "Chunks by visual/layout blocks and preserves page/bbox metadata."

    def chunk(self, extraction: ExtractionResult, config: dict | None = None) -> list[Chunk]:
        chunks = []
        for unit in _flatten_units(extraction.units):
            if unit.text.strip():
                chunks.append(self.make_chunk(len(chunks) + 1, _contextual_text(unit), _base_metadata(unit)))
        return _dedupe_chunks(chunks) or RecursiveChunker().chunk(extraction, config)


class DOMAwareChunker(BaseChunker):
    strategy_id = "dom_aware"
    name = "DOM-aware Chunking"
    description = "Treats extracted units as DOM-like nodes for HTML and markdown documents."

    def chunk(self, extraction: ExtractionResult, config: dict | None = None) -> list[Chunk]:
        chunks = []
        for unit in _flatten_units(extraction.units):
            chunks.append(self.make_chunk(len(chunks) + 1, _contextual_text(unit), {"node_type": unit.unit_type, **_base_metadata(unit)}))
        return _dedupe_chunks(chunks) or RecursiveChunker().chunk(extraction, config)


class ParentChildChunker(BaseChunker):
    strategy_id = "parent_child"
    name = "Parent-Child Chunking"
    description = "Creates large parent chunks and smaller retrievable child chunks."
    default_config = {"child_size": 500, "child_overlap": 60}

    def chunk(self, extraction: ExtractionResult, config: dict | None = None) -> list[Chunk]:
        cfg = self.cfg(config)
        chunks: list[Chunk] = []
        child_chunker = RecursiveChunker()
        units = _flatten_units(extraction.units) or []
        source_units = units if units else [ExtractedUnit(id="doc", text=extraction.text, unit_type="document")]
        for unit in source_units:
            parent = self.make_chunk(
                len(chunks) + 1,
                _contextual_text(unit),
                {"role": "parent", "chunk_role": "parent", **_base_metadata(unit)},
            )
            chunks.append(parent)
            child_overlap = 0 if unit.unit_type in {"table", "code_symbol", "code_context"} or len(unit.text) < int(cfg["child_size"]) else int(cfg["child_overlap"])
            child_chunks = child_chunker.chunk(
                ExtractionResult(extraction.method_id, extraction.method_name, _contextual_text(unit), [unit], unit.metadata),
                {"chunk_size": int(cfg["child_size"]), "overlap": child_overlap},
            )
            for child in child_chunks:
                child.id = f"{parent.id}-child-{len(parent.children) + 1}"
                child.strategy_id = self.strategy_id
                child.parent_id = parent.id
                child.metadata.update(
                    {
                        "role": "child",
                        "chunk_role": "child",
                        "parent_chunk_id": parent.id,
                        **_base_metadata(unit),
                    }
                )
                parent.children.append(child.id)
                chunks.append(child)
        return _dedupe_chunks(chunks)
