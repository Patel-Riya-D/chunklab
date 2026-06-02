from __future__ import annotations

import re

from extraction.base import BaseExtractor
from extraction.structured import (
    _attach_unit_context,
    _docx_block_units,
    _docx_page_units,
    _extract_table_units,
    _ordered_units,
    _pdf_pages_without_tables,
)
from utils.models import ExtractedUnit, ExtractionResult, UploadedDocument
from utils.text import normalize_whitespace


class WholeTextExtractor(BaseExtractor):
    method_id = "whole_text"
    name = "Whole Text Extraction"
    description = "Extracts all readable content as a single document unit."

    def extract(self, document: UploadedDocument) -> ExtractionResult:
        if document.suffix == ".docx":
            docx_result = _docx_block_units(document.content)
            if docx_result:
                text, units = docx_result
                return self._result(
                    text=text,
                    units=_attach_unit_context(_ordered_units([ExtractedUnit(id="doc", text=text, unit_type="document", metadata={"source": document.name}), *units])),
                    metadata={"tables_detected": sum(1 for unit in units if unit.unit_type == "table")},
                )

        text = self._read_text(document)
        table_units = _extract_table_units(document, text)
        return self._result(
            text=text,
            units=_attach_unit_context(
                _ordered_units(
                    [
                        ExtractedUnit(id="doc", text=text, unit_type="document", metadata={"source": document.name}),
                        *table_units,
                    ]
                )
            ),
            metadata={"tables_detected": len(table_units)},
        )


class PageWiseExtractor(BaseExtractor):
    method_id = "page_wise"
    name = "Page-wise Extraction"
    description = "Preserves page boundaries for PDFs; falls back to one page for other formats."

    def extract(self, document: UploadedDocument) -> ExtractionResult:
        if document.suffix == ".pdf":
            pages = _pdf_pages_without_tables(document.content) or self._read_pdf_pages(document.content)
        elif document.suffix == ".docx":
            docx_result = _docx_page_units(document.content)
            if docx_result:
                text, units = docx_result
                pages_detected = sum(1 for unit in units if unit.unit_type == "page")
                tables_detected = sum(1 for unit in units if unit.unit_type == "table")
                return self._result(
                    text,
                    _attach_unit_context(units),
                    {"pages": pages_detected, "tables_detected": tables_detected},
                )
            pages = [self._read_text(document)]
        else:
            pages = [self._read_text(document)]
        units = [
            ExtractedUnit(id=f"page-{idx}", text=page, unit_type="page", metadata={"page": idx, "order": idx * 10_000})
            for idx, page in enumerate(pages, start=1)
            if page.strip()
        ]
        text = "\n\n".join(unit.text for unit in units)
        table_units = _extract_table_units(document, text)
        return self._result(
            text,
            _attach_unit_context(_ordered_units([*units, *table_units])),
            {"pages": len(units), "tables_detected": len(table_units)},
        )


class BlockWiseExtractor(BaseExtractor):
    method_id = "block_wise"
    name = "Block-wise Extraction"
    description = "Extracts visual text blocks where available, otherwise paragraph-like blocks."

    def extract(self, document: UploadedDocument) -> ExtractionResult:
        if document.suffix == ".docx":
            docx_result = _docx_block_units(document.content)
            if docx_result:
                text, units = docx_result
                return self._result(
                    text,
                    _attach_unit_context(units),
                    {"blocks": len([unit for unit in units if unit.unit_type != "table"]), "tables_detected": len([unit for unit in units if unit.unit_type == "table"])},
                )

        blocks = self._read_pdf_blocks(document.content) if document.suffix == ".pdf" else []
        if blocks:
            units = [
                ExtractedUnit(
                    id=f"block-{idx}",
                    text=str(block["text"]),
                    unit_type="layout_block",
                    metadata={k: v for k, v in block.items() if k != "text"},
                )
                for idx, block in enumerate(blocks, start=1)
            ]
        else:
            text = self._read_text(document)
            paragraphs = [normalize_whitespace(p) for p in re.split(r"\n\s*\n", text) if p.strip()]
            units = [
                ExtractedUnit(id=f"block-{idx}", text=paragraph, unit_type="paragraph", metadata={"block": idx})
                for idx, paragraph in enumerate(paragraphs, start=1)
            ]
        text = "\n\n".join(unit.text for unit in units)
        table_units = _extract_table_units(document, text)
        return self._result(
            text,
            _attach_unit_context(_ordered_units([*units, *table_units])),
            {"blocks": len(units), "tables_detected": len(table_units)},
        )
