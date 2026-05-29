from __future__ import annotations

import re

from extraction.base import BaseExtractor
from utils.models import ExtractedUnit, ExtractionResult, UploadedDocument
from utils.text import normalize_whitespace


class WholeTextExtractor(BaseExtractor):
    method_id = "whole_text"
    name = "Whole Text Extraction"
    description = "Extracts all readable content as a single document unit."

    def extract(self, document: UploadedDocument) -> ExtractionResult:
        text = self._read_text(document)
        return self._result(
            text=text,
            units=[ExtractedUnit(id="doc", text=text, unit_type="document", metadata={"source": document.name})],
        )


class PageWiseExtractor(BaseExtractor):
    method_id = "page_wise"
    name = "Page-wise Extraction"
    description = "Preserves page boundaries for PDFs; falls back to one page for other formats."

    def extract(self, document: UploadedDocument) -> ExtractionResult:
        if document.suffix == ".pdf":
            pages = self._read_pdf_pages(document.content)
        else:
            pages = [self._read_text(document)]
        units = [
            ExtractedUnit(id=f"page-{idx}", text=page, unit_type="page", metadata={"page": idx})
            for idx, page in enumerate(pages, start=1)
            if page.strip()
        ]
        return self._result("\n\n".join(unit.text for unit in units), units, {"pages": len(units)})


class BlockWiseExtractor(BaseExtractor):
    method_id = "block_wise"
    name = "Block-wise Extraction"
    description = "Extracts visual text blocks where available, otherwise paragraph-like blocks."

    def extract(self, document: UploadedDocument) -> ExtractionResult:
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
        return self._result("\n\n".join(unit.text for unit in units), units, {"blocks": len(units)})

