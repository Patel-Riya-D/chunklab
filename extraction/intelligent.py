from __future__ import annotations

import ast
import re

from extraction.base import BaseExtractor
from extraction.structured import _attach_unit_context, _extract_table_units, _ordered_units, _section_units
from utils.models import ExtractedUnit, ExtractionResult, UploadedDocument
from utils.text import split_sentences


class OCRExtractor(BaseExtractor):
    method_id = "ocr"
    name = "OCR-based Extraction"
    description = "OCR-ready pipeline placeholder with text fallback for born-digital documents."

    def extract(self, document: UploadedDocument) -> ExtractionResult:
        text = self._read_text(document)
        unit = ExtractedUnit(id="ocr-layer-1", text=text, unit_type="ocr_text", metadata={"ocr_engine": "placeholder"})
        return self._result(text, [unit], {"ocr_status": "placeholder_text_fallback"})


class HandwrittenOCRExtractor(OCRExtractor):
    method_id = "handwritten_ocr"
    name = "Handwritten OCR"
    description = "Architecture placeholder for future handwriting recognition models."

    def extract(self, document: UploadedDocument) -> ExtractionResult:
        result = super().extract(document)
        result.metadata["handwriting_status"] = "placeholder"
        result.method_id = self.method_id
        result.method_name = self.name
        return result


class SemanticSegmentationExtractor(BaseExtractor):
    method_id = "semantic_segmentation"
    name = "Semantic Segmentation"
    description = "Groups sentences into topical segments using a lightweight heuristic."

    def extract(self, document: UploadedDocument) -> ExtractionResult:
        text = self._read_text(document)
        table_units = _extract_table_units(document, text)
        sentences = split_sentences(text)
        units: list[ExtractedUnit] = []
        for start in range(0, len(sentences), 5):
            segment = " ".join(sentences[start : start + 5])
            if segment:
                idx = len(units) + 1
                units.append(ExtractedUnit(id=f"semantic-{idx}", text=segment, unit_type="semantic_segment", metadata={"segment": idx}))
        units = units or [ExtractedUnit(id="semantic-1", text=text, unit_type="semantic_segment")]
        return self._result(
            text,
            _attach_unit_context(_ordered_units([*units, *table_units])),
            {"tables_detected": len(table_units)},
        )


class ClauseAwareExtractor(BaseExtractor):
    method_id = "clause_aware"
    name = "Clause-aware Extraction"
    description = "Splits legal and policy-like text into clause-level units."

    def extract(self, document: UploadedDocument) -> ExtractionResult:
        text = self._read_text(document)
        table_units = _extract_table_units(document, text)
        clauses = [c.strip() for c in re.split(r"(?:(?<=;)|(?<=:)|\n\s*(?:\([a-z0-9]+\)|[a-z0-9]+\.)\s+)", text) if c.strip()]
        units = [
            ExtractedUnit(id=f"clause-{idx}", text=clause, unit_type="clause", metadata={"clause": idx})
            for idx, clause in enumerate(clauses, start=1)
        ]
        return self._result(
            text,
            _attach_unit_context(_ordered_units([*units, *table_units])),
            {"clauses": len(units), "tables_detected": len(table_units)},
        )


class CodeAwareExtractor(BaseExtractor):
    method_id = "code_aware"
    name = "Code-aware Extraction"
    description = "Detects functions/classes in code-like documents while preserving source text."

    def extract(self, document: UploadedDocument) -> ExtractionResult:
        text = self._read_text(document)
        table_units = _extract_table_units(document, text)
        if document.suffix == ".py":
            units = _python_code_units(text)
            if units:
                return self._result(
                    text,
                    _attach_unit_context(_ordered_units([*units, *table_units])),
                    {"symbols": len([unit for unit in units if unit.unit_type == "code_symbol"]), "tables_detected": len(table_units)},
                )

        pattern = re.compile(
            r"^\s*(?:(async)\s+)?(?:(def|class|function)\s+([A-Za-z_][\w]*)|"
            r"(const|let|var)\s+([A-Za-z_][\w]*)\s*=|"
            r"(public|private|protected)\s+(?:static\s+)?[\w<>\[\],]+\s+([A-Za-z_][\w]*)\s*\()",
            re.MULTILINE,
        )
        matches = list(pattern.finditer(text))
        units: list[ExtractedUnit] = []
        for idx, match in enumerate(matches, start=1):
            start = match.start()
            end = matches[idx].start() if idx < len(matches) else len(text)
            kind = match.group(2) or match.group(4) or match.group(6) or "symbol"
            if match.group(1) and kind == "def":
                kind = "async def"
            symbol = match.group(3) or match.group(5) or match.group(7)
            units.append(
                ExtractedUnit(
                    id=f"code-{idx}",
                    text=text[start:end].strip(),
                    unit_type="code_symbol",
                    metadata={"symbol": symbol, "kind": kind},
                )
            )
        units = units or _section_units(text)
        return self._result(
            text,
            _attach_unit_context(_ordered_units([*units, *table_units])),
            {"symbols": len(units), "tables_detected": len(table_units)},
        )


def _python_code_units(text: str) -> list[ExtractedUnit]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    lines = text.splitlines()
    spans = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = getattr(node, "lineno", 1)
            end = getattr(node, "end_lineno", start)
            kind = "class" if isinstance(node, ast.ClassDef) else "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            spans.append((start, end, kind, node.name))

    spans.sort(key=lambda item: item[0])
    units: list[ExtractedUnit] = []
    cursor = 1
    for start, end, kind, symbol in spans:
        if cursor < start:
            module_text = _join_code_lines(lines, cursor, start - 1)
            if module_text.strip():
                units.append(
                    ExtractedUnit(
                        id=f"code-{len(units) + 1}",
                        text=module_text,
                        unit_type="code_context",
                        metadata={"symbol": "module", "kind": "module", "line_start": cursor, "line_end": start - 1},
                    )
                )
        code_text = _join_code_lines(lines, start, end)
        if code_text.strip():
            units.append(
                ExtractedUnit(
                    id=f"code-{len(units) + 1}",
                    text=code_text,
                    unit_type="code_symbol",
                    metadata={"symbol": symbol, "kind": kind, "line_start": start, "line_end": end},
                )
            )
        cursor = end + 1

    if cursor <= len(lines):
        module_text = _join_code_lines(lines, cursor, len(lines))
        if module_text.strip():
            units.append(
                ExtractedUnit(
                    id=f"code-{len(units) + 1}",
                    text=module_text,
                    unit_type="code_context",
                    metadata={"symbol": "module", "kind": "module", "line_start": cursor, "line_end": len(lines)},
                )
            )
    return units


def _join_code_lines(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[start - 1 : end]).strip("\n")


class IDPExtractor(BaseExtractor):
    method_id = "idp"
    name = "Intelligent Document Processing"
    description = "Placeholder architecture for form/entity/classification enrichment."

    def extract(self, document: UploadedDocument) -> ExtractionResult:
        text = self._read_text(document)
        table_units = _extract_table_units(document, text)
        units = _section_units(text)
        return self._result(
            text,
            _attach_unit_context(_ordered_units([*units, *table_units])),
            {"idp_status": "placeholder", "entities": [], "tables_detected": len(table_units)},
        )
