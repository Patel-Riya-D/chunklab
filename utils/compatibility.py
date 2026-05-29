from __future__ import annotations


EXTRACTION_METHODS: dict[str, str] = {
    "whole_text": "Whole Text Extraction",
    "page_wise": "Page-wise Extraction",
    "block_wise": "Block-wise Extraction",
    "section_wise": "Section-wise Extraction",
    "hierarchical": "Hierarchical Extraction",
    "layout_aware": "Layout-aware Extraction",
    "table_aware": "Table-aware Extraction",
    "semantic_segmentation": "Semantic Segmentation",
    "clause_aware": "Clause-aware Extraction",
    "code_aware": "Code-aware Extraction",
}


CHUNKING_STRATEGIES: dict[str, str] = {
    "fixed_size": "Fixed-size Chunking",
    "recursive": "Recursive Chunking",
    "header_aware": "Header-aware Chunking",
    "section_aware": "Section-aware Chunking",
    "layout_aware": "Layout-aware Chunking",
    "dom_aware": "DOM-aware Chunking",
    "parent_child": "Parent-Child Chunking",
    "semantic": "Semantic Chunking",
    "adaptive": "Adaptive Chunking",
    "clause_aware": "Clause-aware Chunking",
    "ocr_aware": "OCR-aware Chunking",
    "table_aware": "Table-aware Chunking",
    "code_function": "Code Function-aware Chunking",
    "multi_modal": "Multi-modal Chunking",
}


COMPATIBILITY: dict[str, list[str]] = {
    "whole_text": ["fixed_size", "recursive", "semantic", "adaptive"],
    "page_wise": ["fixed_size", "recursive", "semantic", "ocr_aware"],
    "block_wise": ["recursive", "layout_aware", "semantic", "adaptive"],
    "section_wise": ["section_aware", "header_aware", "parent_child", "semantic"],
    "hierarchical": ["header_aware", "section_aware", "parent_child"],
    "layout_aware": ["layout_aware", "table_aware", "multi_modal", "adaptive"],
    "table_aware": ["table_aware", "layout_aware", "parent_child"],
    "semantic_segmentation": ["semantic", "adaptive", "parent_child"],
    "clause_aware": ["clause_aware", "semantic", "recursive"],
    "code_aware": ["code_function", "recursive", "fixed_size"],
}


def compatible_chunkers(method_id: str) -> list[str]:
    return COMPATIBILITY.get(method_id, ["recursive"])
