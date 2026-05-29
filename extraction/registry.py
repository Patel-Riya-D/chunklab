from __future__ import annotations

from extraction.basic import BlockWiseExtractor, PageWiseExtractor, WholeTextExtractor
from extraction.intelligent import (
    ClauseAwareExtractor,
    CodeAwareExtractor,
    SemanticSegmentationExtractor,
)
from extraction.structured import HierarchicalExtractor, LayoutAwareExtractor, SectionWiseExtractor, TableAwareExtractor


EXTRACTION_REGISTRY = {
    extractor.method_id: extractor
    for extractor in [
        WholeTextExtractor(),
        PageWiseExtractor(),
        BlockWiseExtractor(),
        SectionWiseExtractor(),
        HierarchicalExtractor(),
        LayoutAwareExtractor(),
        TableAwareExtractor(),
        SemanticSegmentationExtractor(),
        ClauseAwareExtractor(),
        CodeAwareExtractor(),
    ]
}


def get_extractor(method_id: str):
    return EXTRACTION_REGISTRY[method_id]
