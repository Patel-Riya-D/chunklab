from __future__ import annotations

from chunking.core import FixedSizeChunker, RecursiveChunker
from chunking.intelligent import AdaptiveChunker, ClauseAwareChunker, SemanticChunker
from chunking.specialized import CodeFunctionChunker, MultiModalChunker, OCRAwareChunker, TableAwareChunker
from chunking.structure import DOMAwareChunker, HeaderAwareChunker, LayoutAwareChunker, ParentChildChunker, SectionAwareChunker


CHUNKER_REGISTRY = {
    chunker.strategy_id: chunker
    for chunker in [
        FixedSizeChunker(),
        RecursiveChunker(),
        HeaderAwareChunker(),
        SectionAwareChunker(),
        LayoutAwareChunker(),
        DOMAwareChunker(),
        ParentChildChunker(),
        SemanticChunker(),
        AdaptiveChunker(),
        ClauseAwareChunker(),
        OCRAwareChunker(),
        TableAwareChunker(),
        CodeFunctionChunker(),
        MultiModalChunker(),
    ]
}


def get_chunker(strategy_id: str):
    return CHUNKER_REGISTRY[strategy_id]

