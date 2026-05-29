from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from utils.models import Chunk, ExtractionResult


class BaseChunker(ABC):
    strategy_id: str
    name: str
    description: str
    default_config: dict[str, Any] = {}

    @abstractmethod
    def chunk(self, extraction: ExtractionResult, config: dict[str, Any] | None = None) -> list[Chunk]:
        raise NotImplementedError

    def cfg(self, config: dict[str, Any] | None) -> dict[str, Any]:
        return {**self.default_config, **(config or {})}

    def make_chunk(
        self,
        index: int,
        text: str,
        metadata: dict[str, Any] | None = None,
        parent_id: str | None = None,
        overlap: int = 0,
    ) -> Chunk:
        return Chunk(
            id=f"{self.strategy_id}-{index}",
            text=text.strip(),
            strategy_id=self.strategy_id,
            metadata=metadata or {},
            parent_id=parent_id,
            overlap_with_previous=overlap,
        )

