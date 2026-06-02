from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DocumentType(str, Enum):
    PDF = "PDF"
    DOCX = "DOCX"
    TXT = "TXT"
    MD = "Markdown"
    HTML = "HTML"
    CODE = "Code"
    UNKNOWN = "Unknown"


@dataclass
class UploadedDocument:
    name: str
    suffix: str
    content: bytes
    doc_type: DocumentType
    size_bytes: int


@dataclass
class ExtractedUnit:
    id: str
    text: str
    unit_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    children: list["ExtractedUnit"] = field(default_factory=list)


@dataclass
class ExtractionResult:
    method_id: str
    method_name: str
    text: str
    units: list[ExtractedUnit]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    text: str
    strategy_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    overlap_with_previous: int = 0


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    rank: int
    score_details: dict[str, float | str] = field(default_factory=dict)


@dataclass
class ChatTurn:
    question: str
    answer: str
    retrieved: list[RetrievalResult]
    latency_ms: float
