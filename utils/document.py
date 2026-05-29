from __future__ import annotations

from pathlib import Path

from utils.models import DocumentType, UploadedDocument


def detect_document_type(filename: str) -> DocumentType:
    suffix = Path(filename).suffix.lower()
    return {
        ".pdf": DocumentType.PDF,
        ".docx": DocumentType.DOCX,
        ".txt": DocumentType.TXT,
        ".md": DocumentType.MD,
        ".markdown": DocumentType.MD,
        ".html": DocumentType.HTML,
        ".htm": DocumentType.HTML,
        ".py": DocumentType.CODE,
        ".js": DocumentType.CODE,
        ".ts": DocumentType.CODE,
        ".tsx": DocumentType.CODE,
        ".jsx": DocumentType.CODE,
        ".java": DocumentType.CODE,
        ".go": DocumentType.CODE,
        ".rs": DocumentType.CODE,
        ".cpp": DocumentType.CODE,
        ".c": DocumentType.CODE,
        ".h": DocumentType.CODE,
        ".cs": DocumentType.CODE,
        ".php": DocumentType.CODE,
        ".rb": DocumentType.CODE,
    }.get(suffix, DocumentType.UNKNOWN)


def make_uploaded_document(name: str, content: bytes) -> UploadedDocument:
    suffix = Path(name).suffix.lower()
    return UploadedDocument(
        name=name,
        suffix=suffix,
        content=content,
        doc_type=detect_document_type(name),
        size_bytes=len(content),
    )


def human_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"
