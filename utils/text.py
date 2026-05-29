from __future__ import annotations

import re
from html.parser import HTMLParser


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def estimate_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


class _HTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if stripped:
            self.parts.append(stripped)


def html_to_text(html: str) -> str:
    parser = _HTMLTextParser()
    parser.feed(html)
    return "\n".join(parser.parts)


def markdown_headers(text: str) -> list[tuple[int, str, int]]:
    headers: list[tuple[int, str, int]] = []
    for idx, line in enumerate(text.splitlines()):
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if match:
            headers.append((len(match.group(1)), match.group(2).strip(), idx))
    return headers

