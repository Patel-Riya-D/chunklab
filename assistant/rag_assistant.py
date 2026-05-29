from __future__ import annotations

import re
from dataclasses import dataclass, field

from assistant.azure_llm import AzureAnswerGenerator
from retrieval.retriever import Retriever
from utils.models import ChatTurn, Chunk
from utils.text import normalize_whitespace, split_sentences
from vectorstore.memory import tokenize


@dataclass
class RAGAssistant:
    retriever: Retriever = field(default_factory=Retriever)

    def build_index(self, chunks: list[Chunk]) -> None:
        self.retriever.index(chunks)

    def answer(self, question: str, top_k: int = 5) -> ChatTurn:
        response = self.retriever.retrieve(question, top_k=top_k)
        if not response.results:
            answer = "I could not retrieve any chunks yet. Generate chunks and build the retrieval index first."
        else:
            best = response.results[0]
            direct_answer = self._answer_with_llm(question, response.results[:top_k])
            if not direct_answer:
                direct_answer = self._direct_answer(question, best.chunk)
            context_lines = []
            for result in response.results[:top_k]:
                snippet = result.chunk.text.strip().replace("\n", " ")
                context_lines.append(f"[{result.chunk.id}, score {result.score:.3f}] {snippet[:450]}")
            answer = (
                f"{direct_answer}\n\n"
                f"Retrieval details: {best.chunk.id} "
                f"({best.chunk.metadata.get('title', 'untitled')}, score {best.score:.3f}).\n\n"
                "Retrieved context:\n\n"
                + "\n\n".join(context_lines)
            )
        return ChatTurn(question=question, answer=answer, retrieved=response.results, latency_ms=response.latency_ms)

    def _answer_with_llm(self, question: str, retrieved) -> str:
        try:
            return AzureAnswerGenerator().generate(question, retrieved).strip()
        except Exception:
            return ""

    def _direct_answer(self, question: str, chunk: Chunk) -> str:
        question_terms = set(tokenize(question))
        text = normalize_whitespace(chunk.text)
        title = str(chunk.metadata.get("title", "")).strip()

        directive_match = re.search(r"(Strategic Mandate Directive:\s*.+?)(?:\s+Section\s+\d+:|$)", text)
        if directive_match and {"strategic", "mandate", "directive"} & question_terms:
            directive = normalize_whitespace(directive_match.group(1))
            return f"Answer: {directive}"

        sentences = split_sentences(text)
        scored_sentences = []
        for sentence in sentences:
            terms = set(tokenize(sentence))
            overlap = len(question_terms & terms)
            if overlap:
                scored_sentences.append((overlap, len(sentence), sentence))

        scored_sentences.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        selected = [sentence for _, _, sentence in scored_sentences[:3]]
        if not selected:
            selected = sentences[:3]

        answer_text = " ".join(selected).strip()
        if title:
            return f"Answer: {title}: {answer_text}"
        return f"Answer: {answer_text}"
