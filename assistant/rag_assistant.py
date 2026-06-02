from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from assistant.azure_llm import AzureAnswerGenerator
from retrieval.retriever import Retriever
from utils.models import ChatTurn, Chunk, RetrievalResult
from utils.text import normalize_whitespace, split_sentences
from vectorstore.memory import cosine_similarity, tokenize


@dataclass
class RAGAssistant:
    retriever: Retriever = field(default_factory=Retriever)
    term_document_frequency: Counter[str] = field(default_factory=Counter)
    indexed_chunk_count: int = 0
    chunks_by_id: dict[str, Chunk] = field(default_factory=dict)

    def build_index(self, chunks: list[Chunk]) -> None:
        self.retriever.index(chunks)
        self.chunks_by_id = {chunk.id: chunk for chunk in chunks}
        self.indexed_chunk_count = len([chunk for chunk in chunks if chunk.text.strip()])
        self.term_document_frequency = Counter()
        for chunk in chunks:
            terms = self._expanded_terms(self._chunk_support_text(chunk))
            self.term_document_frequency.update(terms)

    def answer(self, question: str, top_k: int = 5, answer_mode: str = "extractive") -> ChatTurn:
        response = self.retriever.retrieve(question, top_k=top_k)
        self._annotate_parent_child_results(response.results)
        if not response.results:
            answer = "I could not retrieve any chunks yet. Generate chunks and build the retrieval index first."
        elif answer_mode == "llm":
            answer = self._llm_answer(question, response.results[:top_k])
        elif self._is_acronym_question(question):
            for result in response.results[:top_k]:
                answer_chunk = self._answer_context_chunk(result.chunk)
                acronym_answer = self._answer_acronym_question(question, answer_chunk)
                if acronym_answer:
                    answer = f"Answer: {acronym_answer}\n\nSource: {self._source_label(answer_chunk)}"
                    break
            else:
                answer = (
                    "Answer: I could not find an explicit full-form expansion in the retrieved chunks.\n\n"
                    f"Closest source: {self._source_label(self._answer_context_chunk(response.results[0].chunk))}"
                )
        else:
            best = self._best_answer_result(question, response.results[:top_k])
            answer_chunk = self._answer_context_chunk(best.chunk)
            if not self._has_answer_support(question, answer_chunk):
                answer = (
                    "Answer: I could not find this answer in the retrieved chunks. "
                    "Try a different chunking/extraction method or ask with terms that appear in the document.\n\n"
                    f"Closest source: {self._source_label(answer_chunk)}"
                )
            else:
                direct_answer = self._direct_answer(question, answer_chunk)
                answer = f"{direct_answer}\n\nSource: {self._source_label(answer_chunk)}"
        return ChatTurn(question=question, answer=answer, retrieved=response.results, latency_ms=response.latency_ms)

    def _llm_answer(self, question: str, retrieved: list[RetrievalResult]) -> str:
        try:
            generated = AzureAnswerGenerator().generate(question, retrieved).strip()
        except Exception as exc:
            best = self._answer_context_chunk(retrieved[0].chunk)
            fallback = self._direct_answer(question, best)
            return (
                f"{fallback}\n\n"
                f"Source: {self._source_label(best)}\n\n"
                f"LLM answer mode could not run: {exc}"
            )
        return generated or "Answer: The retrieved context did not contain enough information."

    def _best_answer_result(self, question: str, retrieved: list[RetrievalResult]) -> RetrievalResult:
        for result in retrieved:
            if self._has_answer_support(question, self._answer_context_chunk(result.chunk)):
                return result
        return retrieved[0]

    def _source_label(self, chunk: Chunk) -> str:
        title = str(chunk.metadata.get("title") or chunk.metadata.get("section_path") or chunk.metadata.get("source_unit") or chunk.id)
        page = chunk.metadata.get("page")
        return f"{title}, page {page}" if page else title

    def _chunk_role(self, chunk: Chunk) -> str:
        role = str(chunk.metadata.get("chunk_role", chunk.metadata.get("role", ""))).lower()
        if role:
            return role
        if chunk.parent_id:
            return "child"
        if chunk.children:
            return "parent"
        return "chunk"

    def _answer_context_chunk(self, chunk: Chunk) -> Chunk:
        if self._chunk_role(chunk) == "child" and chunk.parent_id:
            return self.chunks_by_id.get(chunk.parent_id, chunk)
        return chunk

    def _annotate_parent_child_results(self, results: list[RetrievalResult]) -> None:
        result_by_chunk_id = {result.chunk.id: result for result in results}
        for result in results:
            chunk = result.chunk
            role = self._chunk_role(chunk)
            result.score_details["retrieved_role"] = role
            if role == "child" and chunk.parent_id:
                parent = self.chunks_by_id.get(chunk.parent_id)
                result.score_details["matched_child"] = chunk.id
                result.score_details["expanded_parent"] = chunk.parent_id
                if parent:
                    result.score_details["expanded_parent_source"] = self._source_label(parent)
                    result.score_details["expanded_parent_children"] = str(len(parent.children))
                    result.score_details["expanded_parent_text"] = parent.text
                    parent_result = result_by_chunk_id.get(parent.id)
                    if parent_result:
                        result.score_details["expanded_parent_rank"] = str(parent_result.rank)
                        result.score_details["expanded_parent_score"] = f"{parent_result.score:.3f}"
                        parent_score_details = {
                            key: value
                            for key, value in parent_result.score_details.items()
                            if key not in {"retrieved_role", "matched_parent", "parent_children"}
                        }
                        if parent_score_details:
                            result.score_details["expanded_parent_score_details"] = ", ".join(
                                f"{key}: {value}" for key, value in parent_score_details.items()
                            )
            elif role == "parent":
                result.score_details["matched_parent"] = chunk.id
                result.score_details["parent_children"] = str(len(chunk.children))

    def _direct_answer(self, question: str, chunk: Chunk) -> str:
        question_terms = self._anchor_terms(self._important_terms(question))
        title = str(chunk.metadata.get("title", "")).strip()
        answerable_text = self._answerable_text(chunk)
        text = normalize_whitespace(answerable_text)

        directive_match = re.search(r"(Strategic Mandate Directive:\s*.+?)(?:\s+Section\s+\d+:|$)", text)
        if directive_match and {"strategic", "mandate", "directive"} & question_terms:
            directive = normalize_whitespace(directive_match.group(1))
            return f"Answer: {directive}"

        table_answer = self._table_answer(question, chunk)
        if table_answer:
            return table_answer

        table_items = self._matching_table_items(question, chunk)
        if table_items:
            return "Answer: " + "; ".join(table_items)

        list_items = self._matching_list_items(question, chunk)
        if list_items:
            return "Answer: " + "; ".join(list_items)

        sentences = self._answer_units(answerable_text)
        selected = self._best_answer_sentences(question, question_terms, sentences)
        if len(selected) == 1 and self._looks_like_heading_only(selected[0]):
            next_index = sentences.index(selected[0]) + 1
            if next_index < len(sentences) and not self._looks_like_heading_only(sentences[next_index]):
                selected.append(sentences[next_index])
        if not selected:
            selected = sentences[:3]

        answer_text = " ".join(selected).strip()
        if title:
            return f"Answer: {title}: {answer_text}"
        return f"Answer: {answer_text}"

    def _matching_list_items(self, question: str, chunk: Chunk) -> list[str]:
        question_terms = self._important_terms(question)
        title_terms = self._expanded_terms(str(chunk.metadata.get("title", "")))
        asks_for_list = bool({"list", "show", "name", "names"} & question_terms)
        title_matches = bool(question_terms & title_terms)
        if not asks_for_list and not title_matches:
            return []

        items = []
        for line in chunk.text.splitlines():
            stripped = self._clean_candidate_line(line)
            if not stripped or self._expanded_terms(stripped) == title_terms:
                continue
            if self._looks_like_heading_only(stripped):
                continue
            if len(stripped.split()) <= 18:
                items.append(stripped)
        return items[:8]

    def _best_answer_sentences(self, question: str, question_terms: set[str], sentences: list[str]) -> list[str]:
        if not sentences:
            return []

        semantic_scores = self._semantic_sentence_scores(question, sentences)
        scored_sentences = []
        for index, sentence in enumerate(sentences):
            terms = self._expanded_terms(sentence)
            overlap = len(question_terms & terms)
            coverage = overlap / max(1, len(question_terms))
            density = overlap / max(1, len(terms))
            semantic = semantic_scores[index] if index < len(semantic_scores) else 0.0
            score = (0.70 * semantic) + (0.20 * coverage) + (0.10 * density)
            scored_sentences.append((score, semantic, coverage, -len(sentence), index, sentence))

        scored_sentences.sort(key=lambda item: item[:4], reverse=True)
        best = scored_sentences[0]
        selected = [best]
        for candidate in scored_sentences[1:]:
            is_neighbor = abs(candidate[4] - best[4]) == 1
            is_close = candidate[0] >= best[0] * 0.92
            if is_neighbor and is_close:
                selected.append(candidate)
            if len(selected) >= 2:
                break
        return [item[5] for item in sorted(selected, key=lambda item: item[4])]

    def _semantic_sentence_scores(self, question: str, sentences: list[str]) -> list[float]:
        try:
            embedder = self.retriever.vector_store.embedder
            vectors = embedder.embed([question, *sentences])
        except Exception:
            return [0.0 for _ in sentences]
        if len(vectors) < len(sentences) + 1:
            return [0.0 for _ in sentences]
        query_vector = vectors[0]
        return [cosine_similarity(query_vector, sentence_vector) for sentence_vector in vectors[1:]]

    def _has_answer_support(self, question: str, chunk: Chunk) -> bool:
        question_terms = self._important_terms(question)
        if not question_terms:
            return True
        chunk_terms = self._expanded_terms(self._chunk_support_text(chunk))
        anchor_terms = self._anchor_terms(question_terms)
        if self.indexed_chunk_count and any(
            self.term_document_frequency.get(term, 0) == 0 and not self._has_present_variant(term)
            for term in anchor_terms
        ):
            return False
        overlap = anchor_terms & chunk_terms
        if not overlap:
            return False
        if len(anchor_terms) == 1:
            return True
        return len(overlap) / len(anchor_terms) >= 0.5

    def _important_terms(self, text: str) -> set[str]:
        return {term for term in self._expanded_terms(text) if len(term) > 1 and not term.isdigit()}

    def _table_answer(self, question: str, chunk: Chunk) -> str:
        rows = chunk.metadata.get("rows")
        if not isinstance(rows, list) or len(rows) < 2 or not all(isinstance(row, list) for row in rows):
            return ""

        question_terms = self._important_terms(question)
        title = str(chunk.metadata.get("title", "")).strip()
        title_terms = self._expanded_terms(title)
        table_terms = self._expanded_terms(" ".join(str(cell) for row in rows for cell in row))
        if question_terms and not (question_terms & (title_terms | table_terms)):
            return ""

        table = self._markdown_table(rows)
        if not table:
            return ""
        heading = f"Answer: {title}" if title else "Answer"
        return f"{heading}:\n\n{table}"

    def _markdown_table(self, rows: list[list[object]]) -> str:
        normalized_rows = [[normalize_whitespace(str(cell)) for cell in row] for row in rows]
        column_count = max((len(row) for row in normalized_rows), default=0)
        if column_count < 2 or len(normalized_rows) < 2:
            return ""
        padded_rows = [row + [""] * (column_count - len(row)) for row in normalized_rows]
        header = [cell or f"Column {index}" for index, cell in enumerate(padded_rows[0], start=1)]
        body = padded_rows[1:]
        return "\n".join(
            [
                "| " + " | ".join(self._markdown_cell(cell) for cell in header) + " |",
                "| " + " | ".join("---" for _ in header) + " |",
                *[
                    "| " + " | ".join(self._markdown_cell(cell) for cell in row) + " |"
                    for row in body
                    if any(cell for cell in row)
                ],
            ]
        )

    def _markdown_cell(self, text: str) -> str:
        return text.replace("\\", "\\\\").replace("|", "\\|").strip()

    def _matching_table_items(self, question: str, chunk: Chunk) -> list[str]:
        rows = chunk.metadata.get("rows")
        if not isinstance(rows, list) or len(rows) < 2 or not all(isinstance(row, list) for row in rows):
            return []

        question_terms = self._important_terms(question)
        table_terms = self._expanded_terms(" ".join(str(cell) for row in rows for cell in row))
        if not (question_terms & table_terms):
            return []

        headers = [str(cell).strip() or f"Column {index}" for index, cell in enumerate(rows[0], start=1)]
        items = []
        for row in rows[1:]:
            cells = [str(cell).strip() for cell in row]
            if not any(cells):
                continue
            first = cells[0]
            pairs = [
                f"{header}: {cell}"
                for header, cell in zip(headers[1:], cells[1:])
                if cell
            ]
            item = f"{first} - " + ", ".join(pairs) if first and pairs else " - ".join(cell for cell in cells if cell)
            if item:
                items.append(item)
        return items[:8]

    def _anchor_terms(self, terms: set[str]) -> set[str]:
        if not self.indexed_chunk_count:
            return terms
        rare_terms = {
            term
            for term in terms
            if self.term_document_frequency.get(term, 0) > 0
            and self.term_document_frequency.get(term, 0) / max(1, self.indexed_chunk_count) <= 0.35
        }
        present_terms = {term for term in terms if self.term_document_frequency.get(term, 0) > 0}
        return rare_terms or present_terms or terms

    def _has_present_variant(self, term: str) -> bool:
        variants = {term}
        if term.endswith("s") and len(term) > 3:
            variants.add(term[:-1])
        else:
            variants.add(f"{term}s")
        return any(self.term_document_frequency.get(variant, 0) > 0 for variant in variants)

    def _clean_candidate_line(self, line: str) -> str:
        stripped = line.strip(" -*•\t")
        stripped = re.sub(r"^section\s+path:\s*.*?\s+page:\s*\d+\s*", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"^page:\s*\d+\s*", "", stripped, flags=re.IGNORECASE).strip()
        if re.match(r"^(section path|source|chunk id|score):", stripped, re.IGNORECASE):
            return ""
        stripped = stripped.replace("↓", " -> ")
        stripped = re.sub(r"\s*;\s*->\s*;?\s*", " -> ", stripped)
        stripped = re.sub(r"\s*->\s*$", "", stripped)
        stripped = re.sub(r"\s+", " ", stripped).strip()
        return stripped

    def _looks_like_heading_only(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True
        if stripped.endswith(":") and len(stripped.split()) <= 6:
            return True
        words = re.findall(r"[A-Za-z]+", stripped)
        if not words or len(words) > 6:
            return False
        if all(word.isupper() for word in words):
            return True
        title_words = sum(1 for word in words if word[:1].isupper())
        return title_words / len(words) >= 0.65 and not re.search(r"[.!?;]", stripped)

    def _answer_units(self, text: str) -> list[str]:
        units = []
        protected = (
            text.replace("Mr.", "Mr<dot>")
            .replace("Ms.", "Ms<dot>")
            .replace("Mrs.", "Mrs<dot>")
            .replace("Dr.", "Dr<dot>")
        )
        for line in protected.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"\s*•\s*", "\n", line)
            for part in line.splitlines():
                part = part.strip(" -*\t")
                if not part or self._looks_like_heading_only(part):
                    continue
                for sentence in split_sentences(part):
                    restored = sentence.replace("<dot>", ".").strip()
                    if restored:
                        units.append(restored)
        return units

    def _answer_acronym_question(self, question: str, chunk: Chunk) -> str:
        if not self._is_acronym_question(question):
            return ""
        acronyms = re.findall(r"\b[A-Z]{2,}\b", question)
        if not acronyms:
            return ""
        text = self._chunk_support_text(chunk)
        for acronym in acronyms:
            before = re.search(rf"([A-Z][A-Za-z&.,'’/ -]{{3,120}}?)\s*\(\s*{re.escape(acronym)}\s*\)", text)
            if before:
                return f"{acronym}: {normalize_whitespace(before.group(1))}"
            after = re.search(rf"\b{re.escape(acronym)}\b\s*(?:means|stands for|[:=-])\s*([A-Z][A-Za-z&.,'’/ -]{{3,120}})", text, re.IGNORECASE)
            if after:
                return f"{acronym}: {normalize_whitespace(after.group(1))}"
        return ""

    def _is_acronym_question(self, question: str) -> bool:
        return bool(re.search(r"\b(full\s*form|stands?\s+for|meaning\s+of)\b", question, re.IGNORECASE))

    def _content_without_repeated_title(self, text: str, title: str) -> str:
        if not title:
            return text
        lines = text.splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        if lines and lines[0].strip().casefold() == title.casefold():
            return "\n".join(lines[1:]).strip()
        return text

    def _answerable_text(self, chunk: Chunk) -> str:
        title = str(chunk.metadata.get("title", "")).strip()
        text = self._content_without_repeated_title(chunk.text, title)
        kept_lines = []
        for line in text.splitlines():
            cleaned = self._clean_candidate_line(line)
            if cleaned:
                kept_lines.append(cleaned)
        return "\n".join(kept_lines).strip()

    def _chunk_support_text(self, chunk: Chunk) -> str:
        parts = [str(chunk.metadata.get("title", "")), self._answerable_text(chunk)]
        rows = chunk.metadata.get("rows")
        if isinstance(rows, list):
            parts.extend(" ".join(str(cell) for cell in row) for row in rows if isinstance(row, list))
        return "\n".join(part for part in parts if part)

    def _expanded_terms(self, text: str) -> set[str]:
        terms = set(tokenize(text))
        expanded = set(terms)
        for term in terms:
            if term.endswith("s") and len(term) > 3:
                expanded.add(term[:-1])
            if "-" in term:
                expanded.update(part for part in term.split("-") if part)
        return expanded
