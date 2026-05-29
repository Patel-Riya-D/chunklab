from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.models import Chunk, ExtractedUnit, RetrievalResult
from utils.text import estimate_tokens


def _table_dataframe(rows: list[list[str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    header, body = rows[0], rows[1:]
    column_names = [name or f"Column {index}" for index, name in enumerate(header, start=1)]
    return pd.DataFrame(body, columns=column_names)


def _show_table_rows(rows: object) -> bool:
    if not isinstance(rows, list) or not rows:
        return False
    if not all(isinstance(row, list) for row in rows):
        return False
    st.table(_table_dataframe(rows))
    return True


def _show_text(unit_type: str, text: str) -> None:
    if unit_type in {"code_symbol", "code_context"}:
        st.code(text, language="python")
    else:
        st.write(text)


def show_units(units: list[ExtractedUnit], depth: int = 0) -> None:
    for unit in units:
        label = f"{unit.id} | {unit.unit_type}"
        if unit.metadata.get("title"):
            label += f" | {unit.metadata['title']}"
        with st.container(border=True):
            st.markdown(f"**{label}**")
            if unit.metadata:
                visible_metadata = {key: value for key, value in unit.metadata.items() if key != "rows"}
                st.caption(", ".join(f"{key}: {value}" for key, value in visible_metadata.items()))
            if not _show_table_rows(unit.metadata.get("rows")):
                _show_text(unit.unit_type, unit.text)
            if unit.children:
                show_units(unit.children, depth + 1)


def chunk_table(chunks: list[Chunk]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": chunk.id,
                "role": chunk.metadata.get("chunk_role", chunk.metadata.get("role", "")),
                "parent": chunk.parent_id or "",
                "children": len(chunk.children),
                "chars": len(chunk.text),
                "tokens_est": estimate_tokens(chunk.text),
                "overlap": chunk.overlap_with_previous,
                "source": chunk.metadata.get("source_unit", chunk.metadata.get("title", "")),
            }
            for chunk in chunks
        ]
    )


def show_chunk(chunk: Chunk, index: int | None = None, total: int | None = None) -> None:
    with st.container(border=True):
        role = str(chunk.metadata.get("chunk_role", chunk.metadata.get("role", ""))).lower()
        role_label = "Parent" if role == "parent" else "Child" if role == "child" else "Chunk"
        prefix = f"Chunk {index} of {total} - " if index is not None and total is not None else ""
        label = f"{prefix}{role_label}: {chunk.id} | {len(chunk.text)} chars"
        if chunk.parent_id:
            label += f" | child of {chunk.parent_id}"
        if chunk.children:
            label += f" | {len(chunk.children)} children"
        st.markdown(f"**{label}**")
        cols = st.columns(5)
        cols[0].metric("Role", role_label)
        cols[1].metric("Characters", len(chunk.text))
        cols[2].metric("Estimated tokens", estimate_tokens(chunk.text))
        cols[3].metric("Overlap", chunk.overlap_with_previous)
        cols[4].metric("Children", len(chunk.children))
        meta_parts = []
        if chunk.parent_id:
            meta_parts.append(f"parent: {chunk.parent_id}")
        if chunk.metadata.get("parent_section_title"):
            meta_parts.append(f"parent section: {chunk.metadata['parent_section_title']}")
        if chunk.metadata.get("section_path"):
            meta_parts.append(f"path: {chunk.metadata['section_path']}")
        if chunk.metadata.get("title"):
            meta_parts.append(f"section: {chunk.metadata['title']}")
        if chunk.metadata.get("page"):
            meta_parts.append(f"page: {chunk.metadata['page']}")
        if chunk.metadata.get("source_unit"):
            meta_parts.append(f"source: {chunk.metadata['source_unit']}")
        if meta_parts:
            st.caption(" | ".join(meta_parts))
        if not _show_table_rows(chunk.metadata.get("rows")):
            _show_text("code_symbol" if chunk.strategy_id == "code_function" else "", chunk.text)


def show_retrieved(results: list[RetrievalResult]) -> None:
    for result in results:
        with st.container(border=True):
            st.markdown(f"**Rank {result.rank}: {result.chunk.id} | score {result.score:.3f}**")
            if result.chunk.metadata:
                visible_metadata = {key: value for key, value in result.chunk.metadata.items() if key != "rows"}
                st.caption(", ".join(f"{key}: {value}" for key, value in visible_metadata.items()))
            if not _show_table_rows(result.chunk.metadata.get("rows")):
                _show_text("code_symbol" if result.chunk.strategy_id == "code_function" else "", result.chunk.text)
