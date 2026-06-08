from __future__ import annotations

import pandas as pd
import streamlit as st

from analytics.metrics import chunk_metrics
from assistant.rag_assistant import RAGAssistant
from chunking.registry import get_chunker
from extraction.registry import EXTRACTION_REGISTRY, get_extractor
from ui.components import show_chunk, show_original_document, show_retrieved, show_units
from utils.compatibility import CHUNKING_STRATEGIES, compatible_chunkers
from utils.document import human_size, make_uploaded_document


st.set_page_config(page_title="ChunkLab", page_icon="CL", layout="wide")

st.markdown(
    """
    <style>
    .stMain .block-container {
        padding-bottom: 2rem !important;
    }
    section[data-testid="stBottom"] {
        background: transparent !important;
        border: 0 !important;
        padding: 0 !important;
        min-height: 0 !important;
    }
    section[data-testid="stBottom"] > div {
        width: 100% !important;
        margin: 0 !important;
    }
    [data-testid="stChatInput"] {
        width: 100% !important;
        margin: 0 !important;
        background: rgb(14, 17, 23) !important;
        padding: 0.25rem !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 -0.5rem 1rem rgba(14, 17, 23, 0.75);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults = {
        "document": None,
        "extraction_result": None,
        "chunks": [],
        "assistant": RAGAssistant(),
        "chat_history": [],
        "last_strategy": None,
        "pending_question": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_downstream() -> None:
    st.session_state.extraction_result = None
    st.session_state.chunks = []
    st.session_state.chat_history = []
    st.session_state.pending_question = None
    st.session_state.assistant = RAGAssistant()


def strategy_settings(strategy_id: str) -> dict[str, object]:
    if strategy_id in {"fixed_size", "recursive", "ocr_aware"}:
        st.subheader("Strategy Settings")
        return {
            "chunk_size": st.slider("Chunk size", 200, 4000, 1200, 100),
            "overlap": st.slider("Overlap", 0, 800, 150, 25),
        }
    if strategy_id == "parent_child":
        st.subheader("Strategy Settings")
        return {
            "child_size": st.slider("Child chunk size", 200, 2000, 500, 50),
            "child_overlap": st.slider("Child overlap", 0, 400, 60, 20),
        }
    if strategy_id == "semantic":
        st.subheader("Strategy Settings")
        return {"sentences_per_chunk": st.slider("Sentences per chunk", 2, 12, 5, 1)}
    if strategy_id == "adaptive":
        st.subheader("Strategy Settings")
        return {"target_chars": st.slider("Target characters", 300, 2500, 900, 100)}
    return {}


init_state()

st.title("ChunkLab")
st.caption("Document intelligence, chunking visualization, and retrieval evaluation in one modular Streamlit platform.")

with st.sidebar:
    st.header("Pipeline Status")
    st.write("Upload -> Extract -> Chunk -> Index -> Retrieve")
    st.divider()

    if st.session_state.document:
        st.success(f"Document loaded: {st.session_state.document.name}")
    else:
        st.info("No document uploaded.")

    if st.session_state.extraction_result:
        st.info(f"Extraction: {st.session_state.extraction_result.method_name}")
    else:
        st.warning("Extraction not run.")

    if st.session_state.chunks:
        st.metric("Indexed Chunks", len(st.session_state.chunks))
    else:
        st.warning("No chunks generated.")

document_tab, assistant_tab = st.tabs(["Document Lab", "Retrieval Assistant"])

with document_tab:
    st.header("Document Processing Pipeline")

    # Step 1
    st.subheader("Step 1 — Upload")
    uploaded_file = st.file_uploader(
        "Choose a document",
        type=[
            "pdf", "docx", "txt", "md", "markdown", "html",
            "py"
        ],
        label_visibility="collapsed",
    )
    if uploaded_file:
        content = uploaded_file.getvalue()
        current_name = st.session_state.document.name if st.session_state.document else None
        if uploaded_file.name != current_name:
            st.session_state.document = make_uploaded_document(uploaded_file.name, content)
            reset_downstream()

    document = st.session_state.document
    if document:
        cols = st.columns(4)
        cols[0].metric("File", document.name)
        cols[1].metric("Type", document.doc_type.value)
        cols[2].metric("Size", human_size(document.size_bytes))
        cols[3].metric("Suffix", document.suffix or "none")
        show_original_document(document)
    else:
        st.info("Upload a PDF, DOCX, TXT, Markdown, HTML, or code file to begin.")

    st.divider()

    # Step 2
    st.subheader("Step 2 — Extraction")
    if not st.session_state.document:
        st.warning("Upload a document first.")
    else:
        method_labels = {extractor.name: method_id for method_id, extractor in EXTRACTION_REGISTRY.items()}
        selected_label = st.radio("Extraction method", list(method_labels.keys()), horizontal=True)
        method_id = method_labels[selected_label]
        extractor = get_extractor(method_id)
        st.caption(extractor.description)

        compatible = compatible_chunkers(method_id)
        st.info("Compatible chunkers: " + ", ".join(CHUNKING_STRATEGIES[item] for item in compatible))

        if st.button("Run Extraction", type="primary"):
            with st.spinner("Extracting document structure..."):
                st.session_state.extraction_result = extractor.extract(st.session_state.document)
                st.session_state.chunks = []
                st.session_state.chat_history = []
                st.session_state.pending_question = None
                st.session_state.assistant = RAGAssistant()

        result = st.session_state.extraction_result
        if result:
            cols = st.columns(3)
            cols[0].metric("Characters", len(result.text))
            cols[1].metric("Units", result.metadata.get("unit_count", len(result.units)))
            cols[2].metric("Method", result.method_name)

            st.subheader("Document Units")
            show_units(result.units)

    st.divider()

    # Step 3
    st.subheader("Step 3 — Chunking")
    result = st.session_state.extraction_result
    if not result:
        st.warning("Run extraction first.")
    else:
        compatible = compatible_chunkers(result.method_id)
        options = {CHUNKING_STRATEGIES[item]: item for item in compatible}
        selected = st.radio("Compatible chunking strategy", list(options.keys()), horizontal=True)
        strategy_id = options[selected]
        chunker = get_chunker(strategy_id)
        st.caption(chunker.description)

        settings = strategy_settings(strategy_id)
        if st.button("Generate Chunks", type="primary"):
            with st.spinner("Generating explainable chunks..."):
                chunks = chunker.chunk(result, settings)
                st.session_state.chunks = chunks
                st.session_state.last_strategy = strategy_id
                st.session_state.assistant = RAGAssistant()
                st.session_state.assistant.build_index(chunks)
                st.session_state.chat_history = []
                st.session_state.pending_question = None
            st.success(f"Generated and indexed {len(st.session_state.chunks)} chunks.")

        if st.session_state.chunks:
            metrics = chunk_metrics(st.session_state.chunks)
            cols = st.columns(4)
            cols[0].metric("Chunks", metrics["chunk_count"])
            cols[1].metric("Avg chars", metrics["avg_chars"])
            cols[2].metric("Parents", metrics["parent_chunks"])
            cols[3].metric("Children", metrics["child_chunks"])

    st.divider()

    # Step 4
    st.subheader("Step 4 — Chunk Preview")
    chunks = st.session_state.chunks
    if not chunks:
        st.warning("Generate chunks first.")
    else:
        parent_child_rows = [
            {"parent": chunk.id, "children": ", ".join(chunk.children)}
            for chunk in chunks if chunk.children
        ]
        if parent_child_rows:
            st.subheader("Parent-Child Relationships")
            st.dataframe(pd.DataFrame(parent_child_rows), use_container_width=True, hide_index=True)

        st.subheader("All Chunks")
        for index, chunk in enumerate(chunks, start=1):
            show_chunk(chunk, index, len(chunks))

with assistant_tab:
    st.header("Retrieval Assistant")

    if not st.session_state.chunks:
        st.warning("Generate and index chunks in the Document Lab before using the assistant.")
    else:
        with st.expander("Retrieval settings", expanded=False):
            top_k = st.slider(
                "Retrieved chunks",
                1,
                min(10, len(st.session_state.chunks)),
                min(5, len(st.session_state.chunks)),
            )
            answer_mode_label = st.radio(
                "Answer mode",
                ["LLM from retrieved chunks", "Extractive, no LLM"],
                horizontal=True,
            )
        answer_mode = "llm" if answer_mode_label == "LLM from retrieved chunks" else "extractive"

        message_area = st.container(height=470, border=False)
        with message_area:
            if not st.session_state.chat_history and not st.session_state.pending_question:
                st.caption("Ask a question below to evaluate retrieval quality.")
            for turn in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(turn.question)
                with st.chat_message("assistant"):
                    st.markdown(turn.answer)
                    with st.expander(
                        f"Retrieved chunks ({len(turn.retrieved)}) — {turn.latency_ms:.1f} ms",
                        expanded=False,
                    ):
                        show_retrieved(turn.retrieved)

            if st.session_state.pending_question:
                pending_question = st.session_state.pending_question
                with st.chat_message("user"):
                    st.write(pending_question)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        turn = st.session_state.assistant.answer(
                            pending_question,
                            top_k=top_k,
                            answer_mode=answer_mode,
                        )
                    st.markdown(turn.answer)
                    with st.expander(
                        f"Retrieved chunks ({len(turn.retrieved)}) — {turn.latency_ms:.1f} ms",
                        expanded=False,
                    ):
                        show_retrieved(turn.retrieved)
                st.session_state.chat_history.append(turn)
                st.session_state.pending_question = None

        question = st.chat_input("Ask a question to evaluate retrieval quality")
        if question:
            st.session_state.pending_question = question
            st.rerun()
