# ChunkLab

ChunkLab is a modular Streamlit platform for document extraction, chunking visualization, and retrieval evaluation. It is designed for experimenting with how extraction and chunking choices affect RAG retrieval quality.

## Features

- Upload PDF, DOCX, TXT, Markdown, and HTML files.
- Select extraction methods including whole text, page-wise, block-wise, section-wise, hierarchy-aware, layout-aware, table-aware, OCR placeholders, semantic segmentation, clause-aware, code-aware, and IDP placeholder extraction.
- Dynamically show compatible chunking strategies for the selected extraction method.
- Generate fixed, recursive, header-aware, section-aware, layout-aware, DOM-aware, parent-child, semantic, adaptive, clause-aware, OCR-aware, table-aware, code function-aware, and multi-modal-ready chunks.
- Preview chunk IDs, metadata, hierarchy, parent-child links, overlaps, source sections, pages, and estimated token sizes.
- Ask retrieval questions in a dedicated assistant tab and inspect retrieved chunks, scores, sources, and latency.
- View chunk and retrieval analytics.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The default retrieval path uses a FAISS-backed vector store when FAISS is available and falls back to in-memory search otherwise. The default embedder is a deterministic local hashing embedder so the app starts quickly and runs offline. `SentenceTransformerEmbedder` is included in `vectorstore/embeddings.py` for teams that want to switch to `sentence-transformers` models.

## Architecture

```text
app.py
extraction/    reusable document extraction classes
chunking/      reusable chunking strategy classes
vectorstore/   embedding and vector search abstraction
retrieval/     retriever orchestration
assistant/     retrieval assistant and conversational turn handling
analytics/     chunk and retrieval metrics
ui/            Streamlit rendering helpers
utils/         shared models, compatibility maps, text utilities
data/          reserved for local files
assets/        reserved for visual assets
```

Future support for scanned PDFs, images, code repositories, FAISS, ChromaDB persistence, and LLM-backed answer generation can be added behind the existing module interfaces.
