# ChunkLab

ChunkLab is a Streamlit-based document intelligence lab for testing extraction, chunking, indexing, and retrieval workflows. It helps compare how different document extraction methods and chunking strategies affect retrieval quality in RAG-style systems.

## What It Does

- Upload documents and code files.
- Extract text as whole documents, pages, blocks, sections, layout regions, tables, clauses, semantic units, or code-aware units.
- Show only chunking strategies that are compatible with the selected extraction method.
- Generate explainable chunks with IDs, metadata, hierarchy, parent-child links, pages, source sections, overlaps, and estimated token sizes.
- Build a retrieval index from generated chunks.
- Ask questions in the retrieval assistant and inspect retrieved chunks, scores, sources, and latency.
- Use extractive answers by default, with optional Azure OpenAI answer generation from retrieved chunks.

## Supported File Types

ChunkLab accepts:

- PDF
- DOCX
- TXT
- Markdown
- HTML
- Python, JavaScript, TypeScript, JSX, TSX
- Java, Go, Rust
- C, C++, C#
- PHP, Ruby

## Pipeline

The app follows a simple workflow:

```text
Upload -> Extract -> Chunk -> Index -> Retrieve
```

1. Upload a document.
2. Choose an extraction method.
3. Generate chunks with a compatible chunking strategy.
4. Preview chunks and metadata.
5. Ask retrieval questions in the assistant tab.

## Extraction Methods

- Whole Text Extraction
- Page-wise Extraction
- Block-wise Extraction
- Section-wise Extraction
- Hierarchical Extraction
- Layout-aware Extraction
- Table-aware Extraction
- Semantic Segmentation
- Clause-aware Extraction
- Code-aware Extraction

## Chunking Strategies

- Fixed-size Chunking
- Recursive Chunking
- Header-aware Chunking
- Section-aware Chunking
- Layout-aware Chunking
- DOM-aware Chunking
- Parent-Child Chunking
- Semantic Chunking
- Adaptive Chunking
- Clause-aware Chunking
- OCR-aware Chunking
- Table-aware Chunking
- Code Function-aware Chunking
- Multi-modal Chunking

## Retrieval and Answers

ChunkLab builds a vector retrieval index after chunks are generated.

The default retrieval stack uses FAISS when available and falls back to an in-memory vector store when needed. Embeddings are selected in this order:

1. Azure OpenAI embeddings, if configured.
2. Sentence Transformers, if available.
3. Local hashing embeddings as an offline fallback.

The retrieval assistant supports two answer modes:

- Extractive, no LLM: answers directly from retrieved chunks.
- LLM from retrieved chunks: uses Azure OpenAI to generate an answer from retrieved context.

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Optional Azure OpenAI Configuration

Create a `.env` file or export these variables to enable Azure embeddings:

```bash
AZURE_OPENAI_EMB_KEY=your_embedding_key
AZURE_EMB_ENDPOINT=your_embedding_endpoint
AZURE_EMB_API_VERSION=your_embedding_api_version
AZURE_EMB_DEPLOYMENT=your_embedding_deployment
```

Set these variables to enable LLM answer generation:

```bash
AZURE_OPENAI_LLM_KEY=your_llm_key
AZURE_LLM_ENDPOINT=your_llm_endpoint
AZURE_LLM_API_VERSION=your_llm_api_version
AZURE_LLM_DEPLOYMENT_41_MINI=your_llm_deployment
```

The app still runs without Azure configuration by using local fallback behavior.

## Project Structure

```text
app.py                  Streamlit app and user workflow
analytics/             Chunk and retrieval metrics
assistant/             Retrieval assistant and Azure answer generation
chunking/              Chunking strategy implementations
docs/                  Sample documents
extraction/            Document extraction implementations
retrieval/             Retriever orchestration
ui/                    Streamlit rendering helpers
utils/                 Shared models, compatibility maps, and text utilities
vectorstore/           Embedding and vector search backends
requirements.txt       Python dependencies
```

## Sample Documents

The `docs/` directory contains sample PDFs and Markdown files that can be used to test extraction, chunking, table handling, and retrieval behavior.

## Notes

- FAISS is used when installed and working in the local environment.
- Retrieval falls back gracefully if FAISS or a preferred embedder is unavailable.
- The app is designed for experimentation and evaluation, not production deployment as-is.
