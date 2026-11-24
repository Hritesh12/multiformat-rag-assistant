# Multi-Format RAG Assistant

A local-first Retrieval-Augmented Generation (RAG) application that allows users to upload and query multiple file formats, including PDF documents, Python source code, and CSV datasets.

Unlike a basic document-question-answering system, this project uses format-aware processing and query routing to select an appropriate strategy for each question.

## Features

- **Multi-format document ingestion:** Supports PDF, Python (`.py`), and CSV files.
- **AST-based code chunking:** Uses Python syntax structure to split code around functions and classes.
- **Hybrid retrieval:** Combines dense vector search with BM25 keyword retrieval.
- **Intelligent query routing:** Classifies questions into semantic search, code-focused search, or computation.
- **Exact tabular computation:** Uses pandas to perform supported calculations on CSV data.
- **Citation-grounded responses:** Generates answers using retrieved context and validates referenced chunk IDs.
- **Local LLM support:** Uses Ollama by default, with an optional cloud LLM backend.
- **Interactive interface:** Provides a Streamlit-based interface for uploading files and asking questions.

## Architecture

The application follows a modular pipeline:

```text
                User Uploads Files
                       |
                       v
                Document Ingestion
                       |
          +------------+------------+
          |            |            |
          v            v            v
         PDF         Python        CSV
       Chunking     AST Parsing   Metadata
          |            |         Extraction
          +------------+------------+
                       |
                       v
                Retrieval System
                       |
                       v
                  User Query
                       |
                       v
                  Query Router
                       |
          +------------+------------+
          |            |            |
          v            v            v
       Semantic       Code       Computation
        Search       Search       (pandas)
          |            |            |
          +------------+------------+
                       |
                       v
                 LLM Generation
                       |
                       v
              Citation-Checked Answer
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Vector Database | ChromaDB |
| Keyword Retrieval | BM25 |
| Tabular Processing | Pandas |
| PDF Processing | pypdf |
| Code Processing | Python AST |
| Local LLM | Ollama |
| Cloud LLM | Anthropic API |
| Testing | pytest / Python unittest |

## Project Structure

```text
multiformat-rag/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── sample_docs/
│
├── docs/
│   └── BUILD_LOG.md
│
├── src/
│   ├── ingest/
│   │   ├── pdf_chunker.py
│   │   ├── code_chunker.py
│   │   └── csv_handler.py
│   │
│   ├── retrieval/
│   │   ├── backend.py
│   │   ├── chroma_bm25_backend.py
│   │   ├── router.py
│   │   └── compute_executor.py
│   │
│   ├── generation/
│   │   ├── llm_backend.py
│   │   ├── ollama_backend.py
│   │   ├── cloud_backend.py
│   │   ├── generator.py
│   │   └── mock_backend.py
│   │
│   └── pipeline.py
│
├── tests/
│   ├── test_backend_logic.py
│   ├── test_generator.py
│   └── test_e2e_pipeline.py
│
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd multiformat-rag
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the local LLM

Install Ollama and download a supported model:

```bash
ollama pull llama3.1
```

Ensure Ollama is running before starting the application.

### 5. Start the application

```bash
streamlit run app/streamlit_app.py
```

Upload your documents through the sidebar and start asking questions.

## Optional: Cloud LLM Backend

The application also includes an optional Anthropic API backend.

Configure your API key as an environment variable.

**Windows PowerShell:**

```powershell
$env:ANTHROPIC_API_KEY="your-api-key"
```

**Linux / macOS:**

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

Select the cloud backend from the application sidebar.

Never commit API keys or `.env` files to the repository.

## Running Tests

Run the project's test files using:

```bash
python -m unittest discover -s tests -v
```

Confirm that the test command matches the actual test framework and project configuration before relying on it.

## Design Decisions

### Format-aware chunking

Different file formats require different processing strategies.

- PDF documents use paragraph-, sentence-, and line-aware chunking.
- Python files use AST-based chunking to preserve function and class boundaries.
- CSV files use schema-aware metadata and retain tabular data for computation.

### Hybrid retrieval

The retrieval system combines:

- Dense vector retrieval for semantic similarity.
- BM25 keyword retrieval for exact-term matching.
- Reciprocal Rank Fusion (RRF) to combine retrieval results.

### Query routing

Questions are routed according to their requirements:

- **SEMANTIC:** Retrieves relevant document context.
- **CODE:** Prioritizes code-related retrieval.
- **COMPUTE:** Uses pandas for supported tabular calculations.

### Modular architecture

The project uses backend interfaces to separate retrieval and LLM implementations from the main application pipeline.

This makes it easier to experiment with alternative backends without rewriting the entire application.

## Current Limitations

- Query routing is rule-based and may misclassify unfamiliar question patterns.
- Tabular computation currently supports a limited set of operations.
- BM25 indexing is rebuilt during index updates.
- AST-based chunking is Python-specific; other languages use a less precise fallback.

## Future Improvements

- Add an OpenSearch-backed retrieval implementation.
- Improve query routing with confidence-based fallback.
- Support additional programming languages.
- Improve incremental indexing.
- Expand supported analytical operations.
- Add more integration and evaluation tests.

## License

Add an appropriate open-source license before distributing the project publicly.