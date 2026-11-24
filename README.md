# Multi-Format RAG System

A Retrieval-Augmented Generation system that indexes and answers questions across **PDFs, Python code, and CSV/tabular data** in one unified pipeline — with format-aware chunking, hybrid retrieval, query routing (semantic search vs. exact computation vs. code-weighted search), and citation-grounded generation.

Runs fully local by default (Ollama), with a one-line swap to a cloud LLM API.

---

## Why this project is different from a typical beginner RAG tutorial

Most RAG tutorials (including the two open-source repos this project was inspired by) handle one document type — usually PDFs — with fixed-size character chunking and a single retrieval-then-generate step. That breaks down the moment you mix in code or spreadsheets:

- **Fixed-size chunking cuts functions in half.** A 500-character chunk boundary doesn't know or care that it's slicing through the middle of a function body.
- **Embedding a CSV as raw text is close to useless.** Vector similarity search can't reliably answer "what's the average cost" — that's a computation, not a retrieval problem.
- **Not every question needs the same retrieval strategy.** A question about a function's logic and a question about a numeric aggregate and a question about what a report says are three different problems wearing the same "ask a question" costume.

This project handles each of those directly:

| Problem | This project's approach |
|---|---|
| Code gets cut mid-function by naive chunking | AST-based chunking (`src/ingest/code_chunker.py`) — chunks along actual function/class boundaries |
| CSVs can't be usefully embedded as text | Schema-aware summarization for retrieval + a live DataFrame for exact pandas computation (`src/ingest/csv_handler.py`, `src/retrieval/compute_executor.py`) |
| One retrieval strategy doesn't fit every question | A query router classifies each query as COMPUTE / CODE / SEMANTIC before retrieval happens (`src/retrieval/router.py`) |
| RAG answers look confident whether they're grounded or hallucinated | Every answer must cite chunk IDs; citations are checked post-hoc against what was actually retrieved (`src/generation/generator.py`) |

---

## Setup (run this on your own machine — not in a sandboxed environment)

```bash
git clone <your-repo-url>
cd multiformat-rag
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Local LLM (default)
1. Install Ollama: https://ollama.com
2. Pull a model: `ollama pull llama3.1`
3. Start Ollama (usually runs automatically after install; if not: `ollama serve`)

### Cloud LLM (optional alternative)
```bash
export ANTHROPIC_API_KEY=your_key_here
```
Then select "Cloud API" in the app's sidebar instead of "Local (Ollama)".

### Run the app
```bash
streamlit run app/streamlit_app.py
```
Upload a PDF, a `.py` file, and a `.csv` file in the sidebar, then ask questions in the chat box.

### Run the tests
```bash
python3 tests/test_backend_logic.py
python3 tests/test_generator.py
python3 tests/test_e2e_pipeline.py
```

---

## Architecture

```
User query
    │
    ▼
QueryRouter ──────────► classifies as COMPUTE / CODE / SEMANTIC
    │
    ├── COMPUTE ──► ComputeExecutor ──► exact pandas aggregation (no LLM math)
    │
    ├── CODE ─────► hybrid retrieval, re-ranked toward code-type chunks
    │
    └── SEMANTIC ─► hybrid retrieval (BM25 + dense embeddings, RRF-fused)
                         │
                         ▼
                    Generator ──► LLM (Ollama or Cloud, swappable)
                         │
                         ▼
                 Citation-checked answer
```

**Why ChromaDB + rank_bm25 instead of OpenSearch:** OpenSearch requires running a separate server (Docker/JVM), which works well for a production deployment story but makes the project harder to demo standalone. This is a deliberate trade-off, made explicit rather than hidden: the retrieval logic depends on an abstract `RetrievalBackend` interface (`src/retrieval/backend.py`), not on ChromaDB directly — swapping in an OpenSearch-backed implementation later is a new class implementing that interface, not a rewrite.

**Why rule-based query routing instead of an LLM-based router:** An LLM-based router is more flexible but adds a full extra generation call (latency + cost) to every query just to classify it, and introduces its own hallucination risk in the classification step itself. A rule-based router is fast, free, deterministic, and its failure modes are enumerable — documented directly in `src/retrieval/router.py`.

---

## What was actually hard to build (and how I know, not just claim, this works)

Every non-trivial piece of this project was tested against real inputs during development, and several genuine bugs were found and fixed — not hypothetical edge cases, but things that actually broke when run. The full list with root causes is in **`docs/BUILD_LOG.md`**. Highlights, useful for interview conversation:

1. **AST chunker initially duplicated the entire file** into a "module-level" chunk because of an off-by-range-boundary bug in how top-level statement ranges were collected — caught by printing actual chunk output against a test file, not by inspection alone.
2. **PDF chunker produced one 4945-character oversized chunk** on a real resume PDF that had no blank-line paragraph structure AND no sentence-terminal punctuation (bullet-heavy layout) — both of my two fallback layers failed simultaneously. Fixed by adding a third fallback (line-grouping). This was found by testing against real uploaded documents, not synthetic ones — synthetic test files didn't have this property.
3. **BM25 keyword search silently dropped true matches** on small corpora, because BM25's IDF term can go negative when a term appears in most/all documents (guaranteed on a 1-document corpus) — and the filter logic wrongly treated "negative score" as "no match." Fixed by checking actual token overlap instead of score sign.
4. **A compute query ("average cost across all restaurants") matched the wrong column** because a heuristic meant to deprioritize identifier-like columns (e.g., a restaurant-name column) didn't distinguish between a genuine string identifier and a numeric column that happened to have all-unique values in a small test dataset. This bug only showed up in **end-to-end integration testing** — isolated unit tests of the same component, with slightly different queries, had kept passing the whole time.
5. **ChromaDB's `EphemeralClient` shares state across separate instances** when given the same collection name within a process — an undocumented-feeling gotcha discovered by a failing test, confirmed with a minimal isolated repro before accepting it as the explanation (rather than assuming the bug was in my own code).

If asked in an interview "walk me through a bug you hit building this" — any of the above is a real, specific, honestly-earned answer.

---

## How to explain this project in an interview

**One-liner:** "A RAG system that handles PDFs, code, and CSVs in one pipeline, with a query router that decides whether a question needs semantic search, exact computation, or code-weighted retrieval — because treating every question as a text-similarity problem breaks down once you mix in tabular data and source code."

**If asked "why not just use LangChain/LlamaIndex end-to-end":** "I used focused libraries for specific jobs (ChromaDB for vectors, rank_bm25 for keyword search, pypdf for extraction) behind my own interfaces, rather than a heavyweight framework, so I fully understand and can explain every part of the retrieval and routing logic — which matters more to me right now than saving a day of glue code."

**If asked about limitations (be honest — these are real, not rhetorical):**
- The query router is rule-based and will misroute a query phrased in a way its patterns don't anticipate.
- The compute executor handles single-aggregate and simple groupby queries; genuinely complex multi-condition analytical questions are reported as unsupported rather than guessed at.
- BM25 is rebuilt from the full corpus on each index update rather than incrementally updated — fine at this project's scale, a real constraint at much larger scale.
- The AST chunker is Python-specific; other languages fall back to a conservative line-based chunker, clearly labeled as lower-precision.

**If asked "what would you do with more time":** OpenSearch swap for production deployment, an LLM-based router as a fallback when rule-based routing has low confidence, incremental BM25 indexing, multi-language AST-aware chunking (tree-sitter would be the natural next tool).

---

## Project structure

```
src/
  ingest/
    pdf_chunker.py       # paragraph/sentence/line-aware PDF chunking
    code_chunker.py       # AST-based Python chunking
    csv_handler.py        # schema-aware CSV summarization
  retrieval/
    backend.py             # abstract retrieval interface + RRF fusion
    chroma_bm25_backend.py # ChromaDB (dense) + rank_bm25 (keyword) implementation
    router.py              # query classification: COMPUTE / CODE / SEMANTIC
    compute_executor.py    # safe, constrained pandas execution (no eval/exec)
  generation/
    llm_backend.py          # abstract LLM interface
    ollama_backend.py       # local LLM via Ollama REST API
    cloud_backend.py        # Anthropic API backend
    generator.py             # prompt assembly + citation/hallucination checking
    mock_backend.py          # test double for generator logic tests
  pipeline.py             # wires router + retrieval + compute + generation together
app/
  streamlit_app.py        # UI — thin, no business logic
tests/
  test_backend_logic.py   # retrieval backend unit tests
  test_generator.py        # generation logic unit tests
  test_e2e_pipeline.py     # full pipeline integration tests (real ingestion, real retrieval)
docs/
  BUILD_LOG.md             # every real bug found during development, with root cause and fix
```
