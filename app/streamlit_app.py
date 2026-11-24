"""
Streamlit UI for the multi-format RAG system.

Run with: streamlit run app/streamlit_app.py

This file is intentionally thin — retrieval, routing, and generation
logic all live in src/, tested independently of Streamlit.
"""

from __future__ import annotations
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.ingest.pdf_chunker import PDFChunker
from src.ingest.code_chunker import PythonASTChunker
from src.ingest.csv_handler import CSVHandler
from src.retrieval.backend import IndexableDocument
from src.retrieval.chroma_bm25_backend import ChromaBM25Backend
from src.retrieval.router import QueryRouter, QueryStrategy
from src.retrieval.compute_executor import ComputeExecutor
from src.generation.generator import Generator
from src.generation.ollama_backend import OllamaBackend
from src.generation.cloud_backend import CloudBackend
from src.pipeline import RAGPipeline


st.set_page_config(page_title="Multi-Format RAG", page_icon="📚", layout="wide")


def init_session_state():
    if "backend" not in st.session_state:
        st.session_state.backend = ChromaBM25Backend(
            collection_name="streamlit_session",
            persist_directory=None,
        )
    if "dataframes" not in st.session_state:
        st.session_state.dataframes = {}
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "csv_handler" not in st.session_state:
        st.session_state.csv_handler = CSVHandler()


def get_llm_backend(choice: str):
    if choice == "Local (Ollama)":
        return OllamaBackend()
    return CloudBackend()


def _collect_csv_metadata(dataframes: dict) -> tuple[list[str], list[str]]:
    """Gathers column names and categorical values across all
    indexed CSVs, for the router's relevance-checking (see
    docs/BUILD_LOG.md, bugs #11 and #14). Only LOW-cardinality
    columns contribute values — a name/ID column would add hundreds
    of near-useless single-use values to check against every query."""
    all_columns = []
    all_values = []
    for df in dataframes.values():
        all_columns.extend(str(c) for c in df.columns)
        for col in df.columns:
            if df[col].dtype == object and df[col].nunique() <= min(50, len(df) * 0.5 + 1):
                all_values.extend(str(v) for v in df[col].dropna().unique())
    return all_columns, all_values


def ingest_uploaded_file(uploaded_file) -> tuple[list[IndexableDocument], str]:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    docs: list[IndexableDocument] = []

    if suffix == ".pdf":
        chunks = PDFChunker().chunk_file(tmp_path)
        for c in chunks:
            docs.append(IndexableDocument(
                doc_id=c.chunk_id, content=c.content, source_file=uploaded_file.name,
                source_type="pdf", metadata={"page": c.page_number},
            ))
        return docs, "pdf"

    if suffix == ".py":
        chunks = PythonASTChunker().chunk_file(tmp_path)
        for c in chunks:
            docs.append(IndexableDocument(
                doc_id=c.chunk_id, content=c.content, source_file=uploaded_file.name,
                source_type="code", metadata={"chunk_type": c.chunk_type, "name": c.name},
            ))
        return docs, "code"

    if suffix == ".csv":
        # CSVHandler.load returns (schema_chunk, row_chunks, df) — a
        # schema/summary chunk PLUS one chunk per row, so citations
        # for different rows point to different chunk_ids instead of
        # sharing one blob (docs/BUILD_LOG.md — the original bug).
        schema_chunk, row_chunks, df = st.session_state.csv_handler.load(tmp_path)
        docs.append(IndexableDocument(
            doc_id=schema_chunk.chunk_id, content=schema_chunk.content,
            source_file=uploaded_file.name, source_type="csv",
        ))
        for rc in row_chunks:
            docs.append(IndexableDocument(
                doc_id=rc.chunk_id, content=rc.content,
                source_file=uploaded_file.name, source_type="csv",
                metadata={"row_index": rc.row_index},
            ))
        st.session_state.dataframes[uploaded_file.name] = df
        return docs, "csv"

    raise ValueError(f"Unsupported file type: {suffix}. Supported: .pdf, .py, .csv")


def render_sidebar():
    with st.sidebar:
        st.header("📁 Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, Python, or CSV files",
            type=["pdf", "py", "csv"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            for uf in uploaded_files:
                if uf.name in st.session_state.indexed_files:
                    continue
                with st.spinner(f"Ingesting {uf.name}..."):
                    try:
                        docs, source_type = ingest_uploaded_file(uf)
                        st.session_state.backend.index(docs)
                        st.session_state.indexed_files.append(uf.name)
                        st.success(f"Indexed {uf.name} — {len(docs)} chunk(s) ({source_type})")
                    except Exception as e:
                        st.error(f"Failed to ingest {uf.name}: {e}")

        if st.session_state.indexed_files:
            st.caption("Indexed files:")
            for fname in st.session_state.indexed_files:
                st.caption(f"  • {fname}")

        st.divider()
        st.header("⚙️ Settings")
        llm_choice = st.radio("LLM backend", ["Local (Ollama)", "Cloud API"], index=0)
        top_k = st.slider("Chunks to retrieve", min_value=2, max_value=10, value=5)

        return llm_choice, top_k


def render_routing_badge(strategy: QueryStrategy, reasoning: str):
    icons = {QueryStrategy.COMPUTE: "🔢", QueryStrategy.CODE: "💻", QueryStrategy.SEMANTIC: "📄"}
    st.caption(f"{icons.get(strategy, '❓')} **Routed to: {strategy.value}** — {reasoning}")


def render_citations(answer):
    if answer.sources_cited:
        with st.expander(f"📎 {len(answer.sources_cited)} source(s) cited"):
            for cid in answer.sources_cited:
                st.code(f"chunk_{cid}", language=None)
    if answer.hallucinated_citations:
        st.warning(
            f"⚠️ The model cited {len(answer.hallucinated_citations)} chunk ID(s) that weren't actually "
            f"in the retrieved context: {answer.hallucinated_citations}. Treat this answer with extra caution."
        )


def main():
    init_session_state()
    st.title("📚 Multi-Format RAG System")
    st.caption("Query across PDFs, code, and spreadsheets — with format-aware routing and citation-grounded answers.")

    llm_choice, top_k = render_sidebar()

    if not st.session_state.indexed_files:
        st.info("👈 Upload a PDF, Python file, or CSV in the sidebar to get started.")
        return

    llm_backend = get_llm_backend(llm_choice)
    available, reason = llm_backend.is_available()
    if not available:
        st.warning(f"LLM backend not available: {reason}")

    csv_columns, csv_categorical_values = _collect_csv_metadata(st.session_state.dataframes)
    router = QueryRouter(
        csv_files_indexed=list(st.session_state.dataframes.keys()),
        csv_columns=csv_columns,
        csv_categorical_values=csv_categorical_values,
    )
    compute_executor = ComputeExecutor(dataframes=st.session_state.dataframes)
    generator = Generator(llm_backend)
    pipeline = RAGPipeline(st.session_state.backend, generator, compute_executor, router, top_k=top_k)

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn["role"] == "assistant" and "routing" in turn:
                render_routing_badge(turn["routing"].strategy, turn["routing"].reasoning)

    if query := st.chat_input("Ask about your documents..."):
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = pipeline.query(query)
            st.markdown(result.answer.text)
            render_routing_badge(result.answer.routing_used.strategy, result.answer.routing_used.reasoning)
            render_citations(result.answer)

        st.session_state.chat_history.append({
            "role": "assistant", "content": result.answer.text, "routing": result.answer.routing_used,
        })


if __name__ == "__main__":
    main()
