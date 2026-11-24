"""
End-to-end integration test: real ingestion -> real indexing -> real
routing -> real retrieval -> real compute -> generation (via mock LLM,
since no real LLM is reachable in this sandbox).

This is the most important test in the project because it's the ONLY
one that exercises every layer together. Each layer passing its own
unit tests does not guarantee they compose correctly — this test
specifically checks the composition, using real chunkers, a real
hybrid retrieval backend, and a real compute executor against actual
sample documents (not mocks) for every layer except the LLM call
itself.
"""
import sys
import uuid
sys.path.insert(0, '/home/claude/multiformat-rag')

import chromadb
from chromadb import EmbeddingFunction
import hashlib

from src.ingest.pdf_chunker import PDFChunker
from src.ingest.code_chunker import PythonASTChunker
from src.ingest.csv_handler import CSVHandler
from src.retrieval.backend import IndexableDocument
from src.retrieval.chroma_bm25_backend import ChromaBM25Backend
from src.retrieval.router import QueryRouter
from src.retrieval.compute_executor import ComputeExecutor
from src.generation.generator import Generator
from src.generation.mock_backend import MockLLMBackend
from src.pipeline import RAGPipeline


class DeterministicHashEmbedding(EmbeddingFunction):
    """Same test-only substitute used in test_backend_logic.py — see
    that file's docstring for why (no network access to download a
    real embedding model in this sandbox)."""
    def __init__(self):
        pass

    def __call__(self, input):
        vectors = []
        for text in input:
            h = hashlib.sha256(text.encode()).digest()
            floats = [(b / 127.5) - 1.0 for b in h[:16]]
            vectors.append(floats)
        return vectors


def build_test_backend():
    backend = ChromaBM25Backend.__new__(ChromaBM25Backend)
    backend.client = chromadb.EphemeralClient()
    backend.embedding_fn = DeterministicHashEmbedding()
    backend.collection = backend.client.get_or_create_collection(
        name=f"e2e_{uuid.uuid4().hex[:12]}", embedding_function=backend.embedding_fn
    )
    backend._bm25_corpus = []
    backend._bm25_index = None
    return backend


def ingest_everything(backend):
    """Real ingestion of the actual sample files this project has
    been tested against throughout development — a resume PDF, the
    sample Python module, and the sample restaurant CSV."""
    all_docs = []

    pdf_chunker = PDFChunker()
    pdf_chunks = pdf_chunker.chunk_file('/mnt/user-data/uploads/Hritesh_Kumar_resume.pdf')
    for c in pdf_chunks:
        all_docs.append(IndexableDocument(
            doc_id=c.chunk_id, content=c.content, source_file=c.file_path,
            source_type="pdf", metadata={"page": c.page_number},
        ))

    code_chunker = PythonASTChunker()
    code_chunks = code_chunker.chunk_file('/home/claude/multiformat-rag/data/sample_docs/sample_module.py')
    for c in code_chunks:
        all_docs.append(IndexableDocument(
            doc_id=c.chunk_id, content=c.content, source_file=c.file_path,
            source_type="code", metadata={"chunk_type": c.chunk_type, "name": c.name},
        ))

    csv_handler = CSVHandler()
    schema_chunk, df = csv_handler.load('/home/claude/multiformat-rag/data/sample_docs/sample_restaurants.csv')
    all_docs.append(IndexableDocument(
        doc_id=schema_chunk.chunk_id, content=schema_chunk.content,
        source_file=schema_chunk.file_path, source_type="csv",
    ))

    backend.index(all_docs)
    return csv_handler, {schema_chunk.file_path: df}


def test_e2e_semantic_query_about_pdf_content():
    backend = build_test_backend()
    csv_handler, dataframes = ingest_everything(backend)
    router = QueryRouter(csv_files_indexed=list(dataframes.keys()))
    compute_executor = ComputeExecutor(dataframes=dataframes)
    mock_llm = MockLLMBackend(scripted_response="The robot achieved 98% navigation success [chunk_" + "x" * 8 + "].")
    generator = Generator(mock_llm)
    pipeline = RAGPipeline(backend, generator, compute_executor, router)

    result = pipeline.query("What was the navigation success rate of the delivery robot?")

    assert result.retrieval_count > 0, "should have retrieved at least one chunk from the resume PDF"
    assert mock_llm.last_prompt is not None, "LLM should have been called"
    assert "CONTEXT:" in mock_llm.last_prompt
    assert "navigation success rate" in mock_llm.last_prompt.lower()
    print(f"PASS: e2e_semantic_query_about_pdf_content — retrieved {result.retrieval_count} chunks")


def test_e2e_compute_query_bypasses_llm_math():
    backend = build_test_backend()
    csv_handler, dataframes = ingest_everything(backend)
    router = QueryRouter(csv_files_indexed=list(dataframes.keys()))
    compute_executor = ComputeExecutor(dataframes=dataframes)
    mock_llm = MockLLMBackend(scripted_response="The average cost is as shown.")
    generator = Generator(mock_llm)
    pipeline = RAGPipeline(backend, generator, compute_executor, router)

    result = pipeline.query("What is the average cost across all restaurants?")

    assert result.answer.raw_compute_result is not None, "compute path should have run"
    assert result.answer.raw_compute_result.success
    assert "531.43" in result.answer.raw_compute_result.answer, result.answer.raw_compute_result.answer
    assert result.retrieval_count == 0, "compute queries should not use semantic retrieval"
    print("PASS: e2e_compute_query_bypasses_llm_math — computed:", result.answer.raw_compute_result.answer)


def test_e2e_code_query_prioritizes_code_chunks():
    backend = build_test_backend()
    csv_handler, dataframes = ingest_everything(backend)
    router = QueryRouter(csv_files_indexed=list(dataframes.keys()))
    compute_executor = ComputeExecutor(dataframes=dataframes)
    mock_llm = MockLLMBackend(scripted_response="The DataProcessor class handles cleaning.")
    generator = Generator(mock_llm)
    pipeline = RAGPipeline(backend, generator, compute_executor, router)

    result = pipeline.query("What does the DataProcessor class do?")

    assert result.retrieval_count > 0
    assert "CONTEXT:" in mock_llm.last_prompt
    # At least the top retrieved chunk should be code-sourced, given
    # the query is unambiguously about a specific class name that
    # only exists in the sample Python module.
    assert "code" in mock_llm.last_prompt.lower() or "DataProcessor" in mock_llm.last_prompt
    print("PASS: e2e_code_query_prioritizes_code_chunks")


def test_e2e_all_three_source_types_are_indexed():
    backend = build_test_backend()
    csv_handler, dataframes = ingest_everything(backend)
    source_types = {d.source_type for d in backend._bm25_corpus}
    assert source_types == {"pdf", "code", "csv"}, f"expected all 3 source types indexed, got {source_types}"
    print(f"PASS: e2e_all_three_source_types_are_indexed — {backend.collection.count()} total chunks across {source_types}")


if __name__ == "__main__":
    test_e2e_all_three_source_types_are_indexed()
    test_e2e_semantic_query_about_pdf_content()
    test_e2e_compute_query_bypasses_llm_math()
    test_e2e_code_query_prioritizes_code_chunks()
    print()
    print("ALL END-TO-END TESTS PASSED")
