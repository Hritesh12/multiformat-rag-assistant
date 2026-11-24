"""
Test the ChromaBM25Backend LOGIC (indexing, hybrid search, RRF, metadata
handling, reindexing) using a minimal HAND-ROLLED embedding function
instead of any downloaded model.

Why: sentence-transformers needs torch, which could not be installed
in this sandbox (disk space ran out mid-install). ChromaDB's own
built-in "default" embedder also needs to download an ONNX model from
an S3 bucket not reachable from this sandboxed environment's network
allowlist. Both are documented in docs/BUILD_LOG.md.

The embedding function below is NOT semantically meaningful (it's a
deterministic hash-based vector, not a trained model) — it exists
only to exercise the actual code paths in chroma_bm25_backend.py:
does indexing work, does metadata round-trip correctly, does hybrid
search actually combine both retrieval methods, does re-indexing the
same doc_id replace rather than duplicate. Those are the things that
can genuinely break due to a coding bug. Semantic retrieval QUALITY
(does the real model correctly rank "pandas" close to "dataframes")
is a property of the sentence-transformers model itself, a
well-established library, and will be exercised for real the first
time this runs on a machine with normal internet access — see
README.md setup instructions.
"""
import sys
import hashlib
import struct
sys.path.insert(0, '/home/claude/multiformat-rag')

import chromadb
from chromadb import EmbeddingFunction
from src.retrieval.backend import IndexableDocument
from src.retrieval.chroma_bm25_backend import ChromaBM25Backend, _tokenize


class DeterministicHashEmbedding(EmbeddingFunction):
    """Turns text into a fixed-length, bounded vector via hashing. Not
    semantically meaningful — same-text-in gives same-vector-out
    (which is all indexing/round-trip tests need), but does NOT
    cluster similar meanings together the way a real model does.
    Used only to test code paths without a network-downloaded model.

    Values are normalized into [-1, 1] by mapping each hash byte
    (0-255) rather than reinterpreting raw bytes as IEEE-754 floats —
    an earlier version of this test helper did the latter and produced
    denormal/overflow values (e.g. 2.8e+28) that made distance
    computations return inf. Fixed by using bounded, well-behaved
    values instead."""
    def __init__(self):
        pass

    def __call__(self, input):
        vectors = []
        for text in input:
            h = hashlib.sha256(text.encode()).digest()
            # Each byte (0-255) mapped to [-1, 1] — bounded and finite.
            floats = [(b / 127.5) - 1.0 for b in h[:16]]
            vectors.append(floats)
        return vectors


def make_test_backend():
    """Builds a ChromaBM25Backend but swaps in a lightweight
    deterministic embedder instead of sentence-transformers.

    IMPORTANT: uses a unique collection name per call. Testing
    revealed that separate chromadb.EphemeralClient() instances share
    underlying collection state when given the SAME collection name
    within a single process — a genuine gotcha, confirmed directly:
    creating two EphemeralClients, adding one document via the first
    client's collection, then reading count() from the second
    client's same-named collection (which never had .add() called on
    it) returned 1, not 0. Without unique names per test, later tests
    in this file silently saw documents left over from earlier tests —
    exactly what happened before this was fixed. See docs/BUILD_LOG.md."""
    import uuid
    collection_name = f"test_{uuid.uuid4().hex[:12]}"
    backend = ChromaBM25Backend.__new__(ChromaBM25Backend)  # skip __init__, no real embedder
    backend.client = chromadb.EphemeralClient()
    backend.embedding_fn = DeterministicHashEmbedding()
    backend.collection = backend.client.get_or_create_collection(
        name=collection_name, embedding_function=backend.embedding_fn
    )
    backend._bm25_corpus = []
    backend._bm25_index = None
    return backend


def test_tokenizer():
    tokens = _tokenize("Hello, World! This is a TEST_123.")
    assert tokens == ["hello", "world", "this", "is", "a", "test_123"], tokens
    print("PASS: tokenizer")


def test_index_and_keyword_search():
    backend = make_test_backend()
    docs = [
        IndexableDocument(doc_id="1", content="Python data analysis with pandas", source_file="a.py", source_type="code"),
        IndexableDocument(doc_id="2", content="Restaurant ratings in Bangalore zones", source_file="b.csv", source_type="csv"),
        IndexableDocument(doc_id="3", content="Deep reinforcement learning for robotics", source_file="c.pdf", source_type="pdf"),
    ]
    backend.index(docs)
    results = backend.keyword_search("pandas data analysis", top_k=3)
    assert len(results) > 0, "keyword search returned nothing"
    assert results[0].document.doc_id == "1", f"expected doc 1 to rank first (real BM25, unaffected by embedder), got {results[0].document.doc_id}"
    print("PASS: index_and_keyword_search — top result:", results[0].document.doc_id, "score:", round(results[0].score, 3))


def test_dense_search_runs_and_returns_correct_shape():
    backend = make_test_backend()
    docs = [
        IndexableDocument(doc_id="1", content="Python data analysis with pandas", source_file="a.py", source_type="code"),
        IndexableDocument(doc_id="2", content="Restaurant ratings in Bangalore zones", source_file="b.csv", source_type="csv"),
    ]
    backend.index(docs)
    results = backend.dense_search("some query text", top_k=2)
    assert len(results) == 2, f"expected 2 results, got {len(results)}"
    for r in results:
        assert r.retrieval_method == "dense"
        assert r.document.source_file in ("a.py", "b.csv")
        assert isinstance(r.score, float)
    print("PASS: dense_search_runs_and_returns_correct_shape — got", len(results), "results, scores:", [round(r.score, 3) for r in results])


def test_hybrid_search_combines_both():
    backend = make_test_backend()
    docs = [
        IndexableDocument(doc_id="1", content="Python data analysis with pandas dataframes", source_file="a.py", source_type="code"),
        IndexableDocument(doc_id="2", content="Restaurant ratings in Bangalore zones and cuisine trends", source_file="b.csv", source_type="csv"),
        IndexableDocument(doc_id="3", content="Deep reinforcement learning for robotic manipulation", source_file="c.pdf", source_type="pdf"),
    ]
    backend.index(docs)
    results = backend.hybrid_search("pandas dataframe analysis", top_k=3)
    assert len(results) > 0
    assert all(r.retrieval_method == "hybrid" for r in results)
    # doc 1 should rank highly since it strongly matches on BM25 (exact
    # keyword overlap) even though the fake embedder contributes noise
    assert results[0].document.doc_id == "1", f"expected doc 1 first due to strong BM25 signal, got {results[0].document.doc_id}"
    print("PASS: hybrid_search_combines_both — ranked:", [r.document.doc_id for r in results])


def test_metadata_roundtrip():
    backend = make_test_backend()
    docs = [
        IndexableDocument(
            doc_id="1", content="test content", source_file="x.py",
            source_type="code", metadata={"chunk_type": "function", "start_line": 5}
        ),
    ]
    backend.index(docs)
    results = backend.dense_search("test", top_k=1)
    assert results[0].document.source_type == "code"
    assert results[0].document.source_file == "x.py"
    print("PASS: metadata_roundtrip")


def test_reindex_same_doc_id_replaces_not_duplicates():
    backend = make_test_backend()
    doc_v1 = IndexableDocument(doc_id="1", content="version one content", source_file="a.py", source_type="code")
    doc_v2 = IndexableDocument(doc_id="1", content="version two updated content", source_file="a.py", source_type="code")
    backend.index([doc_v1])
    backend.index([doc_v2])
    assert backend.collection.count() == 1, f"expected 1 doc after reindex, got {backend.collection.count()}"
    assert len(backend._bm25_corpus) == 1, f"expected 1 bm25 doc after reindex, got {len(backend._bm25_corpus)}"
    results = backend.keyword_search("updated", top_k=1)
    assert len(results) == 1 and "updated" in results[0].document.content
    print("PASS: reindex_same_doc_id_replaces_not_duplicates")


def test_empty_index_does_not_crash():
    backend = make_test_backend()
    assert backend.keyword_search("anything", top_k=5) == []
    assert backend.dense_search("anything", top_k=5) == []
    assert backend.hybrid_search("anything", top_k=5) == []
    print("PASS: empty_index_does_not_crash")


if __name__ == "__main__":
    test_tokenizer()
    test_index_and_keyword_search()
    test_dense_search_runs_and_returns_correct_shape()
    test_hybrid_search_combines_both()
    test_metadata_roundtrip()
    test_reindex_same_doc_id_replaces_not_duplicates()
    test_empty_index_does_not_crash()
    print()
    print("ALL TESTS PASSED")
