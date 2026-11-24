"""
Abstract interface for the retrieval backend.

Why this exists as its own file:
This project uses an embedded stack (ChromaDB + rank_bm25) instead of
a server-based one (OpenSearch) — a deliberate trade-off for demo
portability and zero-infrastructure setup (see docs/ARCHITECTURE.md
for the reasoning). But "swappable" is only a real claim if the rest
of the codebase depends on THIS interface, not on ChromaDB or
rank_bm25 directly. If retrieval.py imported chromadb directly
everywhere, swapping backends would mean rewriting the retrieval
logic, not just writing a new adapter — that's the difference between
an actual abstraction and a documentation-only claim.

Every chunk type (PDF, code, CSV schema) gets normalized into an
IndexableDocument before it reaches this layer, so the backend never
needs to know about source-format-specific fields.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class IndexableDocument:
    """Backend-agnostic representation of a single retrievable unit,
    regardless of whether it originated from a PDF chunk, a code
    chunk, or a CSV schema summary."""
    doc_id: str
    content: str
    source_file: str
    source_type: str          # "pdf" | "code" | "csv"
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedResult:
    document: IndexableDocument
    score: float
    retrieval_method: str     # "bm25" | "dense" | "hybrid"


class RetrievalBackend(ABC):
    """Anything implementing this interface can sit behind the query
    router without the router needing to change. A future OpenSearch-
    backed implementation would implement this same interface."""

    @abstractmethod
    def index(self, documents: list[IndexableDocument]) -> None:
        ...

    @abstractmethod
    def keyword_search(self, query: str, top_k: int) -> list[RetrievedResult]:
        ...

    @abstractmethod
    def dense_search(self, query: str, top_k: int) -> list[RetrievedResult]:
        ...

    def hybrid_search(self, query: str, top_k: int, alpha: float = 0.5) -> list[RetrievedResult]:
        """Default hybrid implementation: run both, merge by
        Reciprocal Rank Fusion. Backends can override this if they
        have a native hybrid mode (OpenSearch does) — but a correct
        default here means a backend only HAS to implement the two
        methods above to be usable."""
        keyword_results = self.keyword_search(query, top_k=top_k * 2)
        dense_results = self.dense_search(query, top_k=top_k * 2)
        return reciprocal_rank_fusion(keyword_results, dense_results, top_k=top_k, alpha=alpha)


def reciprocal_rank_fusion(
    keyword_results: list[RetrievedResult],
    dense_results: list[RetrievedResult],
    top_k: int,
    alpha: float = 0.5,
    rrf_k: int = 60,
) -> list[RetrievedResult]:
    """Combines two ranked lists into one using Reciprocal Rank Fusion
    rather than naively averaging raw scores. This matters because
    BM25 scores and cosine-similarity scores live on completely
    different numeric scales (BM25 is unbounded, cosine similarity is
    roughly [0,1]) — averaging them directly would let whichever
    method happens to produce larger numbers dominate the ranking
    regardless of which one is actually more relevant. RRF instead
    uses each result's RANK POSITION in its own list, which is scale-
    free by construction.

    alpha controls the keyword-vs-dense weighting: alpha=1.0 is pure
    keyword, alpha=0.0 is pure dense, 0.5 is balanced.
    """
    scores: dict[str, float] = {}
    doc_lookup: dict[str, IndexableDocument] = {}

    for rank, result in enumerate(keyword_results):
        doc_id = result.document.doc_id
        scores[doc_id] = scores.get(doc_id, 0.0) + alpha * (1.0 / (rrf_k + rank + 1))
        doc_lookup[doc_id] = result.document

    for rank, result in enumerate(dense_results):
        doc_id = result.document.doc_id
        scores[doc_id] = scores.get(doc_id, 0.0) + (1 - alpha) * (1.0 / (rrf_k + rank + 1))
        doc_lookup[doc_id] = result.document

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [
        RetrievedResult(document=doc_lookup[doc_id], score=score, retrieval_method="hybrid")
        for doc_id, score in ranked
    ]
