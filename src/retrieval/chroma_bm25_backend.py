"""
Concrete retrieval backend: ChromaDB (dense/vector) + rank_bm25 (keyword).

See backend.py for why this interface exists. See docs/ARCHITECTURE.md
for why this embedded stack was chosen over a server-based one like
OpenSearch, and what would change to swap it in later.

Embedding model: sentence-transformers/all-MiniLM-L6-v2 by default —
small (80MB), fast on CPU, good enough quality for a project at this
scale, and it's a genuinely common production choice, not a toy
model picked just to avoid downloading something bigger. Swappable
via the `embedding_model_name` constructor argument.
"""

from __future__ import annotations
import re
import chromadb
from chromadb.utils import embedding_functions

from src.retrieval.backend import (
    RetrievalBackend,
    IndexableDocument,
    RetrievedResult,
)

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


def _tokenize(text: str) -> list[str]:
    # Simple whitespace + punctuation-stripping tokenizer. BM25 doesn't
    # need anything fancier than this — it's not sensitive to the same
    # things a neural model is.
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


class ChromaBM25Backend(RetrievalBackend):
    def __init__(
        self,
        collection_name: str = "multiformat_rag",
        persist_directory: str | None = "./chroma_store",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        # NOTE on collection_name: if you run multiple ChromaBM25Backend
        # instances in the same process with the SAME collection_name,
        # they will share underlying chromadb state even with separate
        # client objects — confirmed directly during development (two
        # EphemeralClients, same collection name: writes via one were
        # immediately visible via the other, with no .add() call on the
        # second). Not an issue for this app's normal single-backend
        # usage, but relevant if you extend this into a multi-session
        # or multi-tenant setup — use a distinct collection_name per
        # tenant/session in that case.
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.EphemeralClient()

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )

        # BM25 is in-memory and rebuilt from the full corpus rather than
        # incrementally updated — rank_bm25 doesn't support incremental
        # updates natively. Fine at this project's scale (thousands of
        # chunks); documented as a scaling limit in ARCHITECTURE.md
        # rather than silently ignored.
        self._bm25_corpus: list[IndexableDocument] = []
        self._bm25_index: BM25Okapi | None = None

    def index(self, documents: list[IndexableDocument]) -> None:
        if not documents:
            return

        self.collection.upsert(
            ids=[d.doc_id for d in documents],
            documents=[d.content for d in documents],
            metadatas=[
                {"source_file": d.source_file, "source_type": d.source_type, **_flatten_metadata(d.metadata)}
                for d in documents
            ],
        )

        # Rebuild BM25 over the full corpus. We track documents by
        # doc_id so re-indexing the same document (e.g. after a file
        # edit) replaces rather than duplicates it.
        existing_ids = {d.doc_id for d in self._bm25_corpus}
        new_ids = {d.doc_id for d in documents}
        self._bm25_corpus = [d for d in self._bm25_corpus if d.doc_id not in new_ids] + documents
        tokenized = [_tokenize(d.content) for d in self._bm25_corpus]
        self._bm25_index = BM25Okapi(tokenized) if tokenized else None

    def keyword_search(self, query: str, top_k: int) -> list[RetrievedResult]:
        if self._bm25_index is None or not self._bm25_corpus:
            return []

        query_tokens = set(_tokenize(query))
        scores = self._bm25_index.get_scores(list(query_tokens))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for i in ranked_indices:
            # We filter on ACTUAL TOKEN OVERLAP, not score sign.
            # Originally this filtered on `scores[i] <= 0`, on the
            # assumption that a non-positive BM25 score meant zero
            # query-term overlap. That assumption is false: BM25's IDF
            # term can go negative for a term that appears in a large
            # fraction of the corpus (the classic case is a small or
            # single-document corpus, where every term in that one
            # document is — by definition — in 100% of documents).
            # Verified directly: a 1-document corpus containing the
            # query term "updated" produced a BM25 score of -0.27 for
            # that exact match, which the score-sign filter incorrectly
            # discarded as a non-match. Token overlap is what we
            # actually mean by "did this document match" — the score
            # is for RANKING among matches, not for deciding whether
            # something matched at all.
            doc_tokens = set(_tokenize(self._bm25_corpus[i].content))
            if not (query_tokens & doc_tokens):
                continue
            results.append(RetrievedResult(
                document=self._bm25_corpus[i],
                score=float(scores[i]),
                retrieval_method="bm25",
            ))
        return results

    def dense_search(self, query: str, top_k: int) -> list[RetrievedResult]:
        if self.collection.count() == 0:
            return []

        raw = self.collection.query(query_texts=[query], n_results=min(top_k, self.collection.count()))
        results = []
        ids = raw["ids"][0]
        docs = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]

        for doc_id, content, meta, distance in zip(ids, docs, metadatas, distances):
            # Chroma returns a distance (lower = more similar); we
            # convert to a similarity-style score (higher = better) so
            # it behaves consistently with the BM25 side for anything
            # that inspects .score directly, even though RRF itself
            # only cares about rank order, not the raw score value.
            similarity = 1.0 / (1.0 + distance)
            doc = IndexableDocument(
                doc_id=doc_id,
                content=content,
                source_file=meta.get("source_file", "unknown"),
                source_type=meta.get("source_type", "unknown"),
                metadata=meta,
            )
            results.append(RetrievedResult(document=doc, score=similarity, retrieval_method="dense"))
        return results


def _flatten_metadata(metadata: dict) -> dict:
    # Chroma metadata values must be str/int/float/bool — not None,
    # not nested dicts/lists. Silently dropping incompatible values
    # would lose information without any signal; instead we coerce
    # what's safely coercible and stringify the rest.
    flat = {}
    for k, v in metadata.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            flat[k] = v
        else:
            flat[k] = str(v)
    return flat
