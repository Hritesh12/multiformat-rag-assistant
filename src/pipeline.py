"""
End-to-end pipeline: wires together ingestion (already-indexed data),
routing, retrieval, computation, and generation into a single
query-in/answer-out entry point.

This is what app.py (the Streamlit UI) calls. Keeping this
orchestration in its own module, separate from the UI, means the
whole pipeline is testable without Streamlit involved at all — which
is exactly how it's tested below.
"""

from __future__ import annotations
from dataclasses import dataclass

from src.retrieval.backend import RetrievalBackend
from src.retrieval.router import QueryRouter, QueryStrategy
from src.retrieval.compute_executor import ComputeExecutor
from src.generation.generator import Generator, GeneratedAnswer


@dataclass
class PipelineResult:
    answer: GeneratedAnswer
    retrieval_count: int


class RAGPipeline:
    def __init__(
        self,
        retrieval_backend: RetrievalBackend,
        generator: Generator,
        compute_executor: ComputeExecutor,
        router: QueryRouter,
        top_k: int = 5,
    ):
        self.retrieval_backend = retrieval_backend
        self.generator = generator
        self.compute_executor = compute_executor
        self.router = router
        self.top_k = top_k

    def query(self, user_query: str) -> PipelineResult:
        routing = self.router.route(user_query)

        if routing.strategy == QueryStrategy.COMPUTE:
            compute_result = self.compute_executor.execute(user_query)
            answer = self.generator.generate(user_query, routing, retrieved=[], compute_result=compute_result)
            return PipelineResult(answer=answer, retrieval_count=0)

        if routing.strategy == QueryStrategy.CODE:
            # Weight retrieval toward code by using a higher alpha
            # in hybrid search isn't quite right here (alpha controls
            # keyword-vs-dense, not source-type) — instead we retrieve
            # normally, then re-rank to prefer source_type == "code"
            # results, without excluding other types entirely (a
            # question ostensibly about code might still benefit from
            # a PDF chunk that documents the function, for instance).
            retrieved = self.retrieval_backend.hybrid_search(user_query, top_k=self.top_k * 2)
            retrieved = self._prioritize_source_type(retrieved, preferred_type="code", keep=self.top_k)
        else:
            retrieved = self.retrieval_backend.hybrid_search(user_query, top_k=self.top_k)

        answer = self.generator.generate(user_query, routing, retrieved=retrieved)
        return PipelineResult(answer=answer, retrieval_count=len(retrieved))

    def _prioritize_source_type(self, retrieved, preferred_type: str, keep: int):
        preferred = [r for r in retrieved if r.document.source_type == preferred_type]
        other = [r for r in retrieved if r.document.source_type != preferred_type]
        # Keep ALL preferred-type matches up to `keep`, then fill any
        # remaining slots with other types — rather than dropping
        # other types entirely, since a code question can genuinely
        # need supporting context from a PDF or CSV chunk too.
        combined = preferred[:keep] + other[:max(0, keep - len(preferred[:keep]))]
        return combined[:keep]
