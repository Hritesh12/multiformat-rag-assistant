"""
Generator: turns (query + retrieved context) into a grounded answer.

Why "citation-grounded" is a deliberate design choice, not decoration:
The single biggest trust problem with RAG systems is that a fluent,
confident-sounding answer LOOKS the same whether it's accurately
grounded in the retrieved documents or subtly (or badly) hallucinated.
Without forcing the model to cite which chunk each claim came from,
there's no way for the user to verify an answer without re-reading
every source document themselves — which defeats the point of RAG.

This module enforces grounding two ways:
1. The prompt explicitly instructs the model to cite chunk IDs inline
   for every factual claim, and to say "I don't have enough
   information" rather than fill a gap with something plausible.
2. After generation, cited chunk IDs are checked against what was
   ACTUALLY retrieved — if the model cites a chunk ID that wasn't in
   the context it was given, that's flagged. This doesn't prevent a
   model from hallucinating a claim attached to a REAL chunk ID (no
   post-hoc check can fully prevent that), but it does catch the
   cheaper, more common failure of citing something that was never
   there at all.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field

from src.generation.llm_backend import LLMBackend
from src.retrieval.backend import RetrievedResult
from src.retrieval.router import RoutingDecision, QueryStrategy
from src.retrieval.compute_executor import ComputeResult


@dataclass
class GeneratedAnswer:
    text: str
    routing_used: RoutingDecision
    sources_cited: list[str] = field(default_factory=list)
    sources_available_but_not_cited: list[str] = field(default_factory=list)
    hallucinated_citations: list[str] = field(default_factory=list)   # cited IDs not actually retrieved
    backend_used: str = ""
    raw_compute_result: ComputeResult | None = None


SYSTEM_PROMPT = """You are a research assistant answering questions about the user's own documents (PDFs, code, and spreadsheets).

Rules you must follow:
1. Answer ONLY using the CONTEXT provided below. Do not use outside knowledge about the general topic.
2. For every factual claim, cite the source using its chunk ID in square brackets, like [chunk_a1b2c3d4]. The chunk ID is shown before each context block.
3. If the context doesn't contain enough information to answer, say so plainly instead of guessing or filling the gap with plausible-sounding content.
4. If different chunks disagree, point out the disagreement rather than picking one silently.
5. Keep the answer focused and avoid restating the entire context back to the user."""


class Generator:
    def __init__(self, llm_backend: LLMBackend):
        self.llm = llm_backend

    def generate(self, query: str, routing: RoutingDecision, retrieved: list[RetrievedResult], compute_result: ComputeResult | None = None) -> GeneratedAnswer:
        available, reason = self.llm.is_available()
        if not available:
            return GeneratedAnswer(
                text=f"Cannot generate an answer — the LLM backend isn't available: {reason}",
                routing_used=routing,
                backend_used="none",
            )

        if routing.strategy == QueryStrategy.COMPUTE and compute_result is not None:
            return self._generate_for_compute(query, routing, compute_result)

        return self._generate_for_retrieval(query, routing, retrieved)

    def _generate_for_compute(self, query: str, routing: RoutingDecision, compute_result: ComputeResult) -> GeneratedAnswer:
        # For COMPUTE queries, we do NOT ask the LLM to re-derive the
        # number — the pandas computation already produced an exact,
        # verified answer. The LLM's job here is limited to phrasing
        # that answer naturally in response to the user's question,
        # with an explicit instruction not to alter the computed value.
        if not compute_result.success:
            return GeneratedAnswer(
                text=f"I couldn't compute an answer to that: {compute_result.error}",
                routing_used=routing,
                backend_used="none",
                raw_compute_result=compute_result,
            )

        prompt = (
            f"The user asked: \"{query}\"\n\n"
            f"An exact computation was already performed and produced this result:\n{compute_result.answer}\n\n"
            f"Phrase this as a direct, natural answer to the user's question. "
            f"Do NOT recompute or alter the numeric value — just present it clearly. "
            f"Keep it to 1-3 sentences."
        )
        response = self.llm.generate(prompt, system_prompt="You phrase pre-computed exact numeric results as natural answers. Never alter the numbers given to you.")
        return GeneratedAnswer(
            text=response.text,
            routing_used=routing,
            backend_used=response.backend_used,
            raw_compute_result=compute_result,
        )

    def _generate_for_retrieval(self, query: str, routing: RoutingDecision, retrieved: list[RetrievedResult]) -> GeneratedAnswer:
        if not retrieved:
            return GeneratedAnswer(
                text="I couldn't find any relevant content in the indexed documents to answer that.",
                routing_used=routing,
                backend_used="none",
            )

        context_blocks = []
        available_chunk_ids = []
        for r in retrieved:
            chunk_id = r.document.doc_id
            available_chunk_ids.append(chunk_id)
            context_blocks.append(
                f"[chunk_{chunk_id}] (source: {r.document.source_file}, type: {r.document.source_type})\n{r.document.content}"
            )
        context_text = "\n\n---\n\n".join(context_blocks)

        prompt = f"CONTEXT:\n{context_text}\n\n---\n\nQUESTION: {query}"
        response = self.llm.generate(prompt, system_prompt=SYSTEM_PROMPT)

        cited_ids = self._extract_cited_chunk_ids(response.text)
        hallucinated = [cid for cid in cited_ids if cid not in available_chunk_ids]
        not_cited = [cid for cid in available_chunk_ids if cid not in cited_ids]

        return GeneratedAnswer(
            text=response.text,
            routing_used=routing,
            sources_cited=cited_ids,
            sources_available_but_not_cited=not_cited,
            hallucinated_citations=hallucinated,
            backend_used=response.backend_used,
        )

    def _extract_cited_chunk_ids(self, text: str) -> list[str]:
        # Matches "[chunk_a1b2c3d4]" and extracts "a1b2c3d4" — must
        # stay in sync with the "[chunk_{chunk_id}]" format used when
        # building context_blocks above.
        return list(dict.fromkeys(re.findall(r"\[chunk_([a-f0-9]+)\]", text)))
