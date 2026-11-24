import sys
sys.path.insert(0, '/home/claude/multiformat-rag')

from src.generation.generator import Generator
from src.generation.mock_backend import MockLLMBackend
from src.retrieval.backend import IndexableDocument, RetrievedResult
from src.retrieval.router import RoutingDecision, QueryStrategy
from src.retrieval.compute_executor import ComputeResult


def make_retrieved(doc_id, content, source_file="test.pdf", source_type="pdf"):
    doc = IndexableDocument(doc_id=doc_id, content=content, source_file=source_file, source_type=source_type)
    return RetrievedResult(document=doc, score=1.0, retrieval_method="hybrid")


def test_citation_extraction_normal_case():
    retrieved = [
        make_retrieved("abc123", "The system achieved 98% accuracy."),
        make_retrieved("def456", "It used YOLO for object detection."),
    ]
    mock = MockLLMBackend(scripted_response="The system achieved 98% accuracy [chunk_abc123], using YOLO for detection [chunk_def456].")
    gen = Generator(mock)
    routing = RoutingDecision(strategy=QueryStrategy.SEMANTIC, confidence="medium", matched_pattern=None, reasoning="test")

    result = gen.generate("What did the system achieve?", routing, retrieved)

    assert set(result.sources_cited) == {"abc123", "def456"}, result.sources_cited
    assert result.hallucinated_citations == [], result.hallucinated_citations
    assert result.sources_available_but_not_cited == [], result.sources_available_but_not_cited
    print("PASS: citation_extraction_normal_case")


def test_hallucinated_citation_detected():
    retrieved = [make_retrieved("abc123", "Some real content.")]
    # Mock returns a citation to a chunk ID that was NEVER in the retrieved context
    mock = MockLLMBackend(scripted_response="This claim cites a real chunk [chunk_abc123] and a fake one [chunk_ffffff99].")
    gen = Generator(mock)
    routing = RoutingDecision(strategy=QueryStrategy.SEMANTIC, confidence="medium", matched_pattern=None, reasoning="test")

    result = gen.generate("query", routing, retrieved)

    assert "ffffff99" in result.hallucinated_citations, result.hallucinated_citations
    assert "abc123" not in result.hallucinated_citations
    print("PASS: hallucinated_citation_detected —", result.hallucinated_citations)


def test_uncited_available_source_tracked():
    retrieved = [
        make_retrieved("abc123", "Content that gets cited."),
        make_retrieved("def456", "Content that gets retrieved but never cited."),
    ]
    mock = MockLLMBackend(scripted_response="Only citing this one [chunk_abc123].")
    gen = Generator(mock)
    routing = RoutingDecision(strategy=QueryStrategy.SEMANTIC, confidence="medium", matched_pattern=None, reasoning="test")

    result = gen.generate("query", routing, retrieved)

    assert result.sources_available_but_not_cited == ["def456"], result.sources_available_but_not_cited
    print("PASS: uncited_available_source_tracked")


def test_empty_retrieval_does_not_crash():
    mock = MockLLMBackend(scripted_response="should not be reached")
    gen = Generator(mock)
    routing = RoutingDecision(strategy=QueryStrategy.SEMANTIC, confidence="medium", matched_pattern=None, reasoning="test")

    result = gen.generate("query", routing, [])

    assert "couldn't find any relevant content" in result.text.lower()
    assert mock.last_prompt is None, "LLM should not have been called with empty retrieval"
    print("PASS: empty_retrieval_does_not_crash")


def test_compute_strategy_uses_compute_result_not_llm_math():
    mock = MockLLMBackend(scripted_response="The average cost is 531.43 rupees.")
    gen = Generator(mock)
    routing = RoutingDecision(strategy=QueryStrategy.COMPUTE, confidence="high", matched_pattern="average", reasoning="test")
    compute_result = ComputeResult(success=True, answer="Mean of 'cost' in restaurants.csv: 531.43", operation_used="mean", column_used="cost")

    result = gen.generate("What is the average cost?", routing, retrieved=[], compute_result=compute_result)

    assert result.raw_compute_result is compute_result
    assert "531.43" in mock.last_prompt, "the exact computed value should be in the prompt sent to the LLM"
    assert "do not recompute" in mock.last_system_prompt.lower() or "never alter" in mock.last_system_prompt.lower()
    print("PASS: compute_strategy_uses_compute_result_not_llm_math")


def test_compute_failure_does_not_call_llm():
    mock = MockLLMBackend(scripted_response="should not be reached")
    gen = Generator(mock)
    routing = RoutingDecision(strategy=QueryStrategy.COMPUTE, confidence="high", matched_pattern="average", reasoning="test")
    compute_result = ComputeResult(success=False, answer="", error="No matching column found")

    result = gen.generate("What is the average xyz?", routing, retrieved=[], compute_result=compute_result)

    assert "No matching column found" in result.text
    assert mock.last_prompt is None, "LLM should not be called when compute already failed"
    print("PASS: compute_failure_does_not_call_llm")


def test_unavailable_backend_reported_clearly():
    class UnavailableMock(MockLLMBackend):
        def is_available(self):
            return False, "test: server not running"
    mock = UnavailableMock()
    gen = Generator(mock)
    routing = RoutingDecision(strategy=QueryStrategy.SEMANTIC, confidence="medium", matched_pattern=None, reasoning="test")

    result = gen.generate("query", routing, [make_retrieved("abc", "content")])

    assert "server not running" in result.text
    print("PASS: unavailable_backend_reported_clearly")


if __name__ == "__main__":
    test_citation_extraction_normal_case()
    test_hallucinated_citation_detected()
    test_uncited_available_source_tracked()
    test_empty_retrieval_does_not_crash()
    test_compute_strategy_uses_compute_result_not_llm_math()
    test_compute_failure_does_not_call_llm()
    test_unavailable_backend_reported_clearly()
    print()
    print("ALL TESTS PASSED")
