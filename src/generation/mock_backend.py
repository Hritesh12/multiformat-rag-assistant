"""
A fake LLMBackend that returns pre-scripted responses, used ONLY for
testing generator.py's own logic (prompt assembly, citation
extraction, hallucination detection) — not for testing whether an
actual LLM produces good answers, which requires a real backend.

This is a legitimate, standard testing pattern (a test double), used
here for the same reason DeterministicHashEmbedding was used earlier:
this sandbox cannot run a real Ollama server or reach a paid cloud
API, so the parts of the system that don't depend on real model
output are tested directly, and the parts that DO depend on it are
clearly marked as requiring the developer's own machine to exercise.
"""

from __future__ import annotations
from typing import Iterator

from src.generation.llm_backend import LLMBackend, LLMResponse


class MockLLMBackend(LLMBackend):
    def __init__(self, scripted_response: str = ""):
        self.scripted_response = scripted_response
        self.last_prompt: str | None = None
        self.last_system_prompt: str | None = None

    def is_available(self) -> tuple[bool, str]:
        return True, "OK (mock)"

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        return LLMResponse(text=self.scripted_response, model_used="mock-model", backend_used="mock")

    def generate_stream(self, prompt: str, system_prompt: str | None = None) -> Iterator[str]:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        for word in self.scripted_response.split(" "):
            yield word + " "
