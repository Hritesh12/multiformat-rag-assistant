"""
Abstract interface for the LLM generation backend.

Why this exists (same reasoning as retrieval/backend.py):
"Local-first, swappable to a cloud API" is only a real architectural
claim if the rest of the codebase depends on an interface, not on
Ollama or a specific cloud SDK directly. This file defines that
interface; ollama_backend.py and cloud_backend.py both implement it.

The interface is intentionally small — just "given a prompt, return
text" plus a streaming variant — because the interesting logic (how
retrieved chunks get turned into a prompt, how citations get attached)
lives in generator.py and should work identically regardless of which
LLM is actually answering.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass
class LLMResponse:
    text: str
    model_used: str
    backend_used: str        # "ollama" | "cloud" | "mock"


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        ...

    @abstractmethod
    def generate_stream(self, prompt: str, system_prompt: str | None = None) -> Iterator[str]:
        """Yields text chunks as they're produced, for a responsive UI
        instead of waiting for the full response. Every backend must
        implement real streaming if the underlying API supports it
        (both Ollama and the major cloud APIs do) — this is not a
        fake-it-with-one-yield method, since streaming is specifically
        what makes a live interview demo feel responsive rather than
        making the interviewer stare at a spinner."""
        ...

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """Checks whether this backend can actually be used right now
        (e.g. is the Ollama server running? is an API key set?) and
        returns (available, reason). Called before generation, so a
        misconfigured backend fails with a clear message instead of a
        raw connection-refused traceback surfacing in the UI."""
        ...
