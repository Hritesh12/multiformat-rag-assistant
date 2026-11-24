"""
Cloud LLM backend: talks to Anthropic's API as the swap-in alternative
to the local Ollama backend.

Why Anthropic specifically, not "any cloud provider": picking one
concrete, real API to implement (rather than a generic multi-provider
wrapper) is what makes this a genuine, testable implementation of the
LLMBackend interface instead of unverifiable placeholder code. The
architecture itself doesn't depend on it being Anthropic — implementing
LLMBackend for a different provider (OpenAI, etc.) would be adding a
new file, not restructuring anything, which is the actual point being
demonstrated.

Requires ANTHROPIC_API_KEY to be set as an environment variable — this
backend never hardcodes or asks the user to paste a key into code.
"""

from __future__ import annotations
import os
from typing import Iterator

from src.generation.llm_backend import LLMBackend, LLMResponse

try:
    import anthropic
except ImportError:
    anthropic = None


class CloudBackend(LLMBackend):
    def __init__(self, model: str = "claude-sonnet-4-5-20250929", api_key: str | None = None):
        self.model = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None
        if anthropic is not None and self._api_key:
            self._client = anthropic.Anthropic(api_key=self._api_key)

    def is_available(self) -> tuple[bool, str]:
        if anthropic is None:
            return False, "The 'anthropic' package isn't installed. Run: pip install anthropic"
        if not self._api_key:
            return False, "No API key found. Set the ANTHROPIC_API_KEY environment variable."
        # We deliberately do NOT make a real API call here just to
        # check availability (unlike Ollama's is_available, which
        # hits a free local endpoint) — a cloud API call costs money
        # and adds latency, so "is configured" is the check, not "is
        # confirmed reachable". A misconfigured key still surfaces
        # clearly, just on the first real generate() call instead.
        return True, "OK"

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        if self._client is None:
            raise RuntimeError("CloudBackend is not configured — check is_available() before calling generate().")

        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if hasattr(block, "text"))
        return LLMResponse(text=text, model_used=self.model, backend_used="cloud")

    def generate_stream(self, prompt: str, system_prompt: str | None = None) -> Iterator[str]:
        if self._client is None:
            raise RuntimeError("CloudBackend is not configured — check is_available() before calling generate_stream().")

        with self._client.messages.stream(
            model=self.model,
            max_tokens=1024,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text
