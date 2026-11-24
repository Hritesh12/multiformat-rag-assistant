"""
Ollama backend: talks to a locally-running Ollama server over HTTP.

Ollama runs as a local service (default: http://localhost:11434) once
installed and a model is pulled (e.g. `ollama pull llama3.1`). This
backend does NOT install or manage Ollama itself — that's a one-time
setup step documented in README.md — it just talks to whatever's
already running there via Ollama's REST API.

This is the DEFAULT backend for the app (see generator.py), matching
the project's privacy-first framing: with this backend, no document
content or query text ever leaves your machine.
"""

from __future__ import annotations
import json
from typing import Iterator
import requests

from src.generation.llm_backend import LLMBackend, LLMResponse


class OllamaBackend(LLMBackend):
    def __init__(self, model: str = "llama3.1", host: str = "http://localhost:11434", timeout: int = 60):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> tuple[bool, str]:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            if resp.status_code != 200:
                return False, f"Ollama server responded with status {resp.status_code}"
            models = [m["name"] for m in resp.json().get("models", [])]
            # Ollama's model listing includes a ":latest" tag suffix by
            # default; a user who ran `ollama pull llama3.1` has a
            # model literally named "llama3.1:latest" in this list, so
            # we check both the exact name and the name+":latest" form
            # rather than requiring an exact match that would
            # incorrectly report "not available" for the common case.
            if not any(self.model == m or f"{self.model}:latest" == m for m in models):
                return False, f"Ollama is running, but model '{self.model}' isn't pulled. Run: ollama pull {self.model}. Available models: {models}"
            return True, "OK"
        except requests.exceptions.ConnectionError:
            return False, f"Could not connect to Ollama at {self.host}. Is Ollama running? (Start it with: ollama serve)"
        except requests.exceptions.Timeout:
            return False, f"Ollama at {self.host} did not respond within 3s"
        except Exception as e:
            return False, f"Unexpected error checking Ollama availability: {e}"

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt

        resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return LLMResponse(text=data.get("response", ""), model_used=self.model, backend_used="ollama")

    def generate_stream(self, prompt: str, system_prompt: str | None = None) -> Iterator[str]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
        }
        if system_prompt:
            payload["system"] = system_prompt

        with requests.post(f"{self.host}/api/generate", json=payload, timeout=self.timeout, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                # Ollama's streaming format is newline-delimited JSON,
                # one object per line, each with a "response" field
                # holding that chunk's text and a "done" boolean.
                chunk = json.loads(line)
                if chunk.get("response"):
                    yield chunk["response"]
                if chunk.get("done"):
                    break
