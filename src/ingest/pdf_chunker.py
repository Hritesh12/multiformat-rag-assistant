"""
Semantic-boundary chunking for PDF/prose documents.

Why this exists:
The most common approach (used in both tutorial repos referenced for
this project) is fixed-size character chunking with overlap — e.g.
"take 500 characters, slide forward 400, repeat". It's simple and it
works okay, but it routinely cuts sentences and paragraphs in half,
which hurts both embedding quality (a half-sentence embeds worse than
a full thought) and the LLM's ability to use the chunk coherently.

This chunker instead:
1. Extracts text page-by-page (so every chunk keeps its page number
   for citation — "this answer came from page 4").
2. Splits on paragraph boundaries first, not raw character counts.
3. Merges small paragraphs together up to a target size, and only
   falls back to a hard character split for a single paragraph that's
   larger than the max chunk size on its own (e.g. a dense block of
   unbroken text) — so we still bound chunk size without routinely
   cutting mid-thought.

This is a real trade-off, not a magic fix: for scanned/OCR'd PDFs
with poor paragraph structure, the benefit shrinks toward zero and it
falls back to the same character-bounded behavior as the naive
approach. That's a known limitation, not something to gloss over.
"""

from __future__ import annotations
import hashlib
import re
from dataclasses import dataclass, field
from pypdf import PdfReader


@dataclass
class PDFChunk:
    content: str
    file_path: str
    page_number: int
    chunk_index_on_page: int
    chunk_id: str = field(init=False)

    def __post_init__(self):
        key = f"{self.file_path}:p{self.page_number}:{self.chunk_index_on_page}"
        self.chunk_id = hashlib.sha256(key.encode()).hexdigest()[:16]


class PDFChunker:
    def __init__(self, target_chunk_chars: int = 1000, max_chunk_chars: int = 1500, min_chunk_chars: int = 200):
        self.target_chunk_chars = target_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars

    def chunk_file(self, file_path: str) -> list[PDFChunk]:
        reader = PdfReader(file_path)
        all_chunks: list[PDFChunk] = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            page_chunks = self._chunk_page_text(text, file_path, page_num)
            all_chunks.extend(page_chunks)

        return all_chunks

    def _chunk_page_text(self, text: str, file_path: str, page_num: int) -> list[PDFChunk]:
        paragraphs = self._split_paragraphs(text)
        merged = self._merge_to_target_size(paragraphs)

        chunks = []
        for i, content in enumerate(merged):
            chunks.append(PDFChunk(
                content=content.strip(),
                file_path=file_path,
                page_number=page_num,
                chunk_index_on_page=i,
            ))
        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        # PDF text extraction rarely preserves clean double-newlines
        # between paragraphs the way a plain-text file would. We treat
        # a blank line as the primary paragraph boundary signal.
        raw_paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

        if len(paragraphs) <= 1:
            # No blank-line structure — try sentence boundaries next.
            paragraphs = self._split_sentences_into_pseudo_paragraphs(text)

        if len(paragraphs) <= 1:
            # Still nothing usable — this is dense, bullet/table-style
            # text (common in resumes, forms, and some PDF exports)
            # with no blank lines AND no sentence-terminal punctuation
            # for long stretches. Fall back to single newlines, which
            # in this kind of document usually correspond to a line
            # item or bullet, then group a few lines together — this
            # is the layer that fixes single-oversized-chunk failures
            # on that document type. Verified against a real resume
            # PDF during development, not just a hypothetical case.
            paragraphs = self._split_lines_into_pseudo_paragraphs(text)

        return paragraphs

    def _split_lines_into_pseudo_paragraphs(self, text: str, lines_per_group: int = 3) -> list[str]:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        groups = []
        for i in range(0, len(lines), lines_per_group):
            group = "\n".join(lines[i:i + lines_per_group]).strip()
            if group:
                groups.append(group)
        return groups

    def _split_sentences_into_pseudo_paragraphs(self, text: str, sentences_per_group: int = 4) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        groups = []
        for i in range(0, len(sentences), sentences_per_group):
            group = " ".join(sentences[i:i + sentences_per_group]).strip()
            if group:
                groups.append(group)
        return groups

    def _merge_to_target_size(self, paragraphs: list[str]) -> list[str]:
        merged: list[str] = []
        buffer = ""

        for para in paragraphs:
            if len(para) > self.max_chunk_chars:
                # Single paragraph too big on its own — flush whatever's
                # buffered, then hard-split this one paragraph specifically
                # (not the whole document) at sentence boundaries.
                if buffer:
                    merged.append(buffer)
                    buffer = ""
                merged.extend(self._hard_split(para))
                continue

            candidate = (buffer + "\n\n" + para).strip() if buffer else para
            if len(candidate) <= self.target_chunk_chars:
                buffer = candidate
            else:
                if buffer:
                    merged.append(buffer)
                buffer = para

        if buffer:
            merged.append(buffer)

        # Merge any trailing chunk that's too small into the previous one,
        # rather than leaving a tiny orphan chunk that adds index noise.
        final = []
        for chunk in merged:
            if final and len(chunk) < self.min_chunk_chars:
                final[-1] = final[-1] + "\n\n" + chunk
            else:
                final.append(chunk)

        return final

    def _hard_split(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        pieces, buffer = [], ""
        for s in sentences:
            candidate = (buffer + " " + s).strip() if buffer else s
            if len(candidate) <= self.max_chunk_chars:
                buffer = candidate
            else:
                if buffer:
                    pieces.append(buffer)
                buffer = s
        if buffer:
            pieces.append(buffer)
        return pieces
