"""
Code-aware chunking using Python's AST module.

Why this exists:
Naive RAG systems chunk everything (PDFs, code, CSVs) with the same
fixed-size text splitter. That's fine for prose, but it's actively
wrong for code: a 500-character chunk boundary will slice a function
in half, separate a docstring from its function, or split a class
from its methods. When that half-function gets embedded and retrieved,
the LLM generates answers from incomplete logic.

This module parses Python source into its actual syntax tree and chunks
along real boundaries: functions, classes, and methods stay whole.
Each chunk keeps its enclosing context (e.g. "method inside class Foo")
as metadata, which we use later for citation and for retrieval routing.

This does NOT try to be a general multi-language parser (that's a much
bigger project). It handles Python natively via `ast`, and falls back
to a conservative line-based chunker for other languages, clearly
labeled as a fallback rather than pretending to be equally precise.
"""

from __future__ import annotations
import ast
import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CodeChunk:
    content: str
    chunk_type: str          # "function" | "class" | "module_level" | "fallback_block"
    name: str                # function/class name, or "module" for top-level code
    parent: str | None       # enclosing class name, if any
    file_path: str
    start_line: int
    end_line: int
    docstring: str | None = field(default=None)
    chunk_id: str = field(init=False)

    def __post_init__(self):
        # Stable id so re-ingesting the same chunk doesn't duplicate it
        # in the index — hash of file+name+line range, not content,
        # so trivial reformatting doesn't force a full re-embed.
        key = f"{self.file_path}:{self.name}:{self.start_line}-{self.end_line}"
        self.chunk_id = hashlib.sha256(key.encode()).hexdigest()[:16]


class PythonASTChunker:
    """Chunks Python source files along function/class boundaries."""

    def chunk_file(self, file_path: str, source: str | None = None) -> list[CodeChunk]:
        if source is None:
            source = Path(file_path).read_text(encoding="utf-8", errors="replace")

        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Not valid Python (or a different language with a .py-like
            # extension) — fall back rather than silently dropping the file.
            return self._fallback_chunk(file_path, source)

        source_lines = source.splitlines()
        chunks: list[CodeChunk] = []

        # Walk only top-level nodes, then recurse one level into classes
        # for their methods. We deliberately don't recurse into nested
        # functions past that — a method-level chunk is the right grain
        # for retrieval; going deeper produces chunks too small to be
        # individually meaningful.
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.append(self._function_chunk(node, source_lines, file_path, parent=None))
            elif isinstance(node, ast.ClassDef):
                chunks.append(self._class_chunk(node, source_lines, file_path))
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        chunks.append(self._function_chunk(sub, source_lines, file_path, parent=node.name))

        # Anything at module level that isn't a def/class (imports,
        # constants, top-level script logic) — group into one chunk so
        # it's still searchable, e.g. "what does this script's __main__
        # block do".
        module_level = self._extract_module_level(tree, source_lines, file_path)
        if module_level:
            chunks.append(module_level)

        return chunks

    def _function_chunk(self, node, source_lines, file_path, parent) -> CodeChunk:
        start, end = node.lineno, node.end_lineno
        content = "\n".join(source_lines[start - 1:end])
        docstring = ast.get_docstring(node)
        return CodeChunk(
            content=content,
            chunk_type="function",
            name=node.name,
            parent=parent,
            file_path=file_path,
            start_line=start,
            end_line=end,
            docstring=docstring,
        )

    def _class_chunk(self, node, source_lines, file_path) -> CodeChunk:
        # For the class itself, we chunk just the signature + docstring,
        # NOT the full body (that's redundant with the per-method chunks
        # we already emit) — this keeps the class-level chunk focused on
        # "what is this class for" rather than duplicating every method's
        # text into the index twice.
        start = node.lineno
        docstring = ast.get_docstring(node)
        sig_end = start
        for sub in node.body:
            if isinstance(sub, ast.Expr):  # the docstring expr itself
                sig_end = sub.end_lineno
                break
        content = "\n".join(source_lines[start - 1:sig_end])
        return CodeChunk(
            content=content,
            chunk_type="class",
            name=node.name,
            parent=None,
            file_path=file_path,
            start_line=start,
            end_line=node.end_lineno,
            docstring=docstring,
        )

    def _extract_module_level(self, tree, source_lines, file_path) -> CodeChunk | None:
        """Collect only the source lines that belong to top-level statements
        which are NOT function/class definitions (imports, constants, the
        __main__ block, etc). We can't just span from the first such node's
        line to the last, because that span would swallow any function/class
        defined in between. Instead we collect each qualifying node's own
        line range individually and stitch them together, skipping gaps."""
        top_nodes = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        if not top_nodes:
            return None

        pieces = []
        for n in top_nodes:
            start, end = n.lineno, n.end_lineno
            pieces.append("\n".join(source_lines[start - 1:end]))
        content = "\n".join(pieces)
        if not content.strip():
            return None

        return CodeChunk(
            content=content,
            chunk_type="module_level",
            name="module",
            parent=None,
            file_path=file_path,
            start_line=top_nodes[0].lineno,
            end_line=top_nodes[-1].end_lineno,
        )

    def _fallback_chunk(self, file_path: str, source: str, lines_per_chunk: int = 40) -> list[CodeChunk]:
        """Conservative fallback for non-Python or unparseable code: chunk
        by line count, but at least keep the chunk boundaries on blank
        lines where possible, instead of an arbitrary character count."""
        lines = source.splitlines()
        chunks = []
        i = 0
        while i < len(lines):
            end = min(i + lines_per_chunk, len(lines))
            # Nudge the boundary to the next blank line within a small
            # window, so we don't cut mid-statement if avoidable.
            while end < len(lines) and end < i + lines_per_chunk + 10 and lines[end].strip() != "":
                end += 1
            content = "\n".join(lines[i:end])
            if content.strip():
                chunks.append(CodeChunk(
                    content=content,
                    chunk_type="fallback_block",
                    name=f"block_{i}_{end}",
                    parent=None,
                    file_path=file_path,
                    start_line=i + 1,
                    end_line=end,
                ))
            i = end
        return chunks
