"""
Query router: decides HOW to answer a query, not just WHAT to retrieve.

Why this exists:
A naive RAG system treats every query identically: embed it, do a
similarity search, stuff the top-k chunks into a prompt, generate.
That's fine for "what does the report say about X" — but it's the
wrong approach for "what's the average cost in the restaurant
dataset", because that's not a search problem, it's a computation
problem. No amount of semantic retrieval will reliably produce a
correct numeric aggregate; you either compute it exactly with pandas,
or you're asking an LLM to eyeball arithmetic over retrieved text
snippets and hope it's right.

This router classifies each query into one of three strategies BEFORE
retrieval happens:

  - COMPUTE:  route to CSVHandler + pandas for an exact numeric answer
  - CODE:     route to hybrid search but weighted toward code chunks
  - SEMANTIC: standard hybrid search across all indexed content

The classification is intentionally rule-based (keyword/pattern
matching) rather than an LLM call. Trade-off: fast, free, fully
deterministic, and its failure modes are enumerable — but it will
misroute a query phrased in a way the rules don't anticipate. See
docs/BUILD_LOG.md for real misrouting bugs found via testing and how
they were fixed.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum


class QueryStrategy(Enum):
    COMPUTE = "compute"
    CODE = "code"
    SEMANTIC = "semantic"


@dataclass
class RoutingDecision:
    strategy: QueryStrategy
    confidence: str
    matched_pattern: str | None
    reasoning: str


_COMPUTE_PATTERNS = [
    r"\baverage\b", r"\bmean\b", r"\bmedian\b", r"\bsum\b", r"\btotal\b",
    r"\bmax(imum)?\b", r"\bmin(imum)?\b", r"\bhow many\b", r"\bcount\b",
    r"\bhighest\b", r"\blowest\b", r"\btop \d+\b", r"\bbottom \d+\b",
    r"\bcorrelat", r"\bstandard deviation\b", r"\bpercentage\b",
    r"\bgroup(ed)? by\b", r"\bcompare .* (rows|records|entries)\b",
]

_CODE_PATTERNS = [
    r"\bfunction\b", r"\bclass\b", r"\bmethod\b", r"\bimplement", r"\bdef \b",
    r"\bhow does .* (work|handle|process)\b", r"\bwhat does .* do\b",
    r"\bbug\b", r"\berror\b", r"\bexception\b", r"\brefactor\b",
    r"\bparameter", r"\bargument", r"\breturn(s)? \b", r"\bimport\b",
]

# Patterns where code is clearly the SUBJECT of the sentence, not just
# an incidental word — checked before compute patterns so a sentence
# with a numeric-sounding word ("maximum", "count") but a clear code
# subject still routes to CODE. See docs/BUILD_LOG.md bugs #6, #10.
_CODE_SUBJECT_PATTERNS = [
    r"\bin (this|the) function\b", r"\bin (this|the) class\b",
    r"\bin (this|the) method\b", r"\bin (this|the) code(base)?\b",
    r"\bthis function\b", r"\bthis class\b", r"\bthis method\b",
    r"\bof (this|the) function\b", r"\bof (this|the) class\b",
    r"\w+\(\)",                          # any name followed by () — e.g. "average()"
    r"\bthe (\w+) function\b",           # "the average function"
]


class QueryRouter:
    def __init__(self, csv_files_indexed: list[str] | None = None, csv_columns: list[str] | None = None, csv_categorical_values: list[str] | None = None):
        self.csv_files_indexed = csv_files_indexed or []
        self.csv_columns = set(c.lower().replace("_", " ") for c in (csv_columns or []))
        # csv_categorical_values: actual VALUES found in categorical
        # columns (e.g. "BTM", "Italian"), not just column names.
        # Needed because "How many restaurants are in BTM?" mentions
        # "BTM" (a value) but no column name at all — checking columns
        # alone would wrongly reject this as irrelevant. Bug #11/#14.
        self.csv_categorical_values = set(v.lower() for v in (csv_categorical_values or []))

    def route(self, query: str) -> RoutingDecision:
        query_lower = query.lower()

        code_subject_match = self._first_match(query_lower, _CODE_SUBJECT_PATTERNS)
        if code_subject_match:
            return RoutingDecision(
                strategy=QueryStrategy.CODE,
                confidence="high",
                matched_pattern=code_subject_match,
                reasoning=f"Query has a clear code subject ('{code_subject_match}') — routing to code even though it may also contain incidental numeric-sounding words.",
            )

        compute_match = self._first_match(query_lower, _COMPUTE_PATTERNS)
        if compute_match and self.csv_files_indexed:
            # "how many" / "count" are generic enough to match
            # questions unrelated to the indexed table ("how many
            # branches did X have" is a document fact, not a table
            # question) — require column OR categorical-value
            # relevance before committing to COMPUTE for just these
            # two patterns. Other patterns are specific enough alone.
            generic_patterns = {"how many", "count"}
            if compute_match in generic_patterns and (self.csv_columns or self.csv_categorical_values):
                has_column_relevance = any(col_word in query_lower for col_word in self.csv_columns)
                has_value_relevance = any(val in query_lower for val in self.csv_categorical_values)
                if not (has_column_relevance or has_value_relevance):
                    return RoutingDecision(
                        strategy=QueryStrategy.SEMANTIC,
                        confidence="medium",
                        matched_pattern=compute_match,
                        reasoning=f"Query matches '{compute_match}' but doesn't mention any column or value from the indexed table — likely a plain document fact, not a table computation. Using semantic search instead.",
                    )
            return RoutingDecision(
                strategy=QueryStrategy.COMPUTE,
                confidence="high",
                matched_pattern=compute_match,
                reasoning=f"Query matches a numeric/aggregate pattern ('{compute_match}') and tabular data is available — routing to exact computation instead of semantic retrieval.",
            )
        if compute_match and not self.csv_files_indexed:
            return RoutingDecision(
                strategy=QueryStrategy.SEMANTIC,
                confidence="low",
                matched_pattern=compute_match,
                reasoning=f"Query looks like it wants a computation ('{compute_match}'), but no tabular data is indexed — falling back to semantic search, which may not answer this well.",
            )

        code_match = self._first_match(query_lower, _CODE_PATTERNS)
        if code_match:
            return RoutingDecision(
                strategy=QueryStrategy.CODE,
                confidence="medium",
                matched_pattern=code_match,
                reasoning=f"Query matches a code-related pattern ('{code_match}') — weighting retrieval toward code chunks.",
            )

        return RoutingDecision(
            strategy=QueryStrategy.SEMANTIC,
            confidence="medium",
            matched_pattern=None,
            reasoning="No compute or code pattern matched — using standard hybrid semantic search across all indexed content.",
        )

    def _first_match(self, text: str, patterns: list[str]) -> str | None:
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                return m.group(0)
        return None
