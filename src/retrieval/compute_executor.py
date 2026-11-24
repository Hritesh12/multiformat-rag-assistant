"""
Executes COMPUTE-routed queries against indexed CSV data using pandas.

Deliberately does NOT ask an LLM to generate a pandas expression and
eval()/exec() it — that's a security risk and unreliable. Instead,
this maps queries to a small, safe set of pandas operations via a
constrained parser, and reports clearly when a query doesn't fit.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
import pandas as pd


@dataclass
class ComputeResult:
    success: bool
    answer: str
    operation_used: str | None = None
    column_used: str | None = None
    error: str | None = None


class ComputeExecutor:
    def __init__(self, dataframes: dict[str, pd.DataFrame]):
        self.dataframes = dataframes

    def execute(self, query: str) -> ComputeResult:
        if not self.dataframes:
            return ComputeResult(success=False, answer="", error="No CSV data is currently indexed.")

        query_lower = query.lower()

        # "how many" / "count" phrasing is an unambiguous signal that
        # the question wants a count across matches, never a single-
        # row field lookup — so row-lookup is skipped whenever that
        # phrasing is present, even if an entity value also appears.
        # Bug #14: "How many restaurants are in BTM?" was being
        # wrongly intercepted as a row lookup for "BTM".
        is_count_style = bool(re.search(r"\bhow many\b|\bcount\b", query_lower))
        if not is_count_style:
            for file_path, df in self.dataframes.items():
                lookup_result = self._try_row_lookup(query_lower, df, file_path)
                if lookup_result is not None:
                    return lookup_result

        operation = self._detect_operation(query_lower)
        if operation is None:
            return ComputeResult(
                success=False, answer="",
                error="Could not map this query to a supported computation (mean/max/min/sum/count/groupby). Try rephrasing, e.g. 'what is the average <column>?'",
            )

        for file_path, df in self.dataframes.items():
            groupby_col = self._match_groupby_column(query_lower, df, exclude=None)
            column = self._match_column(query_lower, df, exclude={groupby_col} if groupby_col else set())
            if column:
                return self._run_operation(operation, df, column, file_path, query_lower, groupby_col=groupby_col)

        return ComputeResult(
            success=False, answer="",
            error=f"Detected a '{operation}' computation, but couldn't match any indexed CSV's column names to your query. Available columns: {self._all_columns_summary()}",
        )

    def _try_row_lookup(self, query_lower: str, df: pd.DataFrame, file_path: str) -> ComputeResult | None:
        """Checks whether the query names a specific row (by an
        entity-like column's value — e.g. a restaurant name) and, if
        so, looks up a requested field from that one row. Bug #12:
        'average cost of Spice Route' isn't a column-wide average,
        it's a single-row lookup where 'average' is part of the
        column's natural-language name."""
        entity_col, matched_value, row = self._match_entity_row(query_lower, df)
        if entity_col is None:
            return None

        target_col = self._match_column(query_lower, df, exclude={entity_col})
        if target_col is None:
            return ComputeResult(
                success=False, answer="",
                error=f"Found '{matched_value}' in {file_path}, but couldn't tell which field you want. Available columns: {list(df.columns)}",
            )

        value = row[target_col]
        value_str = f"{value:,.2f}" if isinstance(value, float) else str(value)
        answer = f"{target_col} for '{matched_value}' in {file_path}: {value_str}"
        return ComputeResult(success=True, answer=answer, operation_used="row_lookup", column_used=target_col)

    def _match_entity_row(self, query_lower: str, df: pd.DataFrame):
        """Finds a column that looks like an entity/name column (a
        non-numeric column with mostly-unique string values) whose
        VALUE appears in the query text."""
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            if df[col].nunique() < max(3, len(df) * 0.5):
                continue
            for idx, value in df[col].items():
                value_str = str(value).lower()
                if len(value_str) >= 3 and value_str in query_lower:
                    return col, value, df.loc[idx]
        return None, None, None

    def _detect_operation(self, query_lower: str) -> str | None:
        if re.search(r"\baverage\b|\bmean\b", query_lower):
            return "mean"
        if re.search(r"\bmax(imum)?\b|\bhighest\b", query_lower):
            return "max"
        if re.search(r"\bmin(imum)?\b|\blowest\b", query_lower):
            return "min"
        if re.search(r"\bsum\b|\btotal\b", query_lower):
            return "sum"
        if re.search(r"\bhow many\b|\bcount\b", query_lower):
            return "count"
        if re.search(r"\bmedian\b", query_lower):
            return "median"
        return None

    def _match_column(self, query_lower: str, df: pd.DataFrame, exclude: set[str] | None = None) -> str | None:
        exclude = exclude or set()
        candidates = [c for c in df.columns if c not in exclude]

        def is_deprioritized(col):
            if pd.api.types.is_numeric_dtype(df[col]):
                return False
            return df[col].nunique() == len(df)

        for col in sorted(candidates, key=is_deprioritized):
            col_variants = [col.lower(), col.lower().replace("_", " ")]
            if any(v in query_lower for v in col_variants):
                return col

        # Fallback: significant-word matching, not requiring the
        # exact phrase. Bug #15: 'average_cost_inr' doesn't literally
        # appear as a phrase in 'average cost of Spice Route'.
        _GENERIC_SUFFIXES = {"inr", "usd", "id", "value", "amount", "num", "count"}
        for col in sorted(candidates, key=is_deprioritized):
            words = [w for w in col.lower().split("_") if w not in _GENERIC_SUFFIXES and len(w) > 2]
            if words and all(w in query_lower for w in words):
                return col
        return None

    def _run_operation(self, operation: str, df: pd.DataFrame, column: str, file_path: str, query_lower: str, groupby_col: str | None = None) -> ComputeResult:
        try:
            if operation == "count" and not pd.api.types.is_numeric_dtype(df[column]):
                # Check for a specific VALUE to filter by (e.g. "BTM")
                # before falling back to a full breakdown. Bug #17:
                # "How many restaurants are in BTM?" was returning a
                # count-of-1 for every individual restaurant_id.
                filter_col, filter_value = self._match_filter_value(query_lower, df, exclude={column})
                if filter_col is not None:
                    matching_count = int((df[filter_col] == filter_value).sum())
                    answer = f"{matching_count} row(s) in {file_path} have {filter_col} = '{filter_value}'"
                    return ComputeResult(success=True, answer=answer, operation_used="filtered_count", column_used=filter_col)

                if groupby_col:
                    result = df.groupby(groupby_col)[column].count()
                    answer = self._format_series(result, f"Count of {column} by {groupby_col}")
                    return ComputeResult(success=True, answer=answer, operation_used="count_groupby", column_used=column)
                result = df[column].value_counts()
                answer = self._format_series(result, f"Count of each value in {column}")
                return ComputeResult(success=True, answer=answer, operation_used="value_counts", column_used=column)

            if not pd.api.types.is_numeric_dtype(df[column]) and operation != "count":
                return ComputeResult(
                    success=False, answer="",
                    error=f"'{column}' is not numeric, so {operation} isn't a valid operation on it. It contains categories: {df[column].dropna().unique().tolist()[:10]}",
                )

            if groupby_col:
                grouped = getattr(df.groupby(groupby_col)[column], operation)()
                answer = self._format_series(grouped, f"{operation} of {column} by {groupby_col}")
                return ComputeResult(success=True, answer=answer, operation_used=f"{operation}_groupby", column_used=column)

            value = getattr(df[column], operation)()

            # For max/min, check if the query asks WHICH row that
            # value belongs to. Bug #13: "Which restaurant has the
            # highest rating?" returned only the number.
            entity_label = None
            if operation in ("max", "min") and re.search(r"\bwhich\b|\bwhat\b.*\b(restaurant|item|row|record|entry)\b", query_lower):
                entity_col = self._find_entity_column(df, exclude={column})
                if entity_col is not None:
                    matching_rows = df[df[column] == value]
                    if len(matching_rows) > 0:
                        entity_label = matching_rows.iloc[0][entity_col]

            if entity_label is not None:
                value_str = f"{value:,.2f}" if isinstance(value, float) else str(value)
                answer = f"{entity_label} has the {operation} {column} in {file_path}: {value_str}"
                return ComputeResult(success=True, answer=answer, operation_used=f"{operation}_with_entity", column_used=column)

            answer = f"{operation.capitalize()} of '{column}' in {file_path}: {value:,.2f}" if isinstance(value, float) else f"{operation.capitalize()} of '{column}' in {file_path}: {value}"
            return ComputeResult(success=True, answer=answer, operation_used=operation, column_used=column)

        except Exception as e:
            return ComputeResult(success=False, answer="", error=f"Computation failed: {e}")

    def _find_entity_column(self, df: pd.DataFrame, exclude: set[str] | None = None) -> str | None:
        """Finds the most likely 'name/identity' column. Bug #16:
        prefer a column literally called 'name' over an equally-
        unique ID column (e.g. 'restaurant_id')."""
        exclude = exclude or set()
        candidates = [c for c in df.columns if c not in exclude and not pd.api.types.is_numeric_dtype(df[c])]
        if not candidates:
            return None
        name_like = [c for c in candidates if "name" in c.lower()]
        if name_like:
            return max(name_like, key=lambda c: df[c].nunique())
        return max(candidates, key=lambda c: df[c].nunique())

    def _match_filter_value(self, query_lower: str, df: pd.DataFrame, exclude: set[str] | None = None):
        """Finds a categorical column whose specific VALUE (not
        column name) appears in the query — e.g. 'BTM' as a value of
        `zone`. Mirror image of _match_entity_row: wants LOW
        cardinality (a repeated category), not high (a unique name)."""
        exclude = exclude or set()
        for col in df.columns:
            if col in exclude or pd.api.types.is_numeric_dtype(df[col]):
                continue
            if df[col].nunique() >= max(2, len(df) * 0.8):
                continue
            for value in df[col].dropna().unique():
                value_str = str(value).lower()
                if len(value_str) >= 2 and value_str in query_lower:
                    return col, value
        return None, None

    def _match_groupby_column(self, query_lower: str, df: pd.DataFrame, exclude: str | None) -> str | None:
        by_match = re.search(r"\bby (\w+)\b|\bper (\w+)\b", query_lower)
        candidate = None
        if by_match:
            candidate = by_match.group(1) or by_match.group(2)
        for col in df.columns:
            if col == exclude:
                continue
            if candidate and candidate in col.lower():
                return col
        return None

    def _format_series(self, series: pd.Series, label: str) -> str:
        lines = [label + ":"]
        for idx, val in series.items():
            val_str = f"{val:,.2f}" if isinstance(val, float) else str(val)
            lines.append(f"  {idx}: {val_str}")
        return "\n".join(lines)

    def _all_columns_summary(self) -> str:
        parts = []
        for file_path, df in self.dataframes.items():
            parts.append(f"{file_path}: {list(df.columns)}")
        return " | ".join(parts)
