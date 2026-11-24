"""
Schema-aware handling for CSV/tabular data.

Produces TWO kinds of chunks per CSV:
1. A schema/summary chunk (columns, dtypes, sample rows, stats) —
   embeddable, used for dataset-level questions ("what's in this
   dataset").
2. One chunk PER ROW — each with its own unique, stable chunk_id.
   Needed for row-level citation accuracy: without this, a question
   like "which restaurants are in BTM" cites the SAME chunk_id for
   every restaurant mentioned, because they'd all come from one
   shared schema-summary blob. That's a real citation bug, not just
   a cosmetic one — a citation should point to where a specific fact
   came from, not "somewhere in this whole file." See docs/BUILD_LOG.md.

The live DataFrame is kept separately for ComputeExecutor's exact
numeric/lookup queries — row chunks are for semantic retrieval and
citation, not for computation.
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class CSVSchemaChunk:
    content: str
    file_path: str
    n_rows: int
    n_cols: int
    columns: list[str]
    chunk_id: str = ""

    def __post_init__(self):
        key = f"{self.file_path}:schema"
        self.chunk_id = hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass
class CSVRowChunk:
    """One indexable, individually-citable chunk per CSV row."""
    content: str
    file_path: str
    row_index: int
    chunk_id: str = field(init=False)

    def __post_init__(self):
        key = f"{self.file_path}:row:{self.row_index}"
        self.chunk_id = hashlib.sha256(key.encode()).hexdigest()[:16]


class CSVHandler:
    def __init__(self, max_sample_rows: int = 5, max_unique_for_categorical: int = 20):
        self.max_sample_rows = max_sample_rows
        self.max_unique_for_categorical = max_unique_for_categorical
        self._loaded_frames: dict[str, pd.DataFrame] = {}

    def load(self, file_path: str, max_row_chunks: int = 500) -> tuple[CSVSchemaChunk, list[CSVRowChunk], pd.DataFrame]:
        df = pd.read_csv(file_path)
        self._loaded_frames[file_path] = df
        schema_chunk = self._build_schema_chunk(file_path, df)
        row_chunks = self._build_row_chunks(file_path, df, max_row_chunks=max_row_chunks)
        return schema_chunk, row_chunks, df

    def get_dataframe(self, file_path: str) -> pd.DataFrame | None:
        return self._loaded_frames.get(file_path)

    def _build_row_chunks(self, file_path: str, df: pd.DataFrame, max_row_chunks: int) -> list[CSVRowChunk]:
        # Capped so a CSV with tens of thousands of rows doesn't
        # bloat the index. Exact AGGREGATE questions never need row
        # chunks — those go through ComputeExecutor against the full
        # DataFrame regardless of row count. This cap only affects
        # retrieval-based row lookup for very large files.
        n_rows = min(len(df), max_row_chunks)
        chunks = []
        for i in range(n_rows):
            row = df.iloc[i]
            fields = ", ".join(f"{col}: {row[col]}" for col in df.columns)
            content = f"Row {i} from {file_path} — {fields}"
            chunks.append(CSVRowChunk(content=content, file_path=file_path, row_index=i))
        return chunks

    def _build_schema_chunk(self, file_path: str, df: pd.DataFrame) -> CSVSchemaChunk:
        lines = [
            f"Dataset: {file_path}",
            f"Rows: {len(df)}, Columns: {len(df.columns)}",
            "",
            "Columns:",
        ]

        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = int(df[col].isna().sum())
            line = f"  - {col} ({dtype}, {null_count} nulls)"

            if pd.api.types.is_numeric_dtype(df[col]):
                desc = df[col].describe()
                line += f" — min={desc.get('min', 'NA'):.2f}, max={desc.get('max', 'NA'):.2f}, mean={desc.get('mean', 'NA'):.2f}"
            elif df[col].nunique() <= self.max_unique_for_categorical and df[col].nunique() < len(df):
                uniques = df[col].dropna().unique().tolist()
                line += f" — categories: {uniques}"
            elif df[col].nunique() == len(df):
                line += " — appears to be a unique identifier column"

            lines.append(line)

        lines.append("")
        lines.append(f"Sample rows (first {self.max_sample_rows}):")
        lines.append(df.head(self.max_sample_rows).to_string(index=False))

        content = "\n".join(lines)
        return CSVSchemaChunk(
            content=content,
            file_path=file_path,
            n_rows=len(df),
            n_cols=len(df.columns),
            columns=list(df.columns),
        )
