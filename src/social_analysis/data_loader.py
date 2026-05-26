"""Reusable data loading and saving helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


PathLike = str | Path


class DataLoader:
    """Small wrapper around common notebook data I/O operations."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    def load_csv(self, path: PathLike, **kwargs: Any) -> pd.DataFrame:
        """Load a CSV file into a DataFrame."""
        return pd.read_csv(Path(path), **kwargs)

    def save_dataframe(
        self,
        df: pd.DataFrame,
        path: PathLike,
        index: bool = False,
        **kwargs: Any,
    ) -> Path:
        """Save a DataFrame as CSV, creating the parent directory first."""
        output_path = Path(path)
        self.ensure_directory(output_path.parent)
        df.to_csv(output_path, index=index, **kwargs)
        return output_path

    def ensure_directory(self, path: PathLike) -> Path:
        """Create a directory if it does not already exist."""
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def load_sqlite_table(self, db_path: PathLike, table_name: str) -> pd.DataFrame:
        """Load a full SQLite table into a DataFrame."""
        with sqlite3.connect(Path(db_path)) as conn:
            return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
