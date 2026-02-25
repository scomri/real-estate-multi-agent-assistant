"""
Data-loading utilities for the Cortex Real Estate AI agent.

Both public functions are memoised with "functools.lru_cache" so the
Parquet file is read from disk only once per interpreter session, 
regardless of how many graph nodes call them.

The dataset path is resolved from the "DATA_PATH" environment variable,
defaulting to "data/cortex.parquet" relative to the working directory.
"""

import logging
import os
from functools import lru_cache

import pandas as pd


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce dataset columns to standard types and formats.

    - year  : stored as str ('2024') → cast to int (matches Pydantic Optional[int])
    - month : stored as 'YYYY-Mnn'  → convert to ISO 'YYYY-MM' (e.g. '2024-M01' → '2024-01')

    Centralising format normalisation here keeps all pipeline nodes and prompts format-agnostic: 
    only this function ever needs to change when a new dataset ships with a different storage convention.
    """
    df = df.copy()
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    if "month" in df.columns:
        df["month"] = df["month"].str.replace(r"-M(\d)", r"-\1", regex=True)
    return df

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    """
    Load and cache the Cortex real estate dataset.

    Reads the Parquet file located at the path specified by the "DATA_PATH"
    environment variable (default: "data/cortex.parquet").

    Returns
    -------
    pd.DataFrame
        The full dataset as a DataFrame.  
        Subsequent calls return the cached result without re-reading the file.
    """
    path = os.getenv("DATA_PATH", "data/cortex.parquet")
    logger.info("Loading data from: %s", path)
    df = pd.read_parquet(path)
    df = _normalize(df)
    logger.info("Data loaded — shape: %s", df.shape)
    return df


@lru_cache(maxsize=1)
def get_data_dict() -> str:
    """
    Build a compact schema string injected into LLM prompts.
    For low-cardinality columns, it lists the unique values.
    For high-cardinality columns, it lists the number of unique values.

    Returns
    -------
    str
        A multi-line string describing the columns and their values.
    """
    logger.debug("Building schema string for LLM prompts")
    df = load_data()
    lines = []
    for col in df.columns:
        nuniq = df[col].nunique()
        if nuniq <= 40:
            vals = sorted(df[col].dropna().astype(str).unique().tolist())
            lines.append(f"- {col}: {vals}")
        else:
            lines.append(f"- {col}: (high cardinality, {nuniq} unique values)")
    return "\n".join(lines)
