"""
src/utils/io.py
----------------
File I/O helper utilities for reading/writing ML artifacts.
Supports Parquet, CSV, JSON, YAML, and joblib formats.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def read_dataframe(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """
    Read a DataFrame from Parquet or CSV.

    Args:
        path: File path (.parquet, .csv, .tsv)
        **kwargs: Passed to pd.read_parquet or pd.read_csv

    Returns:
        Loaded DataFrame
    """
    p = Path(path)
    if not p.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)

    if p.suffix == ".parquet":
        df = pd.read_parquet(p, **kwargs)
    elif p.suffix in {".csv", ".tsv"}:
        sep = "\t" if p.suffix == ".tsv" else ","
        df = pd.read_csv(p, sep=sep, **kwargs)
    else:
        msg = f"Unsupported format: {p.suffix}. Use .parquet or .csv"
        raise ValueError(msg)

    logger.info("Read %d × %d from %s", df.shape[0], df.shape[1], path)
    return df


def write_dataframe(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    """
    Write a DataFrame to Parquet or CSV.

    Args:
        df: DataFrame to write
        path: Destination path (.parquet, .csv)
        **kwargs: Passed to underlying writer
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.suffix == ".parquet":
        df.to_parquet(p, index=False, engine="pyarrow", compression="snappy", **kwargs)
    elif p.suffix in {".csv", ".tsv"}:
        sep = "\t" if p.suffix == ".tsv" else ","
        df.to_csv(p, index=False, sep=sep, **kwargs)
    else:
        msg = f"Unsupported format: {p.suffix}"
        raise ValueError(msg)

    size_mb = p.stat().st_size / 1024 / 1024
    logger.info("Wrote %d rows → %s (%.1f MB)", len(df), path, size_mb)


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file and return as dict."""
    p = Path(path)
    if not p.exists():
        msg = f"JSON file not found: {path}"
        raise FileNotFoundError(msg)
    with p.open() as f:
        return json.load(f)


def write_json(data: dict[str, Any] | list[Any], path: str | Path, indent: int = 2) -> None:
    """Write data to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(data, f, indent=indent, default=str)
    logger.info("Wrote JSON → %s", path)


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file."""
    p = Path(path)
    if not p.exists():
        msg = f"YAML file not found: {path}"
        raise FileNotFoundError(msg)
    with p.open() as f:
        return yaml.safe_load(f) or {}


def write_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Write data to a YAML file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    logger.info("Wrote YAML → %s", path)


def save_artifact(obj: Any, path: str | Path) -> None:
    """Serialize any Python object with joblib."""
    import joblib

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, p)
    size_kb = p.stat().st_size / 1024
    logger.info("Artifact saved → %s (%.1f KB)", path, size_kb)


def load_artifact(path: str | Path) -> Any:
    """Deserialize a joblib artifact."""
    import joblib

    p = Path(path)
    if not p.exists():
        msg = f"Artifact not found: {path}"
        raise FileNotFoundError(msg)
    return joblib.load(p)


def ensure_dirs(*paths: str | Path) -> None:
    """Create directories if they don't exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def file_size_mb(path: str | Path) -> float:
    """Return file size in MB."""
    p = Path(path)
    if not p.exists():
        return 0.0
    return p.stat().st_size / 1024 / 1024
