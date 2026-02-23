"""
src/data/ingestion/batch_ingester.py
--------------------------------------
Batch data ingestion from local files and cloud storage (S3/GCS).
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
import pandas as pd
from src.core.interfaces import BaseDataIngester

logger = logging.getLogger(__name__)

REQUIRED_SCHEMA = {
    "transaction_id": "object",
    "user_id": "object",
    "amount": "float64",
    "currency": "object",
    "merchant_id": "object",
    "merchant_category": "object",
    "timestamp": "datetime64[ns]",
    "is_fraud": "int64",
}

class BatchIngester(BaseDataIngester):
    """Ingest transaction data from Parquet or CSV files."""

    def ingest(self, source: str, **kwargs: Any) -> pd.DataFrame:
        path = Path(source)
        if not path.exists():
            msg = f"Source file not found: {source}"
            raise FileNotFoundError(msg)
        if path.suffix == ".parquet":
            df = pd.read_parquet(source, **kwargs)
        elif path.suffix in {".csv", ".tsv"}:
            df = pd.read_csv(source, **kwargs)
        else:
            msg = f"Unsupported format: {path.suffix}"
            raise ValueError(msg)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        logger.info("Ingested %d rows from %s", len(df), source)
        return df

    def validate_schema(self, df: pd.DataFrame) -> bool:
        required = set(REQUIRED_SCHEMA.keys())
        missing = required - set(df.columns)
        if missing:
            logger.error("Missing columns: %s", missing)
            return False
        return True
