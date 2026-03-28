"""
src/data/validation/validator.py
---------------------------------
Schema and quality validation for incoming transaction data.
Runs lightweight checks before feature engineering to catch bad data early.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "transaction_id",
    "user_id",
    "amount",
    "currency",
    "merchant_id",
    "merchant_category",
    "timestamp",
]

AMOUNT_MIN = 0.01
AMOUNT_MAX = 1_000_000.0


@dataclass
class ValidationReport:
    passed: bool
    n_rows: int
    n_rows_valid: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def invalid_row_count(self) -> int:
        return self.n_rows - self.n_rows_valid


class TransactionValidator:
    """
    Lightweight, fast data quality checks for transaction DataFrames.

    Checks:
    - Required columns present
    - No null transaction_id / user_id
    - Amount in valid range [0.01, 1M]
    - Timestamp parseable
    - Duplicate transaction IDs
    - Fraud rate sanity (training only)

    Usage:
        validator = TransactionValidator()
        report = validator.validate(df)
        if not report.passed:
            raise DataValidationError(report.errors)
    """

    def validate(
        self,
        df: pd.DataFrame,
        *,
        training: bool = False,
        fail_fast: bool = False,
    ) -> ValidationReport:
        """
        Run all validation checks.

        Args:
            df: Raw transaction DataFrame
            training: If True, validates 'is_fraud' column and fraud rate
            fail_fast: If True, raise on first error instead of collecting all

        Returns:
            ValidationReport with passed status and any errors/warnings
        """
        errors: list[str] = []
        warnings: list[str] = []
        n_rows = len(df)

        # 1. Required columns
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            if fail_fast:
                return ValidationReport(passed=False, n_rows=n_rows, n_rows_valid=0, errors=errors)

        if training and "is_fraud" not in df.columns:
            errors.append("Training data missing 'is_fraud' target column")

        # 2. Empty DataFrame
        if n_rows == 0:
            errors.append("DataFrame is empty")
            return ValidationReport(passed=False, n_rows=0, n_rows_valid=0, errors=errors)

        # 3. Null checks on key identifiers
        if "transaction_id" in df.columns:
            null_txn = df["transaction_id"].isna().sum()
            if null_txn > 0:
                errors.append(f"{null_txn} rows have null transaction_id")

        if "user_id" in df.columns:
            null_user = df["user_id"].isna().sum()
            if null_user > 0:
                errors.append(f"{null_user} rows have null user_id")

        # 4. Amount range
        if "amount" in df.columns:
            invalid_amounts = ((df["amount"] < AMOUNT_MIN) | (df["amount"] > AMOUNT_MAX)).sum()
            if invalid_amounts > 0:
                errors.append(
                    f"{invalid_amounts} rows have amount outside valid range "
                    f"[{AMOUNT_MIN}, {AMOUNT_MAX}]"
                )
            null_amounts = df["amount"].isna().sum()
            if null_amounts > 0:
                errors.append(f"{null_amounts} rows have null amount")

        # 5. Duplicate transaction IDs
        if "transaction_id" in df.columns:
            dupes = df["transaction_id"].duplicated().sum()
            if dupes > 0:
                warnings.append(f"{dupes} duplicate transaction_id values found")

        # 6. Timestamp parseable
        if "timestamp" in df.columns:
            try:
                pd.to_datetime(df["timestamp"])
            except Exception as exc:
                errors.append(f"Timestamp column not parseable: {exc}")

        # 7. Fraud rate sanity (training only)
        if training and "is_fraud" in df.columns:
            fraud_rate = df["is_fraud"].mean()
            if fraud_rate < 0.001:
                warnings.append(
                    f"Very low fraud rate: {fraud_rate:.4%}. " "Model training may be unreliable."
                )
            elif fraud_rate > 0.20:
                warnings.append(
                    f"Unusually high fraud rate: {fraud_rate:.2%}. "
                    "Verify data is not pre-filtered."
                )
            null_labels = df["is_fraud"].isna().sum()
            if null_labels > 0:
                errors.append(f"{null_labels} rows have null is_fraud label")

        passed = len(errors) == 0
        n_valid = n_rows - sum(
            [
                df["transaction_id"].isna().sum() if "transaction_id" in df.columns else 0,
                (
                    ((df["amount"] < AMOUNT_MIN) | (df["amount"] > AMOUNT_MAX)).sum()
                    if "amount" in df.columns
                    else 0
                ),
            ]
        )

        report = ValidationReport(
            passed=passed,
            n_rows=n_rows,
            n_rows_valid=max(0, int(n_valid)),
            errors=errors,
            warnings=warnings,
        )

        if errors:
            logger.error("Validation failed: %s", "; ".join(errors))
        if warnings:
            logger.warning("Validation warnings: %s", "; ".join(warnings))
        if passed:
            logger.info(
                "Validation passed: %d rows, fraud_rate=%.2f%%",
                n_rows,
                df["is_fraud"].mean() * 100 if training and "is_fraud" in df.columns else 0,
            )

        return report

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Quick schema check — returns True/False without full report."""
        return self.validate(df, fail_fast=True).passed
