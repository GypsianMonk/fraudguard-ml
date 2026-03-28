#!/usr/bin/env python3
"""
scripts/compare_models.py
--------------------------
Phase 4 of the ML roadmap: train LR → RF → XGBoost and compare properly.
Uses AUC-PR and recall as primary metrics — never accuracy.

Usage:
    python scripts/compare_models.py --data data/raw/transactions.parquet
    python scripts/compare_models.py --data data/raw/transactions_small.parquet --output reports/model_comparison.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.engineer import FraudFeatureEngineer
from src.utils.logging import configure_logging

configure_logging(level="INFO", format_="console")
logger = logging.getLogger(__name__)


MODELS = {
    "logistic_regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    C=0.1,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    ),
    "xgboost": xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    ),
}


def cross_validate_model(
    name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = 5,
) -> dict:
    """5-fold stratified CV, returning mean ± std of AUC-PR, AUC-ROC, recall."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    auc_prs, auc_rocs, recalls_at_80 = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        neg = (y_tr == 0).sum()
        pos = (y_tr == 1).sum()

        # XGBoost needs scale_pos_weight set per-fold
        if name == "xgboost":
            if hasattr(model, "set_params"):
                with contextlib.suppress(Exception):
                    model.set_params(scale_pos_weight=neg / max(pos, 1))

            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_tr, y_tr)

        y_prob = model.predict_proba(X_val)[:, 1]

        auc_pr = average_precision_score(y_val, y_prob)
        auc_roc = roc_auc_score(y_val, y_prob)

        # Precision/recall at target recall=0.80
        precision, recall, _ = precision_recall_curve(y_val, y_prob)
        eligible = precision[recall >= 0.80]
        prec_at_80 = float(eligible.max()) if len(eligible) > 0 else 0.0

        auc_prs.append(auc_pr)
        auc_rocs.append(auc_roc)
        recalls_at_80.append(prec_at_80)

        logger.info("  %s fold %d: AUC-PR=%.4f AUC-ROC=%.4f", name, fold + 1, auc_pr, auc_roc)

    return {
        "model": name,
        "auc_pr_mean": round(float(np.mean(auc_prs)), 4),
        "auc_pr_std": round(float(np.std(auc_prs)), 4),
        "auc_roc_mean": round(float(np.mean(auc_rocs)), 4),
        "auc_roc_std": round(float(np.std(auc_rocs)), 4),
        "prec_at_80recall_mean": round(float(np.mean(recalls_at_80)), 4),
        "n_folds": n_folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare LR, RF, XGBoost on fraud detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", default="data/raw/transactions.parquet")
    parser.add_argument("--sample", type=int, default=50_000, help="Subsample for speed")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if not Path(args.data).exists():
        logger.error("Data not found: %s", args.data)
        logger.info("Run: python scripts/generate_synthetic_data.py")
        sys.exit(1)

    logger.info("Loading data...")
    df = pd.read_parquet(args.data) if args.data.endswith(".parquet") else pd.read_csv(args.data)
    if len(df) > args.sample:
        df = df.sample(args.sample, random_state=42)
        logger.info("Sampled %d rows", len(df))

    logger.info("Engineering features...")
    engineer = FraudFeatureEngineer(mode="training")
    X = engineer.fit_transform(df)
    y = X.pop("is_fraud") if "is_fraud" in X.columns else df["is_fraud"]

    drop_cols = ["transaction_id", "user_id", "merchant_id", "timestamp"]
    X.drop(columns=[c for c in drop_cols if c in X.columns], inplace=True)
    X = X.fillna(0)

    fraud_rate = y.mean()
    logger.info("Dataset: %d rows | fraud_rate=%.3f%%", len(X), fraud_rate * 100)

    # Compare models
    print("\n" + "=" * 70)
    print(f"MODEL COMPARISON  |  {'AUC-PR':>12}  {'AUC-ROC':>12}  {'Prec@80Rec':>12}")
    print("=" * 70)

    results = []
    for name, model in MODELS.items():
        logger.info("Cross-validating %s...", name)
        result = cross_validate_model(name, model, X, y, n_folds=args.folds)
        results.append(result)
        print(
            f"  {name:<25}  "
            f"{result['auc_pr_mean']:.4f} ±{result['auc_pr_std']:.3f}  "
            f"{result['auc_roc_mean']:.4f} ±{result['auc_roc_std']:.3f}  "
            f"{result['prec_at_80recall_mean']:.4f}"
        )

    best = max(results, key=lambda r: r["auc_pr_mean"])
    print("=" * 70)
    print(f"\n✅ Best model: {best['model']} (AUC-PR={best['auc_pr_mean']:.4f})")
    print("\nRemember: use AUC-PR and recall — NOT accuracy — for fraud.")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output).open("w") as f:
            json.dump({"results": results, "best_model": best["model"]}, f, indent=2)
        logger.info("Results → %s", args.output)


if __name__ == "__main__":
    main()
