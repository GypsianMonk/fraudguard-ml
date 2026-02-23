"""
src/api/routes/predict.py
--------------------------
Fraud prediction endpoints.
Handles real-time single predictions and async batch predictions.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends

from src.api.dependencies import (
    get_container,
    get_feature_engineer,
    get_feature_store,
    verify_api_key,
    AppContainer,
)
from src.core.schemas import (
    BatchPredictionResponse,
    BatchTransactionRequest,
    FeatureContributions,
    PredictionResponse,
    RiskTier,
    TransactionRequest,
)
from src.monitoring.metrics_collector import FraudMetricsCollector

logger = logging.getLogger(__name__)
router = APIRouter()


def _classify_risk_tier(probability: float, settings: dict | None = None) -> RiskTier:
    """Classify fraud probability into risk tier."""
    if probability >= 0.85:
        return RiskTier.CRITICAL
    elif probability >= 0.60:
        return RiskTier.HIGH
    elif probability >= 0.30:
        return RiskTier.MEDIUM
    return RiskTier.LOW


def _transaction_to_dataframe(txn: TransactionRequest) -> pd.DataFrame:
    """Convert a TransactionRequest to a DataFrame row for feature engineering."""
    row = {
        "transaction_id": txn.transaction_id,
        "user_id": txn.user_id,
        "amount": txn.amount,
        "currency": txn.currency,
        "merchant_id": txn.merchant_id,
        "merchant_category": txn.merchant_category.value,
        "payment_method": txn.payment_method.value,
        "timestamp": txn.timestamp,
        "card_present": txn.card_present,
        "device_fingerprint": txn.device_fingerprint,
        "country": txn.location.country if txn.location else "US",
        "latitude": txn.location.latitude if txn.location else None,
        "longitude": txn.location.longitude if txn.location else None,
    }
    return pd.DataFrame([row])


def _extract_feature_contributions(
    feature_df: pd.DataFrame,
    shap_values: np.ndarray | None,
) -> FeatureContributions:
    """Extract top SHAP feature contributions for explainability."""
    if shap_values is None:
        return FeatureContributions()

    feature_names = list(feature_df.columns)
    shap_row = shap_values[0] if len(shap_values.shape) > 1 else shap_values

    # Map SHAP values to contribution fields
    contribs: dict[str, float] = {}
    for i, name in enumerate(feature_names):
        if i < len(shap_row):
            abs_val = float(abs(shap_row[i]))
            if "velocity_1h" in name or "txn_count_1h" in name:
                contribs["velocity_1h"] = max(contribs.get("velocity_1h", 0), abs_val)
            elif "velocity_24h" in name or "txn_count_24h" in name:
                contribs["velocity_24h"] = max(contribs.get("velocity_24h", 0), abs_val)
            elif "amount_zscore" in name:
                contribs["amount_zscore"] = abs_val
            elif "device" in name:
                contribs["new_device"] = max(contribs.get("new_device", 0), abs_val)
            elif "country" in name or "geo" in name or "international" in name:
                contribs["geo_anomaly"] = max(contribs.get("geo_anomaly", 0), abs_val)
            elif "merchant_risk" in name:
                contribs["merchant_risk_score"] = abs_val
            elif "is_night" in name or "hour" in name:
                contribs["time_of_day_anomaly"] = max(contribs.get("time_of_day_anomaly", 0), abs_val)
            elif "card_not_present" in name:
                contribs["card_not_present"] = abs_val

    return FeatureContributions(**contribs)


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Real-time fraud prediction",
    description=(
        "Predict fraud probability for a single transaction. "
        "Returns result in <50ms P99 latency. Includes risk tier, "
        "confidence, and SHAP-based feature contributions."
    ),
    responses={
        401: {"description": "Invalid or missing API key"},
        422: {"description": "Invalid request payload"},
        500: {"description": "Model inference failed"},
    },
)
async def predict_single(
    request: TransactionRequest,
    container: Annotated[AppContainer, Depends(get_container)],
    _api_key: Annotated[str, Depends(verify_api_key)],
) -> PredictionResponse:
    """
    Real-time single transaction fraud prediction.

    Performance target: P99 < 50ms end-to-end.
    """
    start_time = time.monotonic()
    metrics: FraudMetricsCollector = container.metrics

    try:
        # 1. Convert to DataFrame
        df = _transaction_to_dataframe(request)

        # 2. Enrich with historical features from feature store
        if container.feature_store is not None:
            try:
                user_feats = await container.feature_store.get_user_features(request.user_id)
                # Merge historical features into df
                for k, v in user_feats.items():
                    df[k] = v
            except Exception as exc:
                logger.warning("Feature store fetch failed, using transaction-only features: %s", exc)

        # 3. Feature engineering
        feature_df = container.feature_engineer.transform(df)

        # 4. Model inference
        fraud_proba = container.model.predict_proba(feature_df)[0]

        # 5. SHAP explanations (async, non-blocking in prod; sync for simplicity here)
        shap_values = None
        try:
            shap_values = container.model._xgb.get_shap_values(feature_df)
        except Exception:
            pass  # Explanations are best-effort

        # 6. Build response
        risk_tier = _classify_risk_tier(float(fraud_proba))
        latency_ms = int((time.monotonic() - start_time) * 1000)

        response = PredictionResponse(
            transaction_id=request.transaction_id,
            fraud_probability=float(fraud_proba),
            fraud_label=fraud_proba >= 0.5,
            risk_tier=risk_tier,
            model_version=container.model_version,
            feature_contributions=_extract_feature_contributions(feature_df, shap_values),
            latency_ms=latency_ms,
        )

        # 7. Record metrics
        metrics.record_prediction(
            fraud_prob=float(fraud_proba),
            risk_tier=risk_tier.value,
            latency_ms=latency_ms,
        )

        # 8. Update user features in background
        if container.feature_store is not None:
            await container.feature_store.update_user_features(
                user_id=request.user_id,
                features={
                    "last_txn_amount": request.amount,
                    "last_txn_timestamp": request.timestamp.isoformat(),
                    "last_txn_country": request.location.country if request.location else "US",
                },
            )

        logger.info(
            "Prediction: txn=%s user=%s prob=%.3f tier=%s latency=%dms",
            request.transaction_id, request.user_id, fraud_proba, risk_tier.value, latency_ms,
        )

        return response

    except Exception as exc:
        metrics.record_error("inference_error")
        logger.exception("Prediction failed for txn=%s: %s", request.transaction_id, exc)
        raise


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch fraud prediction",
    description="Score multiple transactions in a single request (up to 1000). "
                "Use async=true for large batches with callback webhook.",
)
async def predict_batch(
    request: BatchTransactionRequest,
    background_tasks: BackgroundTasks,
    container: Annotated[AppContainer, Depends(get_container)],
    _api_key: Annotated[str, Depends(verify_api_key)],
) -> BatchPredictionResponse:
    """
    Batch fraud prediction for up to 1000 transactions.
    """
    start_time = time.monotonic()
    batch_id = str(uuid4())

    if request.async_mode:
        # Queue for async processing and return immediately
        background_tasks.add_task(
            _process_batch_async,
            batch_id=batch_id,
            transactions=request.transactions,
            callback_url=request.callback_url,
            container=container,
        )
        return BatchPredictionResponse(
            batch_id=batch_id,
            total=len(request.transactions),
            status="processing",
            callback_url=request.callback_url,
            latency_ms=int((time.monotonic() - start_time) * 1000),
        )

    # Synchronous batch processing
    dfs = [_transaction_to_dataframe(txn) for txn in request.transactions]
    batch_df = pd.concat(dfs, ignore_index=True)

    feature_df = container.feature_engineer.transform(batch_df)
    fraud_probas = container.model.predict_proba(feature_df)

    predictions = []
    for txn, proba in zip(request.transactions, fraud_probas):
        risk_tier = _classify_risk_tier(float(proba))
        predictions.append(PredictionResponse(
            transaction_id=txn.transaction_id,
            fraud_probability=float(proba),
            fraud_label=proba >= 0.5,
            risk_tier=risk_tier,
            model_version=container.model_version,
            feature_contributions=FeatureContributions(),
            latency_ms=0,
        ))

    latency_ms = int((time.monotonic() - start_time) * 1000)
    logger.info("Batch prediction: %d transactions in %dms", len(request.transactions), latency_ms)

    return BatchPredictionResponse(
        batch_id=batch_id,
        total=len(predictions),
        predictions=predictions,
        status="completed",
        latency_ms=latency_ms,
    )


async def _process_batch_async(
    batch_id: str,
    transactions: list[TransactionRequest],
    callback_url: str | None,
    container: AppContainer,
) -> None:
    """Process batch asynchronously and POST results to callback URL."""
    import httpx

    dfs = [_transaction_to_dataframe(txn) for txn in transactions]
    batch_df = pd.concat(dfs, ignore_index=True)
    feature_df = container.feature_engineer.transform(batch_df)
    fraud_probas = container.model.predict_proba(feature_df)

    results = [
        {"transaction_id": txn.transaction_id, "fraud_probability": float(p)}
        for txn, p in zip(transactions, fraud_probas)
    ]

    if callback_url:
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                await client.post(callback_url, json={"batch_id": batch_id, "results": results})
            except Exception as exc:
                logger.error("Callback POST failed for batch %s: %s", batch_id, exc)
