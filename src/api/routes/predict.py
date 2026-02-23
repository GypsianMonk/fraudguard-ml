"""
src/api/routes/predict.py
--------------------------
Fraud prediction endpoints.
"""
from __future__ import annotations

import contextlib
import logging
import time
from typing import TYPE_CHECKING, Annotated
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends

from src.api.dependencies import AppContainer, get_container, verify_api_key
from src.core.schemas import (
    BatchPredictionResponse,
    BatchTransactionRequest,
    FeatureContributions,
    PredictionResponse,
    RiskTier,
    TransactionRequest,
)

if TYPE_CHECKING:
    import numpy as np
    from src.monitoring.metrics_collector import FraudMetricsCollector

logger = logging.getLogger(__name__)
router = APIRouter()


def _classify_risk_tier(probability: float) -> RiskTier:
    if probability >= 0.85:
        return RiskTier.CRITICAL
    if probability >= 0.60:
        return RiskTier.HIGH
    if probability >= 0.30:
        return RiskTier.MEDIUM
    return RiskTier.LOW


def _transaction_to_dataframe(txn: TransactionRequest) -> pd.DataFrame:
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
    if shap_values is None:
        return FeatureContributions()

    feature_names = list(feature_df.columns)
    shap_row = shap_values[0] if len(shap_values.shape) > 1 else shap_values
    contribs: dict[str, float] = {}

    for i, name in enumerate(feature_names):
        if i >= len(shap_row):
            continue
        abs_val = float(abs(shap_row[i]))
        if "velocity_1h" in name or "txn_count_1h" in name:
            contribs["velocity_1h"] = max(contribs.get("velocity_1h", 0), abs_val)
        elif "velocity_24h" in name or "txn_count_24h" in name:
            contribs["velocity_24h"] = max(contribs.get("velocity_24h", 0), abs_val)
        elif "amount_zscore" in name:
            contribs["amount_zscore"] = abs_val
        elif "device" in name:
            contribs["new_device"] = max(contribs.get("new_device", 0), abs_val)
        elif any(k in name for k in ("country", "geo", "international")):
            contribs["geo_anomaly"] = max(contribs.get("geo_anomaly", 0), abs_val)
        elif "merchant_risk" in name:
            contribs["merchant_risk_score"] = abs_val
        elif "is_night" in name or "hour" in name:
            contribs["time_of_day_anomaly"] = max(
                contribs.get("time_of_day_anomaly", 0), abs_val
            )
        elif "card_not_present" in name:
            contribs["card_not_present"] = abs_val

    return FeatureContributions(**contribs)


@router.post("/predict", response_model=PredictionResponse, summary="Real-time fraud prediction")
async def predict_single(
    request: TransactionRequest,
    container: Annotated[AppContainer, Depends(get_container)],
    _api_key: Annotated[str, Depends(verify_api_key)],
) -> PredictionResponse:
    """Real-time single transaction fraud prediction. P99 < 50ms."""
    start_time = time.monotonic()
    metrics: FraudMetricsCollector = container.metrics

    try:
        df = _transaction_to_dataframe(request)

        if container.feature_store is not None:
            with contextlib.suppress(Exception):
                user_feats = await container.feature_store.get_user_features(
                    request.user_id
                )
                for k, v in user_feats.items():
                    df[k] = v

        feature_df = container.feature_engineer.transform(df)
        fraud_proba = container.model.predict_proba(feature_df)[0]

        shap_values = None
        with contextlib.suppress(Exception):
            shap_values = container.model._xgb.get_shap_values(feature_df)

        risk_tier = _classify_risk_tier(float(fraud_proba))
        latency_ms = int((time.monotonic() - start_time) * 1000)

        response = PredictionResponse(
            transaction_id=request.transaction_id,
            fraud_probability=float(fraud_proba),
            fraud_label=fraud_proba >= 0.5,
            risk_tier=risk_tier,
            model_version=container.model_version,
            feature_contributions=_extract_feature_contributions(
                feature_df, shap_values
            ),
            latency_ms=latency_ms,
        )

        metrics.record_prediction(
            fraud_prob=float(fraud_proba),
            risk_tier=risk_tier.value,
            latency_ms=latency_ms,
        )

        if container.feature_store is not None:
            await container.feature_store.update_user_features(
                user_id=request.user_id,
                features={
                    "last_txn_amount": request.amount,
                    "last_txn_timestamp": request.timestamp.isoformat(),
                    "last_txn_country": (
                        request.location.country if request.location else "US"
                    ),
                },
            )

        logger.info(
            "Prediction: txn=%s prob=%.3f tier=%s latency=%dms",
            request.transaction_id, fraud_proba, risk_tier.value, latency_ms,
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
)
async def predict_batch(
    request: BatchTransactionRequest,
    background_tasks: BackgroundTasks,
    container: Annotated[AppContainer, Depends(get_container)],
    _api_key: Annotated[str, Depends(verify_api_key)],
) -> BatchPredictionResponse:
    """Batch fraud prediction for up to 1000 transactions."""
    start_time = time.monotonic()
    batch_id = str(uuid4())

    if request.async_mode:
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

    dfs = [_transaction_to_dataframe(txn) for txn in request.transactions]
    batch_df = pd.concat(dfs, ignore_index=True)
    feature_df = container.feature_engineer.transform(batch_df)
    fraud_probas = container.model.predict_proba(feature_df)

    predictions = [
        PredictionResponse(
            transaction_id=txn.transaction_id,
            fraud_probability=float(proba),
            fraud_label=proba >= 0.5,
            risk_tier=_classify_risk_tier(float(proba)),
            model_version=container.model_version,
            feature_contributions=FeatureContributions(),
            latency_ms=0,
        )
        for txn, proba in zip(request.transactions, fraud_probas, strict=True)
    ]

    latency_ms = int((time.monotonic() - start_time) * 1000)
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
    import httpx

    dfs = [_transaction_to_dataframe(txn) for txn in transactions]
    batch_df = pd.concat(dfs, ignore_index=True)
    feature_df = container.feature_engineer.transform(batch_df)
    fraud_probas = container.model.predict_proba(feature_df)

    results = [
        {"transaction_id": txn.transaction_id, "fraud_probability": float(p)}
        for txn, p in zip(transactions, fraud_probas, strict=True)
    ]

    if callback_url:
        async with httpx.AsyncClient(timeout=30) as client:
            with contextlib.suppress(Exception):
                await client.post(callback_url, json={"batch_id": batch_id, "results": results})
