"""
src/core/schemas.py
--------------------
Pydantic request/response schemas for the FraudGuard ML API.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class RiskTier(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PaymentMethod(StrEnum):
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    DIGITAL_WALLET = "digital_wallet"
    BANK_TRANSFER = "bank_transfer"
    BNPL = "bnpl"
    CRYPTO = "crypto"
    OTHER = "other"


class MerchantCategory(StrEnum):
    ELECTRONICS = "electronics"
    GROCERIES = "groceries"
    RESTAURANTS = "restaurants"
    TRAVEL = "travel"
    ONLINE_GAMBLING = "online_gambling"
    JEWELRY = "jewelry"
    FINANCIAL_SERVICES = "financial_services"
    HEALTHCARE = "healthcare"
    ENTERTAINMENT = "entertainment"
    OTHER = "other"


class GeoLocation(BaseModel):
    country: str = Field(..., min_length=2, max_length=3)
    city: str | None = None
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)


class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., min_length=1, max_length=100)
    user_id: str = Field(..., min_length=1, max_length=100)
    amount: Annotated[float, Field(gt=0, le=1_000_000)]
    currency: str = Field(default="USD", min_length=3, max_length=3)
    merchant_id: str = Field(..., min_length=1, max_length=100)
    merchant_category: MerchantCategory = MerchantCategory.OTHER
    payment_method: PaymentMethod = PaymentMethod.CREDIT_CARD
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    card_present: bool = False
    device_fingerprint: str | None = None
    ip_address: str | None = None
    location: GeoLocation | None = None

    @field_validator("currency")
    @classmethod
    def currency_uppercase(cls, v: str) -> str:
        return v.upper()


class FeatureContributions(BaseModel):
    velocity_1h: float | None = None
    velocity_24h: float | None = None
    amount_zscore: float | None = None
    new_device: float | None = None
    geo_anomaly: float | None = None
    merchant_risk_score: float | None = None
    time_of_day_anomaly: float | None = None
    card_not_present: float | None = None


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    fraud_label: bool
    risk_tier: RiskTier
    model_version: str
    feature_contributions: FeatureContributions
    latency_ms: int
    decision_id: str = Field(
        default_factory=lambda: __import__("uuid").uuid4().hex
    )


class BatchTransactionRequest(BaseModel):
    transactions: Annotated[list[TransactionRequest], Field(min_length=1, max_length=1000)]
    async_mode: bool = False
    callback_url: str | None = None


class BatchPredictionResponse(BaseModel):
    batch_id: str
    total: int
    predictions: list[PredictionResponse] | None = None
    status: str
    callback_url: str | None = None
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime


class ReadinessResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    feature_store_connected: bool
    mlflow_connected: bool
    timestamp: datetime
