"""
src/core/schemas.py
-------------------
Pydantic v2 request/response schemas for the inference API.
All schemas include strict validation with detailed error messages.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, IPvAnyAddress, field_validator, model_validator


class RiskTier(str, Enum):
    """Fraud risk classification tiers."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class PaymentMethod(str, Enum):
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    BANK_TRANSFER = "bank_transfer"
    DIGITAL_WALLET = "digital_wallet"
    CRYPTO = "crypto"
    BUY_NOW_PAY_LATER = "bnpl"


class MerchantCategory(str, Enum):
    ELECTRONICS = "electronics"
    TRAVEL = "travel"
    GROCERIES = "groceries"
    RESTAURANTS = "restaurants"
    ENTERTAINMENT = "entertainment"
    HEALTHCARE = "healthcare"
    FINANCIAL_SERVICES = "financial_services"
    ONLINE_GAMBLING = "online_gambling"
    JEWELRY = "jewelry"
    OTHER = "other"


class Location(BaseModel):
    country: Annotated[str, Field(min_length=2, max_length=2, description="ISO 3166-1 alpha-2 country code")]
    city: str | None = None
    latitude: Annotated[float, Field(ge=-90.0, le=90.0)] | None = None
    longitude: Annotated[float, Field(ge=-180.0, le=180.0)] | None = None
    postal_code: str | None = None

    @field_validator("country")
    @classmethod
    def uppercase_country(cls, v: str) -> str:
        return v.upper()


class TransactionRequest(BaseModel):
    """
    Real-time fraud prediction request.
    All PII fields are handled via tokenization — no raw card/account numbers.
    """

    transaction_id: Annotated[str, Field(min_length=1, max_length=128, description="Unique transaction identifier")]
    user_id: Annotated[str, Field(min_length=1, max_length=128, description="Tokenized user identifier")]
    amount: Annotated[float, Field(gt=0.0, lt=1_000_000.0, description="Transaction amount")]
    currency: Annotated[str, Field(min_length=3, max_length=3, description="ISO 4217 currency code")]
    merchant_id: Annotated[str, Field(min_length=1, max_length=128)]
    merchant_category: MerchantCategory = MerchantCategory.OTHER
    payment_method: PaymentMethod = PaymentMethod.CREDIT_CARD
    timestamp: datetime
    card_present: bool = False
    device_fingerprint: str | None = None
    ip_address: IPvAnyAddress | None = None
    location: Location | None = None
    is_international: bool | None = None  # Auto-computed if location provided
    user_agent: str | None = None

    @field_validator("currency")
    @classmethod
    def uppercase_currency(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def compute_international_flag(self) -> TransactionRequest:
        """Auto-derive is_international from user's home country vs transaction country."""
        if self.is_international is None and self.location is not None:
            # In production this would query user profile service
            self.is_international = self.location.country not in {"US", "CA"}
        return self


class FeatureContributions(BaseModel):
    """SHAP-based top feature contributions for explainability."""
    velocity_1h: float = 0.0
    velocity_24h: float = 0.0
    amount_zscore: float = 0.0
    new_device: float = 0.0
    geo_anomaly: float = 0.0
    merchant_risk_score: float = 0.0
    time_of_day_anomaly: float = 0.0
    card_not_present: float = 0.0


class PredictionResponse(BaseModel):
    """Fraud prediction response with risk explanation."""

    transaction_id: str
    fraud_probability: Annotated[float, Field(ge=0.0, le=1.0)]
    fraud_label: bool
    risk_tier: RiskTier
    model_version: str
    feature_contributions: FeatureContributions
    latency_ms: int
    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"json_schema_extra": {
        "example": {
            "transaction_id": "txn_abc123",
            "fraud_probability": 0.847,
            "fraud_label": True,
            "risk_tier": "HIGH",
            "model_version": "v2.1.0",
            "feature_contributions": {
                "velocity_1h": 0.312,
                "amount_zscore": 0.198,
                "new_device": 0.145,
                "geo_anomaly": 0.192,
            },
            "latency_ms": 23,
            "decision_id": "dec_789xyz",
        }
    }}


class BatchTransactionRequest(BaseModel):
    """Batch prediction request — up to 1000 transactions."""

    transactions: Annotated[list[TransactionRequest], Field(min_length=1, max_length=1000)]
    async_mode: bool = Field(default=False, alias="async")
    callback_url: str | None = None

    @model_validator(mode="after")
    def validate_async_callback(self) -> BatchTransactionRequest:
        if self.async_mode and not self.callback_url:
            msg = "callback_url required when async=true"
            raise ValueError(msg)
        return self


class BatchPredictionResponse(BaseModel):
    batch_id: str = Field(default_factory=lambda: str(uuid4()))
    total: int
    predictions: list[PredictionResponse] | None = None
    status: str = "completed"  # completed | processing (async)
    callback_url: str | None = None
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ReadinessResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None
    feature_store_connected: bool
    mlflow_connected: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelInfoResponse(BaseModel):
    model_version: str
    model_stage: str
    training_date: str | None
    metrics: dict[str, float]
    feature_count: int
    ensemble_weights: dict[str, float]


class ModelReloadRequest(BaseModel):
    version: str = "latest"
    dry_run: bool = False


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    request_id: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
