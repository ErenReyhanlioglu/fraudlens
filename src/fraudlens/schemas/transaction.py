"""Pydantic v2 schemas for transaction ingestion and scoring response."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TransactionType(StrEnum):
    TRANSFER = "transfer"
    PAYMENT = "payment"
    WITHDRAWAL = "withdrawal"
    DEPOSIT = "deposit"
    PURCHASE = "purchase"


class Channel(StrEnum):
    ONLINE = "online"
    MOBILE = "mobile"
    ATM = "atm"
    BRANCH = "branch"
    API = "api"


class ShapFeature(BaseModel):
    """Single SHAP contribution for a model feature."""
    model_config = ConfigDict(frozen=True)
    feature: str
    value: float
    contribution: float


class TransactionRequest(BaseModel):
    """Incoming transaction payload for fraud scoring."""

    model_config = ConfigDict(populate_by_name=True)

    # Primitive types for JSON boundary compatibility
    transaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str
    amount: float = Field(gt=0.0)
    currency: str = Field(min_length=3, max_length=3)
    transaction_type: TransactionType
    channel: Channel

    # sender
    sender_account_id: str = Field(min_length=1, max_length=64)
    sender_bank_code: str = Field(min_length=1, max_length=11)
    sender_country: str = Field(min_length=2, max_length=2)

    # receiver
    receiver_account_id: str = Field(min_length=1, max_length=64)
    receiver_bank_code: str = Field(min_length=1, max_length=11)
    receiver_country: str = Field(min_length=2, max_length=2)

    # network & device
    ip_address: str | None = None
    device_fingerprint: str | None = None
    user_agent: str | None = None

    # merchant
    merchant_id: str | None = None
    merchant_category_code: str | None = Field(default=None, min_length=4, max_length=4)
    merchant_name: str | None = None

    # metadata & bypass
    metadata: dict[str, Any] | None = None
    raw_features: dict[str, Any] | None = None

    @field_validator("currency")
    @classmethod
    def currency_uppercase(cls, v: str) -> str:
        upper = v.upper()
        if not upper.isalpha():
            raise ValueError("currency must contain only letters")
        return upper

    @field_validator("sender_country", "receiver_country")
    @classmethod
    def country_uppercase(cls, v: str) -> str:
        upper = v.upper()
        if not upper.isalpha():
            raise ValueError("country must be a 2-letter code")
        return upper

    @model_validator(mode="after")
    def device_required_for_digital_channels(self) -> TransactionRequest:
        if self.channel in (Channel.ONLINE, Channel.MOBILE) and self.ip_address is None:
            raise ValueError("ip_address required for online/mobile")
        return self

    @model_validator(mode="after")
    def mcc_required_for_purchases(self) -> TransactionRequest:
        if self.transaction_type is TransactionType.PURCHASE and self.merchant_category_code is None:
            raise ValueError("mcc required for purchase")
        return self


class TransactionResponse(BaseModel):
    """Scoring result returned immediately after a POST /transactions call."""

    model_config = ConfigDict(frozen=True)

    transaction_id: str | uuid.UUID
    decision_id: str | uuid.UUID
    
    received_at: datetime
    
    fraud_probability: float = Field(ge=0.0, le=1.0)
    risk_tier: str
    triage_action: str
    shap_top_features: list[ShapFeature]
    processing_time_ms: float