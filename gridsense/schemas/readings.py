"""Pydantic v2 schemas for sensor readings."""
from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class TransformerReadingSchema(BaseModel):
    """DTM sensor reading from a distribution transformer."""

    model_config = ConfigDict(from_attributes=True)

    transformer_id: str
    timestamp: datetime
    Va: float = Field(description="Phase A voltage (V)")
    Vb: float = Field(description="Phase B voltage (V)")
    Vc: float = Field(description="Phase C voltage (V)")
    Ia: float = Field(description="Phase A current (A)")
    Ib: float = Field(description="Phase B current (A)")
    Ic: float = Field(description="Phase C current (A)")
    oil_temp: float = Field(description="Transformer oil temperature (°C)")
    power_factor: float = Field(description="Power factor (0-1)")
    thd_pct: float = Field(description="Total harmonic distortion (%)")
    active_power_kw: float = Field(description="Active power (kW)")
    reactive_power_kvar: float = Field(description="Reactive power (kvar)")
    tamper_flag: bool = Field(default=False, description="Tamper detection flag")
    anomaly_score: Optional[float] = Field(default=None, description="ML anomaly score (0-1)")


class MeterReadingSchema(BaseModel):
    """Smart meter reading."""

    model_config = ConfigDict(from_attributes=True)

    meter_id: str
    transformer_id: str
    timestamp: datetime
    active_power_kw: float
    reactive_power_kvar: float
    tamper_flag: bool = False
    consumption_drop_pct: Optional[float] = None
