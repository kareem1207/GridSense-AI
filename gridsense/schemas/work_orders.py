"""Pydantic v2 schemas for work orders."""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict


class WorkOrderPriority(str, Enum):
    """Work order priority level."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class WorkOrderStatus(str, Enum):
    """Work order lifecycle status."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"


class WorkOrderSchema(BaseModel):
    """AI-generated work order for field engineers."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    transformer_id: str
    alert_id: Optional[str] = None
    created_at: datetime
    status: WorkOrderStatus = WorkOrderStatus.PENDING
    fault_type: str
    severity: str
    evidence: str
    recommended_action: str
    tools_needed: str
    estimated_repair_hours: float
    priority: WorkOrderPriority
