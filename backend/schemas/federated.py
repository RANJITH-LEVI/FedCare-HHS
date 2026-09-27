"""
Pydantic schemas for federated learning rounds, model metrics, and global explainability.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class FederatedRoundTriggerRequest(BaseModel):
    local_epochs: int = Field(default=12, ge=1, le=50, description="Local training epochs per client")
    learning_rate: float = Field(default=0.06, gt=0.0, le=0.5, description="Client optimizer learning rate")
    dp_enabled: bool = Field(default=True, description="Enable Differential Privacy weight delta clipping & noise")
    noise_multiplier: float = Field(default=0.05, ge=0.0, le=1.0, description="Gaussian noise scale factor")
    clip_norm: float = Field(default=1.0, gt=0.0, le=10.0, description="L2 Gradient clipping norm")


class HospitalRoundMetric(BaseModel):
    hospital_id: str
    name: str
    location: str
    train_samples: int
    val_samples: int
    local_accuracy: float
    local_f1: float
    local_loss: float
    delta_norm: float


class FederatedRoundResponse(BaseModel):
    round: int
    timestamp: str
    duration_sec: float
    global_accuracy: float
    global_precision: float
    global_recall: float
    global_f1: float
    global_loss: float
    global_roc_auc: float
    dp_epsilon: Optional[float] = None
    dp_delta: Optional[float] = None
    noise_multiplier: float
    clip_norm: float
    centralized_benchmark: Dict[str, float]
    hospital_metrics: Dict[str, HospitalRoundMetric]


class ModelMetricsResponse(BaseModel):
    total_rounds: int
    latest_round: int
    last_trained_timestamp: str
    global_metrics: Dict[str, float]
    centralized_baseline: Dict[str, float]
    differential_privacy: Dict[str, Any]
    hospital_stats: Dict[str, Any]
    round_history: List[Dict[str, Any]]


class GlobalFeatureImportance(BaseModel):
    feature: str
    label: str
    importance: float
    relative_pct: float


class GlobalExplainResponse(BaseModel):
    base_value: float
    features: List[GlobalFeatureImportance]
