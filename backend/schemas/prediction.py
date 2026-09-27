"""
Pydantic schemas for clinical predictions and SHAP explainability.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from backend.schemas.patient import ClinicalInput


class PredictionRequest(ClinicalInput):
    """Prediction intake payload (13 clinical fields + optional tracking IDs)."""
    patient_id: Optional[int] = Field(default=None, description="Optional existing database patient ID")
    hospital_id: Optional[str] = Field(default="HOSP-01", description="Originating hospital node ID")


class FeatureAttribution(BaseModel):
    feature: str
    label: str
    raw_value: Optional[Any] = None
    display_value: str
    shap_value: float
    abs_shap: float
    effect: str


class WaterfallStep(BaseModel):
    label: str
    feature: Optional[str] = None
    delta: float
    cumulative: float
    type: str  # 'base', 'risk', 'protective', 'final'


class InteractionItem(BaseModel):
    feature_1: str
    label_1: str
    feature_2: str
    label_2: str
    synergy: float
    type: str  # 'Synergistic', 'Antagonistic', 'Additive'


class PredictionResponse(BaseModel):
    prediction_id: Optional[int] = None
    patient_id: Optional[int] = None
    risk_score: float
    risk_percentage: float
    risk_tier: str
    risk_badge: str
    base_value: float
    plain_language_narrative: str
    feature_attributions: List[FeatureAttribution]
    top_risk_factors: List[FeatureAttribution]
    top_protective_factors: List[FeatureAttribution]
    waterfall_steps: List[WaterfallStep]
    interactions: List[InteractionItem] = []
    selected_features: List[str]
    created_at: Optional[datetime] = None
