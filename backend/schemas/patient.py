"""
Pydantic schemas for patient intake, validation, and database responses.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class ClinicalInput(BaseModel):
    """The canonical 13 UCI Heart Disease clinical indicators."""
    age: float = Field(..., ge=18, le=100, description="Age in years", example=58.0)
    sex: int = Field(..., ge=0, le=1, description="0 = Female, 1 = Male", example=1)
    cp: int = Field(..., ge=1, le=4, description="Chest Pain Type (1: Typical, 2: Atypical, 3: Non-Anginal, 4: Asymptomatic)", example=4)
    trestbps: float = Field(..., ge=70, le=240, description="Resting Blood Pressure in mm Hg", example=140.0)
    chol: float = Field(..., ge=80, le=650, description="Serum Cholesterol in mg/dl", example=260.0)
    fbs: int = Field(..., ge=0, le=1, description="Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", example=0)
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG Results (0: Normal, 1: ST-T Abnormality, 2: LV Hypertrophy)", example=0)
    thalach: float = Field(..., ge=50, le=240, description="Maximum Heart Rate Achieved (bpm)", example=135.0)
    exang: int = Field(..., ge=0, le=1, description="Exercise Induced Angina (1 = Yes, 0 = No)", example=1)
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST Depression induced by exercise relative to rest", example=2.2)
    slope: int = Field(..., ge=1, le=3, description="Slope of the peak exercise ST segment (1: Up, 2: Flat, 3: Down)", example=2)
    ca: float = Field(..., ge=0.0, le=3.0, description="Number of major vessels (0-3) colored by fluoroscopy", example=1.0)
    thal: int = Field(..., ge=3, le=7, description="Thalassemia (3: Normal, 6: Fixed defect, 7: Reversible defect)", example=7)


class PatientIngestRequest(ClinicalInput):
    """Request schema for ingesting a patient from a hospital source."""
    hospital_id: str = Field(default="HOSP-01", description="Hospital Node ID (e.g. HOSP-01)", example="HOSP-01")
    patient_identifier: Optional[str] = Field(default=None, description="Optional hospital chart MRN/ID", example="PT-82914")
    target: Optional[int] = Field(default=None, ge=0, le=1, description="Confirmed diagnosis if known (0: Healthy, 1: CAD)")


class PatientResponse(ClinicalInput):
    id: int
    patient_identifier: str
    hospital_id: str
    target: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True
