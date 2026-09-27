"""
SQLAlchemy ORM models for hospitals, patients, predictions, and training rounds.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from backend.db.database import Base


class HospitalModel(Base):
    __tablename__ = "hospitals"

    id = Column(String(32), primary_key=True, index=True)
    name = Column(String(128), nullable=False)
    location = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
    sample_count = Column(Integer, default=0)
    local_accuracy = Column(Float, default=0.0)
    last_active = Column(DateTime, default=datetime.utcnow)

    patients = relationship("PatientModel", back_populates="hospital")


class PatientModel(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_identifier = Column(String(64), unique=True, index=True, nullable=False)
    hospital_id = Column(String(32), ForeignKey("hospitals.id"), nullable=False)
    
    # 13 Clinical Attributes
    age = Column(Float, nullable=False)
    sex = Column(Integer, nullable=False)
    cp = Column(Integer, nullable=False)
    trestbps = Column(Float, nullable=False)
    chol = Column(Float, nullable=False)
    fbs = Column(Integer, nullable=False)
    restecg = Column(Integer, nullable=False)
    thalach = Column(Float, nullable=False)
    exang = Column(Integer, nullable=False)
    oldpeak = Column(Float, nullable=False)
    slope = Column(Integer, nullable=False)
    ca = Column(Float, nullable=False)
    thal = Column(Integer, nullable=False)
    
    target = Column(Integer, nullable=True)  # True label if known
    created_at = Column(DateTime, default=datetime.utcnow)

    hospital = relationship("HospitalModel", back_populates="patients")
    predictions = relationship("PredictionModel", back_populates="patient")


class PredictionModel(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)
    risk_score = Column(Float, nullable=False)
    risk_tier = Column(String(32), nullable=False)
    plain_language_narrative = Column(Text, nullable=False)
    shap_explanation_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("PatientModel", back_populates="predictions")


class TrainingRoundModel(Base):
    __tablename__ = "training_rounds"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    round_number = Column(Integer, index=True, nullable=False)
    strategy = Column(String(32), default="FedAvg")
    global_accuracy = Column(Float, nullable=False)
    global_precision = Column(Float, nullable=False)
    global_recall = Column(Float, nullable=False)
    global_f1 = Column(Float, nullable=False)
    global_loss = Column(Float, nullable=False)
    global_roc_auc = Column(Float, nullable=False)
    dp_epsilon = Column(Float, nullable=True)
    dp_delta = Column(Float, nullable=True)
    per_hospital_metrics_json = Column(Text, nullable=False)
    duration_seconds = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow)
