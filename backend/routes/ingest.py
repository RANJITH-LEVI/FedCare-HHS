"""
FastAPI Route: /ingest
Accepts new patient records from simulated hospital sources, validates schema, and stores in SQLite.
"""

import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import PatientModel, HospitalModel
from backend.schemas.patient import PatientIngestRequest, PatientResponse

router = APIRouter(tags=["Patient Ingestion"])


@router.post("/ingest", response_model=PatientResponse)
def ingest_patient_record(req: PatientIngestRequest, db: Session = Depends(get_db)):
    """
    Ingests a new patient clinical record from a hospital node into the decentralized repository.
    Validates clinical parameters, checks hospital existence, and commits to SQLite.
    """
    # Verify or register hospital
    hospital = db.query(HospitalModel).filter(HospitalModel.id == req.hospital_id).first()
    if not hospital:
        # Auto-register if not yet present
        hospital = HospitalModel(
            id=req.hospital_id,
            name=f"Hospital Node {req.hospital_id}",
            location="Clinical Network",
            sample_count=0,
            local_accuracy=0.80,
            last_active=datetime.utcnow()
        )
        db.add(hospital)
        db.commit()

    patient_id_str = req.patient_identifier or f"PT-{uuid.uuid4().hex[:6].upper()}"

    db_patient = PatientModel(
        patient_identifier=patient_id_str,
        hospital_id=req.hospital_id,
        age=req.age,
        sex=req.sex,
        cp=req.cp,
        trestbps=req.trestbps,
        chol=req.chol,
        fbs=req.fbs,
        restecg=req.restecg,
        thalach=req.thalach,
        exang=req.exang,
        oldpeak=req.oldpeak,
        slope=req.slope,
        ca=req.ca,
        thal=req.thal,
        target=req.target,
        created_at=datetime.utcnow()
    )
    db.add(db_patient)

    # Increment hospital count
    hospital.sample_count += 1
    hospital.last_active = datetime.utcnow()
    db.commit()
    db.refresh(db_patient)

    return db_patient


@router.get("/patients", response_model=List[PatientResponse])
def list_patients(limit: int = 50, hospital_id: Optional[str] = None, db: Session = Depends(get_db)):
    """Retrieves recent ingested patients across hospital sites."""
    query = db.query(PatientModel)
    if hospital_id:
        query = query.filter(PatientModel.hospital_id == hospital_id)
    return query.order_by(PatientModel.created_at.desc()).limit(limit).all()
