"""
FastAPI Route: /predict
Accepts the 13 clinical fields, performs global RBFN inference,
generates local SHAP waterfall attribution & narrative, and persists in SQLite.
"""

import json
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import PredictionModel, PatientModel
from backend.schemas.prediction import PredictionRequest, PredictionResponse
from federated.simulate import FederatedSimulationManager

router = APIRouter(tags=["Clinical Prediction & SHAP"])


@router.post("/predict", response_model=PredictionResponse)
def predict_cardiac_risk(req: PredictionRequest, db: Session = Depends(get_db)):
    """
    Evaluates patient cardiovascular disease risk using the federated HHS-RBFN global model.
    Returns:
    - Risk score probability & categorical tier
    - Full local SHAP waterfall attribution (risk drivers vs protective factors)
    - Plain-language clinical reasoning narrative
    - Feature interaction synergies
    """
    mgr = FederatedSimulationManager.get_instance()
    
    # Extract clinical features dict
    patient_dict = {
        'age': req.age,
        'sex': req.sex,
        'cp': req.cp,
        'trestbps': req.trestbps,
        'chol': req.chol,
        'fbs': req.fbs,
        'restecg': req.restecg,
        'thalach': req.thalach,
        'exang': req.exang,
        'oldpeak': req.oldpeak,
        'slope': req.slope,
        'ca': req.ca,
        'thal': req.thal
    }

    try:
        explanation = mgr.predict_and_explain(patient_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction and explanation failed: {str(e)}")

    # Store in database
    db_pred = PredictionModel(
        patient_id=req.patient_id,
        risk_score=explanation['risk_score'],
        risk_tier=explanation['risk_tier'],
        plain_language_narrative=explanation['plain_language_narrative'],
        shap_explanation_json=json.dumps(explanation, default=str),
        created_at=datetime.utcnow()
    )
    db.add(db_pred)
    db.commit()
    db.refresh(db_pred)

    explanation['prediction_id'] = db_pred.id
    explanation['patient_id'] = req.patient_id
    explanation['created_at'] = db_pred.created_at
    return explanation


@router.get("/predict/history")
def get_prediction_history(limit: int = 20, db: Session = Depends(get_db)):
    """Retrieves recent clinical risk assessments."""
    preds = db.query(PredictionModel).order_by(PredictionModel.created_at.desc()).limit(limit).all()
    results = []
    for p in preds:
        try:
            exp_data = json.loads(p.shap_explanation_json)
        except Exception:
            exp_data = {}
        results.append({
            'prediction_id': p.id,
            'patient_id': p.patient_id,
            'risk_score': p.risk_score,
            'risk_tier': p.risk_tier,
            'plain_language_narrative': p.plain_language_narrative,
            'created_at': p.created_at,
            'top_risk_factors': exp_data.get('top_risk_factors', []),
            'top_protective_factors': exp_data.get('top_protective_factors', [])
        })
    return results
