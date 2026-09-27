"""
FastAPI Routes for Federated Learning:
- POST /federated/round: Triggers one federated training round across simulated hospital nodes
- GET /model/metrics: Global accuracy/precision/recall/F1, round history, per-hospital stats
- GET /model/explain/global: Current global SHAP feature importance recomputed per round
"""

import json
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import TrainingRoundModel, HospitalModel
from backend.schemas.federated import (
    FederatedRoundTriggerRequest,
    FederatedRoundResponse,
    ModelMetricsResponse,
    GlobalExplainResponse
)
from federated.simulate import FederatedSimulationManager

router = APIRouter(tags=["Federated Learning Operations"])


@router.post("/federated/round", response_model=FederatedRoundResponse)
def trigger_federated_round(req: FederatedRoundTriggerRequest = FederatedRoundTriggerRequest(),
                            db: Session = Depends(get_db)):
    """
    Triggers one on-demand federated learning round across the simulated hospital nodes:
    1. Distributes current global RBFN weights
    2. Hospital clients train locally on non-IID clinical records
    3. Clients apply Differential Privacy (norm clipping + calibrated Gaussian noise)
    4. Flower server executes sample-weighted FedAvg aggregation
    5. Evaluates multi-center global validation and per-hospital metrics
    6. Recomputes global SHAP feature importance
    """
    mgr = FederatedSimulationManager.get_instance()
    
    try:
        round_res = mgr.trigger_round(
            local_epochs=req.local_epochs,
            lr=req.learning_rate,
            dp_enabled=req.dp_enabled,
            clip_norm=req.clip_norm,
            noise_multiplier=req.noise_multiplier
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Federated round execution failed: {str(e)}")

    # Store round in database
    db_round = TrainingRoundModel(
        round_number=round_res['round'],
        strategy="FedAvg",
        global_accuracy=round_res['global_accuracy'],
        global_precision=round_res['global_precision'],
        global_recall=round_res['global_recall'],
        global_f1=round_res['global_f1'],
        global_loss=round_res['global_loss'],
        global_roc_auc=round_res['global_roc_auc'],
        dp_epsilon=round_res.get('dp_epsilon'),
        dp_delta=round_res.get('dp_delta'),
        per_hospital_metrics_json=json.dumps(round_res.get('hospital_metrics', {})),
        duration_seconds=round_res.get('duration_sec', 0.0),
        timestamp=datetime.utcnow()
    )
    db.add(db_round)

    # Update hospital local accuracies in DB
    for site_key, h_data in round_res.get('hospital_metrics', {}).items():
        hosp_id = h_data.get('hospital_id', site_key)
        hosp = db.query(HospitalModel).filter(HospitalModel.id == hosp_id).first()
        if hosp:
            hosp.local_accuracy = h_data.get('local_accuracy', hosp.local_accuracy)
            hosp.last_active = datetime.utcnow()

    db.commit()
    return round_res


@router.get("/model/metrics", response_model=ModelMetricsResponse)
def get_model_metrics(db: Session = Depends(get_db)):
    """
    Returns current global model performance metrics,
    centralized benchmark comparison, round history, and per-hospital contribution stats.
    """
    mgr = FederatedSimulationManager.get_instance()
    return mgr.get_metrics_summary()


@router.get("/model/explain/global", response_model=GlobalExplainResponse)
def get_global_explanation():
    """
    Returns global SHAP feature importance across the federated cardiovascular model,
    recomputed after every training round.
    """
    mgr = FederatedSimulationManager.get_instance()
    return mgr.get_global_explanation()
