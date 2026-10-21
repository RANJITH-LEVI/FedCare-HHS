"""
FastAPI Route: /hospitals
Lists simulated hospital network nodes with clinical profile metadata and metrics,
provides dataset viewing by hospital, and supports uploading new hospital datasets
with automated federated training round execution.
"""

import io
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Body
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import HospitalModel, TrainingRoundModel
from model.preprocessing import (
    HOSPITALS_INFO,
    FEATURE_NAMES,
    TARGET_NAME,
    FEATURE_METADATA,
    load_raw_dataset
)
from federated.simulate import FederatedSimulationManager

router = APIRouter(tags=["Hospital Network & Datasets"])


class AddDatasetJSONRequest(BaseModel):
    site_key: str
    name: str
    location: str
    records: Optional[List[Dict[str, Any]]] = None
    csv_text: Optional[str] = None
    local_epochs: int = 15


@router.get("/hospitals")
def list_hospitals(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """Returns all simulated participating hospital nodes and their latest telemetry."""
    mgr = FederatedSimulationManager.get_instance()
    summary = mgr.get_metrics_summary()
    hosp_stats = summary.get('hospital_stats', {})

    db_hospitals = {h.id: h for h in db.query(HospitalModel).all()}
    results = []

    for site_key, static_info in HOSPITALS_INFO.items():
        h_id = static_info['id']
        dynamic_stats = hosp_stats.get(site_key, {})
        db_record = db_hospitals.get(h_id)

        sample_count = db_record.sample_count if db_record else dynamic_stats.get('train_samples', 0)
        local_acc = db_record.local_accuracy if db_record else dynamic_stats.get('local_accuracy', 0.82)

        results.append({
            'id': h_id,
            'site_key': site_key,
            'name': static_info['name'],
            'location': static_info['location'],
            'description': static_info.get('description', ''),
            'sample_count': sample_count,
            'local_accuracy': round(float(local_acc), 4),
            'local_f1': dynamic_stats.get('local_f1', 0.80),
            'local_loss': dynamic_stats.get('local_loss', 0.38)
        })

    return results


@router.get("/hospitals/datasets")
def get_hospital_datasets() -> List[Dict[str, Any]]:
    """
    Returns summaries and sample records for all participating hospital datasets
    for the mobile app and web dataset explorer.
    """
    results = []
    for site_key, info in list(HOSPITALS_INFO.items()):
        try:
            df = load_raw_dataset(site_key)
            total = len(df)
            positives = int((df[TARGET_NAME] > 0).sum())
            negatives = total - positives
            prevalence = round((positives / max(total, 1)) * 100, 1)

            # Convert first 40 records to dict list for preview
            sample_df = df.head(40).fillna(-1)
            records = sample_df.to_dict(orient="records")

            results.append({
                'id': info['id'],
                'site_key': site_key,
                'name': info['name'],
                'location': info['location'],
                'description': info.get('description', ''),
                'total_records': total,
                'positive_records': positives,
                'negative_records': negatives,
                'disease_prevalence_pct': prevalence,
                'feature_names': FEATURE_NAMES,
                'records': records
            })
        except Exception as e:
            results.append({
                'id': info.get('id', site_key),
                'site_key': site_key,
                'name': info.get('name', site_key),
                'location': info.get('location', ''),
                'error': str(e),
                'total_records': 0,
                'records': []
            })
    return results


@router.post("/hospitals/dataset/add")
async def add_hospital_dataset_file(
    site_key: str = Form(...),
    name: str = Form(...),
    location: str = Form(...),
    local_epochs: int = Form(15),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Uploads a new hospital dataset via CSV file, registers the hospital node,
    and automatically executes a federated training round so the global RBFN model learns immediately.
    """
    content = await file.read()
    try:
        csv_str = content.decode('utf-8')
    except UnicodeDecodeError:
        csv_str = content.decode('latin-1')

    # Read CSV
    try:
        # Check if header is present
        first_line = csv_str.strip().split('\n')[0].lower()
        if 'age' in first_line:
            df = pd.read_csv(io.StringIO(csv_str))
        else:
            cols = FEATURE_NAMES + [TARGET_NAME]
            df = pd.read_csv(io.StringIO(csv_str), names=cols, na_values='?')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    return _process_and_train_dataset(site_key, name, location, df, local_epochs, db)


@router.post("/hospitals/dataset/add-json")
def add_hospital_dataset_json(
    payload: AddDatasetJSONRequest,
    db: Session = Depends(get_db)
):
    """
    Accepts JSON records or raw CSV text for adding a hospital dataset,
    and automatically triggers federated learning on the updated network.
    """
    if payload.records:
        df = pd.DataFrame(payload.records)
    elif payload.csv_text:
        try:
            first_line = payload.csv_text.strip().split('\n')[0].lower()
            if 'age' in first_line:
                df = pd.read_csv(io.StringIO(payload.csv_text))
            else:
                cols = FEATURE_NAMES + [TARGET_NAME]
                df = pd.read_csv(io.StringIO(payload.csv_text), names=cols, na_values='?')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"CSV parse error: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Must provide either 'records' or 'csv_text'.")

    return _process_and_train_dataset(
        payload.site_key,
        payload.name,
        payload.location,
        df,
        payload.local_epochs,
        db
    )


def _process_and_train_dataset(
    site_key: str,
    name: str,
    location: str,
    df: pd.DataFrame,
    local_epochs: int,
    db: Session
) -> Dict[str, Any]:
    """Helper to register dataset and run automatic training."""
    # Ensure all required features are present
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = FEATURE_METADATA.get(col, {}).get('default', 0)
    if TARGET_NAME not in df.columns:
        df[TARGET_NAME] = 0

    df[TARGET_NAME] = (df[TARGET_NAME] > 0).astype(int)

    mgr = FederatedSimulationManager.get_instance()
    try:
        result = mgr.add_hospital_dataset(
            site_key=site_key,
            name=name,
            location=location,
            df=df,
            local_epochs=local_epochs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training round failed with new dataset: {str(e)}")

    round_res = result['round_metrics']

    # Persist round to DB
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
        per_hospital_metrics_json=json.dumps(round_res.get('hospital_metrics', {}), default=str),
        duration_seconds=round_res.get('duration_sec', 0.0),
        timestamp=datetime.utcnow()
    )
    db.add(db_round)

    # Update or add HospitalModel
    hosp = db.query(HospitalModel).filter(HospitalModel.id == result['hospital_id']).first()
    if not hosp:
        hosp = HospitalModel(
            id=result['hospital_id'],
            name=name,
            location=location,
            sample_count=result['total_samples'],
            local_accuracy=round_res['global_accuracy'],
            last_active=datetime.utcnow()
        )
        db.add(hosp)
    else:
        hosp.sample_count = result['total_samples']
        hosp.last_active = datetime.utcnow()

    db.commit()

    return {
        "status": "success",
        "message": f"Successfully registered {name} and trained round #{round_res['round']} on new dataset!",
        "hospital_id": result['hospital_id'],
        "hospital_name": name,
        "site_key": result['site_key'],
        "total_samples": result['total_samples'],
        "round_number": round_res['round'],
        "global_accuracy": round_res['global_accuracy'],
        "global_f1": round_res['global_f1'],
        "global_loss": round_res['global_loss'],
        "duration_seconds": round_res['duration_sec'],
        "dp_epsilon": round_res.get('dp_epsilon')
    }
