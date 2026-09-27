"""
FastAPI Route: /hospitals
Lists simulated hospital network nodes with clinical profile metadata and metrics.
"""

from typing import List, Dict, Any
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import HospitalModel
from model.preprocessing import HOSPITALS_INFO
from federated.simulate import FederatedSimulationManager

router = APIRouter(tags=["Hospital Network"])


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
            'description': static_info['description'],
            'sample_count': sample_count,
            'local_accuracy': round(float(local_acc), 4),
            'local_f1': dynamic_stats.get('local_f1', 0.80),
            'local_loss': dynamic_stats.get('local_loss', 0.38)
        })

    return results
