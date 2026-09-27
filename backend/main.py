"""
FedCare-HHS Backend Application
FastAPI REST API providing:
- POST /predict: RBFN inference + SHAP explainability
- POST /ingest: Patient record validation and storage
- POST /federated/round: On-demand federated learning execution
- GET /model/metrics: Global accuracy, history, and centralized benchmark
- GET /model/explain/global: Dynamic global SHAP feature importance
- GET /hospitals: Multi-hospital network directory
"""

import os
import sys
import threading
import traceback
from contextlib import asynccontextmanager
from datetime import datetime

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.db.database import Base, engine, SessionLocal
from backend.db.models import HospitalModel, TrainingRoundModel
from backend.routes.predict import router as predict_router
from backend.routes.ingest import router as ingest_router
from backend.routes.federated import router as federated_router
from backend.routes.hospitals import router as hospitals_router
from model.preprocessing import HOSPITALS_INFO
from federated.simulate import FederatedSimulationManager

# Ensure DB tables exist on import
Base.metadata.create_all(bind=engine)


def _init_manager_background():
    """Runs heavy ML init in a daemon thread so Render health-check passes immediately."""
    try:
        mgr = FederatedSimulationManager.get_instance()
        acc = mgr.get_metrics_summary()['global_metrics']['accuracy']
        print(f"[OK] FedCare-HHS model ready. Global Accuracy: {acc:.4f}")
    except Exception:
        print(f"[ERROR] FederatedSimulationManager init failed:\n{traceback.format_exc()}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes database schema and hospital nodes on startup."""
    # 1. Create DB tables
    Base.metadata.create_all(bind=engine)

    # 2. Populate Hospital Nodes in SQLite
    db = SessionLocal()
    try:
        for site_key, info in HOSPITALS_INFO.items():
            existing = db.query(HospitalModel).filter(HospitalModel.id == info['id']).first()
            if not existing:
                hosp = HospitalModel(
                    id=info['id'],
                    name=info['name'],
                    location=info['location'],
                    description=info['description'],
                    sample_count=250,
                    local_accuracy=0.83,
                    last_active=datetime.utcnow()
                )
                db.add(hosp)
        db.commit()
    except Exception as e:
        print(f"[WARN] Hospital DB seed failed (non-fatal): {e}")
    finally:
        db.close()

    # 3. Heavy ML warmup in background — health check passes immediately
    t = threading.Thread(target=_init_manager_background, daemon=True, name="fedcare-init")
    t.start()
    print("[INFO] FedCare-HHS backend started. ML model initialising in background...")

    yield


app = FastAPI(
    title="FedCare-HHS Cardiovascular Prediction & Federated Learning API",
    description=(
        "Full-stack federated, explainable cardiovascular disease prediction system. "
        "Integrates Harris Hawks Search (HHS) feature selection, Radial Basis Function Network (RBFN) "
        "classifier from scratch, Flower Federated Learning with Differential Privacy, "
        "and SHAP game-theoretic explainability."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for Streamlit frontend and local/cloud deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(predict_router)
app.include_router(ingest_router)
app.include_router(federated_router)
app.include_router(hospitals_router)


@app.get("/", tags=["System Health"])
def root():
    return {
        "system": "FedCare-HHS",
        "description": "Federated Explainable Cardiovascular Disease Prediction Platform",
        "status": "online",
        "docs_url": "/docs",
        "model_architecture": "Harris Hawks Search (HHS) + Radial Basis Function Network (RBFN) + Flower FL + DP",
        "federation_status": "Simulated multi-hospital network on public UCI benchmark datasets"
    }


@app.get("/health", tags=["System Health"])
def health_check():
    mgr_ready = FederatedSimulationManager._instance is not None
    return {
        "status": "healthy",
        "model_ready": mgr_ready,
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
