"""
Full End-to-End System Verification for FedCare-HHS
Tests all backend REST endpoints and frontend server health.
"""

import urllib.request
import json

base_url = 'http://127.0.0.1:8000'

print("=== 1. HEALTH PROBE ===")
with urllib.request.urlopen(f'{base_url}/health') as res:
    data = json.loads(res.read().decode('utf-8'))
    print(f"Status: {res.status} | Payload: {data}")

print("\n=== 2. PATIENT INGESTION (POST /ingest) ===")
patient_data = {
    'age': 60.0, 'sex': 1, 'cp': 4, 'trestbps': 140.0, 'chol': 260.0,
    'fbs': 0, 'restecg': 1, 'thalach': 140.0, 'exang': 1, 'oldpeak': 2.0,
    'slope': 2, 'ca': 1.0, 'thal': 7, 'hospital_id': 'HOSP-01'
}
req = urllib.request.Request(
    f'{base_url}/ingest',
    data=json.dumps(patient_data).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
with urllib.request.urlopen(req) as res:
    ingested = json.loads(res.read().decode('utf-8'))
    print(f"Status: {res.status} | Patient ID: {ingested['patient_identifier']} | Node: {ingested['hospital_id']}")

print("\n=== 3. CLINICAL RISK PREDICTION & SHAP (POST /predict) ===")
req_pred = urllib.request.Request(
    f'{base_url}/predict',
    data=json.dumps(patient_data).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
with urllib.request.urlopen(req_pred) as res:
    pred = json.loads(res.read().decode('utf-8'))
    print(f"Status: {res.status} | CAD Probability: {pred['risk_percentage']}% ({pred['risk_tier']})")
    print(f"Base Prevalence: {pred['base_value'] * 100:.1f}%")
    print(f"Top Risk Drivers: {[r['label'] for r in pred['top_risk_factors']]}")
    print(f"Top Protective: {[p['label'] for p in pred['top_protective_factors']]}")
    print(f"Waterfall Steps Count: {len(pred['waterfall_steps'])}")
    print(f"Clinical Narrative: {pred['plain_language_narrative']}")

print("\n=== 4. ON-DEMAND FEDERATED LEARNING ROUND (POST /federated/round) ===")
round_req = urllib.request.Request(
    f'{base_url}/federated/round',
    data=json.dumps({'local_epochs': 5, 'dp_enabled': True, 'noise_multiplier': 0.05}).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
with urllib.request.urlopen(round_req) as res:
    rnd = json.loads(res.read().decode('utf-8'))
    print(f"Status: {res.status} | Round: #{rnd['round']} | Global Acc: {rnd['global_accuracy'] * 100:.2f}% | DP Epsilon: {rnd['dp_epsilon']}")
    for site, h in rnd['hospital_metrics'].items():
        print(f"  - {h['name']}: {h['train_samples']} samples, local acc: {h['local_accuracy'] * 100:.1f}%")

print("\n=== 5. DYNAMIC GLOBAL SHAP (GET /model/explain/global) ===")
with urllib.request.urlopen(f'{base_url}/model/explain/global') as res:
    g_shap = json.loads(res.read().decode('utf-8'))
    print(f"Status: {res.status} | Features Count: {len(g_shap['features'])}")
    for f in g_shap['features'][:4]:
        print(f"  - {f['label']}: mean |SHAP| = {f['importance']:.4f} ({f['relative_pct']}%)")

print("\n=== 6. FRONTEND STREAMLIT HEALTH ===")
with urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health') as res:
    print(f"Status: {res.status} | Response: {res.read().decode('utf-8')}")

print("\n>>> ALL SYSTEM VERIFICATIONS PASSED SUCCESSFULLY! <<<")
