import urllib.request
import json

base_url = 'http://127.0.0.1:8000'

archetypes = [
    {
        'name': 'Archetype 1 (Severe CAD)',
        'data': {'age': 63.0, 'sex': 1, 'cp': 4, 'trestbps': 150.0, 'chol': 270.0, 'fbs': 1, 'restecg': 2, 'thalach': 125.0, 'exang': 1, 'oldpeak': 2.8, 'slope': 2, 'ca': 2.0, 'thal': 7}
    },
    {
        'name': 'Archetype 2 (Healthy Routine)',
        'data': {'age': 45.0, 'sex': 0, 'cp': 2, 'trestbps': 118.0, 'chol': 185.0, 'fbs': 0, 'restecg': 0, 'thalach': 172.0, 'exang': 0, 'oldpeak': 0.2, 'slope': 1, 'ca': 0.0, 'thal': 3}
    },
    {
        'name': 'Archetype 3 (Borderline Risk)',
        'data': {'age': 58.0, 'sex': 1, 'cp': 3, 'trestbps': 136.0, 'chol': 288.0, 'fbs': 0, 'restecg': 1, 'thalach': 148.0, 'exang': 0, 'oldpeak': 1.2, 'slope': 2, 'ca': 1.0, 'thal': 6}
    }
]

print("=== 1. TESTING ARCHETYPE PREDICTIONS & WATERFALLS ===")
for arch in archetypes:
    req = urllib.request.Request(
        f'{base_url}/predict',
        data=json.dumps(arch['data']).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as res:
        p = json.loads(res.read().decode('utf-8'))
        print(f"[{arch['name']}] -> Risk: {p['risk_percentage']}% ({p['risk_tier']}) | Waterfall steps: {len(p['waterfall_steps'])} | Narrative: {p['plain_language_narrative'][:60]}...")

print("\n=== 2. TESTING HOSPITAL NETWORK ===")
with urllib.request.urlopen(f'{base_url}/hospitals') as res:
    hosps = json.loads(res.read().decode('utf-8'))
    print(f"Total Hospitals: {len(hosps)}")
    for h in hosps:
        print(f"  - {h['id']}: {h['name']} | Local Acc: {h['local_accuracy']*100:.1f}%")

print("\n=== 3. TESTING PATIENTS REGISTRY & FILTERING ===")
with urllib.request.urlopen(f'{base_url}/patients') as res:
    all_pts = json.loads(res.read().decode('utf-8'))
    print(f"Total Patients in Registry: {len(all_pts)}")

with urllib.request.urlopen(f'{base_url}/patients?hospital_id=HOSP-01') as res:
    h1_pts = json.loads(res.read().decode('utf-8'))
    print(f"Patients in HOSP-01: {len(h1_pts)}")

print("\n=== 4. TESTING GLOBAL SHAP IMPORTANCE ===")
with urllib.request.urlopen(f'{base_url}/model/explain/global') as res:
    g = json.loads(res.read().decode('utf-8'))
    print(f"Global SHAP Features: {len(g['features'])} | Base: {g['base_value']}")

print("\n=== 5. TESTING MODEL METRICS & PRIVACY ===")
with urllib.request.urlopen(f'{base_url}/model/metrics') as res:
    m = json.loads(res.read().decode('utf-8'))
    print(f"Total Rounds: {m['total_rounds']} | Global Acc: {m['global_metrics']['accuracy']*100:.2f}% | DP Eps: {m['differential_privacy']['epsilon']}")

print("\n>>> ALL TEST CASES PASSED CLEANLY! <<<")
