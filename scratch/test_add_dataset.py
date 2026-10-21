import urllib.request
import json

# Synthetic test records for a new clinical center: "Boston Cardiology Institute"
test_records = [
    {"age": 55, "sex": 1, "cp": 2, "trestbps": 135, "chol": 230, "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0, "oldpeak": 0.8, "slope": 1, "ca": 0, "thal": 3, "target": 0},
    {"age": 62, "sex": 1, "cp": 4, "trestbps": 150, "chol": 270, "fbs": 1, "restecg": 2, "thalach": 125, "exang": 1, "oldpeak": 2.2, "slope": 2, "ca": 2, "thal": 7, "target": 1},
    {"age": 48, "sex": 0, "cp": 1, "trestbps": 120, "chol": 210, "fbs": 0, "restecg": 0, "thalach": 172, "exang": 0, "oldpeak": 0.2, "slope": 1, "ca": 0, "thal": 3, "target": 0},
    {"age": 70, "sex": 1, "cp": 4, "trestbps": 160, "chol": 285, "fbs": 0, "restecg": 1, "thalach": 110, "exang": 1, "oldpeak": 3.0, "slope": 3, "ca": 3, "thal": 6, "target": 1},
    {"age": 58, "sex": 1, "cp": 3, "trestbps": 140, "chol": 245, "fbs": 0, "restecg": 0, "thalach": 148, "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 1, "thal": 3, "target": 0},
    {"age": 65, "sex": 0, "cp": 4, "trestbps": 155, "chol": 310, "fbs": 1, "restecg": 2, "thalach": 118, "exang": 1, "oldpeak": 2.6, "slope": 2, "ca": 2, "thal": 7, "target": 1},
    {"age": 44, "sex": 1, "cp": 2, "trestbps": 125, "chol": 220, "fbs": 0, "restecg": 0, "thalach": 168, "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3, "target": 0},
    {"age": 67, "sex": 1, "cp": 4, "trestbps": 145, "chol": 260, "fbs": 0, "restecg": 1, "thalach": 130, "exang": 1, "oldpeak": 1.8, "slope": 2, "ca": 1, "thal": 7, "target": 1},
    {"age": 52, "sex": 0, "cp": 3, "trestbps": 130, "chol": 235, "fbs": 0, "restecg": 0, "thalach": 155, "exang": 0, "oldpeak": 0.6, "slope": 1, "ca": 0, "thal": 3, "target": 0},
    {"age": 61, "sex": 1, "cp": 4, "trestbps": 148, "chol": 290, "fbs": 1, "restecg": 2, "thalach": 122, "exang": 1, "oldpeak": 2.4, "slope": 2, "ca": 2, "thal": 7, "target": 1}
]

payload = {
    "site_key": "boston",
    "name": "Boston Cardiology Institute",
    "location": "Boston, MA, USA",
    "records": test_records,
    "local_epochs": 10
}

data_bytes = json.dumps(payload).encode('utf-8')
req = urllib.request.Request(
    "http://127.0.0.1:8000/hospitals/dataset/add-json",
    data=data_bytes,
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode('utf-8'))
        print("Success! Dataset added and trained automatically:")
        print(f"Hospital: {res['hospital_name']} ({res['hospital_id']})")
        print(f"New Training Round: #{res['round_number']}")
        print(f"Updated Global Accuracy: {res['global_accuracy'] * 100:.2f}%")
        print(f"Duration: {res['duration_seconds']}s")
except Exception as e:
    print(f"Error testing add dataset: {e}")
