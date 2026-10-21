import urllib.request
import json

try:
    res = urllib.request.urlopen("http://127.0.0.1:8000/hospitals/datasets")
    data = json.loads(res.read().decode('utf-8'))
    print(f"Loaded {len(data)} hospital datasets:")
    for d in data:
        print(f" * {d['name']} ({d['site_key']}): {d['total_records']} total records, {d['disease_prevalence_pct']}% prevalence, {len(d['records'])} sample records")
except Exception as e:
    print(f"Error testing datasets API: {e}")
