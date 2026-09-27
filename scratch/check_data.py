import urllib.request
import os
import pandas as pd
import numpy as np

urls = {
    'cleveland': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
    'hungarian': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data',
    'switzerland': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data',
    'va': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data'
}

cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

os.makedirs('data/raw', exist_ok=True)

for name, url in urls.items():
    raw_path = f'data/raw/{name}.csv'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read().decode('utf-8')
        with open(raw_path, 'w', encoding='utf-8') as f:
            f.write(content)
        df = pd.read_csv(raw_path, names=cols, na_values='?')
        print(f"SUCCESS {name}: shape={df.shape}, missing={df.isna().sum().sum()}")
    except Exception as e:
        print(f"FAILED {name}: {e}")
