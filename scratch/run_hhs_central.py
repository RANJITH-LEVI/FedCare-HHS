"""
Runs Harris Hawks Search (HHS) centrally on the reference UCI Cleveland dataset
to fix the feature schema and establish the baseline benchmark.
"""

import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.preprocessing import load_raw_dataset, get_reference_uci_split, ClinicalDataPipeline, FEATURE_NAMES
from model.hhs import HarrisHawksSelector
from model.rbfn import RBFNClassifier

print("Step 1: Loading UCI Cleveland reference dataset...")
X_train, X_test, y_train, y_test = get_reference_uci_split(test_size=0.2, random_state=42)

print("Step 2: Fitting initial preprocessing pipeline...")
pipeline_full = ClinicalDataPipeline(selected_features=FEATURE_NAMES, use_smote=True)
X_train_trans, y_train_res = pipeline_full.fit_transform(X_train, y_train)
X_test_trans = pipeline_full.transform(X_test)

print("Step 3: Executing Harris Hawks Search (HHS) Metaheuristic...")
hhs = HarrisHawksSelector(n_hawks=20, max_iter=25, alpha=0.92, random_state=42)
hhs.fit(X_train_trans, y_train_res, feature_names=FEATURE_NAMES, evaluator_cls=RBFNClassifier)

selected_features = hhs.selected_features_
print(f"HHS Completed! Selected {len(selected_features)}/{len(FEATURE_NAMES)} features:")
for f in selected_features:
    print(f"  + {f}")

schema_path = "data/processed/selected_features.json"
hhs.save_selection_schema(schema_path)
print(f"Feature schema saved to {schema_path}")

print("\nStep 4: Training baseline RBFN on fixed HHS-selected features...")
pipeline_hhs = ClinicalDataPipeline(selected_features=selected_features, use_smote=True)
X_tr_hhs, y_tr_hhs = pipeline_hhs.fit_transform(X_train, y_train)
X_te_hhs = pipeline_hhs.transform(X_test)
pipeline_hhs.save("data/processed/pipeline.pkl")

rbfn_baseline = RBFNClassifier(n_centers=16, learning_rate=0.08, l2_reg=1e-3, max_iter=350, random_state=42)
rbfn_baseline.fit(X_tr_hhs, y_tr_hhs)

metrics = rbfn_baseline.evaluate(X_te_hhs, y_test)
print("\n=== HHS-RBFN STANDALONE BASELINE METRICS ===")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

# Save baseline model state
baseline_state = {
    'metrics': metrics,
    'model_state': rbfn_baseline.export_state(),
    'selected_features': selected_features
}
with open("data/processed/baseline_model.json", "w", encoding="utf-8") as f:
    json.dump(baseline_state, f, indent=2)
print("Baseline model state saved to data/processed/baseline_model.json")
