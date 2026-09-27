"""
FedCare-HHS Federated Simulation Orchestrator
Coordinates multi-hospital federated rounds on demand, manages global model state,
calculates differential privacy budget, and exposes metrics for FastAPI and Streamlit.
"""

import os
import json
import time
import pickle
import numpy as np
from typing import Dict, List, Any, Optional

from model.preprocessing import (
    load_all_datasets,
    get_federated_partitions,
    ClinicalDataPipeline,
    HOSPITALS_INFO,
    FEATURE_NAMES
)
from model.rbfn import RBFNClassifier
from model.explain import ModelExplainer
from federated.client import HospitalNumPyClient
from federated.server import DifferentialPrivacyTracker
import threading

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class FederatedSimulationManager:
    """
    Singleton simulation manager for multi-hospital federated learning.
    Allows on-demand triggering of training rounds, tracking metrics, and real-time explainability.
    """
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> 'FederatedSimulationManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.data_dir = os.path.join(PROJECT_ROOT, "data", "processed")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.selected_features_path = os.path.join(self.data_dir, "selected_features.json")
        self.pipeline_path = os.path.join(self.data_dir, "pipeline.pkl")
        self.global_model_path = os.path.join(self.data_dir, "global_model.json")
        self.history_path = os.path.join(self.data_dir, "federated_history.json")

        # Load fixed HHS selected features
        if os.path.exists(self.selected_features_path):
            with open(self.selected_features_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
                self.selected_features = schema['selected_features']
        else:
            # Default to full features if HHS not run yet
            self.selected_features = FEATURE_NAMES.copy()

        # Load or fit preprocessing pipeline
        if os.path.exists(self.pipeline_path):
            with open(self.pipeline_path, 'rb') as f:
                self.pipeline = pickle.load(f)
        else:
            self.pipeline = ClinicalDataPipeline(selected_features=self.selected_features, use_smote=True)

        # Prepare 4 hospital client partitions and global test set
        self.partitions = get_federated_partitions(
            selected_features=self.selected_features,
            test_ratio=0.2,
            random_state=42
        )
        self.pipeline = self.partitions['pipeline']
        self.global_test_X = self.partitions['global_test']['X_test']
        self.global_test_y = self.partitions['global_test']['y_test']

        # Initialize Global RBFN Model
        self.global_model = RBFNClassifier(
            n_centers=16,
            learning_rate=0.06,
            l2_reg=1e-3,
            max_iter=300,
            random_state=42
        )
        
        # Initialize centers on global reference background
        self.global_model._init_centers_and_sigmas(self.global_test_X)
        n_centers = self.global_model.centers_.shape[0]
        rng = np.random.RandomState(42)
        limit = 1.0 / np.sqrt(n_centers)
        self.global_model.weights_ = rng.uniform(-limit, limit, size=n_centers)
        self.global_model.bias_ = 0.0
        self.global_model.is_fitted = True

        # Initialize Centralized Baseline Model for benchmark comparison
        self.centralized_baseline_metrics = self._train_centralized_baseline()

        # Build simulated hospital clients
        self.clients: Dict[str, HospitalNumPyClient] = {}
        for site_key, client_data in self.partitions['clients'].items():
            # Clone model architecture with identical centers and sigmas
            local_model = RBFNClassifier(
                n_centers=16,
                learning_rate=0.06,
                l2_reg=1e-3,
                max_iter=50,
                random_state=42
            )
            local_model.centers_ = self.global_model.centers_.copy()
            local_model.sigmas_ = self.global_model.sigmas_.copy()
            local_model.weights_ = self.global_model.weights_.copy()
            local_model.bias_ = self.global_model.bias_
            local_model.is_fitted = True

            self.clients[site_key] = HospitalNumPyClient(
                hospital_id=client_data['info']['id'],
                site_key=site_key,
                model=local_model,
                X_train=client_data['X_train'],
                y_train=client_data['y_train'],
                X_val=client_data['X_val'],
                y_val=client_data['y_val'],
                dp_enabled=True,
                clip_norm=1.0,
                noise_multiplier=0.05
            )

        # Differential privacy tracker
        self.dp_tracker = DifferentialPrivacyTracker(target_delta=1e-4)

        # Initialize Explainer before seeding rounds
        self.explainer = ModelExplainer(
            model=self.global_model,
            background_data=self.global_test_X,
            feature_names=self.selected_features
        )
        self.latest_global_shap = None

        # History
        self.round_history: List[Dict[str, Any]] = []
        self._load_or_init_history()

        # Compute initial global SHAP feature importance
        if self.latest_global_shap is None:
            self.latest_global_shap = self.explainer.compute_global_importance(self.global_test_X)

    def _train_centralized_baseline(self) -> Dict[str, float]:
        """Trains an RBFN centrally on pooled training data across all hospitals."""
        X_pooled = np.vstack([c['X_train'] for c in self.partitions['clients'].values()])
        y_pooled = np.concatenate([c['y_train'] for c in self.partitions['clients'].values()])

        central_model = RBFNClassifier(n_centers=16, learning_rate=0.07, l2_reg=1e-3, max_iter=300, random_state=42)
        central_model.fit(X_pooled, y_pooled)
        metrics = central_model.evaluate(self.global_test_X, self.global_test_y)
        return {
            'accuracy': round(float(metrics['accuracy']), 4),
            'precision': round(float(metrics['precision']), 4),
            'recall': round(float(metrics['recall']), 4),
            'f1': round(float(metrics['f1']), 4),
            'loss': round(float(metrics['loss']), 4),
            'roc_auc': round(float(metrics['roc_auc']), 4)
        }

    def _load_or_init_history(self) -> None:
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, 'r', encoding='utf-8') as f:
                    self.round_history = json.load(f)
            except Exception:
                self.round_history = []
        
        # If no history exists, pre-seed with an initial 3 rounds so charts display immediately
        if not self.round_history:
            self._preseed_initial_rounds(num_rounds=3)

    def _preseed_initial_rounds(self, num_rounds: int = 3) -> None:
        """Runs initial training rounds so the system starts with active metrics."""
        for _ in range(num_rounds):
            self.trigger_round(local_epochs=12, dp_enabled=True, noise_multiplier=0.05)

    def trigger_round(self,
                      local_epochs: int = 15,
                      lr: float = 0.06,
                      dp_enabled: bool = True,
                      clip_norm: float = 1.0,
                      noise_multiplier: float = 0.05) -> Dict[str, Any]:
        """
        Executes exactly one federated training round across all simulated hospital nodes.
        1. Distributes global weights to clients
        2. Clients train locally and apply Differential Privacy
        3. Server performs sample-weighted FedAvg
        4. Global & local models are evaluated
        5. Global SHAP is updated
        """
        start_time = time.time()
        current_round = len(self.round_history) + 1
        global_weights = self.global_model.get_weights()

        client_updates = []
        total_samples = 0
        hospital_stats = {}

        # 1. Local Training on Clients
        for site_key, client in self.clients.items():
            client.dp_enabled = dp_enabled
            client.clip_norm = clip_norm
            client.noise_multiplier = noise_multiplier

            fit_config = {"local_epochs": local_epochs, "lr": lr}
            weights_priv, n_samples, train_meta = client.fit(global_weights, fit_config)
            
            # Evaluate client on its local validation partition
            eval_loss, eval_count, eval_meta = client.evaluate(weights_priv, {})

            client_updates.append((weights_priv, n_samples))
            total_samples += n_samples

            info = HOSPITALS_INFO.get(site_key, {})
            hospital_stats[site_key] = {
                'hospital_id': info.get('id', site_key),
                'name': info.get('name', site_key.capitalize()),
                'location': info.get('location', 'Simulated'),
                'train_samples': n_samples,
                'val_samples': eval_count,
                'local_accuracy': round(float(eval_meta.get('accuracy', 0.0)), 4),
                'local_f1': round(float(eval_meta.get('f1', 0.0)), 4),
                'local_loss': round(float(eval_loss), 4),
                'delta_norm': round(float(train_meta.get('delta_norm_raw', 0.0)), 4)
            }

        # 2. Server FedAvg Parameter Aggregation
        # theta_global = sum_k (n_k / total_samples) * theta_k
        agg_w = np.zeros_like(global_weights[0])
        agg_b = 0.0

        for (w_client, b_client), n_samples in client_updates:
            weight_factor = n_samples / max(total_samples, 1)
            agg_w += weight_factor * w_client
            agg_b += weight_factor * float(b_client[0])

        new_global_weights = [agg_w, np.array([agg_b], dtype=float)]
        self.global_model.set_weights(new_global_weights)

        # 3. Global Evaluation on Multi-Center Held-Out Test Set
        global_eval = self.global_model.evaluate(self.global_test_X, self.global_test_y)

        # 4. Differential Privacy Accounting
        current_eps = self.dp_tracker.compute_epsilon(current_round, noise_multiplier) if dp_enabled else None
        duration = round(time.time() - start_time, 2)

        round_record = {
            'round': current_round,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'duration_sec': duration,
            'global_accuracy': round(float(global_eval['accuracy']), 4),
            'global_precision': round(float(global_eval['precision']), 4),
            'global_recall': round(float(global_eval['recall']), 4),
            'global_f1': round(float(global_eval['f1']), 4),
            'global_loss': round(float(global_eval['loss']), 4),
            'global_roc_auc': round(float(global_eval['roc_auc']), 4),
            'dp_epsilon': current_eps,
            'dp_delta': 1e-4 if dp_enabled else None,
            'noise_multiplier': noise_multiplier if dp_enabled else 0.0,
            'clip_norm': clip_norm if dp_enabled else 0.0,
            'centralized_benchmark': self.centralized_baseline_metrics,
            'hospital_metrics': hospital_stats
        }

        self.round_history.append(round_record)
        self._save_state()

        # 5. Recompute Global SHAP Feature Importance
        self.latest_global_shap = self.explainer.compute_global_importance(self.global_test_X)

        return round_record

    def _save_state(self) -> None:
        """Persists global model weights and training round history."""
        with open(self.history_path, 'w', encoding='utf-8') as f:
            json.dump(self.round_history, f, indent=2)

        model_state = {
            'metrics': self.round_history[-1] if self.round_history else {},
            'model_state': self.global_model.export_state(),
            'selected_features': self.selected_features
        }
        with open(self.global_model_path, 'w', encoding='utf-8') as f:
            json.dump(model_state, f, indent=2)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Returns comprehensive model metrics, comparison with centralized baseline, and history."""
        latest = self.round_history[-1] if self.round_history else {}
        return {
            'total_rounds': len(self.round_history),
            'latest_round': latest.get('round', 0),
            'last_trained_timestamp': latest.get('timestamp', 'N/A'),
            'global_metrics': {
                'accuracy': latest.get('global_accuracy', 0.0),
                'precision': latest.get('global_precision', 0.0),
                'recall': latest.get('global_recall', 0.0),
                'f1': latest.get('global_f1', 0.0),
                'loss': latest.get('global_loss', 0.0),
                'roc_auc': latest.get('global_roc_auc', 0.0)
            },
            'centralized_baseline': self.centralized_baseline_metrics,
            'differential_privacy': {
                'dp_enabled': latest.get('dp_epsilon') is not None,
                'epsilon': latest.get('dp_epsilon'),
                'delta': latest.get('dp_delta', 1e-4),
                'noise_multiplier': latest.get('noise_multiplier', 0.05),
                'clip_norm': latest.get('clip_norm', 1.0)
            },
            'hospital_stats': latest.get('hospital_metrics', {}),
            'round_history': self.round_history
        }

    def get_global_explanation(self) -> Dict[str, Any]:
        """Returns the latest global SHAP feature importance."""
        return self.latest_global_shap

    def predict_and_explain(self, raw_patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts raw patient dictionary with clinical fields,
        normalizes via the fixed pipeline, executes global RBFN inference,
        and generates full local SHAP waterfall explanation and clinical narrative.
        """
        x_norm = self.pipeline.transform_single(raw_patient_dict)
        explanation = self.explainer.explain_patient(x_norm, raw_patient_dict)
        
        # Add interaction bonus values for top features
        interactions = self.explainer.compute_shap_interactions(x_norm, top_k=4)
        explanation['interactions'] = interactions['interactions']
        explanation['selected_features'] = self.selected_features
        return explanation
