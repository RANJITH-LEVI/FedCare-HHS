"""
FedCare-HHS Federated Strategy & Server Coordination
Implements FedAvg strategy with DP privacy budget tracking and server-side global evaluation.
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from model.rbfn import RBFNClassifier


class DifferentialPrivacyTracker:
    """
    Computes formal (epsilon, delta) Differential Privacy expenditure
    using Gaussian Mechanism composition over federated training rounds.
    """
    def __init__(self, target_delta: float = 1e-4):
        self.target_delta = target_delta

    def compute_epsilon(self, rounds: int, noise_multiplier: float) -> float:
        """
        Calculates epsilon under Gaussian mechanism composition:
        eps = sqrt(2 * T * ln(1.25 / delta)) / noise_multiplier
        """
        if rounds <= 0 or noise_multiplier <= 0:
            return float('inf')
        
        log_term = math.log(1.25 / self.target_delta)
        eps = math.sqrt(2.0 * rounds * log_term) / noise_multiplier
        return round(float(eps), 3)


class FedCareStrategy(fl.server.strategy.FedAvg):
    """
    Custom Flower FedAvg strategy supporting per-hospital tracking,
    DP budget accounting, and multi-center global validation.
    """
    def __init__(self,
                 global_model: RBFNClassifier,
                 eval_data: Tuple[np.ndarray, np.ndarray],
                 dp_enabled: bool = True,
                 noise_multiplier: float = 0.05,
                 min_fit_clients: int = 4,
                 min_available_clients: int = 4,
                 **kwargs):
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            **kwargs
        )
        self.global_model = global_model
        self.eval_X, self.eval_y = eval_data
        self.dp_enabled = dp_enabled
        self.noise_multiplier = noise_multiplier
        self.dp_tracker = DifferentialPrivacyTracker(target_delta=1e-4)

        # Telemetry
        self.round_history: List[Dict[str, Any]] = []
        self.latest_hospital_metrics: Dict[str, Dict[str, float]] = {}

    def aggregate_fit(self,
                      server_round: int,
                      results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
                      failures: List[BaseException]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregates client weights via sample-weighted FedAvg and records per-hospital training stats.
        """
        if not results:
            return None, {}

        # 1. Standard FedAvg parameter aggregation
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_params is None:
            return None, metrics

        # 2. Convert to ndarrays and update global model
        ndarrays = parameters_to_ndarrays(aggregated_params)
        self.global_model.set_weights(ndarrays)

        # 3. Extract and log per-hospital local results
        for _, fit_res in results:
            custom_metrics = fit_res.metrics
            hosp_id = str(custom_metrics.get("site_key", custom_metrics.get("hospital_id", "unknown")))
            self.latest_hospital_metrics[hosp_id] = {
                'loss': float(custom_metrics.get('loss', 0.0)),
                'accuracy': float(custom_metrics.get('accuracy', 0.0)),
                'f1': float(custom_metrics.get('f1', 0.0)),
                'num_samples': int(fit_res.num_examples)
            }

        # 4. Server-side evaluation on held-out multi-center test set
        eval_metrics = self.global_model.evaluate(self.eval_X, self.eval_y)

        # 5. Differential Privacy tracking
        if self.dp_enabled:
            current_eps = self.dp_tracker.compute_epsilon(server_round, self.noise_multiplier)
        else:
            current_eps = None

        round_record = {
            'round': server_round,
            'global_accuracy': round(eval_metrics['accuracy'], 4),
            'global_precision': round(eval_metrics['precision'], 4),
            'global_recall': round(eval_metrics['recall'], 4),
            'global_f1': round(eval_metrics['f1'], 4),
            'global_loss': round(eval_metrics['loss'], 4),
            'global_roc_auc': round(eval_metrics['roc_auc'], 4),
            'dp_epsilon': current_eps,
            'dp_delta': 1e-4 if self.dp_enabled else None,
            'hospital_metrics': self.latest_hospital_metrics.copy()
        }
        self.round_history.append(round_record)

        return aggregated_params, eval_metrics
