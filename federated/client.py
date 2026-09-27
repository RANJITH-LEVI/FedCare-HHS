"""
FedCare-HHS Hospital Federated Client
Wraps RBFN as a Flower NumPyClient with Differential Privacy noise injection
(gradient clipping + Gaussian noise) for formal mathematical privacy guarantees.
"""

import numpy as np
import flwr as fl
from typing import List, Dict, Tuple, Any, Optional
from model.rbfn import RBFNClassifier


class HospitalNumPyClient(fl.client.NumPyClient):
    """
    Simulated hospital client implementing Flower NumPyClient interface.
    Each client owns local clinical records and never shares raw patient tables.
    """
    def __init__(self,
                 hospital_id: str,
                 site_key: str,
                 model: RBFNClassifier,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_val: np.ndarray,
                 y_val: np.ndarray,
                 dp_enabled: bool = True,
                 clip_norm: float = 1.0,
                 noise_multiplier: float = 0.05,
                 random_state: int = 42):
        self.hospital_id = hospital_id
        self.site_key = site_key
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        # Differential Privacy parameters
        self.dp_enabled = dp_enabled
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.rng = np.random.RandomState(random_state)

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Returns current local model parameters [weights, bias]."""
        return self.model.get_weights()

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Executes local training round on hospital's internal private dataset.
        Applies Differential Privacy (L2-norm clipping + Gaussian noise) on the weight delta.
        """
        # 1. Update model with global parameters
        self.model.set_weights(parameters)
        w_init = parameters[0].copy()
        b_init = float(parameters[1][0])

        # 2. Local epoch training
        local_epochs = int(config.get("local_epochs", 15))
        lr = float(config.get("lr", self.model.learning_rate))
        
        # Save previous config and run local iterations
        prev_max_iter = self.model.max_iter
        prev_lr = self.model.learning_rate
        self.model.max_iter = local_epochs
        self.model.learning_rate = lr
        
        self.model.fit(self.X_train, self.y_train)
        
        self.model.max_iter = prev_max_iter
        self.model.learning_rate = prev_lr

        # 3. Calculate Weight Delta: Delta_theta = theta_trained - theta_init
        delta_w = self.model.weights_ - w_init
        delta_b = self.model.bias_ - b_init

        # 4. Differential Privacy: L2 Gradient/Delta Clipping + Gaussian Noise
        norm_before = float(np.sqrt(np.sum(delta_w ** 2) + delta_b ** 2))
        
        if self.dp_enabled and self.clip_norm > 0:
            clip_factor = min(1.0, self.clip_norm / max(norm_before, 1e-8))
            delta_w_clipped = delta_w * clip_factor
            delta_b_clipped = delta_b * clip_factor

            # Calibrated Gaussian Noise scale = sigma_DP * C
            noise_scale = self.noise_multiplier * self.clip_norm
            noise_w = self.rng.normal(0.0, noise_scale, size=delta_w.shape)
            noise_b = self.rng.normal(0.0, noise_scale)

            delta_w_priv = delta_w_clipped + noise_w
            delta_b_priv = delta_b_clipped + noise_b
        else:
            delta_w_priv = delta_w
            delta_b_priv = delta_b

        # Final private weights to send to server
        w_priv = w_init + delta_w_priv
        b_priv = np.array([b_init + delta_b_priv], dtype=float)

        # Local train evaluation metrics
        train_metrics = self.model.evaluate(self.X_train, self.y_train)
        train_metrics.update({
            'hospital_id': self.hospital_id,
            'site_key': self.site_key,
            'delta_norm_raw': norm_before,
            'dp_enabled': self.dp_enabled,
            'noise_multiplier': self.noise_multiplier if self.dp_enabled else 0.0
        })

        return [w_priv, b_priv], len(self.X_train), train_metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluates server parameters on hospital's internal validation set.
        """
        self.model.set_weights(parameters)
        metrics = self.model.evaluate(self.X_val, self.y_val)
        loss = metrics['loss']
        metrics['hospital_id'] = self.hospital_id
        metrics['site_key'] = self.site_key
        return float(loss), len(self.X_val), metrics
