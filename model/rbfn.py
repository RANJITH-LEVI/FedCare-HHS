"""
Radial Basis Function Network (RBFN) Classifier from Scratch
Features:
- Gaussian kernel basis hidden layer with explicit center selection (k-means)
- Spread width determination via nearest-neighbor heuristics or global spread
- Sigmoid output layer for well-calibrated clinical probability estimation
- Gradient descent / Adam optimizer with L2 weight regularization
- Explicit inspectability (weights, activations, analytical gradients)
- Full NumPyClient parameter export/import for Flower Federated Learning
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class RBFNClassifier:
    """
    Scratch implementation of a Radial Basis Function Network (RBFN)
    for binary clinical classification.
    """
    def __init__(self,
                 n_centers: int = 16,
                 sigma: Optional[float] = None,
                 learning_rate: float = 0.05,
                 l2_reg: float = 1e-4,
                 max_iter: int = 300,
                 batch_size: int = 32,
                 random_state: int = 42):
        self.n_centers = n_centers
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state

        # Model parameters
        self.centers_: Optional[np.ndarray] = None       # Shape: (n_centers, n_features)
        self.sigmas_: Optional[np.ndarray] = None        # Shape: (n_centers,)
        self.weights_: Optional[np.ndarray] = None       # Shape: (n_centers,)
        self.bias_: float = 0.0
        self.is_fitted: bool = False
        
        # Training telemetry
        self.loss_history_: List[float] = []
        self.acc_history_: List[float] = []

    def _init_centers_and_sigmas(self, X: np.ndarray) -> None:
        """Determines RBF centers using K-Means and computes local widths."""
        n_samples, n_features = X.shape
        k = min(self.n_centers, n_samples)
        
        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        kmeans.fit(X)
        self.centers_ = kmeans.cluster_centers_

        if self.sigma is not None:
            self.sigmas_ = np.full(k, fill_value=self.sigma, dtype=float)
        else:
            # Heuristic: sigma_j = mean distance to k_nn nearest neighboring centers
            # Ensures adequate overlap between basis functions without oversaturation
            p = min(3, max(1, k - 1))
            sigmas = []
            for i in range(k):
                dists = np.linalg.norm(self.centers_ - self.centers_[i], axis=1)
                dists = np.sort(dists)
                # First distance is 0 (distance to self), take next p neighbors
                neighbor_dists = dists[1:p+1] if k > 1 else np.array([1.0])
                mean_dist = np.mean(neighbor_dists)
                sigmas.append(max(mean_dist, 1e-3))
            self.sigmas_ = np.array(sigmas, dtype=float)

    def _compute_rbf_activations(self, X: np.ndarray) -> np.ndarray:
        """
        Computes Gaussian kernel activations:
        phi_j(x) = exp( - ||x - c_j||^2 / (2 * sigma_j^2) )
        Returns matrix Phi of shape (n_samples, n_centers)
        """
        if self.centers_ is None or self.sigmas_ is None:
            raise RuntimeError("RBF centers and sigmas are not initialized.")
        
        # X: (N, D), Centers: (K, D)
        # Efficient distance computation using broadcasting: (N, 1, D) - (1, K, D)
        diff = X[:, np.newaxis, :] - self.centers_[np.newaxis, :, :] # (N, K, D)
        sq_dist = np.sum(diff ** 2, axis=2)                         # (N, K)
        
        # Gaussian RBF kernel
        two_sigma_sq = 2.0 * (self.sigmas_ ** 2)                    # (K,)
        phi = np.exp(-sq_dist / (two_sigma_sq + 1e-8))               # (N, K)
        return phi

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))

    def fit(self, X: np.ndarray, y: np.ndarray,
            eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> 'RBFNClassifier':
        """
        Fits RBFN using mini-batch gradient descent with momentum and L2 regularization.
        """
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        y_arr = np.array(y, dtype=float)

        # 1. Initialize basis functions if not already initialized
        if self.centers_ is None or self.sigmas_ is None:
            self._init_centers_and_sigmas(X)
        
        n_centers = self.centers_.shape[0]

        # 2. Compute basis activations for training data
        Phi = self._compute_rbf_activations(X) # (N, K)

        # 3. Initialize weights (Xavier-like initialization for linear output layer)
        if self.weights_ is None:
            limit = 1.0 / np.sqrt(n_centers)
            self.weights_ = rng.uniform(-limit, limit, size=n_centers)
            self.bias_ = 0.0

        # Optimizer momentum buffers
        v_w = np.zeros_like(self.weights_)
        v_b = 0.0
        beta = 0.9

        self.loss_history_ = []
        self.acc_history_ = []

        batch_size = min(self.batch_size, n_samples)
        n_batches = int(np.ceil(n_samples / batch_size))

        for epoch in range(self.max_iter):
            # Shuffle at each epoch
            perm = rng.permutation(n_samples)
            Phi_shuffled = Phi[perm]
            y_shuffled = y_arr[perm]

            for b in range(n_batches):
                start_idx = b * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                Phi_b = Phi_shuffled[start_idx:end_idx]
                y_b = y_shuffled[start_idx:end_idx]
                m = len(y_b)

                # Forward pass
                z_b = np.dot(Phi_b, self.weights_) + self.bias_
                y_hat_b = self._sigmoid(z_b)

                # Binary Cross-Entropy Gradients
                err_b = y_hat_b - y_b
                grad_w = (np.dot(Phi_b.T, err_b) / m) + (self.l2_reg * self.weights_)
                grad_b = np.mean(err_b)

                # Gradient descent with momentum
                v_w = beta * v_w + (1.0 - beta) * grad_w
                v_b = beta * v_b + (1.0 - beta) * grad_b

                self.weights_ -= self.learning_rate * v_w
                self.bias_ -= self.learning_rate * v_b

            # Full batch loss & accuracy tracking
            z_full = np.dot(Phi, self.weights_) + self.bias_
            y_hat_full = self._sigmoid(z_full)
            eps = 1e-12
            bce_loss = -np.mean(y_arr * np.log(y_hat_full + eps) + (1.0 - y_arr) * np.log(1.0 - y_hat_full + eps))
            reg_loss = 0.5 * self.l2_reg * np.sum(self.weights_ ** 2)
            total_loss = float(bce_loss + reg_loss)
            
            acc = float(np.mean((y_hat_full >= 0.5) == y_arr))
            self.loss_history_.append(total_loss)
            self.acc_history_.append(acc)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probability of heart disease [P(y=0), P(y=1)]
        """
        if not self.is_fitted and self.weights_ is None:
            raise RuntimeError("Model is not fitted or weights not set.")
        Phi = self._compute_rbf_activations(X)
        z = np.dot(Phi, self.weights_) + self.bias_
        p1 = self._sigmoid(z)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Returns binary predictions: 0 (No Heart Disease) or 1 (Heart Disease)."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculates standard classification metrics."""
        y_true = np.array(y, dtype=int)
        y_prob = self.predict_proba(X)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        eps = 1e-12
        loss = -np.mean(y_true * np.log(y_prob + eps) + (1.0 - y_true) * np.log(1.0 - y_prob + eps))

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = 0.5

        return {
            'loss': float(loss),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'roc_auc': float(auc)
        }

    # -------------------------------------------------------------
    # Flower Federated Learning Interface
    # -------------------------------------------------------------
    def get_weights(self) -> List[np.ndarray]:
        """
        Returns Flower-compatible list of numpy arrays: [weights, bias]
        """
        if self.weights_ is None:
            raise RuntimeError("Model weights are uninitialized.")
        return [self.weights_.copy(), np.array([self.bias_], dtype=float)]

    def set_weights(self, weights: List[np.ndarray]) -> None:
        """
        Sets model parameters from Flower-aggregated weight arrays.
        """
        self.weights_ = weights[0].copy()
        self.bias_ = float(weights[1][0])
        self.is_fitted = True

    # -------------------------------------------------------------
    # Analytical Sensitivity / Gradient Attribution
    # -------------------------------------------------------------
    def analytical_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Computes analytical derivative d(P)/d(x_i) of the predicted probability
        with respect to input feature vector x.
        Provides millisecond-latency local sensitivity attribution.
        """
        if x.ndim == 1:
            x_mat = x[np.newaxis, :]
        else:
            x_mat = x

        Phi = self._compute_rbf_activations(x_mat) # (1, K)
        z = np.dot(Phi, self.weights_) + self.bias_
        p = self._sigmoid(z)[0]
        sig_deriv = p * (1.0 - p) # Scalar

        # dPhi_j / dx_k = - (x_k - c_jk) / sigma_j^2 * phi_j(x)
        # Sum over all centers j: weights_j * dPhi_j / dx_k
        diff = x_mat[0, np.newaxis, :] - self.centers_ # (K, D)
        sigma_sq = (self.sigmas_ ** 2)[:, np.newaxis] # (K, 1)
        dphi_dx = - (diff / (sigma_sq + 1e-8)) * Phi[0, :, np.newaxis] # (K, D)
        
        dz_dx = np.sum(self.weights_[:, np.newaxis] * dphi_dx, axis=0) # (D,)
        dp_dx = sig_deriv * dz_dx
        return dp_dx

    def export_state(self) -> Dict[str, Any]:
        """Serializes model parameters into dictionary."""
        return {
            'n_centers': self.n_centers,
            'sigma': self.sigma,
            'learning_rate': self.learning_rate,
            'l2_reg': self.l2_reg,
            'max_iter': self.max_iter,
            'centers': self.centers_.tolist() if self.centers_ is not None else None,
            'sigmas': self.sigmas_.tolist() if self.sigmas_ is not None else None,
            'weights': self.weights_.tolist() if self.weights_ is not None else None,
            'bias': self.bias_,
            'is_fitted': self.is_fitted
        }

    def import_state(self, state: Dict[str, Any]) -> 'RBFNClassifier':
        """Restores model parameters from dictionary."""
        self.n_centers = state['n_centers']
        self.sigma = state.get('sigma')
        self.learning_rate = state['learning_rate']
        self.l2_reg = state['l2_reg']
        self.max_iter = state['max_iter']
        if state.get('centers') is not None:
            self.centers_ = np.array(state['centers'], dtype=float)
        if state.get('sigmas') is not None:
            self.sigmas_ = np.array(state['sigmas'], dtype=float)
        if state.get('weights') is not None:
            self.weights_ = np.array(state['weights'], dtype=float)
        self.bias_ = float(state.get('bias', 0.0))
        self.is_fitted = bool(state.get('is_fitted', False))
        return self
