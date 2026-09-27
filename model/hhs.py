"""
Harris Hawks Search (HHS) Metaheuristic Feature Selector
Implements the biological predation dynamics of Harris Hawks (Heidari et al., 2019):
- Exploration phase (random perching & population centroid tracking)
- Escaping energy transition
- Exploitation phase (Soft besiege, Hard besiege, Surprise pounce with Levy flight dives)
- Sigmoid transfer function mapping continuous search space to binary feature masks
"""

import math
import json
import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from sklearn.model_selection import StratifiedKFold
from model.preprocessing import FEATURE_NAMES


class HarrisHawksSelector:
    """
    Harris Hawks Optimization (HHO/HHS) for binary feature selection.
    """
    def __init__(self,
                 n_hawks: int = 25,
                 max_iter: int = 35,
                 alpha: float = 0.92,
                 lb: float = -4.0,
                 ub: float = 4.0,
                 random_state: int = 42):
        self.n_hawks = n_hawks
        self.max_iter = max_iter
        self.alpha = alpha  # Weight for classification accuracy vs parsimony
        self.lb = lb
        self.ub = ub
        self.random_state = random_state

        self.best_rabbit_pos_: Optional[np.ndarray] = None
        self.best_mask_: Optional[np.ndarray] = None
        self.best_fitness_: float = -1.0
        self.best_accuracy_: float = 0.0
        self.selected_features_: List[str] = []
        self.history_: Dict[str, List[float]] = {
            'iteration': [],
            'best_fitness': [],
            'best_accuracy': [],
            'num_features': []
        }

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10.0, 10.0)))

    def _pos_to_mask(self, pos: np.ndarray) -> np.ndarray:
        """Converts continuous position vector to binary feature mask."""
        prob = self._sigmoid(pos)
        mask = (prob >= 0.5).astype(int)
        # Prevent degenerate all-zero mask: select top 2 features by probability
        if np.sum(mask) == 0:
            top_indices = np.argsort(prob)[-2:]
            mask[top_indices] = 1
        return mask

    @staticmethod
    def _levy_flight(dim: int, beta: float = 1.5, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """Computes Mantegna's algorithm for Levy flight step."""
        if rng is None:
            rng = np.random.RandomState()
        sigma_u = (
            math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
            (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))
        ) ** (1 / beta)
        u = rng.normal(0, sigma_u, size=dim)
        v = rng.normal(0, 1.0, size=dim)
        step = u / (np.abs(v) ** (1 / beta) + 1e-10)
        return 0.01 * step

    def _evaluate_fitness(self, mask: np.ndarray,
                          X_train: np.ndarray, y_train: np.ndarray,
                          evaluator_cls: Any) -> Tuple[float, float]:
        """
        Evaluates a candidate feature mask using Stratified 3-Fold Cross-Validation.
        Fitness = alpha * Accuracy + (1 - alpha) * (1 - num_features / D)
        """
        feat_indices = np.where(mask == 1)[0]
        d_total = len(mask)
        d_selected = len(feat_indices)

        if d_selected == 0:
            return 0.0, 0.0

        X_sub = X_train[:, feat_indices]
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        acc_scores = []

        for tr_idx, val_idx in cv.split(X_sub, y_train):
            X_tr, X_val = X_sub[tr_idx], X_sub[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            model = evaluator_cls(n_centers=min(12, len(X_tr)), max_iter=150, random_state=self.random_state)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            acc_scores.append(float(np.mean(preds == y_val)))

        mean_acc = float(np.mean(acc_scores))
        parsimony = 1.0 - (d_selected / d_total)
        fitness = (self.alpha * mean_acc) + ((1.0 - self.alpha) * parsimony)
        return float(fitness), float(mean_acc)

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None,
            evaluator_cls: Optional[Any] = None) -> 'HarrisHawksSelector':
        """
        Executes Harris Hawks Search on reference training dataset.
        """
        if evaluator_cls is None:
            from model.rbfn import RBFNClassifier
            evaluator_cls = RBFNClassifier

        rng = np.random.RandomState(self.random_state)
        feat_names = feature_names or FEATURE_NAMES
        dim = X.shape[1]

        # 1. Initialize hawk population positions randomly in [lb, ub]
        hawks = rng.uniform(self.lb, self.ub, size=(self.n_hawks, dim))
        fitnesses = np.zeros(self.n_hawks)
        accuracies = np.zeros(self.n_hawks)

        # Initial evaluation
        for i in range(self.n_hawks):
            mask = self._pos_to_mask(hawks[i])
            fit, acc = self._evaluate_fitness(mask, X, y, evaluator_cls)
            fitnesses[i] = fit
            accuracies[i] = acc

        # Best hawk (Prey / Rabbit)
        best_idx = np.argmax(fitnesses)
        self.best_rabbit_pos_ = hawks[best_idx].copy()
        self.best_fitness_ = float(fitnesses[best_idx])
        self.best_accuracy_ = float(accuracies[best_idx])
        self.best_mask_ = self._pos_to_mask(self.best_rabbit_pos_)

        # 2. Main Iteration Loop
        for t in range(self.max_iter):
            # Population centroid X_m
            X_m = np.mean(hawks, axis=0)

            for i in range(self.n_hawks):
                # Escaping energy E
                E0 = 2.0 * rng.uniform() - 1.0  # [-1, 1]
                E = 2.0 * E0 * (1.0 - (t / self.max_iter))  # Decreases from ~2 to 0

                # ----------------- EXPLORATION PHASE (|E| >= 1) -----------------
                if abs(E) >= 1.0:
                    q = rng.uniform()
                    r1 = rng.uniform()
                    r2 = rng.uniform()
                    r3 = rng.uniform()
                    r4 = rng.uniform()

                    if q >= 0.5:
                        # Perch based on other random hawks
                        rand_hawk_idx = rng.randint(0, self.n_hawks)
                        X_rand = hawks[rand_hawk_idx]
                        new_pos = X_rand - r1 * np.abs(X_rand - 2.0 * r2 * hawks[i])
                    else:
                        # Perch based on rabbit position & population average
                        new_pos = (self.best_rabbit_pos_ - X_m) - r3 * (self.lb + r4 * (self.ub - self.lb))

                    new_pos = np.clip(new_pos, self.lb, self.ub)
                    new_mask = self._pos_to_mask(new_pos)
                    new_fit, new_acc = self._evaluate_fitness(new_mask, X, y, evaluator_cls)

                    if new_fit > fitnesses[i]:
                        hawks[i] = new_pos
                        fitnesses[i] = new_fit
                        accuracies[i] = new_acc

                # ----------------- EXPLOITATION PHASE (|E| < 1) -----------------
                else:
                    r = rng.uniform() # Chance of prey escaping
                    J = 2.0 * (1.0 - rng.uniform()) # Jump strength of rabbit
                    delta_X = self.best_rabbit_pos_ - hawks[i]

                    # Case 1: Soft Besiege (r >= 0.5 and |E| >= 0.5)
                    if r >= 0.5 and abs(E) >= 0.5:
                        new_pos = delta_X - E * np.abs(J * self.best_rabbit_pos_ - hawks[i])
                        new_pos = np.clip(new_pos, self.lb, self.ub)
                        new_mask = self._pos_to_mask(new_pos)
                        new_fit, new_acc = self._evaluate_fitness(new_mask, X, y, evaluator_cls)
                        if new_fit > fitnesses[i]:
                            hawks[i] = new_pos
                            fitnesses[i] = new_fit
                            accuracies[i] = new_acc

                    # Case 2: Hard Besiege (r >= 0.5 and |E| < 0.5)
                    elif r >= 0.5 and abs(E) < 0.5:
                        new_pos = self.best_rabbit_pos_ - E * np.abs(delta_X)
                        new_pos = np.clip(new_pos, self.lb, self.ub)
                        new_mask = self._pos_to_mask(new_pos)
                        new_fit, new_acc = self._evaluate_fitness(new_mask, X, y, evaluator_cls)
                        if new_fit > fitnesses[i]:
                            hawks[i] = new_pos
                            fitnesses[i] = new_fit
                            accuracies[i] = new_acc

                    # Case 3: Soft Besiege with Progressive Rapid Dives (r < 0.5 and |E| >= 0.5)
                    elif r < 0.5 and abs(E) >= 0.5:
                        Y = self.best_rabbit_pos_ - E * np.abs(J * self.best_rabbit_pos_ - hawks[i])
                        Y = np.clip(Y, self.lb, self.ub)
                        mask_Y = self._pos_to_mask(Y)
                        fit_Y, acc_Y = self._evaluate_fitness(mask_Y, X, y, evaluator_cls)

                        Z = Y + rng.uniform(0, 1, size=dim) * self._levy_flight(dim, rng=rng)
                        Z = np.clip(Z, self.lb, self.ub)
                        mask_Z = self._pos_to_mask(Z)
                        fit_Z, acc_Z = self._evaluate_fitness(mask_Z, X, y, evaluator_cls)

                        if fit_Y > fitnesses[i]:
                            hawks[i] = Y
                            fitnesses[i] = fit_Y
                            accuracies[i] = acc_Y
                        elif fit_Z > fitnesses[i]:
                            hawks[i] = Z
                            fitnesses[i] = fit_Z
                            accuracies[i] = acc_Z

                    # Case 4: Hard Besiege with Progressive Rapid Dives (r < 0.5 and |E| < 0.5)
                    elif r < 0.5 and abs(E) < 0.5:
                        Y = self.best_rabbit_pos_ - E * np.abs(J * self.best_rabbit_pos_ - X_m)
                        Y = np.clip(Y, self.lb, self.ub)
                        mask_Y = self._pos_to_mask(Y)
                        fit_Y, acc_Y = self._evaluate_fitness(mask_Y, X, y, evaluator_cls)

                        Z = Y + rng.uniform(0, 1, size=dim) * self._levy_flight(dim, rng=rng)
                        Z = np.clip(Z, self.lb, self.ub)
                        mask_Z = self._pos_to_mask(Z)
                        fit_Z, acc_Z = self._evaluate_fitness(mask_Z, X, y, evaluator_cls)

                        if fit_Y > fitnesses[i]:
                            hawks[i] = Y
                            fitnesses[i] = fit_Y
                            accuracies[i] = acc_Y
                        elif fit_Z > fitnesses[i]:
                            hawks[i] = Z
                            fitnesses[i] = fit_Z
                            accuracies[i] = acc_Z

            # Update best rabbit
            current_best_idx = np.argmax(fitnesses)
            if fitnesses[current_best_idx] > self.best_fitness_:
                self.best_rabbit_pos_ = hawks[current_best_idx].copy()
                self.best_fitness_ = float(fitnesses[current_best_idx])
                self.best_accuracy_ = float(accuracies[current_best_idx])
                self.best_mask_ = self._pos_to_mask(self.best_rabbit_pos_)

            self.history_['iteration'].append(t + 1)
            self.history_['best_fitness'].append(self.best_fitness_)
            self.history_['best_accuracy'].append(self.best_accuracy_)
            self.history_['num_features'].append(int(np.sum(self.best_mask_)))

        # Selected features list
        selected_indices = np.where(self.best_mask_ == 1)[0]
        self.selected_features_ = [feat_names[idx] for idx in selected_indices]
        return self

    def save_selection_schema(self, filepath: str) -> None:
        """Saves selected feature schema and optimization telemetry to JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        schema = {
            'selected_features': self.selected_features_,
            'all_features': FEATURE_NAMES,
            'best_mask': self.best_mask_.tolist() if self.best_mask_ is not None else [],
            'best_fitness': self.best_fitness_,
            'best_accuracy': self.best_accuracy_,
            'num_selected': len(self.selected_features_),
            'history': self.history_
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=2)

    @classmethod
    def load_selection_schema(cls, filepath: str) -> Dict[str, Any]:
        """Loads fixed feature schema from JSON."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
