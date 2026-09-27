"""
Explainability Engine for FedCare-HHS using SHAP
Provides:
1. Local per-patient SHAP waterfall attributions (risk-increasing vs risk-decreasing)
2. Plain-language clinical reasoning narrative
3. Global SHAP feature importance recomputed per federated round
4. Pairwise feature interaction metrics
5. Pure structured JSON export for FastAPI and frontend rendering
"""

import numpy as np
import shap
from typing import List, Dict, Any, Optional
from model.preprocessing import FEATURE_METADATA


class ModelExplainer:
    """
    Model-agnostic KernelExplainer tailored for RBFN cardiovascular risk predictions.
    """
    def __init__(self, model: Any, background_data: np.ndarray, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        
        # Summarize background dataset using shap.kmeans or sample for high speed
        n_bg = min(len(background_data), 30)
        indices = np.linspace(0, len(background_data) - 1, n_bg, dtype=int)
        self.background_data = background_data[indices]
        
        # Prediction wrapper targeting cardiac risk probability P(y=1)
        self.predict_fn = lambda x: self.model.predict_proba(x)[:, 1]
        
        self.explainer = shap.KernelExplainer(self.predict_fn, self.background_data)
        self.expected_value = float(self.explainer.expected_value)
        if isinstance(self.expected_value, (list, np.ndarray)):
            self.expected_value = float(self.expected_value[-1])

    def explain_patient(self,
                        x_normalized: np.ndarray,
                        raw_patient_dict: Optional[Dict[str, Any]] = None,
                        nsamples: int = 60) -> Dict[str, Any]:
        """
        Computes local SHAP explanation for a single patient.
        Returns full structured JSON with waterfall coordinates and clinical narrative.
        """
        if x_normalized.ndim == 1:
            x_norm = x_normalized[np.newaxis, :]
        else:
            x_norm = x_normalized

        # Compute SHAP values
        shap_vals = self.explainer.shap_values(x_norm, nsamples=nsamples)
        if isinstance(shap_vals, list):
            # If multi-output, take positive class
            s_vals = np.array(shap_vals[-1][0], dtype=float)
        else:
            s_vals = np.array(shap_vals[0], dtype=float)

        risk_score = float(self.predict_fn(x_norm)[0])
        
        # Determine clinical risk category
        if risk_score < 0.25:
            risk_tier = "Low"
            risk_badge = "success"
        elif risk_score < 0.50:
            risk_tier = "Borderline / Moderate"
            risk_badge = "warning"
        elif risk_score < 0.75:
            risk_tier = "High"
            risk_badge = "danger"
        else:
            risk_tier = "Critical"
            risk_badge = "critical"

        # Construct feature impact items
        attributions = []
        top_risk_factors = []
        top_protective_factors = []

        for idx, feat in enumerate(self.feature_names):
            s_val = float(s_vals[idx])
            raw_val = raw_patient_dict.get(feat) if raw_patient_dict else None
            meta = FEATURE_METADATA.get(feat, {})
            label = meta.get('label', feat.upper())
            unit = meta.get('unit', '')

            # Display formatting
            if raw_val is not None:
                if meta.get('type') == 'categorical' and 'options' in meta:
                    val_str = meta['options'].get(int(raw_val), str(raw_val))
                elif isinstance(raw_val, float):
                    val_str = f"{raw_val:.1f} {unit}".strip()
                else:
                    val_str = f"{raw_val} {unit}".strip()
            else:
                val_str = f"{float(x_norm[0, idx]):.2f} (std)"

            effect = "increases_risk" if s_val > 0 else "decreases_risk" if s_val < 0 else "neutral"

            item = {
                'feature': feat,
                'label': label,
                'raw_value': raw_val,
                'display_value': val_str,
                'shap_value': round(s_val, 4),
                'abs_shap': round(abs(s_val), 4),
                'effect': effect
            }
            attributions.append(item)

            if s_val > 0.001:
                top_risk_factors.append(item)
            elif s_val < -0.001:
                top_protective_factors.append(item)

        # Sort attributions by magnitude of impact
        attributions.sort(key=lambda x: x['abs_shap'], reverse=True)
        top_risk_factors.sort(key=lambda x: x['shap_value'], reverse=True)
        top_protective_factors.sort(key=lambda x: x['shap_value'])  # most negative first

        # Generate Waterfall chart coordinates
        cumulative = self.expected_value
        waterfall_steps = [{
            'label': 'Base Prevalence',
            'delta': round(self.expected_value, 4),
            'cumulative': round(cumulative, 4),
            'type': 'base'
        }]

        for attr in attributions:
            prev = cumulative
            cumulative += attr['shap_value']
            waterfall_steps.append({
                'label': attr['label'],
                'feature': attr['feature'],
                'delta': attr['shap_value'],
                'cumulative': round(cumulative, 4),
                'type': 'risk' if attr['shap_value'] > 0 else 'protective'
            })

        waterfall_steps.append({
            'label': 'Final Risk Probability',
            'delta': 0.0,
            'cumulative': round(risk_score, 4),
            'type': 'final'
        })

        # Plain language clinical narrative synthesis
        narrative = self._generate_clinical_narrative(
            risk_score=risk_score,
            risk_tier=risk_tier,
            top_risk=top_risk_factors,
            top_prot=top_protective_factors,
            raw_patient=raw_patient_dict
        )

        return {
            'risk_score': round(risk_score, 4),
            'risk_percentage': round(risk_score * 100, 1),
            'risk_tier': risk_tier,
            'risk_badge': risk_badge,
            'base_value': round(self.expected_value, 4),
            'feature_attributions': attributions,
            'top_risk_factors': top_risk_factors[:3],
            'top_protective_factors': top_protective_factors[:3],
            'waterfall_steps': waterfall_steps,
            'plain_language_narrative': narrative
        }

    def _generate_clinical_narrative(self,
                                     risk_score: float,
                                     risk_tier: str,
                                     top_risk: List[Dict[str, Any]],
                                     top_prot: List[Dict[str, Any]],
                                     raw_patient: Optional[Dict[str, Any]]) -> str:
        """Translates SHAP coefficients into clinical medical terminology."""
        sentences = []
        sentences.append(
            f"Patient is assessed at **{risk_tier.upper()} risk** ({risk_score * 100:.1f}%) "
            f"for angiographically significant coronary artery disease."
        )

        if top_risk:
            risk_items = [f"{r['label']} ({r['display_value']})" for r in top_risk[:3]]
            sentences.append(
                f"**Primary Risk Drivers:** Risk is driven upward predominantly by: {', '.join(risk_items)}."
            )

        if top_prot:
            prot_items = [f"{p['label']} ({p['display_value']})" for p in top_prot[:3]]
            sentences.append(
                f"**Protective Factors:** Risk is moderated/lowered by: {', '.join(prot_items)}."
            )

        # Actionable clinical recommendation
        if risk_score >= 0.75:
            sentences.append(
                "Recommendation: Urgent cardiology referral indicated. Consider stress echocardiography or coronary CTA."
            )
        elif risk_score >= 0.50:
            sentences.append(
                "Recommendation: Intermediate risk profile. Review lipid panel, optimize blood pressure, and schedule exercise stress testing."
            )
        else:
            sentences.append(
                "Recommendation: Low estimated CAD probability. Maintain routine lifestyle guidance and standard cardiovascular risk surveillance."
            )

        return " ".join(sentences)

    def compute_global_importance(self, X_sample: np.ndarray, nsamples: int = 50) -> Dict[str, Any]:
        """
        Computes global feature importance across a reference sample:
        Mean absolute SHAP value per feature.
        """
        n = min(len(X_sample), 40)
        idx = np.linspace(0, len(X_sample) - 1, n, dtype=int)
        sub_X = X_sample[idx]

        shap_vals = self.explainer.shap_values(sub_X, nsamples=nsamples)
        if isinstance(shap_vals, list):
            s_mat = np.array(shap_vals[-1], dtype=float)
        else:
            s_mat = np.array(shap_vals, dtype=float)

        mean_abs_shap = np.mean(np.abs(s_mat), axis=0)
        total_importance = np.sum(mean_abs_shap) + 1e-12

        global_features = []
        for i, feat in enumerate(self.feature_names):
            meta = FEATURE_METADATA.get(feat, {})
            val = float(mean_abs_shap[i])
            pct = float((val / total_importance) * 100)
            global_features.append({
                'feature': feat,
                'label': meta.get('label', feat.upper()),
                'importance': round(val, 4),
                'relative_pct': round(pct, 2)
            })

        global_features.sort(key=lambda x: x['importance'], reverse=True)
        return {
            'features': global_features,
            'base_value': round(self.expected_value, 4)
        }

    def compute_shap_interactions(self,
                                  x_normalized: np.ndarray,
                                  top_k: int = 4) -> Dict[str, Any]:
        """
        Computes pairwise feature interaction effects on the RBFN response:
        delta_ij = f(x_i, x_j) - f(x_i, bg_j) - f(bg_i, x_j) + f(bg_i, bg_j)
        """
        if x_normalized.ndim == 1:
            x_vec = x_normalized
        else:
            x_vec = x_normalized[0]

        bg_mean = np.mean(self.background_data, axis=0)
        f_bg = float(self.predict_fn(bg_mean[np.newaxis, :])[0])

        k = min(top_k, len(self.feature_names))
        top_indices = range(k)
        interactions = []

        for i in top_indices:
            for j in range(i + 1, k):
                feat_i = self.feature_names[i]
                feat_j = self.feature_names[j]

                # Both active
                x_ij = bg_mean.copy()
                x_ij[i] = x_vec[i]
                x_ij[j] = x_vec[j]
                f_ij = float(self.predict_fn(x_ij[np.newaxis, :])[0])

                # Only i active
                x_i = bg_mean.copy()
                x_i[i] = x_vec[i]
                f_i = float(self.predict_fn(x_i[np.newaxis, :])[0])

                # Only j active
                x_j = bg_mean.copy()
                x_j[j] = x_vec[j]
                f_j = float(self.predict_fn(x_j[np.newaxis, :])[0])

                synergy = f_ij - f_i - f_j + f_bg
                interactions.append({
                    'feature_1': feat_i,
                    'label_1': FEATURE_METADATA.get(feat_i, {}).get('label', feat_i),
                    'feature_2': feat_j,
                    'label_2': FEATURE_METADATA.get(feat_j, {}).get('label', feat_j),
                    'synergy': round(float(synergy), 4),
                    'type': 'Synergistic' if synergy > 0.005 else 'Antagonistic' if synergy < -0.005 else 'Additive'
                })

        interactions.sort(key=lambda x: abs(x['synergy']), reverse=True)
        return {'interactions': interactions}
