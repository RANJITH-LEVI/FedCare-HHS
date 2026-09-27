"""
FedCare-HHS Frontend: Clinical Decision Support & Federated Learning Platform
Built with Streamlit.
Connects to FastAPI backend via configurable API_BASE_URL.
"""

import os
import requests
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Page Configuration
st.set_page_config(
    page_title="FedCare-HHS | Federated Cardiovascular AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend URL — reads from Streamlit secrets (cloud) → env var → localhost
def _get_backend_url() -> str:
    try:
        return st.secrets["FEDCARE_BACKEND_URL"].rstrip("/")
    except Exception:
        pass
    return os.getenv("FEDCARE_BACKEND_URL", os.getenv("API_BASE_URL", "http://localhost:8000")).rstrip("/")

DEFAULT_API_BASE_URL = _get_backend_url()
if "backend_url" not in st.session_state:
    st.session_state["backend_url"] = DEFAULT_API_BASE_URL

# Custom CSS for clinical styling
st.markdown("""
<style>
    /* Global styling */
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Header card */
    .clinical-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    .clinical-header h1 {
        color: #38bdf8;
        font-size: 1.85rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .clinical-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Metric cards */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Risk Badges */
    .risk-badge-low {
        background-color: #064e3b;
        color: #34d399;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .risk-badge-moderate {
        background-color: #78350f;
        color: #fbbf24;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: #7c2d12;
        color: #fb923c;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .risk-badge-critical {
        background-color: #881337;
        color: #f43f5e;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }

    /* Factor tags */
    .factor-tag-risk {
        background: rgba(239, 68, 68, 0.15);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.85rem;
        margin: 2px;
        display: inline-block;
    }
    .factor-tag-protective {
        background: rgba(16, 185, 129, 0.15);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.85rem;
        margin: 2px;
        display: inline-block;
    }

    /* Narrative box */
    .narrative-box {
        background-color: #1e293b;
        border-left: 4px solid #38bdf8;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin-top: 14px;
        color: #e2e8f0;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# Helper: Compatibility wrapper for st.dataframe across Streamlit versions
def render_dataframe(df: pd.DataFrame, **kwargs):
    """Renders DataFrame using width='stretch' if supported, fallback to use_container_width."""
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


# Helper: API Client
def fetch_api(endpoint: str, method: str = "GET", payload: dict = None):
    base_url = st.session_state.get("backend_url", DEFAULT_API_BASE_URL).rstrip("/")
    url = f"{base_url}/{endpoint.lstrip('/')}"
    try:
        if method == "GET":
            resp = requests.get(url, timeout=15)
        elif method == "POST":
            resp = requests.post(url, json=payload, timeout=30)
        else:
            return None, "Unsupported HTTP method"
        
        if resp.status_code == 200:
            return resp.json(), None
        else:
            return None, f"HTTP {resp.status_code}: {resp.text}"
    except Exception as e:
        return None, f"Connection Error: {str(e)}"


# Matplotlib Waterfall Plot Generator
def render_shap_waterfall_plot(waterfall_steps: list, risk_score: float, base_val: float):
    """Renders a medical-grade horizontal waterfall chart using Matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')

    # Exclude base and final from delta sorting
    deltas = [step for step in waterfall_steps if step.get('type') in ('risk', 'protective')]
    
    labels = ['Base Risk'] + [s.get('label', '') for s in deltas] + ['Predicted Risk']
    y_pos = np.arange(len(labels))[::-1]  # Top to bottom

    # Plot base bar
    ax.barh(y_pos[0], max(base_val, 0.001), left=0, color='#64748b', alpha=0.8, height=0.55, label='Base Prevalence')
    ax.text(base_val / 2, y_pos[0], f"{base_val*100:.1f}%",
            va='center', ha='center', color='#ffffff', fontsize=8.5, fontweight='bold')

    # Plot delta bars
    running_start = base_val
    for i, s in enumerate(deltas):
        idx = i + 1
        delta = s.get('delta', 0.0)
        color = '#ef4444' if delta > 0 else '#10b981'
        start_left = running_start if delta > 0 else running_start + delta
        ax.barh(y_pos[idx], max(abs(delta), 0.001), left=max(start_left, 0.0), color=color, height=0.55)
        
        # Text annotation on bar
        sign = "+" if delta > 0 else ""
        if abs(delta) >= 0.02:
            ax.text(start_left + abs(delta)/2, y_pos[idx], f"{sign}{delta:.3f}",
                    va='center', ha='center', color='#ffffff', fontsize=8.5, fontweight='bold')
        else:
            offset = 0.015 if delta >= 0 else -0.015
            ha_align = 'left' if delta >= 0 else 'right'
            ax.text(start_left + delta + offset, y_pos[idx], f"{sign}{delta:.3f}",
                    va='center', ha=ha_align, color='#cbd5e1', fontsize=8.0)
        running_start += delta

    # Plot final bar
    final_color = '#f43f5e' if risk_score >= 0.50 else '#10b981'
    ax.barh(y_pos[-1], max(risk_score, 0.001), left=0, color=final_color, height=0.55, alpha=0.9, label='Predicted Risk')
    final_text_pos = max(min(risk_score / 2, 0.85), 0.06)
    ax.text(final_text_pos, y_pos[-1], f"{risk_score*100:.1f}%",
            va='center', ha='center', color='#ffffff', fontsize=9.5, fontweight='bold')

    # Connectors
    ax.axvline(x=base_val, color='#475569', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(x=0.5, color='#eab308', linestyle=':', linewidth=1.0, alpha=0.8, label='Clinical Decision Threshold (50%)')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color='#cbd5e1', fontsize=9.5)
    ax.set_xlabel("Predicted Probability of Coronary Artery Disease", color='#94a3b8', fontsize=10, labelpad=10)
    ax.set_xlim(0.0, 1.0)
    ax.tick_params(axis='x', colors='#94a3b8')
    ax.grid(axis='x', color='#1e293b', linestyle='-', linewidth=0.8)

    for spine in ax.spines.values():
        spine.set_color('#334155')

    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', loc='lower right', fontsize=8.5)
    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/heart-with-pulse.png", width=64)
    st.markdown("### **FedCare-HHS System**")
    st.caption("Federated HHS-RBFN Decision Support")
    
    # API Backend URL configuration
    backend_url_input = st.text_input("Backend API Endpoint", value=st.session_state["backend_url"])
    if backend_url_input.rstrip("/") != st.session_state["backend_url"]:
        st.session_state["backend_url"] = backend_url_input.rstrip("/")
        st.rerun()

    # Health probe
    health_data, health_err = fetch_api("/health")
    if not health_err and health_data and health_data.get("status") == "healthy":
        if health_data.get("model_ready", True):
            st.success("● Backend Connected & Ready", icon="✅")
        else:
            st.warning("● Backend Warming Up Model...", icon="⏳")
    else:
        st.error("● Backend Offline", icon="⚠️")
        st.caption(f"Error: {health_err}")

    st.markdown("---")
    st.markdown("#### **Clinical Demo Presets**")
    st.caption("Select a pre-configured clinical archetype:")
    
    preset_choice = st.selectbox(
        "Select Patient Archetype",
        [
            "Custom Input (Manual)",
            "Archetype 1: Severe CAD (63M, Severe Angina, High ST)",
            "Archetype 2: Healthy Routine Check (45F, Normal)",
            "Archetype 3: Borderline Risk (58M, Hyperlipidemia)"
        ],
        key="selected_preset_choice"
    )

    # Detect preset switch to trigger form recreation and auto-prediction
    if "active_preset" not in st.session_state:
        st.session_state["active_preset"] = preset_choice
        st.session_state["preset_version"] = 0

    if preset_choice != st.session_state["active_preset"]:
        st.session_state["active_preset"] = preset_choice
        st.session_state["preset_version"] = st.session_state.get("preset_version", 0) + 1
        if preset_choice != "Custom Input (Manual)":
            st.session_state["auto_trigger_preset"] = True

    st.markdown("---")
    st.markdown("#### **Federated Learning Quick-Trigger**")
    quick_round_btn = st.button("⚡ Trigger FL Round", help="Runs 1 round across 4 hospitals with Differential Privacy")


# Handle quick round trigger
if quick_round_btn:
    with st.spinner("Executing federated aggregation across 4 hospital nodes..."):
        res, err = fetch_api("/federated/round", method="POST", payload={"local_epochs": 10, "dp_enabled": True})
        if err:
            st.sidebar.error(f"Round failed: {err}")
        else:
            st.sidebar.success(f"Round #{res['round']} completed! Global Acc: {res['global_accuracy']*100:.1f}%")
            st.rerun()


# Preset values mapping
if preset_choice == "Archetype 1: Severe CAD (63M, Severe Angina, High ST)":
    default_vals = {
        'age': 63.0, 'sex': 1, 'cp': 4, 'trestbps': 150.0, 'chol': 270.0,
        'fbs': 1, 'restecg': 2, 'thalach': 125.0, 'exang': 1, 'oldpeak': 2.8,
        'slope': 2, 'ca': 2.0, 'thal': 7
    }
elif preset_choice == "Archetype 2: Healthy Routine Check (45F, Normal)":
    default_vals = {
        'age': 45.0, 'sex': 0, 'cp': 2, 'trestbps': 118.0, 'chol': 185.0,
        'fbs': 0, 'restecg': 0, 'thalach': 172.0, 'exang': 0, 'oldpeak': 0.2,
        'slope': 1, 'ca': 0.0, 'thal': 3
    }
elif preset_choice == "Archetype 3: Borderline Risk (58M, Hyperlipidemia)":
    default_vals = {
        'age': 58.0, 'sex': 1, 'cp': 3, 'trestbps': 136.0, 'chol': 288.0,
        'fbs': 0, 'restecg': 1, 'thalach': 148.0, 'exang': 0, 'oldpeak': 1.2,
        'slope': 2, 'ca': 1.0, 'thal': 6
    }
else:
    default_vals = {
        'age': 55.0, 'sex': 1, 'cp': 3, 'trestbps': 130.0, 'chol': 240.0,
        'fbs': 0, 'restecg': 0, 'thalach': 150.0, 'exang': 0, 'oldpeak': 1.0,
        'slope': 2, 'ca': 0.0, 'thal': 3
    }


# -----------------------------------------------------------------------------
# MAIN HEADER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="clinical-header">
    <h1>🫀 FedCare-HHS Clinical Decision Support System</h1>
    <p>Federated & Explainable Radial Basis Function Network (RBFN) with Harris Hawks Search (HHS) Feature Selection</p>
</div>
""", unsafe_allow_html=True)


# TABS
tab1, tab2, tab3, tab4 = st.tabs([
    "🩺 Patient Risk Assessment & SHAP",
    "🏥 Hospital Network & Federation",
    "🧠 Model Insights & HHS Architecture",
    "📋 Patient Ingestion Registry"
])


# =============================================================================
# TAB 1: PATIENT RISK ASSESSMENT & SHAP EXPLAINABILITY
# =============================================================================
with tab1:
    st.markdown("#### **Individual Patient Intake & Explainable Risk Inference**")
    st.caption("Enter patient clinical markers. The federated HHS-RBFN global model evaluates risk and generates game-theoretic SHAP attributions.")

    col_form, col_results = st.columns([5, 6], gap="large")
    pv = st.session_state.get("preset_version", 0)

    with col_form:
        st.markdown("##### **1. Clinical Intake Form**")
        with st.form(f"patient_intake_form_{pv}"):
            st.markdown("**Demographics & Vitals**")
            c1, c2 = st.columns(2)
            with c1:
                age_val = st.slider("Age (years)", min_value=25, max_value=85, value=int(default_vals['age']), key=f"age_{pv}")
                trestbps_val = st.number_input("Resting BP (mm Hg)", min_value=80, max_value=220, value=int(default_vals['trestbps']), key=f"trestbps_{pv}")
            with c2:
                sex_val = st.selectbox("Biological Sex", options=[1, 0], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)", index=0 if default_vals['sex'] == 1 else 1, key=f"sex_{pv}")
                chol_val = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=550, value=int(default_vals['chol']), key=f"chol_{pv}")

            st.markdown("**Cardiac Symptoms & History**")
            c3, c4 = st.columns(2)
            with c3:
                cp_options = {1: "Typical Angina (1)", 2: "Atypical Angina (2)", 3: "Non-Anginal Pain (3)", 4: "Asymptomatic (4)"}
                cp_val = st.selectbox("Chest Pain Type", options=list(cp_options.keys()), format_func=lambda x: cp_options[x], index=list(cp_options.keys()).index(int(default_vals['cp'])), key=f"cp_{pv}")
                fbs_val = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "False (<= 120)" if x == 0 else "True (> 120)", index=int(default_vals['fbs']), key=f"fbs_{pv}")
            with c4:
                restecg_options = {0: "Normal (0)", 1: "ST-T Abnormality (1)", 2: "LV Hypertrophy (2)"}
                restecg_val = st.selectbox("Resting ECG", options=list(restecg_options.keys()), format_func=lambda x: restecg_options[x], index=list(restecg_options.keys()).index(int(default_vals['restecg'])), key=f"restecg_{pv}")
                exang_val = st.selectbox("Exercise-Induced Angina", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes (1)", index=int(default_vals['exang']), key=f"exang_{pv}")

            st.markdown("**Exercise Stress Test & Advanced Diagnostics**")
            c5, c6 = st.columns(2)
            with c5:
                thalach_val = st.slider("Max Heart Rate (thalach, bpm)", min_value=70, max_value=210, value=int(default_vals['thalach']), key=f"thalach_{pv}")
                oldpeak_val = st.slider("ST Depression (oldpeak, mm)", min_value=0.0, max_value=6.0, value=float(default_vals['oldpeak']), step=0.1, key=f"oldpeak_{pv}")
            with c6:
                slope_options = {1: "Upsloping (1)", 2: "Flat (2)", 3: "Downsloping (3)"}
                slope_val = st.selectbox("Slope of Peak Exercise ST", options=list(slope_options.keys()), format_func=lambda x: slope_options[x], index=list(slope_options.keys()).index(int(default_vals['slope'])), key=f"slope_{pv}")
                ca_val = st.selectbox("Major Vessels Colored (ca)", options=[0.0, 1.0, 2.0, 3.0], index=int(default_vals['ca']), key=f"ca_{pv}")

            thal_options = {3: "Normal (3)", 6: "Fixed Defect (6)", 7: "Reversible Defect (7)"}
            thal_val = st.selectbox("Thalassemia (thal)", options=list(thal_options.keys()), format_func=lambda x: thal_options[x], index=list(thal_options.keys()).index(int(default_vals['thal'])), key=f"thal_{pv}")

            submit_pred_btn = st.form_submit_button("⚡ Evaluate Cardiovascular Risk & Explain", use_container_width=True)

    with col_results:
        st.markdown("##### **2. Explainable Diagnostic Output**")
        
        patient_payload = {
            'age': float(age_val),
            'sex': int(sex_val),
            'cp': int(cp_val),
            'trestbps': float(trestbps_val),
            'chol': float(chol_val),
            'fbs': int(fbs_val),
            'restecg': int(restecg_val),
            'thalach': float(thalach_val),
            'exang': int(exang_val),
            'oldpeak': float(oldpeak_val),
            'slope': int(slope_val),
            'ca': float(ca_val),
            'thal': int(thal_val)
        }

        # Check if auto-triggered by preset change or first load
        auto_triggered = st.session_state.pop("auto_trigger_preset", False)
        should_run_pred = submit_pred_btn or auto_triggered or ("last_prediction" not in st.session_state and health_data and health_data.get("status") == "healthy")

        if should_run_pred:
            with st.spinner("Computing RBFN global inference & SHAP game-theoretic decomposition..."):
                pred_res, pred_err = fetch_api("/predict", method="POST", payload=patient_payload)
                if not pred_err and pred_res:
                    st.session_state["last_prediction"] = pred_res
                    st.session_state["last_patient_input"] = patient_payload
                else:
                    if submit_pred_btn or auto_triggered:
                        st.error(f"Inference failed: {pred_err}")

        if "last_prediction" in st.session_state:
            pred = st.session_state["last_prediction"]
            score = pred["risk_score"]
            tier = pred["risk_tier"]
            badge_class = f"risk-badge-{pred.get('risk_badge', 'moderate')}"

            # Visual summary banner
            col_m1, col_m2 = st.columns([1, 1])
            with col_m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Estimated CAD Probability</div>
                    <div class="metric-value">{score*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Clinical Risk Classification</div>
                    <div style="margin-top: 8px;"><span class="{badge_class}">{tier.upper()}</span></div>
                </div>
                """, unsafe_allow_html=True)

            # Plain-language narrative summary
            st.markdown(f"""
            <div class="narrative-box">
                <b>Clinical Reasoning Summary:</b><br/>
                {pred.get('plain_language_narrative', '')}
            </div>
            """, unsafe_allow_html=True)

            # Top Factors Breakdown
            st.markdown("###### **Key Physiological Drivers**")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("**Risk-Increasing Factors (Pushed Risk Up):**")
                if pred.get("top_risk_factors"):
                    for rf in pred["top_risk_factors"]:
                        st.markdown(f"<span class='factor-tag-risk'>▲ {rf['label']}: {rf['display_value']} (+{rf['shap_value']:.3f})</span>", unsafe_allow_html=True)
                else:
                    st.caption("No significant risk-elevating factors detected.")
            with col_d2:
                st.markdown("**Protective Factors (Moderated Risk Down):**")
                if pred.get("top_protective_factors"):
                    for pf in pred["top_protective_factors"]:
                        st.markdown(f"<span class='factor-tag-protective'>▼ {pf['label']}: {pf['display_value']} ({pf['shap_value']:.3f})</span>", unsafe_allow_html=True)
                else:
                    st.caption("No strong protective factors present.")

            # SHAP Waterfall Chart
            st.markdown("###### **Local SHAP Waterfall Attribution Chart**")
            waterfall_fig = render_shap_waterfall_plot(
                pred.get("waterfall_steps", []),
                risk_score=score,
                base_val=pred.get("base_value", 0.5)
            )
            st.pyplot(waterfall_fig)
            plt.close(waterfall_fig)

            # Bonus: Feature Interaction Matrix
            if pred.get("interactions"):
                with st.expander("🔍 View Top Feature Synergy Interactions (Bonus)"):
                    st.caption("Non-linear cross-feature coupling effects captured by Radial Basis Gaussian kernels:")
                    inter_df = pd.DataFrame(pred["interactions"])
                    render_dataframe(
                        inter_df[['feature_1', 'feature_2', 'synergy', 'type']],
                        column_config={
                            "feature_1": "Feature A",
                            "feature_2": "Feature B",
                            "synergy": st.column_config.NumberColumn("Coupling Strength (Δ)", format="%.4f"),
                            "type": "Interaction Classification"
                        }
                    )

            # Action button to ingest patient into hospital node
            st.markdown("---")
            col_ing1, col_ing2 = st.columns([3, 2])
            with col_ing1:
                target_hosp = st.selectbox(
                    "Hospital Node Destination",
                    options=["HOSP-01", "HOSP-02", "HOSP-03", "HOSP-04"],
                    format_func=lambda x: f"{x} - {x.replace('HOSP-01', 'Cleveland Clinic').replace('HOSP-02', 'Hungarian Institute').replace('HOSP-03', 'Zurich Univ Hosp').replace('HOSP-04', 'Long Beach VA')}",
                    key="target_hosp_select"
                )
            with col_ing2:
                st.write("")
                st.write("")
                if st.button("📥 Ingest into Hospital Node", use_container_width=True, key="btn_ingest_patient"):
                    patient_input = st.session_state.get("last_patient_input", patient_payload)
                    ingest_body = {**patient_input, "hospital_id": target_hosp}
                    i_res, i_err = fetch_api("/ingest", method="POST", payload=ingest_body)
                    if not i_err and i_res:
                        st.session_state["ingest_success_msg"] = f"Record successfully saved as **{i_res['patient_identifier']}** in node **{target_hosp}**!"
                        st.rerun()
                    else:
                        st.error(f"Ingestion failed: {i_err}")

            if "ingest_success_msg" in st.session_state:
                st.success(st.session_state["ingest_success_msg"], icon="✅")
                st.caption("You can inspect this record in the **Patient Ingestion Registry (Tab 4)**.")
                if st.button("Dismiss", key="dismiss_ingest_msg"):
                    del st.session_state["ingest_success_msg"]
                    st.rerun()
        else:
            st.info("👈 Enter patient clinical vitals or select an archetype preset, then click **Evaluate Cardiovascular Risk & Explain**.")


# =============================================================================
# TAB 2: HOSPITAL NETWORK & FEDERATED LEARNING DASHBOARD
# =============================================================================
with tab2:
    st.markdown("#### **Multi-Hospital Federated Network & Privacy-Preserving Coordination**")
    st.caption("Decentralized Flower (flwr) simulation across 4 clinical nodes with Differential Privacy (DP) gradient clipping and Gaussian noise.")

    # Fetch live metrics and hospital status
    metrics_data, m_err = fetch_api("/model/metrics")
    hospitals_data, h_err = fetch_api("/hospitals")

    if not m_err and metrics_data:
        g_metrics = metrics_data.get("global_metrics", {})
        c_baseline = metrics_data.get("centralized_baseline", {})
        dp_info = metrics_data.get("differential_privacy", {})
        history = metrics_data.get("round_history", [])

        # Top summary KPI row
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        with kpi1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Federated Rounds</div>
                <div class="metric-value">{metrics_data.get('total_rounds', 0)}</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Global FL Accuracy</div>
                <div class="metric-value">{g_metrics.get('accuracy', 0.0)*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Centralized Baseline</div>
                <div class="metric-value" style="color: #a855f7;">{c_baseline.get('accuracy', 0.0)*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Global F1-Score</div>
                <div class="metric-value">{g_metrics.get('f1', 0.0):.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi5:
            dp_eps_str = f"ε = {dp_info['epsilon']:.2f}" if dp_info.get('epsilon') else "Disabled"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">DP Privacy Budget Spent</div>
                <div class="metric-value" style="color: #10b981;">{dp_eps_str}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Hospital Nodes Cards
        st.markdown("##### **Participating Hospital Node Status**")
        if hospitals_data:
            h_cols = st.columns(len(hospitals_data))
            for i, h in enumerate(hospitals_data):
                with h_cols[i]:
                    st.markdown(f"""
                    <div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 14px; min-height: 160px;">
                        <span style="font-size: 0.8rem; background-color: #0284c7; color: white; padding: 2px 8px; border-radius: 4px;">{h['id']}</span>
                        <h4 style="margin: 6px 0 2px 0; font-size: 1.05rem; color: #f8fafc;">{h['name']}</h4>
                        <p style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 8px;">📍 {h['location']}</p>
                        <div style="font-size: 0.85rem; color: #cbd5e1;"><b>Samples:</b> {h['sample_count']} patients</div>
                        <div style="font-size: 0.85rem; color: #34d399;"><b>Local Accuracy:</b> {h['local_accuracy']*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # Training Curves & Charts
        st.markdown("##### **Federated Learning Convergence vs. Centralized Benchmark**")
        if history:
            rounds_list = [r['round'] for r in history]
            fl_acc_list = [r['global_accuracy'] * 100 for r in history]
            cent_acc_list = [r.get('centralized_benchmark', {}).get('accuracy', c_baseline.get('accuracy', 0.80)) * 100 for r in history]
            loss_list = [r['global_loss'] for r in history]
            f1_list = [r['global_f1'] for r in history]

            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown("**Accuracy Comparison (Federated vs Centralized Benchmark)**")
                acc_df = pd.DataFrame({
                    "Round": rounds_list,
                    "Federated Global Accuracy (%)": fl_acc_list,
                    "Centralized Baseline Benchmark (%)": cent_acc_list
                }).set_index("Round")
                st.line_chart(acc_df, color=["#38bdf8", "#c084fc"])

            with chart_col2:
                st.markdown("**Global Loss & F1 Progression**")
                loss_df = pd.DataFrame({
                    "Round": rounds_list,
                    "Global Cross-Entropy Loss": loss_list,
                    "Global F1 Score": f1_list
                }).set_index("Round")
                st.line_chart(loss_df, color=["#f43f5e", "#10b981"])

        # Federated Training Control Panel
        st.markdown("---")
        st.markdown("##### **Interactive Federated Round Execution Control**")
        with st.expander("⚙️ Configure & Trigger Next Federated Training Round", expanded=True):
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            with f_col1:
                fl_epochs = st.slider("Local Client Epochs", min_value=1, max_value=30, value=12)
            with f_col2:
                fl_lr = st.select_slider("Learning Rate", options=[0.01, 0.03, 0.05, 0.06, 0.08, 0.10], value=0.06)
            with f_col3:
                fl_dp = st.checkbox("Enable Differential Privacy (DP)", value=True)
                fl_noise = st.slider("DP Noise Multiplier (σ)", min_value=0.01, max_value=0.20, value=0.05, step=0.01, disabled=not fl_dp)
            with f_col4:
                fl_clip = st.slider("L2 Norm Clipping (C)", min_value=0.2, max_value=3.0, value=1.0, step=0.2, disabled=not fl_dp)
                st.write("")
                trigger_btn = st.button("🚀 Execute Federated Round", use_container_width=True, key="btn_trigger_fl_round")

            if trigger_btn:
                with st.spinner("Distributing weights → Training on 4 private hospital nodes → Aggregating via FedAvg..."):
                    payload = {
                        "local_epochs": fl_epochs,
                        "learning_rate": fl_lr,
                        "dp_enabled": fl_dp,
                        "noise_multiplier": fl_noise if fl_dp else 0.0,
                        "clip_norm": fl_clip if fl_dp else 0.0
                    }
                    new_r, r_err = fetch_api("/federated/round", method="POST", payload=payload)
                    if not r_err and new_r:
                        st.success(f"Round #{new_r['round']} successfully aggregated! Global Accuracy: {new_r['global_accuracy']*100:.1f}%")
                        st.rerun()
                    else:
                        st.error(f"Round execution failed: {r_err}")
    else:
        st.warning(f"Unable to fetch federated metrics from backend: {m_err}. Please ensure the backend is running.")


# =============================================================================
# TAB 3: MODEL INSIGHTS & HHS ARCHITECTURE
# =============================================================================
with tab3:
    st.markdown("#### **Harris Hawks Search (HHS) + Radial Basis Function Network (RBFN) Architecture**")
    st.caption("Detailed view of the metaheuristic feature selection optimization and inspectable neural basis architecture.")

    g_shap_data, g_err = fetch_api("/model/explain/global")

    col_ins1, col_ins2 = st.columns([1, 1], gap="large")

    with col_ins1:
        st.markdown("##### **1. Dynamic Global SHAP Feature Importance**")
        st.caption("Recomputed across the global multi-center validation set after each federated round.")
        if not g_err and g_shap_data and "features" in g_shap_data:
            feats = g_shap_data["features"]
            feat_df = pd.DataFrame(feats)
            
            fig_g, ax_g = plt.subplots(figsize=(8, 4.8), facecolor='#0f172a')
            ax_g.set_facecolor('#0f172a')
            
            y_g = np.arange(len(feat_df))[::-1]
            ax_g.barh(y_g, feat_df['importance'], color='#38bdf8', height=0.55, edgecolor='#0284c7')
            ax_g.set_yticks(y_g)
            ax_g.set_yticklabels(feat_df['label'], color='#e2e8f0', fontsize=9.5)
            ax_g.set_xlabel("Mean Absolute SHAP Value |E[|φ_j|]|", color='#94a3b8', fontsize=9.5)
            ax_g.tick_params(axis='x', colors='#94a3b8')
            ax_g.grid(axis='x', color='#1e293b', linestyle='-')
            for spine in ax_g.spines.values():
                spine.set_color('#334155')
            plt.tight_layout()
            st.pyplot(fig_g)
            plt.close(fig_g)

            render_dataframe(
                feat_df[['label', 'importance', 'relative_pct']],
                column_config={
                    "label": "Clinical Feature",
                    "importance": st.column_config.NumberColumn("Mean |SHAP|", format="%.4f"),
                    "relative_pct": st.column_config.NumberColumn("Relative Share (%)", format="%.1f%%")
                }
            )
        else:
            st.info("Global SHAP values are being calculated or backend is offline.")

    with col_ins2:
        st.markdown("##### **2. Harris Hawks Search (HHS) Feature Selection**")
        st.markdown("""
        **HHS Formulation:**
        - Population: 20 candidate binary feature-mask "hawks".
        - Escaping Energy: $E = 2E_0(1 - t/T)$ transitioning from exploration to exploitation.
        - Phases: Soft Besiege, Hard Besiege, Surprise Pounce with Levy Flight rapid dives.
        """)

        # HHS Selected vs Pruned breakdown
        selected_set = ["sex", "cp", "chol", "thalach", "exang", "oldpeak", "thal"]
        all_features = [
            ("cp", "Chest Pain Type", "Selected (High diagnostic salience)"),
            ("oldpeak", "ST Depression (Exercise vs Rest)", "Selected (Direct myocardial ischemia marker)"),
            ("thal", "Thalassemia Defect", "Selected (Perfusion defect indicator)"),
            ("thalach", "Max Heart Rate Achieved", "Selected (Chronotropic competence marker)"),
            ("exang", "Exercise-Induced Angina", "Selected (Exertional ischemia)"),
            ("chol", "Serum Cholesterol", "Selected (Atherosclerotic burden)"),
            ("sex", "Biological Sex", "Selected (Demographic baseline risk)"),
            ("age", "Age", "Pruned (Correlated with max heart rate / chol)"),
            ("trestbps", "Resting Blood Pressure", "Pruned (Subsumed by exercise stress markers)"),
            ("ca", "Vessels Colored (Fluoroscopy)", "Pruned by HHS (High missingness in Europe cohorts)"),
            ("slope", "ST Slope", "Pruned (High redundancy with oldpeak)"),
            ("fbs", "Fasting Blood Sugar", "Pruned (Low standalone CAD predictive power)"),
            ("restecg", "Resting ECG", "Pruned (Superseded by exercise ST metrics)")
        ]

        hhs_df = pd.DataFrame([
            {"Feature": f[1], "Status": "✅ Selected" if f[0] in selected_set else "❌ Pruned", "Clinical Rationale": f[2]}
            for f in all_features
        ])
        render_dataframe(hhs_df)

        st.markdown("##### **3. RBFN Neural Architecture**")
        st.markdown(r"""
        - **Hidden Layer:** 16 Gaussian Radial Basis Centers ($K=16$) initialized in feature space.
        - **Activation Function:** $\phi_j(x) = \exp\left(-\frac{\|x - c_j\|^2}{2\sigma_j^2}\right)$.
        - **Output Layer:** Linear combination with Sigmoid probability calibration $\hat{y} = \sigma(W^T \Phi(x) + b)$.
        - **Federated Averaging:** Linear aggregation on parameters $[W, b]$ across all hospital clients.
        """)


# =============================================================================
# TAB 4: PATIENT INGESTION REGISTRY
# =============================================================================
with tab4:
    st.markdown("#### **Decentralized Patient Record Registry & Audit Trail**")
    st.caption("Browse records ingested across the virtual hospital nodes.")

    c_filter, c_refresh = st.columns([4, 1])
    with c_filter:
        selected_hosp_filter = st.selectbox(
            "Filter by Hospital Node",
            options=["ALL", "HOSP-01", "HOSP-02", "HOSP-03", "HOSP-04"],
            key="registry_filter_select"
        )
    with c_refresh:
        st.write("")
        st.write("")
        refresh_patients = st.button("🔄 Refresh Records", key="btn_refresh_patients")

    # Fetch patients
    endpoint = "/patients" if selected_hosp_filter == "ALL" else f"/patients?hospital_id={selected_hosp_filter}"
    patients_list, p_err = fetch_api(endpoint)

    if not p_err and patients_list:
        p_df = pd.DataFrame(patients_list)
        available_cols = [c for c in ['patient_identifier', 'hospital_id', 'age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'exang', 'oldpeak', 'created_at'] if c in p_df.columns]
        render_dataframe(
            p_df[available_cols],
            column_config={
                "patient_identifier": "Chart ID",
                "hospital_id": "Hospital Node",
                "age": "Age",
                "sex": st.column_config.NumberColumn("Sex (0=F, 1=M)"),
                "cp": "Chest Pain",
                "trestbps": "Resting BP",
                "chol": "Cholesterol",
                "thalach": "Max HR",
                "oldpeak": "ST Depr",
                "created_at": "Ingestion Date"
            }
        )
        st.caption(f"Showing {len(patients_list)} records from the clinical database.")
    else:
        st.info("No patient records found or unable to connect to registry.")
