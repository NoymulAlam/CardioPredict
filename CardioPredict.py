import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from fpdf import FPDF
from datetime import datetime

# ─────────────────────────────────────────
# 1. Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="CardioPredict AI Pro",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# 2. Custom CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    /* General */
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #0f1117; }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    /* Risk cards */
    .risk-card {
        padding: 24px 20px;
        border-radius: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }
    .risk-card h1 { margin: 0; font-size: 3rem; font-weight: 800; }
    .risk-card p  { margin: 6px 0 0; font-size: 1.1rem; font-weight: 600; letter-spacing: 1px; }

    /* Info box */
    .info-box {
        background: #1e2130;
        border-left: 4px solid #4a9eff;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        color: #cfd8e3;
    }

    /* Feature debug box */
    .debug-box {
        background: #1a1a2e;
        border: 1px solid #e63946;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.82rem;
        color: #ffc8c8;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 3. WHO Reference Standards
# ─────────────────────────────────────────
WHO_LIMITS = {
    "trestbps": {"threshold": 140, "unit": "mmHg",   "label": "Resting Blood Pressure",
                 "note": "WHO identifies hypertension (≥140/90 mmHg) as the leading modifiable CVD risk factor."},
    "chol":     {"threshold": 200, "unit": "mg/dL",  "label": "Total Cholesterol",
                 "note": "High lipids promote fatty plaque deposits in coronary arteries."},
    "fbs":      {"threshold": 1,   "unit": "(>120)",  "label": "Fasting Blood Sugar",
                 "note": "FBS > 120 mg/dL may indicate pre-diabetes or Type 2 Diabetes."},
    "age":      {"threshold": 50,  "unit": "years",  "label": "Age",
                 "note": "Risk increases with age; majority of CVD deaths occur in those over 70."},
    "ca":       {"threshold": 0,   "unit": "vessels","label": "Major Vessel Narrowing",
                 "note": "Each narrowed major vessel significantly raises coronary artery disease risk."},
    "oldpeak":  {"threshold": 1.5, "unit": "mm",     "label": "ST Depression (Oldpeak)",
                 "note": "ST depression > 1.5 mm during exercise indicates myocardial ischemia."},
}

# ─────────────────────────────────────────
# 4. Model Loading
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="🔬 Loading Clinical Model...")
def load_model():
    model_path = "best_tuned_logistic_model.pkl"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None
    return None

# ─────────────────────────────────────────
# 5. Preprocessing (BUG FIX)
# ─────────────────────────────────────────
def preprocess_for_model(user_vals: dict, model) -> pd.DataFrame:
    """
    Safely aligns user input to the exact feature set the model was trained on.
    Handles both plain numeric features and one-hot encoded dummy columns.
    """
    input_df = pd.DataFrame([user_vals])

    if not hasattr(model, "feature_names_in_"):
        # Model has no recorded feature names — pass as-is and hope for the best
        return input_df

    expected_cols = list(model.feature_names_in_)
    aligned = pd.DataFrame(columns=expected_cols, index=[0])

    for col in expected_cols:
        if col in input_df.columns:
            # Direct match — copy value
            aligned[col] = input_df[col].values
        elif "_" in col:
            # Possible one-hot encoded column e.g. "cp_2", "thal_1"
            parts = col.rsplit("_", 1)          # rsplit keeps base names with underscores intact
            base_feature = parts[0]
            encoded_val  = parts[1]
            raw_val = user_vals.get(base_feature)
            # Match as string because OHE creates string-like category names
            aligned[col] = 1 if str(raw_val) == encoded_val else 0
        else:
            # Feature not found anywhere — fill with 0 (safe fallback)
            aligned[col] = 0

    # Ensure correct dtypes — model expects floats
    aligned = aligned.astype(float)
    return aligned

# ─────────────────────────────────────────
# 6. PDF Report Generator
# ─────────────────────────────────────────
def create_pdf(probability: float, label: str, user_vals: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_fill_color(30, 30, 50)
    pdf.rect(0, 0, 210, 30, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 30, "CardioPredict AI — Clinical Risk Report", ln=True, align='C')

    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # Date & Summary
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(2)

    # Risk status banner
    if "HIGH" in label:
        pdf.set_fill_color(230, 57, 70)
    elif "MODERATE" in label:
        pdf.set_fill_color(244, 162, 97)
    else:
        pdf.set_fill_color(42, 157, 143)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 12, f"  Risk Classification: {label}   |   Probability: {probability:.2f}%", ln=True, fill=True)

    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    # Input Parameters Table
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Patient Input Parameters", ln=True, border='B')
    pdf.ln(2)

    pdf.set_font("Arial", size=10)
    col_labels = {"age": "Age (yrs)", "sex": "Biological Sex", "cp": "Chest Pain Type",
                  "trestbps": "Resting BP (mmHg)", "chol": "Cholesterol (mg/dL)",
                  "fbs": "Fasting Blood Sugar", "restecg": "Resting ECG",
                  "thalach": "Max Heart Rate", "exang": "Exercise Angina",
                  "oldpeak": "ST Depression", "slope": "ST Slope",
                  "ca": "Major Vessels (CA)", "thal": "Thalassemia"}

    for key, val in user_vals.items():
        display_key = col_labels.get(key, key.upper())
        pdf.set_fill_color(245, 245, 250)
        pdf.cell(90, 7, f"  {display_key}", border=1, fill=True)
        pdf.cell(100, 7, f"  {val}", border=1, ln=True)

    pdf.ln(8)

    # Disclaimer
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5,
        "DISCLAIMER: This report is generated by an AI model for informational purposes only. "
        "It is NOT a substitute for professional medical diagnosis or clinical judgment. "
        "Always consult a qualified healthcare provider.")

    return pdf.output(dest='S').encode('latin-1')

# ─────────────────────────────────────────
# 7. Sidebar Inputs
# ─────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822118.png", width=80)
    st.title("Patient Vitals")
    st.caption("Enter accurate clinical measurements")
    st.divider()

    with st.expander("👤 Basic Information", expanded=True):
        age = st.slider("Age (years)", 18, 100, 55)
        sex = st.radio(
            "Biological Sex",
            options=[1, 0],
            format_func=lambda x: "Male" if x == 1 else "Female",
            horizontal=True,
        )

    with st.expander("🩺 Vital Signs & Labs", expanded=True):
        trestbps = st.number_input("Resting Blood Pressure (mmHg)", 80, 220, 130,
                                   help="Measured at rest in mmHg. WHO hypertension threshold: ≥140 mmHg.")
        chol = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, 210,
                                help="Total serum cholesterol. Healthy: < 200 mg/dL.")
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL?",
            options=[0, 1],
            format_func=lambda x: "Yes (>120 mg/dL)" if x == 1 else "No (≤120 mg/dL)",
            help="1 = True (FBS > 120 mg/dL), 0 = False"
        )
        thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150,
                                   help="Peak heart rate during exercise stress test.")

    with st.expander("🧪 Cardiac Tests", expanded=True):
        cp = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 — Typical Angina",
                1: "1 — Atypical Angina",
                2: "2 — Non-Anginal Pain",
                3: "3 — Asymptomatic",
            }[x],
            help="Type of chest pain experienced."
        )
        exang = st.checkbox("Exercise-Induced Angina", help="Chest pain triggered by physical exertion.")
        oldpeak = st.slider(
            "ST Depression (Oldpeak)",
            min_value=0.0, max_value=6.0, value=1.0, step=0.1,
            help="ST depression induced by exercise relative to rest."
        )
        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            options=[0, 1, 2],
            format_func=lambda x: {0: "0 — Upsloping", 1: "1 — Flat", 2: "2 — Downsloping"}[x],
            help="Shape of the ST segment at peak exercise."
        )
        restecg = st.selectbox(
            "Resting ECG Results",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "0 — Normal",
                1: "1 — ST-T Abnormality",
                2: "2 — Left Ventricular Hypertrophy"
            }[x],
            help="Electrocardiographic results at rest."
        )
        ca = st.select_slider(
            "Major Vessels Narrowed (CA)",
            options=[0, 1, 2, 3],
            help="Number of major vessels (0–3) colored by fluoroscopy."
        )
        thal = st.selectbox(
            "Thalassemia",
            options=[0, 1, 2],
            format_func=lambda x: {0: "0 — Normal", 1: "1 — Fixed Defect", 2: "2 — Reversible Defect"}[x],
            help="Blood disorder affecting haemoglobin."
        )

    st.divider()
    analyze_btn = st.button("🚀 Run Diagnostic Analysis", use_container_width=True, type="primary")

# ─────────────────────────────────────────
# 8. Main Canvas
# ─────────────────────────────────────────
st.title("🫀 CardioPredict AI Pro")
st.caption("Advanced cardiovascular risk assessment · Cleveland Clinical Dataset · Logistic Regression")

model = load_model()

# ── Debug helper (remove in production) ──────────────────────────────────
if model and hasattr(model, "feature_names_in_") and st.checkbox("🔧 Show model feature names (debug)", value=False):
    st.markdown(
        f'<div class="debug-box"><strong>Model expects {len(model.feature_names_in_)} features:</strong><br>'
        + ", ".join(model.feature_names_in_) + "</div>",
        unsafe_allow_html=True,
    )

# ── Welcome state ─────────────────────────────────────────────────────────
if not analyze_btn:
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BP Target",          "< 140 mmHg", "WHO Standard")
    c2.metric("Cholesterol Target", "< 200 mg/dL","WHO Standard")
    c3.metric("FBS Target",         "< 120 mg/dL","WHO Standard")
    c4.metric("Resting HR Target",  "60–100 bpm", "Normal Range")
    st.markdown('<div class="info-box">👈 <strong>Enter patient data</strong> in the sidebar and click <em>Run Diagnostic Analysis</em> to generate an AI-powered cardiovascular risk assessment.</div>', unsafe_allow_html=True)

# ── Analysis state ────────────────────────────────────────────────────────
else:
    if model is None:
        st.error("⚠️ Model file not found. Place `best_tuned_logistic_model.pkl` in the same directory as this app.")
        st.stop()

    # Build input dictionary — ALL features from sidebar (no hardcoded values)
    user_vals = {
        "age":      age,
        "sex":      sex,
        "cp":       cp,
        "trestbps": trestbps,
        "chol":     chol,
        "fbs":      fbs,
        "restecg":  restecg,
        "thalach":  thalach,
        "exang":    int(exang),
        "oldpeak":  oldpeak,
        "slope":    slope,
        "ca":       ca,
        "thal":     thal,
    }

    processed_df = preprocess_for_model(user_vals, model)

    try:
        prob     = model.predict_proba(processed_df)[0][1]
        prob_pct = prob * 100

        if prob_pct > 70:
            label, color, icon, bg_light = "HIGH RISK",     "#e63946", "🚨", "#fdecea"
        elif prob_pct > 35:
            label, color, icon, bg_light = "MODERATE RISK", "#f4a261", "⚠️", "#fff4ec"
        else:
            label, color, icon, bg_light = "LOW RISK",      "#2a9d8f", "✅", "#e8f8f6"

        # ── Result header ─────────────────────────────────────────────────
        st.subheader(f"{icon} Diagnostic Assessment Results")
        col_res1, col_res2 = st.columns([1, 2], gap="large")

        with col_res1:
            st.markdown(f"""
                <div class="risk-card" style="background: {color};">
                    <h1>{prob_pct:.1f}%</h1>
                    <p>{label}</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(prob, text=f"Risk spectrum: 0% → 100%")

            report_bytes = create_pdf(prob_pct, label, user_vals)
            st.download_button(
                label="📥 Download Physician Report (PDF)",
                data=report_bytes,
                file_name=f"CardioReport_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        with col_res2:
            # WHO comparison metrics
            m1, m2, m3, m4 = st.columns(4)
            bp_d   = trestbps - 140
            chol_d = chol - 200
            hr_d   = thalach - 100  # arbitrary reference for display
            m1.metric("Blood Pressure",   f"{trestbps} mmHg", f"{bp_d:+d} vs limit",   delta_color="inverse")
            m2.metric("Cholesterol",       f"{chol} mg/dL",   f"{chol_d:+d} vs limit",  delta_color="inverse")
            m3.metric("Max Heart Rate",    f"{thalach} bpm",  f"{hr_d:+d} vs 100",       delta_color="normal")
            m4.metric("CA Vessels",        str(ca),           "⚠️ Narrowed" if ca > 0 else "Normal", delta_color="inverse")

            st.markdown("<br>", unsafe_allow_html=True)

            # Risk interpretation
            st.markdown(f"""
            <div style="background:{bg_light}; border-left: 5px solid {color};
                        padding: 14px 18px; border-radius: 10px; color: #1a1a1a;">
                {"<strong>Immediate cardiology referral is strongly recommended.</strong> Multiple high-risk indicators are present." if prob_pct > 70
                 else "<strong>Borderline risk detected.</strong> Lifestyle modification and regular monitoring are advised." if prob_pct > 35
                 else "<strong>Current risk profile is within acceptable range.</strong> Continue preventive care and periodic check-ups."}
            </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Clinical Factor Analysis Table ────────────────────────────────
        st.subheader("🔍 Clinical Risk Factor Analysis")

        rows = []
        # Age
        age_flag = age >= WHO_LIMITS["age"]["threshold"]
        rows.append({
            "Feature":       "AGE",
            "User Value":    f"{age} yrs",
            "WHO Threshold": f"{WHO_LIMITS['age']['threshold']} years",
            "Status":        "⚠️ Elevated" if age_flag else "✅ Normal",
            "Interpretation": WHO_LIMITS["age"]["note"],
        })
        # Blood Pressure
        bp_flag = trestbps >= WHO_LIMITS["trestbps"]["threshold"]
        rows.append({
            "Feature":       "TRESTBPS",
            "User Value":    f"{trestbps} mmHg",
            "WHO Threshold": f"= {WHO_LIMITS['trestbps']['threshold']} mmHg",
            "Status":        "⚠️ High" if bp_flag else "✅ Normal",
            "Interpretation": WHO_LIMITS["trestbps"]["note"],
        })
        # Cholesterol
        chol_flag = chol > WHO_LIMITS["chol"]["threshold"]
        rows.append({
            "Feature":       "CHOL",
            "User Value":    f"{chol} mg/dL",
            "WHO Threshold": f"{WHO_LIMITS['chol']['threshold']} mg/dL",
            "Status":        "⚠️ High" if chol_flag else "✅ Normal",
            "Interpretation": WHO_LIMITS["chol"]["note"],
        })
        # FBS
        fbs_flag = fbs == 1
        rows.append({
            "Feature":       "FBS",
            "User Value":    "High (> 120)" if fbs_flag else "Normal (≤ 120)",
            "WHO Threshold": "120 mg/dL",
            "Status":        "⚠️ Elevated" if fbs_flag else "✅ Normal",
            "Interpretation": WHO_LIMITS["fbs"]["note"],
        })
        # CA
        ca_flag = ca > 0
        rows.append({
            "Feature":       "CA",
            "User Value":    str(ca),
            "WHO Threshold": "0",
            "Status":        f"⚠️ {ca} vessel(s) narrowed" if ca_flag else "✅ None",
            "Interpretation": WHO_LIMITS["ca"]["note"],
        })
        # Oldpeak
        op_flag = oldpeak > WHO_LIMITS["oldpeak"]["threshold"]
        rows.append({
            "Feature":       "OLDPEAK",
            "User Value":    f"{oldpeak} mm",
            "WHO Threshold": f"> {WHO_LIMITS['oldpeak']['threshold']} mm",
            "Status":        "⚠️ Significant" if op_flag else "✅ Normal",
            "Interpretation": WHO_LIMITS["oldpeak"]["note"],
        })

        analysis_df = pd.DataFrame(rows)
        st.dataframe(
            analysis_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Feature":       st.column_config.TextColumn("Feature",        width="small"),
                "User Value":    st.column_config.TextColumn("User Value",     width="small"),
                "WHO Threshold": st.column_config.TextColumn("WHO Threshold",  width="small"),
                "Status":        st.column_config.TextColumn("Status",         width="medium"),
                "Interpretation":st.column_config.TextColumn("Interpretation", width="large"),
            }
        )

        # Count flags
        flagged = sum([age_flag, bp_flag, chol_flag, fbs_flag, ca_flag, op_flag])
        if flagged == 0:
            st.success("✅ No major standardized risk triggers identified. Maintain healthy lifestyle.")
        else:
            st.warning(f"⚠️ {flagged} out of 6 monitored risk factor(s) exceed clinical thresholds.")

        # ── Raw input summary (collapsible) ──────────────────────────────
        with st.expander("📋 Full Input Summary (all 13 features sent to model)"):
            summary_df = pd.DataFrame(
                list(user_vals.items()), columns=["Feature", "Value"]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
        st.warning("Verify that your `.pkl` model file matches the feature set defined in this app.")
        st.code(f"Processed dataframe shape: {processed_df.shape}\nColumns: {list(processed_df.columns)}")