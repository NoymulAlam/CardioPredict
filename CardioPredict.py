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
    page_title="CardioPredict",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# 2. Custom CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
    section[data-testid="stSidebar"] { background: #0f1117; }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    .risk-card {
        padding: 24px 20px;
        border-radius: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }
    .risk-card h1 { margin: 0; font-size: 3rem; font-weight: 800; }
    .risk-card p  { margin: 6px 0 0; font-size: 1.1rem; font-weight: 600; letter-spacing: 1px; }
    .info-box {
        background: #1e2130;
        border-left: 4px solid #4a9eff;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
        color: #cfd8e3;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 3. Model Loading
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="🔬 Loading Clinical Model...")
def load_model():
    model_path = "best_tuned_logistic_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# ─────────────────────────────────────────
# 4. Preprocessing
# ─────────────────────────────────────────
def preprocess_for_model(user_vals: dict, model) -> pd.DataFrame:
    input_df = pd.DataFrame([user_vals])
    if not hasattr(model, "feature_names_in_"):
        return input_df

    expected_cols = list(model.feature_names_in_)
    aligned = pd.DataFrame(columns=expected_cols, index=[0])

    for col in expected_cols:
        if col in input_df.columns:
            aligned[col] = input_df[col].values
        elif "_" in col:
            parts = col.rsplit("_", 1)
            base_feature = parts[0]
            encoded_val  = parts[1]
            raw_val = user_vals.get(base_feature)
            aligned[col] = 1 if str(raw_val) == encoded_val else 0
        else:
            aligned[col] = 0

    return aligned.astype(float)

# ─────────────────────────────────────────
# 5. PDF Report Generator
# ─────────────────────────────────────────
def create_pdf(probability: float, label: str, user_vals: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(30, 30, 50)
    pdf.rect(0, 0, 210, 30, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 30, "CardioPredict - Clinical Risk Report", ln=True, align='C')
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    
    if "HIGH" in label: pdf.set_fill_color(230, 57, 70)
    elif "MODERATE" in label: pdf.set_fill_color(244, 162, 97)
    else: pdf.set_fill_color(42, 157, 143)
    
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 12, f"   Risk Status: {label}   |   Result: {probability:.2f}%", ln=True, fill=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Input Summary", ln=True, border='B')
    
    pdf.set_font("Arial", size=10)
    for key, val in user_vals.items():
        clean_val = str(val).replace('—', '-')
        pdf.cell(90, 7, f" {key.upper()}", border=1)
        pdf.cell(100, 7, f" {clean_val}", border=1, ln=True)
        
    return pdf.output(dest='S').encode('latin-1')

# ─────────────────────────────────────────
# 6. Sidebar Inputs
# ─────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822118.png", width=80)
    st.title("Patient Vitals")
    
    age = st.slider("Age", 18, 100, 55)
    sex = st.radio("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female", horizontal=True)
    trestbps = st.number_input("Resting BP", 80, 220, 130)
    chol = st.number_input("Cholesterol", 100, 600, 210)
    fbs = st.selectbox("FBS > 120 mg/dL?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
    exang = st.checkbox("Exercise Angina")
    oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, 0.1)
    slope = st.selectbox("ST Slope", options=[0, 1, 2])
    restecg = st.selectbox("Resting ECG", options=[0, 1, 2])
    ca = st.select_slider("Vessels (CA)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", options=[0, 1, 2])

    st.divider()
    analyze_btn = st.button("🚀 Run Diagnostic Analysis", use_container_width=True, type="primary")

# ─────────────────────────────────────────
# 7. Main Canvas
# ─────────────────────────────────────────
st.title("🫀 CardioPredict")
st.caption("AI Cardiovascular Risk Assessment")

model = load_model()

if not analyze_btn:
    st.info("Fill in the patient details on the left and click 'Run Diagnostic Analysis'.")
else:
    if model is None:
        st.error("Model file not found.")
        st.stop()

    user_vals = {"age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol, "fbs": fbs, 
                 "restecg": restecg, "thalach": thalach, "exang": int(exang), "oldpeak": oldpeak, 
                 "slope": slope, "ca": ca, "thal": thal}

    processed_df = preprocess_for_model(user_vals, model)

    try:
        # If the bar isn't moving, the model likely expects a specific scale.
        # Ensure the column order matches the model's training order:
        if hasattr(model, "feature_names_in_"):
            processed_df = processed_df[model.feature_names_in_]

        prob = model.predict_proba(processed_df)[0][1]
        prob_pct = prob * 100

        if prob_pct > 70: label, color, icon, bg = "HIGH RISK", "#e63946", "🚨", "#fdecea"
        elif prob_pct > 35: label, color, icon, bg = "MODERATE RISK", "#f4a261", "⚠️", "#fff4ec"
        else: label, color, icon, bg = "LOW RISK", "#2a9d8f", "✅", "#e8f8f6"

        st.subheader(f"{icon} Results")
        c1, c2 = st.columns([1, 2])

        with c1:
            st.markdown(f'<div class="risk-card" style="background:{color}"><h1>{prob_pct:.1f}%</h1><p>{label}</p></div>', unsafe_allow_html=True)
            st.progress(prob)
            pdf_bytes = create_pdf(prob_pct, label, user_vals)
            st.download_button("📥 Download Report", pdf_bytes, "Report.pdf", "application/pdf", use_container_width=True)

        with c2:
            st.metric("Blood Pressure", f"{trestbps} mmHg", f"{trestbps-140:+d}", delta_color="inverse")
            st.metric("Cholesterol", f"{chol} mg/dL", f"{chol-200:+d}", delta_color="inverse")
            st.markdown(f'<div style="background:{bg}; border-left:5px solid {color}; padding:15px; border-radius:10px; color:black">Result interpretation based on AI clinical analysis.</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction Error: {e}")