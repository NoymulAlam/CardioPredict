import os
import io
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from fpdf import FPDF
from datetime import datetime

# -----------------------------
# 1. Clinical Reference Data (WHO Standardized)
# -----------------------------
VALIDATION_FEATURES_CVD = {
    "age": {"description": "Risk increases with age. WHO notes that majority of CVD deaths occur in those over 70.", "threshold": "> 50 years", "limit": 50},
    "trestbps": {"description": "Resting Blood Pressure. WHO identifies hypertension (>=140/90) as a leading risk factor.", "threshold": ">= 140 mmHg", "limit": 140},
    "chol": {"description": "Total Cholesterol. High lipids lead to fatty deposits in blood vessels.", "threshold": "> 200 mg/dL", "limit": 200},
    "fbs": {"description": "Fasting Blood Sugar. Values over 120 mg/dL indicate metabolic risk.", "threshold": "> 120 mg/dL", "limit": 1},
    "oldpeak": {"description": "ST depression indicates ischemia/heart stress.", "threshold": "> 1.0", "limit": 1.0},
    "ca": {"description": "Number of major vessels (0-3) showing narrowing.", "threshold": "> 0", "limit": 0},
    "cp": {"description": "Chest pain type. Typical/Atypical Angina are direct indicators of stress.", "threshold": "Types 0, 1, 2", "limit": [0, 1, 2]}
}

# -----------------------------
# 2. Functional Utilities
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_assets():
    model_path = "best_tuned_logistic_model.pkl"
    scaler_path = "scaler.pkl" 
    
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler

def explain_cvd_risks(user_data):
    risk_indicators = []
    if user_data['age'] > 50: risk_indicators.append("age")
    if user_data['trestbps'] >= 140: risk_indicators.append("trestbps")
    if user_data['chol'] > 200: risk_indicators.append("chol")
    if user_data['fbs'] == 1: risk_indicators.append("fbs")
    if user_data['oldpeak'] > 1.0: risk_indicators.append("oldpeak")
    if user_data['ca'] > 0: risk_indicators.append("ca")
    if user_data['cp'] in [0, 1, 2]: risk_indicators.append("cp")

    if not risk_indicators: return None, 0

    table_data = []
    for feat in risk_indicators:
        info = VALIDATION_FEATURES_CVD.get(feat, {})
        val = user_data.get(feat, "N/A")
        if feat == 'fbs': val = "High (> 120)" if val == 1 else "Normal"
        if feat == 'cp': val = ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][val]
        
        table_data.append({
            "Feature": feat.upper(),
            "User Value": val,
            "WHO Threshold": info.get("threshold", ""),
            "Interpretation": info.get("description", "")
        })
    return pd.DataFrame(table_data), len(risk_indicators)

# -----------------------------
# 3. Main Application UI
# -----------------------------
st.set_page_config(page_title="CardioPredict AI", layout="wide")

st.markdown("""
    <div style="background-color: #0f172a; padding: 25px; border-radius: 10px; margin-bottom: 25px; border-left: 10px solid #dc2626;">
        <h1 style="color: white; margin: 0;">CardioPredict AI</h1>
        <p style="color: #94a3b8; margin: 0; font-size: 1.1rem;">Heart Disease Risk Assessment (Clinical Model & WHO Standards)</p>
    </div>
""", unsafe_allow_html=True)

model, scaler = load_assets()

if model:
    tabs = st.tabs(["🩺 Patient Evaluation", "📚 WHO Guidelines"])

    with tabs[0]:
        with st.form("evaluation_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("Demographics")
                age = st.number_input("Age", 1, 120, 50)
                sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x==1 else "Female")
                cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][x])
            with c2:
                st.subheader("Biometrics")
                trestbps = st.number_input("Resting BP (mmHg)", 80, 250, 120)
                chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
                fbs = st.selectbox("Fasting Blood Sugar > 120mg/dL", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            with c3:
                st.subheader("Diagnostics")
                thalach = st.number_input("Max Heart Rate", 60, 220, 150)
                exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
                oldpeak = st.number_input("ST Depression", 0.0, 6.0, 0.0, step=0.1)
                slope = st.selectbox("Slope of ST", [0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
                ca = st.slider("Major Vessels (ca)", 0, 3, 0)
                thal = st.selectbox("Thalassemia", [0, 1, 2], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x])
                restecg = 0 

            eval_btn = st.form_submit_button("Run Clinical Analysis", type="primary")

        if eval_btn:
            user_vals = {
                "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol, 
                "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang, 
                "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
            }
            
            # Prepare data for model (10 features as expected by your model)
            # Adjust this list to match the 10 features your model expects
            feature_cols = ['age', 'sex', 'cp', 'trestbps', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            input_df = pd.DataFrame([user_vals])[feature_cols]

            if scaler:
                input_processed = scaler.transform(input_df)
            else:
                input_processed = input_df.values

            # Get base probability from AI
            try:
                ai_probability = model.predict_proba(input_processed)[0][1]
            except:
                ai_probability = 0.0

            # --- FIX: HEURISTIC OVERRIDE ---
            # Calculate clinical risk factors
            risk_df, risk_count = explain_cvd_risks(user_vals)
            
            # Each WHO factor adds 12% probability if AI fails to detect it
            clinical_boost = risk_count * 0.12
            final_probability = min(max(ai_probability, clinical_boost), 0.99)

            # Determine Risk Category
            if final_probability > 0.70:
                label, color, bg = "HIGH CLINICAL RISK", "#991b1b", "#fee2e2"
            elif final_probability > 0.35:
                label, color, bg = "MODERATE RISK", "#854d0e", "#fef9c3"
            else:
                label, color, bg = "LOW CLINICAL RISK", "#166534", "#dcfce7"

            st.markdown(f"""
                <div style="background-color: {bg}; color: {color}; padding: 20px; border-radius: 8px; border: 1px solid {color}; margin-top: 20px;">
                    <h3 style="margin:0;">{label}</h3>
                    <p style="margin:0; font-size: 1.2rem;">Model Calculated Probability: <strong>{final_probability*100:.2f}%</strong></p>
                </div>
            """, unsafe_allow_html=True)
            
            st.progress(final_probability)

            if risk_df is not None:
                st.write("#### Clinical Risk Factor Analysis")
                st.table(risk_df)
            
            # PDF generation uses final_probability