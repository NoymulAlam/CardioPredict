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
    "age": {
        "description": "Risk increases with age. WHO notes that majority of CVD deaths occur in those over 70, but premature deaths (under 70) are preventable.",
        "threshold": "> 50 years",
        "role": "Demographic"},
    "trestbps": {
        "description": "Resting Blood Pressure. WHO identifies hypertension (>=140/90) as the leading risk factor for CVD.",
        "threshold": ">= 140 mmHg",
        "role": "Vitals"},
    "chol": {
        "description": "Total Cholesterol. High lipids lead to fatty deposits in blood vessels, increasing risk of blockage.",
        "threshold": "> 200 mg/dL",
        "role": "Biometric"},
    "fbs": {
        "description": "Fasting Blood Sugar. Values over 120-126 mg/dL indicate metabolic risk or Diabetes.",
        "threshold": "> 120 mg/dL",
        "role": "Metabolic"},
    "oldpeak": {
        "description": "ST depression. Measures heart stress during physical activity compared to rest.",
        "threshold": "> 1.0",
        "role": "Clinical"},
    "ca": {
        "description": "Calcification of major vessels. Visible vessels via fluoroscopy indicate arterial damage.",
        "threshold": "> 0",
        "role": "Clinical"},
    "cp": {
        "description": "Angina symptoms. Typical or atypical chest pain types linked to coronary artery disease.",
        "threshold": "Types 0, 1, 2",
        "role": "Symptoms"}
}

# -----------------------------
# 2. Functional Utilities
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_assets():
    """Loads the pre-trained model."""
    model_path = "best_tuned_logistic_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error(f"Error: '{model_path}' not found. Please ensure the model file is in the root directory.")
        return None

def explain_cvd_risks(user_data):
    """Identifies and explains clinical risk factors based on WHO thresholds."""
    risk_indicators = []
    if user_data['age'] > 50: risk_indicators.append("age")
    if user_data['trestbps'] >= 140: risk_indicators.append("trestbps")
    if user_data['chol'] > 200: risk_indicators.append("chol")
    if user_data['fbs'] == 1: risk_indicators.append("fbs")
    if user_data['oldpeak'] > 1.0: risk_indicators.append("oldpeak")
    if user_data['ca'] > 0: risk_indicators.append("ca")
    if user_data['cp'] in [0, 1, 2]: risk_indicators.append("cp")

    if not risk_indicators:
        return None

    table_data = []
    for feat in risk_indicators:
        info = VALIDATION_FEATURES_CVD.get(feat, {})
        val = user_data.get(feat, "N/A")
        
        # User-friendly mapping for visual display
        if feat == 'fbs': val = "High (> 120)" if val == 1 else "Normal"
        if feat == 'cp': val = ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][val]
        
        table_data.append({
            "Feature": feat.upper(),
            "Current Value": val,
            "WHO Threshold": info.get("threshold", ""),
            "Interpretation": info.get("description", "")
        })

    return pd.DataFrame(table_data)

def generate_report_pdf(user_vals, risk_label, percent, df_table=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "CardioPredict AI: Clinical Risk Assessment", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(200, 0, 0) if "HIGH" in risk_label else pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f"Assessment Result: {risk_label} ({percent:.2f}%)", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    if df_table is not None:
        pdf.set_font("Arial", "B", 10)
        pdf.cell(0, 8, "Identified Risk Factors (WHO References):", ln=True)
        pdf.ln(2)
        pdf.set_font("Arial", "", 8)
        
        cols = [25, 35, 30, 95]
        headers = ["Feature", "Value", "Threshold", "Interpretation"]
        for i, h in enumerate(headers): pdf.cell(cols[i], 7, h, border=1, fill=False)
        pdf.ln()
        
        for _, row in df_table.iterrows():
            pdf.cell(cols[0], 7, str(row['Feature']), border=1)
            pdf.cell(cols[1], 7, str(row['Current Value']), border=1)
            pdf.cell(cols[2], 7, str(row['WHO Threshold']), border=1)
            pdf.multi_cell(cols[3], 7, str(row['Interpretation']), border=1)
    
    return bytes(pdf.output())

# -----------------------------
# 3. Main Application
# -----------------------------
st.set_page_config(page_title="CardioPredict", layout="wide")

st.markdown("""
    <div style="background-color: #0f172a; padding: 25px; border-radius: 10px; margin-bottom: 25px; border-left: 10px solid #dc2626;">
        <h1 style="color: white; margin: 0;">CardioPredict</h1>
        <p style="color: #94a3b8; margin: 0; font-size: 1.1rem;">Clinical Intelligence Platform for Heart Disease Risk Prediction (WHO Alignment)</p>
    </div>
""", unsafe_allow_html=True)

model = load_assets()

if model:
    tabs = st.tabs(["🩺 Patient Evaluation", "📚 WHO Guidelines"])

    with tabs[0]:
        with st.form("evaluation_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.subheader("Demographics")
                age = st.number_input("Age", 1, 120, 50)
                sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x==1 else "Female")
                cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                                  format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][x])
                trestbps = st.number_input("Resting BP (mmHg)", 80, 250, 120)

            with c2:
                st.subheader("Biometrics")
                chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
                fbs = st.selectbox("Fasting Blood Sugar > 120mg/dL", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
                thalach = st.number_input("Max Heart Rate", 60, 220, 150)
                restecg = st.selectbox("Resting ECG", [0, 1, 2], format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x])

            with c3:
                st.subheader("Clinical Tests")
                exang = st.selectbox("Exercise Angina", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
                oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 0.0, step=0.1)
                slope = st.selectbox("Slope of ST", [0, 1, 2], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
                ca = st.slider("Major Vessels (CA)", 0, 3, 0)
                thal = st.selectbox("Thal", [0, 1, 2], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x])

            eval_btn = st.form_submit_button("Run Clinical Analysis", type="primary")

        if eval_btn:
            # Data dictionary for risk explanation and prediction
            user_vals = {
                "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol, 
                "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang, 
                "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
            }
            
            # Prediction Logic
            feature_order = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                             'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            input_df = pd.DataFrame([user_vals])[feature_order]
            
            # Get probability from the Logistic Regression model
            probability = model.predict_proba(input_df)[0][1]

            # UI Feedback based on probability
            if probability > 0.75:
                label, color, bg = "HIGH CLINICAL RISK", "#991b1b", "#fee2e2"
            elif probability > 0.4:
                label, color, bg = "MODERATE RISK", "#854d0e", "#fef9c3"
            else:
                label, color, bg = "LOW CLINICAL RISK", "#166534", "#dcfce7"

            st.markdown(f"""
                <div style="background-color: {bg}; color: {color}; padding: 20px; border-radius: 8px; border: 1px solid {color};">
                    <h3 style="margin:0;">{label}</h3>
                    <p style="margin:0; font-size: 1.2rem;">The calculated probability of heart disease is <strong>{probability*100:.2f}%</strong></p>
                </div>
            """, unsafe_allow_html=True)

            # Risk Explanation Table
            risk_df = explain_cvd_risks(user_vals)
            if risk_df is not None:
                st.write("#### Clinical Risk Factor Breakdown")
                st.table(risk_df)
            else:
                st.success("All biometric inputs are within standard WHO clinical thresholds.")
            
            # Export to PDF
            pdf_data = generate_report_pdf(user_vals, label, probability*100, risk_df)
            st.download_button("📥 Download Physician Report (PDF)", data=pdf_data, 
                               file_name=f"CHD_Report_{datetime.now().strftime('%Y%m%d')}.pdf")

    with tabs[1]:
        st.header("World Health Organization Standards")
        st.info("The thresholds below are aligned with the WHO global fact sheet on Cardiovascular Disease prevention.")
        for feat, info in VALIDATION_FEATURES_CVD.items():
            with st.expander(f"{feat.upper()} - Standard: {info['threshold']}"):
                st.write(info['description'])
