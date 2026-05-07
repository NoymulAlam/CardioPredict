import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from fpdf import FPDF
from datetime import datetime

# -----------------------------
# 1. Configuration & Assets
# -----------------------------
st.set_page_config(page_title="CardioPredict AI Pro", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource(show_spinner="Loading Clinical Model...")
def load_model():
    model_path = "best_tuned_logistic_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# WHO Standards for reference
WHO_LIMITS = {
    "trestbps": 140,
    "chol": 200,
    "fbs": 120,
    "age": 50
}

# -----------------------------
# 2. Logic & Preprocessing
# -----------------------------
def preprocess_for_model(user_vals, model):
    input_df = pd.DataFrame([user_vals])
    
    # Ensure columns match training order and handle potential One-Hot Encoding
    if hasattr(model, "feature_names_in_"):
        expected = model.feature_names_in_
        final_df = pd.DataFrame(columns=expected)
        for col in expected:
            if col in input_df.columns:
                final_df[col] = input_df[col]
            elif "_" in col: # Handle manual dummy variables if needed
                base, val = col.split("_")[0], col.split("_")[1]
                final_df[col] = 1 if str(user_vals.get(base)) == val else 0
            else:
                final_df[col] = 0
        return final_df
    return input_df

def create_pdf(probability, label, user_vals):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="CardioPredict AI - Clinical Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(200, 10, txt=f"Risk Status: {label}", ln=True)
    pdf.cell(200, 10, txt=f"Calculated Probability: {probability:.2f}%", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt="Input Parameters:", ln=True, border='B')
    for k, v in user_vals.items():
        pdf.cell(200, 8, txt=f"{k.upper()}: {v}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# -----------------------------
# 3. Sidebar UI (Inputs)
# -----------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822118.png", width=100)
    st.title("Patient Vitals")
    st.divider()
    
    with st.expander("👤 Basic Info", expanded=True):
        age = st.slider("Age", 18, 100, 55)
        sex = st.radio("Biological Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
        
    with st.expander("🩺 Clinical Measures", expanded=True):
        trestbps = st.number_input("Resting BP (mmHg)", 80, 220, 130)
        chol = st.number_input("Cholesterol (mg/dL)", 100, 500, 210)
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        fbs = st.selectbox("Fasting Blood Sugar > 120?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        
    with st.expander("🧪 Specialized Tests"):
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                          format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][x])
        exang = st.checkbox("Exercise Induced Angina")
        oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
        ca = st.select_slider("Major Vessels (CA)", options=[0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", [0, 1, 2], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x])

    analyze_btn = st.button("🚀 Run Diagnostic Analysis", use_container_width=True, type="primary")

# -----------------------------
# 4. Main Canvas (Results)
# -----------------------------
st.title("CardioPredict AI")
st.caption("Advanced cardiovascular risk assessment using the Cleveland Clinical Dataset logic.")

model = load_model()

if not analyze_btn:
    # Welcome View
    c1, c2, c3 = st.columns(3)
    c1.metric("BP Target", "< 140", "WHO Std")
    c2.metric("Cholesterol Target", "< 200", "WHO Std")
    c3.metric("FBS Target", "< 120", "WHO Std")
    st.info("👈 Enter patient data in the sidebar and click 'Run Analysis' to begin.")

else:
    if model:
        # Prepare Data
        user_vals = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
            "fbs": fbs, "restecg": 0, "thalach": thalach, "exang": int(exang),
            "oldpeak": oldpeak, "slope": 1, "ca": ca, "thal": thal
        }
        
        processed_df = preprocess_for_model(user_vals, model)
        
        try:
            # Prediction
            prob = model.predict_proba(processed_df)[0][1]
            prob_pct = prob * 100
            
            # Styling based on risk
            if prob_pct > 70:
                label, color, icon = "HIGH RISK", "#e63946", "🚨"
            elif prob_pct > 35:
                label, color, icon = "MODERATE RISK", "#f4a261", "⚠️"
            else:
                label, color, icon = "LOW RISK", "#2a9d8f", "✅"

            # Result Header
            st.subheader(f"{icon} Assessment Results")
            
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                st.markdown(f"""
                    <div style="background-color: {color}; padding: 20px; border-radius: 15px; text-align: center; color: white;">
                        <h1 style="margin:0;">{prob_pct:.1f}%</h1>
                        <p style="margin:0; font-weight: bold;">{label}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # PDF Download
                report_bytes = create_pdf(prob_pct, label, user_vals)
                st.download_button("📥 Download Physician Report", data=report_bytes, 
                                   file_name=f"CardioReport_{datetime.now().strftime('%Y%m%d')}.pdf",
                                   use_container_width=True)

            with col_res2:
                # Comparison Metrics
                m1, m2 = st.columns(2)
                bp_diff = trestbps - WHO_LIMITS["trestbps"]
                chol_diff = chol - WHO_LIMITS["chol"]
                
                m1.metric("Blood Pressure", f"{trestbps} mmHg", f"{bp_diff} vs Limit" if bp_diff > 0 else "Normal", delta_color="inverse")
                m2.metric("Cholesterol", f"{chol} mg/dL", f"{chol_diff} vs Limit" if chol_diff > 0 else "Normal", delta_color="inverse")
                
                st.progress(prob)
                st.caption(f"Risk Probability Spectrum: 0% (Healthy) to 100% (High Risk)")

            st.divider()
            
            # Clinical Insights Table
            st.subheader("🔍 Clinical Factor Analysis")
            factors = []
            if trestbps >= 140: factors.append(["Hypertension", "Resting BP is above the WHO hypertension threshold."])
            if chol > 200: factors.append(["Hyperlipidemia", "Elevated cholesterol increases plaque risk."])
            if ca > 0: factors.append(["Vessel Blockage", f"{ca} major vessels showing narrowing via fluoroscopy."])
            if oldpeak > 1.5: factors.append(["ST Depression", "Significant stress indicated on heart during activity."])
            
            if factors:
                insight_df = pd.DataFrame(factors, columns=["Observation", "Clinical Note"])
                st.table(insight_df)
            else:
                st.success("No major standardized risk triggers identified in biometric data.")

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
            st.warning("Ensure your model (.pkl) matches the feature set used in this app.")
    else:
        st.error("Model not loaded. Check file path.")