import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Set page layout configuration
st.set_page_config(page_title="Customer Churn Predictor", page_icon="🔮", layout="centered")

# Title and description
st.title("🔮 Customer Churn Prediction App")
st.write("Input the customer's demographics and service details below to compute real-time churn risk.")

# Load models and preprocessing pipelines safely
@st.cache_resource
def load_artifacts():
    model_path = os.path.join('models', 'model.pkl')
    scaler_path = os.path.join('models', 'scaler.pkl')
    encoders_path = os.path.join('models', 'encoders.pkl')
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoders_path)):
        st.error("⚠️ Model files not found! Please run `python train.py` first to generate artifacts.")
        st.stop()
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
        
    return model, scaler, encoders

try:
    model, scaler, encoders = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# ----------------- STREAMLIT USER INTERFACE FORM -----------------
st.subheader("📋 Customer Information Form")

with st.form("churn_input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.number_input("Tenure Months", min_value=0, max_value=100, value=12, step=1)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])

    with col2:
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=780.0, step=1.0)
        
    submit_button = st.form_submit_button(label="🔮 Predict Churn Risk")

# ----------------- REAL-TIME MODEL INFERENCE -----------------
if submit_button:
    # 1. Map readable string options back to binary digits matching original dataset format
    senior_num = 1 if senior_citizen == "Yes" else 0
    
    # 2. Build the input feature frame matching pipeline format exactly
    input_data = pd.DataFrame([{
        'Gender': gender,
        'Senior Citizen': senior_num,
        'Partner': partner,
        'Dependents': dependents,
        'Tenure Months': tenure,
        'Phone Service': phone_service,
        'Multiple Lines': multiple_lines,
        'Internet Service': internet_service,
        'Contract': contract,
        'Paperless Billing': paperless_billing,
        'Monthly Charges': monthly_charges,
        'Total Charges': total_charges
    }])
    
    # 3. Apply the fitted LabelEncoders from training step
    for col, encoder in encoders.items():
        try:
            input_data[col] = encoder.transform(input_data[col].astype(str))
        except ValueError:
            # Fallback string handling just in case
            input_data[col] = 0

    # 4. Standardize the data with original feature metrics
    input_scaled = scaler.transform(input_data)
    
    # 5. Extract probabilities
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    st.markdown("---")
    st.subheader("🎯 Prediction Analysis Summary")
    
    # 6. Display custom conditional warning metrics based on calculation threshold
    if prediction == 1:
        st.error(f"🚨 **High Risk Alert:** This customer is predicted to **CHURN** with a probability score of **{probability:.1%}**.")
    else:
        st.success(f"✅ **Low Risk:** This customer is predicted to **STAY** with a churn probability score of only **{probability:.1%}**.")