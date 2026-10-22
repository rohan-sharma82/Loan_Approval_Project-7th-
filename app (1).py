import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Configure page layouts and titles
st.set_page_config(page_title="Loan Approval Predictor", layout="centered", page_icon="🏦")

st.title("🏦 Loan Approval Prediction Web App")
st.write("Enter applicant metrics below to estimate the likelihood of loan approval instantly.")

# Safely load the serialized pipeline
@st.cache_resource
def load_model():
    with open('loan_approval_pipeline.pkl', 'rb') as f:
        return pickle.load(f)

try:
    pipeline = load_model()
except Exception as e:
    st.error(f"Error loading the model file. Please ensure 'loan_approval_pipeline.pkl' is in the same directory. Details: {e}")
    st.stop()

# Create two columns for clean input grouping
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    applicant_income = st.number_input("Applicant Monthly Income ($)", min_value=0, value=5000, step=100)
    coapplicant_income = st.number_input("Coapplicant Monthly Income ($)", min_value=0, value=0, step=100)
    loan_amount = st.number_input("Requested Loan Amount (in thousands $)", min_value=0, value=150, step=10)
    loan_term = st.number_input("Loan Term Duration (in months)", min_value=12, max_value=480, value=360, step=12)
    credit_history = st.selectbox("Does Credit History meet guidelines?", ["Yes (1.0)", "No (0.0)"])

property_area = st.selectbox("Property Area Location", ["Rural", "Semiurban", "Urban"])

# Mapping user inputs back to expected features and pipeline format
credit_val = 1.0 if "Yes" in credit_history else 0.0
# The preprocessor pipeline expects Dependents as a string since the cleaning steps handle it
dep_val = dependents

applicant_data = {
    'Gender': gender,
    'Married': married,
    'Dependents': dep_val,
    'Education': education,
    'Self_Employed': self_employed,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_term,
    'Credit_History': credit_val,
    'Property_Area': property_area
}

# Run Predictions
if st.button("🎯 Run Approval Diagnosis"):
    # Transform dictionary into schema Dataframe
    input_df = pd.DataFrame([applicant_data])

    try:
        # Generate raw predictions and confidence ratings
        prediction = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0][1]

        st.markdown("--- ")
        if prediction == 1:
            st.success(f"🎉 **Loan Recommendation: Approved!** (Approval Probability: {proba * 100:.1f}%)")
        else:
            st.error(f"❌ **Loan Recommendation: Rejected.** (Approval Probability: {proba * 100:.1f}%)")

        # Context Indicators
        st.info("💡 *Decision Note: Credit History checks and EMI-to-Income balances are the primary factors affecting this outcome.*")
    except Exception as err:
        st.error(f"An unexpected error occurred during transformation or inference: {err}")
