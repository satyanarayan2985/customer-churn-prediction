import pickle
import os
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "churn_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
feature_names = pickle.load(open(os.path.join(BASE_DIR, "feature_names.pkl"), "rb"))

st.title("Customer Churn Prediction")

# =========================
# USER INPUTS (ALL FEATURES)
# =========================
age = st.number_input("Age", 18, 100)
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.number_input("Tenure (months)", 0, 120)
usage = st.number_input("Usage Frequency", 0, 100)
support_calls = st.number_input("Support Calls", 0, 50)
payment_delay = st.number_input("Payment Delay (days)", 0, 60)

subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Yearly"])

last_interaction = st.number_input("Last Interaction (days ago)", 0, 365)
total_spend = st.number_input("Total Spend", 0.0, 100000.0)

# =========================
# ENCODING (MUST MATCH TRAINING)
# =========================
gender_map = {"Male": 0, "Female": 1}
subscription_map = {"Basic": 0, "Standard": 1, "Premium": 2}
contract_map = {"Monthly": 0, "Quarterly": 1, "Yearly": 2}

input_data = {
    "Age": age,
    "Gender": gender_map[gender],
    "Tenure": tenure,
    "Usage Frequency": usage,
    "Support Calls": support_calls,
    "Payment Delay": payment_delay,
    "Subscription Type": subscription_map[subscription],
    "Contract Length": contract_map[contract],
    "Last Interaction": last_interaction,
    "Total Spend": total_spend
}

input_df = pd.DataFrame([input_data])

# 🔥 FORCE SAME ORDER AS TRAINING
input_df = input_df[feature_names]

# =========================
# SCALE & PREDICT
# =========================
input_scaled = scaler.transform(input_df)

if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN")
    else:
        st.success("✅ Customer is likely to STAY")
