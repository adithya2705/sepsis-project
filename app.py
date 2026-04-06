import streamlit as st
import pickle
import numpy as np

# ==============================
# Load Model & Features
# ==============================
model = pickle.load(open("models/xgb_model.pkl", "rb"))
feature_names = pickle.load(open("models/features.pkl", "rb"))

# ==============================
# Page Config
# ==============================
st.set_page_config(page_title="Sepsis Prediction", layout="wide")

st.title("🏥 ICU Sepsis Early Warning System")
st.write("Enter patient clinical parameters:")

# ==============================
# Input Fields
# ==============================
input_data = []

col1, col2 = st.columns(2)

for i, feature in enumerate(feature_names):
    if i % 2 == 0:
        val = col1.number_input(feature, value=0.0)
    else:
        val = col2.number_input(feature, value=0.0)
    
    input_data.append(val)

# Debug (can remove later)
st.write("Total features:", len(input_data))

# ==============================
# Prediction
# ==============================
if st.button("🔍 Predict Sepsis Risk"):
    data = np.array(input_data).reshape(1, -1)

    prediction = model.predict(data)
    probability = model.predict_proba(data)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Sepsis - Immediate Attention Required!")
    else:
        st.success("✅ Patient is Stable (Low Sepsis Risk)")

    st.write(f"Sepsis Risk Score: {probability:.2f}")