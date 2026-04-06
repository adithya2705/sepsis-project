import streamlit as st
import pickle
import numpy as np

# Load model and features
model = pickle.load(open("models/xgb_model.pkl", "rb"))
feature_names = pickle.load(open("models/features.pkl", "rb"))

st.set_page_config(page_title="Sepsis Prediction", layout="centered")

st.title("🏥 ICU Sepsis Early Warning System")

st.write("Enter patient details:")

input_data = []

for feature in feature_names:
    if feature == "Gender":
        val = st.selectbox("Gender", ["Male", "Female"])
        val = 1 if val == "Male" else 0
    else:
        val = st.number_input(feature, value=0.0)
    
    input_data.append(val)

# Predict
if st.button("🔍 Predict"):
    data = np.array(input_data).reshape(1, -1)

    prediction = model.predict(data)
    probability = model.predict_proba(data)[0][1]

    st.subheader("Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Sepsis")
    else:
        st.success("✅ Low Risk")

    st.progress(probability)
    st.write(f"Sepsis Probability: {probability*100:.1f}%")