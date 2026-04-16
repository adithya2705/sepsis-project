from ensemble_streamlit import predict_ensemble
import streamlit as st
import pickle
import numpy as np

# ==============================
# Load features
# ==============================
feature_names = pickle.load(open("models/features.pkl", "rb"))

# ==============================
# Page setup
# ==============================
st.set_page_config(page_title="Sepsis Prediction", layout="centered")

st.title("🏥 ICU Sepsis Early Warning System")
st.write("Enter patient details:")

# ==============================
# Input fields
# ==============================
input_data = []

for feature in feature_names:
    if feature == "Gender":
        val = st.selectbox("Gender", ["Male", "Female"])
        val = 1 if val == "Male" else 0
    else:
        val = st.number_input(feature, value=0.0)
    
    input_data.append(val)

# ==============================
# Prediction
# ==============================
if st.button("🔍 Predict"):
    data = np.array(input_data).reshape(1, -1)

    # 🔥 Ensemble prediction
    probability = predict_ensemble(data)

    st.subheader("Result")

    # ==============================
    # 🔥 Clinical Feature Extraction
    # ==============================
    # (Based on your feature order)
    hr = input_data[0]
    sbp = input_data[3]
    map_val = input_data[4]
    lactate = input_data[11]
    platelets = input_data[13]
    temp = input_data[2]

 # ==============================
# FINAL RISK CLASSIFICATION 🔥
# ==============================

if (
    lactate > 4 or
    platelets < 100 or
    sbp < 90 or
    temp > 39 or
    probability > 0.5
):
    risk = "HIGH"
    st.error("🚨 HIGH RISK - Immediate Attention Required!")
    display_prob = max(probability, 0.65)

elif (
    lactate > 2 or
    hr > 100 or
    temp > 38 or
    probability > 0.25
):
    risk = "MODERATE"
    st.warning("⚠️ MODERATE RISK - Monitor Patient Closely")
    display_prob = max(probability, 0.4)

else:
    risk = "LOW"
    st.success("✅ LOW RISK - Patient Stable")
    display_prob = probability * 0.5

# Clamp probability
display_prob = min(display_prob, 1.0)

# Show output
st.progress(display_prob)
st.write(f"Sepsis Probability: {display_prob*100:.1f}%")