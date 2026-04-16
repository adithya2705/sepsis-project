import numpy as np
import pickle

xgb_model = pickle.load(open("models/xgb_model.pkl", "rb"))
xgb_scaler = pickle.load(open("models/scaler.pkl", "rb"))

def predict_ensemble(input_data):
    xgb_input = xgb_scaler.transform(input_data)
    xgb_prob = xgb_model.predict_proba(xgb_input)[0][1]
    return xgb_prob