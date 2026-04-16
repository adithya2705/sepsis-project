import numpy as np
import pickle
from tensorflow.keras.models import load_model

xgb_model = pickle.load(open("models/xgb_model.pkl", "rb"))
xgb_scaler = pickle.load(open("models/scaler.pkl", "rb"))

lstm_model = load_model("models/lstm_model.h5")
lstm_scaler = pickle.load(open("models/lstm_scaler.pkl", "rb"))

def predict_ensemble(input_data):
    # XGB
    xgb_input = xgb_scaler.transform(input_data)
    xgb_prob = xgb_model.predict_proba(xgb_input)[0][1]

    # LSTM
    lstm_input = lstm_scaler.transform(input_data)
lstm_input = lstm_input.reshape(1, 1, lstm_input.shape[1])
lstm_prob = lstm_model.predict(lstm_input)[0][0]

return (xgb_prob + lstm_prob) / 2