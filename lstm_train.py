import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle
import os

# Load dataset
data = pd.read_csv("Dataset.csv")
data.columns = data.columns.str.strip()
data = data.drop(columns=["Unnamed: 0"], errors='ignore')
data = data.fillna(0)

# Same features
selected_features = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp',
    'EtCO2', 'BaseExcess', 'HCO3', 'pH',
    'WBC', 'Lactate', 'Creatinine',
    'Platelets', 'Age', 'Gender'
]

X = data[selected_features]
y = data["SepsisLabel"]

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

if not os.path.exists("models"):
    os.makedirs("models")

pickle.dump(scaler, open("models/lstm_scaler.pkl", "wb"))

# Reshape
X = X.reshape(X.shape[0], 1, X.shape[1])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = Sequential()
model.add(LSTM(64, input_shape=(1, X.shape[2])))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=32)

model.save("models/lstm_model.h5")

print("✅ LSTM trained")