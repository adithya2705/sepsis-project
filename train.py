import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pickle
import os

# Load dataset
data = pd.read_csv("Dataset.csv")
data.columns = data.columns.str.strip()

print("Before cleaning:", data.shape)

# Fill missing values
data = data.fillna(0)

print("After cleaning:", data.shape)

# Split features and target
X = data.drop("SepsisLabel", axis=1)
y = data["SepsisLabel"]

# SAVE FEATURE NAMES 🔥
if not os.path.exists("models"):
    os.makedirs("models")

pickle.dump(list(X.columns), open("models/features.pkl", "wb"))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("models/xgb_model.pkl", "wb"))

print("Model trained and saved successfully!")