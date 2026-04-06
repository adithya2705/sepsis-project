import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pickle
import os

# ==============================
# Load dataset
# ==============================
data = pd.read_csv("Dataset.csv")
data.columns = data.columns.str.strip()

# Remove unwanted column
data = data.drop(columns=["Unnamed: 0"], errors='ignore')

# Fill missing values
data = data.fillna(0)

# ==============================
# Selected features (with Gender)
# ==============================
selected_features = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp',
    'pH', 'WBC', 'Lactate', 'Creatinine',
    'Platelets', 'Age', 'Gender'
]

# Features and target
X = data[selected_features]
y = data["SepsisLabel"]

# ==============================
# Create models folder
# ==============================
if not os.path.exists("models"):
    os.makedirs("models")

# Save feature names
pickle.dump(selected_features, open("models/features.pkl", "wb"))

# ==============================
# Train-test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# Scaling (VERY IMPORTANT 🔥)
# ==============================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
pickle.dump(scaler, open("models/scaler.pkl", "wb"))

# ==============================
# Train model
# ==============================
model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("models/xgb_model.pkl", "wb"))

print("✅ Model trained with scaling + selected features!")