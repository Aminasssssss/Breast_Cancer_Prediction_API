import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

print("Loading data...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
feature_names = X.columns.tolist()

print(f"Data shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=200, 
    random_state=42, 
    class_weight="balanced"
)
model.fit(X_train_scaled, y_train)

print("Evaluating...")
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\n=== RESULTS ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {auc:.4f}")

print("\nSaving models...")
joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(feature_names, "feature_names.joblib")
print("Model saved to model.joblib")
print("Scaler saved to scaler.joblib")
print("Feature names saved to feature_names.joblib")
