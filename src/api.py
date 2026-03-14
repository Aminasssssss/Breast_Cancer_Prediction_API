from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

model = joblib.load(BASE_DIR / "model.joblib")
scaler = joblib.load(BASE_DIR / "scaler.joblib")
feature_names = joblib.load(BASE_DIR / "feature_names.joblib")

app = FastAPI(title="Breast Cancer Prediction API")

class PredictionInput(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"message": "API is running", "model": "Random Forest", "status": "healthy"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True, "features_count": len(feature_names)}

@app.get("/info")
def info():
    return {
        "model": "Random Forest Classifier",
        "dataset": "Breast Cancer Wisconsin",
        "task": "Binary Classification",
        "classes": {"0": "malignant", "1": "benign"},
        "n_features": len(feature_names),
        "feature_names": feature_names[:10],
        "metrics": {"accuracy": 0.9474, "f1_score": 0.9583, "roc_auc": 0.9937}
    }

@app.post("/predict")
def predict(data: PredictionInput):
    if len(data.features) != 30:
        raise HTTPException(status_code=400, detail=f"Expected 30 features, got {len(data.features)}")
    
    X = np.array(data.features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    
    return {
        "prediction": int(prediction),
        "diagnosis": "benign" if prediction == 1 else "malignant",
        "probability": {
            "malignant": round(float(probability[0]), 4),
            "benign": round(float(probability[1]), 4)
        }
    }
