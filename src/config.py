import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

class Settings:
    APP_NAME = "Breast Cancer Prediction API"
    APP_VERSION = "2.0.0"
    MODEL_PATH = BASE_DIR / "model.joblib"
    SCALER_PATH = BASE_DIR / "scaler.joblib"
    FEATURES_PATH = BASE_DIR / "feature_names.joblib"
    REFERENCE_DISTRIBUTIONS_PATH = BASE_DIR / "reference_distributions.json"
    MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
    BATCH_LIMIT = 100
    BATCH_INTERVAL_MIN = 5
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    N_ESTIMATORS = 200
    CLASS_WEIGHT = "balanced"
    ENABLE_DRIFT_MONITORING = True
    DRIFT_THRESHOLD_KS = 0.05
    DRIFT_ALERT_THRESHOLD_PERCENT = 20.0
    LOG_LEVEL = "INFO"

settings = Settings()
