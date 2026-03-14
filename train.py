import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from src.config import settings
from src.reproducibility import set_global_seed
from src.monitoring import compute_reference_distributions

set_global_seed(settings.RANDOM_SEED)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "breast-cancer-classification"
MODEL_NAME = "breast-cancer-rf"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_SEED, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

compute_reference_distributions(X_train_scaled, feature_names)

params = {
    "n_estimators": settings.N_ESTIMATORS,
    "max_depth": settings.MAX_DEPTH,
    "random_state": settings.RANDOM_SEED,
    "class_weight": settings.CLASS_WEIGHT,
}

print("\n" + "="*50)
print("BASELINE COMPARISON")
print("="*50)

dummy = DummyClassifier(strategy="most_frequent", random_state=settings.RANDOM_SEED)
dummy.fit(X_train_scaled, y_train)
y_pred_dummy = dummy.predict(X_test_scaled)
dummy_f1 = f1_score(y_test, y_pred_dummy)
print(f"DummyClassifier F1: {dummy_f1:.4f}")

lr = LogisticRegression(max_iter=1000, random_state=settings.RANDOM_SEED)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
lr_f1 = f1_score(y_test, y_pred_lr)
lr_auc = roc_auc_score(y_test, y_prob_lr)
print(f"LogisticRegression F1: {lr_f1:.4f}, AUC: {lr_auc:.4f}")

with mlflow.start_run(run_name="kfold-validation-run"):
    mlflow.log_params(params)
    skf = StratifiedKFold(n_splits=settings.CV_FOLDS, shuffle=True, random_state=settings.RANDOM_SEED)
    
    cv_results = cross_validate(
        RandomForestClassifier(**params), 
        X_train_scaled, 
        y_train, 
        cv=skf, 
        scoring=['accuracy', 'f1', 'roc_auc'],
        return_train_score=False
    )

    mlflow.log_metric("cv_accuracy_mean", round(cv_results['test_accuracy'].mean(), 4))
    mlflow.log_metric("cv_f1_mean", round(cv_results['test_f1'].mean(), 4))
    mlflow.log_metric("cv_roc_auc_mean", round(cv_results['test_roc_auc'].mean(), 4))
    mlflow.log_metric("cv_accuracy_std", round(cv_results['test_accuracy'].std(), 4))

    model = RandomForestClassifier(**params)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    final_metrics = {
        "final_accuracy": accuracy_score(y_test, y_pred),
        "final_f1": f1_score(y_test, y_pred),
        "final_roc_auc": roc_auc_score(y_test, y_prob)
    }
    mlflow.log_metrics({k: round(v, 4) for k, v in final_metrics.items()})

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, settings.MODEL_PATH)
    joblib.dump(scaler, settings.SCALER_PATH)
    joblib.dump(feature_names, settings.FEATURES_PATH)

    mlflow.log_artifact(str(settings.MODEL_PATH))
    mlflow.log_artifact(str(settings.SCALER_PATH))
    mlflow.log_artifact(str(settings.FEATURES_PATH))

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=MODEL_NAME
    )

    print(f"\nRandom Forest Improvement vs Logistic Regression:")
    print(f"  F1 Score: +{(final_metrics['final_f1'] - lr_f1) * 100:.2f}%")
    print(f"  Final Test F1: {final_metrics['final_f1']:.4f}")

try:
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if versions:
        latest_version = max(int(v.version) for v in versions)
        client.set_registered_model_alias(MODEL_NAME, "production", str(latest_version))
        print(f"Version {latest_version} is now @production")
except Exception as e:
    print(f"Registry error: {e}")