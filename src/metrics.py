"""
Prometheus metrics for monitoring model performance and API health.
"""
import time
from typing import Dict, Any, List, Optional
from collections import defaultdict
from datetime import datetime, timedelta

from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from fastapi import Response

PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Total number of predictions made",
    ["diagnosis"]
)

PREDICTION_ERRORS = Counter(
    "prediction_errors_total",
    "Total number of prediction errors",
    ["error_type"]
)

DRIFT_DETECTIONS = Counter(
    "drift_detections_total",
    "Total number of data drift detections",
    ["feature", "severity"]
)

BATCH_PROCESSED_ROWS = Counter(
    "batch_processed_rows_total",
    "Total number of rows processed by batch pipeline"
)

BATCH_PROCESSING_ERRORS = Counter(
    "batch_processing_errors_total",
    "Total number of batch processing errors"
)

MODEL_REQUESTS = Counter(
    "model_requests_total",
    "Total number of requests to model endpoints",
    ["endpoint", "method"]
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0]
)

BATCH_LATENCY = Histogram(
    "batch_latency_seconds",
    "Batch processing latency in seconds",
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0]
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

MODEL_CONFIDENCE = Gauge(
    "model_confidence",
    "Model confidence score for predictions",
    ["diagnosis"]
)

MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether model is loaded (1) or not (0)"
)

DRIFT_PERCENTAGE = Gauge(
    "drift_percentage",
    "Percentage of features with detected drift"
)

LATEST_MODEL_VERSION = Gauge(
    "latest_model_version",
    "Latest model version number"
)

API_HEALTH = Gauge(
    "api_health",
    "API health status (1=healthy, 0=unhealthy)"
)

ACTIVE_REQUESTS = Gauge(
    "active_requests",
    "Number of currently active requests"
)

class MetricsStore:
    def __init__(self, window_minutes: int = 60):
        self.window_minutes = window_minutes
        self.predictions_history: List[Dict] = []
        self.drift_history: List[Dict] = []
        self.error_history: List[Dict] = []
        
    def add_prediction(self, diagnosis: str, confidence: float, latency_ms: float):
        self.predictions_history.append({
            "timestamp": datetime.now(),
            "diagnosis": diagnosis,
            "confidence": confidence,
            "latency_ms": latency_ms
        })
        self._cleanup()
    
    def add_drift(self, drift_percentage: float):
        self.drift_history.append({
            "timestamp": datetime.now(),
            "drift_percentage": drift_percentage
        })
        self._cleanup()
    
    def add_error(self, error_type: str):
        self.error_history.append({
            "timestamp": datetime.now(),
            "error_type": error_type
        })
        self._cleanup()
    
    def _cleanup(self):
        cutoff = datetime.now() - timedelta(minutes=self.window_minutes)
        self.predictions_history = [p for p in self.predictions_history if p["timestamp"] > cutoff]
        self.drift_history = [d for d in self.drift_history if d["timestamp"] > cutoff]
        self.error_history = [e for e in self.error_history if e["timestamp"] > cutoff]
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.predictions_history:
            return {"status": "no_data"}
        
        confidences = [p["confidence"] for p in self.predictions_history]
        latencies = [p["latency_ms"] for p in self.predictions_history]
        
        benign_count = len([p for p in self.predictions_history if p["diagnosis"] == "benign"])
        malignant_count = len([p for p in self.predictions_history if p["diagnosis"] == "malignant"])
        
        return {
            "total_predictions": len(self.predictions_history),
            "benign_predictions": benign_count,
            "malignant_predictions": malignant_count,
            "benign_percentage": round(benign_count / len(self.predictions_history) * 100, 2) if self.predictions_history else 0,
            "average_confidence": round(float(np.mean(confidences)), 4),
            "median_confidence": round(float(np.median(confidences)), 4),
            "average_latency_ms": round(float(np.mean(latencies)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
            "error_count": len(self.error_history),
            "recent_drift": self.drift_history[-1]["drift_percentage"] if self.drift_history else None
        }

metrics_store = MetricsStore()

def track_prediction(diagnosis: str, confidence: float, latency_seconds: float):
    """Track a prediction for metrics"""
    PREDICTIONS_TOTAL.labels(diagnosis=diagnosis).inc()
    MODEL_CONFIDENCE.labels(diagnosis=diagnosis).set(confidence)
    PREDICTION_LATENCY.observe(latency_seconds)
    
    latency_ms = latency_seconds * 1000
    metrics_store.add_prediction(diagnosis, confidence, latency_ms)

def track_error(error_type: str):
    """Track a prediction error"""
    PREDICTION_ERRORS.labels(error_type=error_type).inc()
    metrics_store.add_error(error_type)

def track_drift(feature_name: str, severity: str = "high"):
    """Track a drift detection"""
    DRIFT_DETECTIONS.labels(feature=feature_name, severity=severity).inc()
    
    current_drift_percentage = DRIFT_PERCENTAGE._value.get()
    if current_drift_percentage is None:
        current_drift_percentage = 0
    metrics_store.add_drift(current_drift_percentage)

def track_batch_processed(rows_count: int):
    """Track batch processed rows"""
    BATCH_PROCESSED_ROWS.inc(rows_count)
    BATCH_LATENCY.observe(0)  # placeholder, actual latency should be passed

def track_batch_error():
    """Track batch processing error"""
    BATCH_PROCESSING_ERRORS.inc()

def track_request(endpoint: str, method: str):
    """Track HTTP request"""
    MODEL_REQUESTS.labels(endpoint=endpoint, method=method).inc()

def set_model_loaded(loaded: bool):
    """Set model loaded status"""
    MODEL_LOADED.set(1 if loaded else 0)

def set_drift_percentage(percentage: float):
    """Set current drift percentage"""
    DRIFT_PERCENTAGE.set(percentage)

def set_api_health(healthy: bool):
    """Set API health status"""
    API_HEALTH.set(1 if healthy else 0)

def increment_active_requests():
    """Increment active requests counter"""
    ACTIVE_REQUESTS.inc()

def decrement_active_requests():
    """Decrement active requests counter"""
    ACTIVE_REQUESTS.dec()

def metrics_endpoint():
    """Return Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")

def get_metrics_summary() -> Dict[str, Any]:
    """Get a summary of current metrics for admin endpoints"""
    return {
        "rolling_window": metrics_store.get_summary(),
        "prometheus": {
            "predictions_total": PREDICTIONS_TOTAL._value.get(),
            "prediction_errors_total": PREDICTION_ERRORS._value.get(),
            "drift_detections_total": DRIFT_DETECTIONS._value.get(),
        },
        "timestamp": datetime.now().isoformat()
    }

import numpy as np