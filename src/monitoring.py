"""
Data drift monitoring and model performance tracking.
"""
import json
import logging
import numpy as np
from scipy.stats import ks_2samp
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from src.config import settings

logger = logging.getLogger(__name__)

REFERENCE_DISTRIBUTIONS_PATH = settings.REFERENCE_DISTRIBUTIONS_PATH

def compute_reference_distributions(
    X_train: np.ndarray, 
    feature_names: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Compute reference distributions from training data.
    
    Args:
        X_train: Training data matrix (n_samples, n_features)
        feature_names: List of feature names
        
    Returns:
        Dictionary with distribution statistics for each feature
    """
    reference = {}
    
    for i, col in enumerate(feature_names):
        feature_data = X_train[:, i]
        
        reference[col] = {
            "mean": float(np.mean(feature_data)),
            "std": float(np.std(feature_data)),
            "min": float(np.min(feature_data)),
            "max": float(np.max(feature_data)),
            "median": float(np.median(feature_data)),
            "q1": float(np.percentile(feature_data, 25)),
            "q3": float(np.percentile(feature_data, 75)),
            "iqr": float(np.percentile(feature_data, 75) - np.percentile(feature_data, 25)),
            "skewness": float(np.mean((feature_data - np.mean(feature_data)) ** 3) / (np.std(feature_data) ** 3 + 1e-8)),
            "kurtosis": float(np.mean((feature_data - np.mean(feature_data)) ** 4) / (np.var(feature_data) ** 2 + 1e-8)),
            "percentiles": {
                "1": float(np.percentile(feature_data, 1)),
                "5": float(np.percentile(feature_data, 5)),
                "10": float(np.percentile(feature_data, 10)),
                "25": float(np.percentile(feature_data, 25)),
                "50": float(np.percentile(feature_data, 50)),
                "75": float(np.percentile(feature_data, 75)),
                "90": float(np.percentile(feature_data, 90)),
                "95": float(np.percentile(feature_data, 95)),
                "99": float(np.percentile(feature_data, 99))
            }
        }
    
    REFERENCE_DISTRIBUTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REFERENCE_DISTRIBUTIONS_PATH, "w") as f:
        json.dump(reference, f, indent=2)
    
    logger.info(f"Reference distributions saved to {REFERENCE_DISTRIBUTIONS_PATH}")
    logger.info(f"Computed distributions for {len(reference)} features")
    
    return reference

def load_reference_distributions() -> Optional[Dict[str, Dict[str, Any]]]:
    """Load reference distributions from file"""
    if not REFERENCE_DISTRIBUTIONS_PATH.exists():
        logger.warning(f"Reference distributions not found at {REFERENCE_DISTRIBUTIONS_PATH}")
        return None
    
    with open(REFERENCE_DISTRIBUTIONS_PATH, "r") as f:
        return json.load(f)

def detect_drift(
    X_batch: np.ndarray,
    feature_names: List[str],
    threshold_ks: float = 0.05,
    threshold_psi: float = 0.1
) -> Dict[str, Any]:
    """
    Detect data drift in a batch of new samples.
    
    Args:
        X_batch: Batch of new samples (n_samples, n_features)
        feature_names: List of feature names
        threshold_ks: Kolmogorov-Smirnov test p-value threshold
        threshold_psi: Population Stability Index threshold
        
    Returns:
        Dictionary with drift detection results
    """
    reference = load_reference_distributions()
    
    if reference is None:
        return {"drift_detected": False, "error": "No reference distribution available"}
    
    drift_report = {}
    drift_count = 0
    psi_values = []
    
    for i, col in enumerate(feature_names):
        if col not in reference:
            continue
            
        current_data = X_batch[:, i]
        ref_data_sample = np.random.normal(
            reference[col]["mean"],
            reference[col]["std"],
            min(len(current_data), 1000)
        )
        
        ks_stat, ks_p_value = ks_2samp(current_data, ref_data_sample)
        
        reference_bins = np.histogram(ref_data_sample, bins=10)[0] / len(ref_data_sample)
        current_bins = np.histogram(current_data, bins=10)[0] / len(current_data)
        psi = np.sum((current_bins - reference_bins) * np.log((current_bins + 1e-8) / (reference_bins + 1e-8)))
        
        drift_detected_ks = ks_p_value < threshold_ks
        drift_detected_psi = abs(psi) > threshold_psi
        drift_detected = drift_detected_ks or drift_detected_psi
        
        if drift_detected:
            drift_count += 1
            
        drift_report[col] = {
            "ks_statistic": float(ks_stat),
            "ks_p_value": float(ks_p_value),
            "psi": float(psi),
            "drift_detected_ks": drift_detected_ks,
            "drift_detected_psi": drift_detected_psi,
            "drift_detected": drift_detected,
            "current_mean": float(np.mean(current_data)),
            "current_std": float(np.std(current_data)),
            "reference_mean": reference[col]["mean"],
            "reference_std": reference[col]["std"],
            "mean_shift_percent": float((np.mean(current_data) - reference[col]["mean"]) / reference[col]["std"] * 100)
        }
        
        psi_values.append(abs(psi))
    
    drift_percentage = (drift_count / len(feature_names)) * 100 if feature_names else 0
    
    drift_report["summary"] = {
        "total_features": len(feature_names),
        "features_with_drift": drift_count,
        "drift_percentage": round(drift_percentage, 2),
        "timestamp": datetime.now().isoformat(),
        "mean_psi": round(float(np.mean(psi_values)), 4),
        "max_psi": round(float(np.max(psi_values)), 4),
        "threshold_ks": threshold_ks,
        "threshold_psi": threshold_psi,
        "recommendation": self._get_drift_recommendation(drift_percentage)
    }
    
    return drift_report

def _get_drift_recommendation(drift_percentage: float) -> str:
    """Generate recommendation based on drift severity"""
    if drift_percentage < 5:
        return "No action needed. Model performance likely stable."
    elif drift_percentage < 15:
        return "Monitor model performance closely. Consider retraining within 3 months."
    elif drift_percentage < 30:
        return "Retraining recommended within 4-6 weeks. Review data pipeline."
    else:
        return "URGENT: Significant drift detected. Retrain model immediately."

class DriftMonitor:
    """Class for continuous drift monitoring"""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.drift_history = []
        
    def track_drift(self, drift_report: Dict[str, Any]):
        """Track drift over time"""
        self.drift_history.append({
            "timestamp": datetime.now().isoformat(),
            "drift_percentage": drift_report.get("summary", {}).get("drift_percentage", 0),
            "features_with_drift": drift_report.get("summary", {}).get("features_with_drift", 0),
            "mean_psi": drift_report.get("summary", {}).get("mean_psi", 0)
        })
        
        if len(self.drift_history) > self.history_size:
            self.drift_history = self.drift_history[-self.history_size:]
    
    def get_trend(self) -> Dict[str, Any]:
        """Analyze drift trend over time"""
        if len(self.drift_history) < 3:
            return {"trend": "insufficient_data"}
        
        recent_drifts = [d["drift_percentage"] for d in self.drift_history[-10:]]
        older_drifts = [d["drift_percentage"] for d in self.drift_history[-20:-10]] if len(self.drift_history) >= 20 else recent_drifts
        
        recent_avg = np.mean(recent_drifts)
        older_avg = np.mean(older_drifts) if older_drifts else recent_avg
        
        trend_direction = "increasing" if recent_avg > older_avg + 2 else "decreasing" if recent_avg < older_avg - 2 else "stable"
        
        return {
            "trend": trend_direction,
            "recent_avg_drift": round(float(recent_avg), 2),
            "older_avg_drift": round(float(older_avg), 2),
            "change_percent": round(float((recent_avg - older_avg) / (older_avg + 1e-8) * 100), 2),
            "samples_count": len(self.drift_history)
        }

drift_monitor = DriftMonitor()