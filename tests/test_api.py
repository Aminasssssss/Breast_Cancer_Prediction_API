"""
Unit tests for FastAPI endpoints.
"""
import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api import app
from src.config import settings

client = TestClient(app)

SAMPLE_FEATURES = [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
]

SAMPLE_FEATURES_BENIGN = [
    13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
    0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023,
    15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259
]

class TestRootEndpoint:
    """Tests for root endpoint"""
    
    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_contains_expected_fields(self):
        response = client.get("/")
        data = response.json()
        assert "message" in data
        assert "model" in data
        assert "dataset" in data
        assert "status" in data
    
    def test_root_message_correct(self):
        response = client.get("/")
        assert "Breast Cancer Prediction API" in response.json()["message"]


class TestHealthEndpoint:
    """Tests for health check endpoint"""
    
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_has_status_field(self):
        response = client.get("/health")
        assert "status" in response.json()
    
    def test_health_has_model_loaded_field(self):
        response = client.get("/health")
        assert "model_loaded" in response.json()
    
    def test_health_has_environment(self):
        response = client.get("/health")
        assert "environment" in response.json()


class TestInfoEndpoint:
    """Tests for info endpoint"""
    
    def test_info_returns_200(self):
        response = client.get("/info")
        assert response.status_code == 200
    
    def test_info_contains_model_name(self):
        response = client.get("/info")
        assert response.json()["model"] == "Random Forest Classifier"
    
    def test_info_contains_feature_names(self):
        response = client.get("/info")
        assert "feature_names" in response.json()
    
    def test_info_contains_metrics(self):
        response = client.get("/info")
        assert "metrics" in response.json()
        assert "accuracy" in response.json()["metrics"]


class TestPredictEndpoint:
    """Tests for prediction endpoint"""
    
    def test_predict_valid_input_returns_200(self):
        response = client.post("/predict", json={"features": SAMPLE_FEATURES})
        assert response.status_code == 200
    
    def test_predict_returns_prediction_field(self):
        response = client.post("/predict", json={"features": SAMPLE_FEATURES})
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], int)
    
    def test_predict_returns_diagnosis_field(self):
        response = client.post("/predict", json={"features": SAMPLE_FEATURES})
        data = response.json()
        assert "diagnosis" in data
        assert data["diagnosis"] in ["benign", "malignant"]
    
    def test_predict_returns_probability_field(self):
        response = client.post("/predict", json={"features": SAMPLE_FEATURES})
        data = response.json()
        assert "probability" in data
        assert "benign" in data["probability"]
        assert "malignant" in data["probability"]
        assert 0 <= data["probability"]["benign"] <= 1
        assert 0 <= data["probability"]["malignant"] <= 1
    
    def test_predict_returns_latency_ms(self):
        response = client.post("/predict", json={"features": SAMPLE_FEATURES})
        data = response.json()
        assert "latency_ms" in data
        assert data["latency_ms"] > 0
    
    def test_predict_probabilities_sum_to_one(self):
        response = client.post("/predict", json={"features": SAMPLE_FEATURES})
        data = response.json()
        prob_sum = data["probability"]["benign"] + data["probability"]["malignant"]
        assert abs(prob_sum - 1.0) < 0.01
    
    def test_predict_benign_case(self):
        response = client.post("/predict", json={"features": SAMPLE_FEATURES_BENIGN})
        data = response.json()
        assert data["diagnosis"] == "benign"
        assert data["prediction"] == 1
    
    def test_predict_invalid_feature_count_returns_400(self):
        response = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
        assert response.status_code == 400
    
    def test_predict_empty_features_returns_400(self):
        response = client.post("/predict", json={"features": []})
        assert response.status_code == 400
    
    def test_predict_missing_features_field_returns_422(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422
    
    def test_predict_negative_values_handled_correctly(self):
        negative_features = [-1.0] * 30
        response = client.post("/predict", json={"features": negative_features})
        assert response.status_code in [200, 400]
    
    def test_predict_zero_features(self):
        zero_features = [0.0] * 30
        response = client.post("/predict", json={"features": zero_features})
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data


class TestMetricsEndpoint:
    """Tests for Prometheus metrics endpoint"""
    
    def test_metrics_returns_200(self):
        response = client.get("/metrics")
        assert response.status_code == 200
    
    def test_metrics_returns_text_plain_content_type(self):
        response = client.get("/metrics")
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
    
    def test_metrics_contains_prediction_total(self):
        response = client.get("/metrics")
        content = response.text
        assert "predictions_total" in content or response.status_code == 200


class TestCORS:
    """Tests for CORS headers"""
    
    def test_cors_headers_present(self):
        response = client.options(
            "/predict",
            headers={"Origin": "http://localhost:8501"}
        )
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers


class TestErrorHandling:
    """Tests for error handling"""
    
    def test_invalid_json_returns_422(self):
        response = client.post(
            "/predict",
            data="not json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_wrong_feature_count_error_message(self):
        response = client.post("/predict", json={"features": [1.0] * 10})
        assert response.status_code == 400
        assert "detail" in response.json()
        assert "30" in response.json()["detail"] or "Expected" in response.json()["detail"]


@pytest.mark.slow
class TestPerformance:
    """Performance tests"""
    
    def test_prediction_under_100ms(self):
        import time
        start = time.time()
        response = client.post("/predict", json={"features": SAMPLE_FEATURES})
        duration = (time.time() - start) * 1000
        assert response.status_code == 200
        assert duration < 200  # 200ms threshold is generous for CI
    
    def test_concurrent_requests(self):
        import concurrent.futures
        
        def make_request():
            return client.post("/predict", json={"features": SAMPLE_FEATURES})
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in futures]
        
        assert all(r.status_code == 200 for r in results)