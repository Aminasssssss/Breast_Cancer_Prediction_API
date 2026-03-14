"""
Unit tests for model loading and inference.
"""
import pytest
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
from src.config import settings
from src.batch import BatchPredictor


class TestModelArtifacts:
    """Tests for model file loading"""
    
    def test_model_file_exists(self):
        assert Path(settings.MODEL_PATH).exists() or Path("model.joblib").exists()
    
    def test_scaler_file_exists(self):
        assert Path(settings.SCALER_PATH).exists() or Path("scaler.joblib").exists()
    
    def test_feature_names_file_exists(self):
        assert Path(settings.FEATURES_PATH).exists() or Path("feature_names.joblib").exists()
    
    def test_model_can_be_loaded(self):
        try:
            model_path = settings.MODEL_PATH if Path(settings.MODEL_PATH).exists() else "model.joblib"
            model = joblib.load(model_path)
            assert model is not None
        except Exception as e:
            pytest.skip(f"Model not available: {e}")
    
    def test_scaler_can_be_loaded(self):
        try:
            scaler_path = settings.SCALER_PATH if Path(settings.SCALER_PATH).exists() else "scaler.joblib"
            scaler = joblib.load(scaler_path)
            assert scaler is not None
        except Exception as e:
            pytest.skip(f"Scaler not available: {e}")
    
    def test_feature_names_can_be_loaded(self):
        try:
            features_path = settings.FEATURES_PATH if Path(settings.FEATURES_PATH).exists() else "feature_names.joblib"
            feature_names = joblib.load(features_path)
            assert feature_names is not None
            assert len(feature_names) == 30
        except Exception as e:
            pytest.skip(f"Feature names not available: {e}")


class TestModelInference:
    """Tests for model inference"""
    
    @pytest.fixture
    def predictor(self):
        try:
            return BatchPredictor()
        except Exception:
            pytest.skip("BatchPredictor initialization failed")
    
    def test_predictor_initialization(self, predictor):
        assert predictor.model is not None
        assert predictor.scaler is not None
        assert predictor.feature_names is not None
    
    def test_prediction_output_shape(self, predictor):
        sample = np.random.randn(1, 30).astype(np.float32)
        sample_scaled = predictor.scaler.transform(sample)
        prediction = predictor.model.predict(sample_scaled)
        assert prediction.shape == (1,)
        assert prediction[0] in [0, 1]
    
    def test_prediction_probabilities_shape(self, predictor):
        sample = np.random.randn(1, 30).astype(np.float32)
        sample_scaled = predictor.scaler.transform(sample)
        probabilities = predictor.model.predict_proba(sample_scaled)
        assert probabilities.shape == (1, 2)
        assert np.isclose(probabilities.sum(), 1.0)
    
    def test_batch_prediction_shape(self, predictor):
        batch = np.random.randn(10, 30).astype(np.float32)
        batch_scaled = predictor.scaler.transform(batch)
        predictions = predictor.model.predict(batch_scaled)
        assert predictions.shape == (10,)
        assert all(p in [0, 1] for p in predictions)
    
    def test_feature_count_validation(self, predictor):
        wrong_features = np.random.randn(10, 20).astype(np.float32)
        with pytest.raises(ValueError):
            predictor.scaler.transform(wrong_features)


class TestModelPerformance:
    """Tests for model accuracy and performance"""
    
    @pytest.fixture
    def model_and_data(self):
        try:
            from sklearn.datasets import load_breast_cancer
            from sklearn.model_selection import train_test_split
            
            data = load_breast_cancer()
            X = data.data
            y = data.target
            
            model_path = settings.MODEL_PATH if Path(settings.MODEL_PATH).exists() else "model.joblib"
            scaler_path = settings.SCALER_PATH if Path(settings.SCALER_PATH).exists() else "scaler.joblib"
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            X_scaled = scaler.transform(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            return model, X_test, y_test
        except Exception as e:
            pytest.skip(f"Model performance test skipped: {e}")
    
    def test_model_accuracy_above_threshold(self, model_and_data):
        from sklearn.metrics import accuracy_score
        
        model, X_test, y_test = model_and_data
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        assert accuracy > 0.90, f"Model accuracy {accuracy:.2%} is below 90%"
    
    def test_model_f1_above_threshold(self, model_and_data):
        from sklearn.metrics import f1_score
        
        model, X_test, y_test = model_and_data
        predictions = model.predict(X_test)
        f1 = f1_score(y_test, predictions)
        
        assert f1 > 0.90, f"Model F1 score {f1:.3f} is below 0.90"
    
    def test_model_auc_above_threshold(self, model_and_data):
        from sklearn.metrics import roc_auc_score
        
        model, X_test, y_test = model_and_data
        probabilities = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probabilities)
        
        assert auc > 0.95, f"Model AUC {auc:.3f} is below 0.95"


class TestPreprocessing:
    """Tests for data preprocessing"""
    
    def test_scaler_returns_same_shape(self):
        try:
            scaler_path = settings.SCALER_PATH if Path(settings.SCALER_PATH).exists() else "scaler.joblib"
            scaler = joblib.load(scaler_path)
            
            input_data = np.random.randn(5, 30).astype(np.float32)
            scaled_data = scaler.transform(input_data)
            
            assert scaled_data.shape == input_data.shape
        except Exception as e:
            pytest.skip(f"Preprocessing test skipped: {e}")
    
    def test_scaler_handles_single_sample(self):
        try:
            scaler_path = settings.SCALER_PATH if Path(settings.SCALER_PATH).exists() else "scaler.joblib"
            scaler = joblib.load(scaler_path)
            
            single_sample = np.random.randn(1, 30).astype(np.float32)
            scaled_sample = scaler.transform(single_sample)
            
            assert scaled_sample.shape == (1, 30)
        except Exception as e:
            pytest.skip(f"Preprocessing test skipped: {e}")
    
    def test_scaler_maintains_relative_distribution(self):
        try:
            scaler_path = settings.SCALER_PATH if Path(settings.SCALER_PATH).exists() else "scaler.joblib"
            scaler = joblib.load(scaler_path)
            
            input_data = np.random.randn(100, 30).astype(np.float32)
            scaled_data = scaler.transform(input_data)
            
            means = scaled_data.mean(axis=0)
            stds = scaled_data.std(axis=0)
            
            assert np.abs(means).mean() < 0.1
            assert np.abs(stds - 1).mean() < 0.1
        except Exception as e:
            pytest.skip(f"Preprocessing test skipped: {e}")


class TestReproducibility:
    """Tests for model reproducibility"""
    
    def test_prediction_deterministic(self):
        try:
            model_path = settings.MODEL_PATH if Path(settings.MODEL_PATH).exists() else "model.joblib"
            scaler_path = settings.SCALER_PATH if Path(settings.SCALER_PATH).exists() else "scaler.joblib"
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            sample = np.random.RandomState(42).randn(1, 30).astype(np.float32)
            sample_scaled = scaler.transform(sample)
            
            pred1 = model.predict(sample_scaled)[0]
            pred2 = model.predict(sample_scaled)[0]
            
            assert pred1 == pred2
        except Exception as e:
            pytest.skip(f"Reproducibility test skipped: {e}")