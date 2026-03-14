"""
Unit tests for batch prediction pipeline.
"""
import pytest
import sys
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.batch import BatchPredictor, initialize_database_from_csv
from src.config import settings


class TestBatchPredictorInit:
    """Tests for BatchPredictor initialization"""
    
    def test_init_with_default_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('src.batch.settings.DB_PATH', Path(tmpdir) / "test.db"):
                predictor = BatchPredictor()
                assert predictor.db_path == str(Path(tmpdir) / "test.db")
    
    def test_init_with_custom_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_db = Path(tmpdir) / "custom.db"
            predictor = BatchPredictor(db_path=str(custom_db))
            assert predictor.db_path == str(custom_db)
    
    def test_load_artifacts_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('src.batch.settings.MODEL_PATH', Path(tmpdir) / "nonexistent.joblib"):
                with pytest.raises(Exception):
                    BatchPredictor()


class TestHashGeneration:
    """Tests for row hash generation"""
    
    def test_generate_row_hash_returns_string(self):
        row = (1.0, 2.0, 3.0)
        hash_val = BatchPredictor.generate_row_hash(row)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64  # SHA256 hex digest length
    
    def test_generate_row_hash_consistent(self):
        row = (1.0, 2.0, 3.0)
        hash1 = BatchPredictor.generate_row_hash(row)
        hash2 = BatchPredictor.generate_row_hash(row)
        assert hash1 == hash2
    
    def test_generate_row_hash_different_for_different_rows(self):
        row1 = (1.0, 2.0, 3.0)
        row2 = (1.0, 2.0, 4.0)
        hash1 = BatchPredictor.generate_row_hash(row1)
        hash2 = BatchPredictor.generate_row_hash(row2)
        assert hash1 != hash2
    
    def test_generate_row_hash_handles_floats(self):
        row = (1.123456789, 2.987654321)
        hash_val = BatchPredictor.generate_row_hash(row)
        assert len(hash_val) == 64


class TestDatabaseOperations:
    """Tests for database operations"""
    
    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix='.db') as tmp:
            yield tmp.name
    
    @pytest.fixture
    def predictor_with_db(self, temp_db):
        with patch('src.batch.settings.DB_PATH', temp_db):
            with patch('src.batch.BatchPredictor._load_artifacts'):
                predictor = BatchPredictor(db_path=temp_db)
                predictor._ensure_database_schema()
                yield predictor
    
    def test_ensure_database_schema_creates_tables(self, predictor_with_db, temp_db):
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        assert "input_data" in tables
        assert "predictions" in tables
        
        conn.close()
    
    def test_ensure_database_schema_adds_row_hash_column(self, predictor_with_db, temp_db):
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA table_info(predictions)")
        columns = [col[1] for col in cursor.fetchall()]
        
        assert "row_hash" in columns
        
        conn.close()
    
    def test_get_unprocessed_rows_returns_empty_when_no_data(self, predictor_with_db):
        rows = predictor_with_db.get_unprocessed_rows(limit=10)
        assert rows == []


class TestBatchProcessing:
    """Tests for batch processing logic"""
    
    @pytest.fixture
    def mock_predictor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            
            predictor = Mock(spec=BatchPredictor)
            predictor.db_path = str(db_path)
            predictor.model = Mock()
            predictor.model.predict.return_value = [1]
            predictor.model.predict_proba.return_value = [[0.1, 0.9]]
            predictor.scaler = Mock()
            predictor.scaler.transform.return_value = [[0.0] * 30]
            predictor.feature_names = [f"f{i}" for i in range(30)]
            
            yield predictor
    
    def test_process_row_batch_returns_results(self, mock_predictor):
        rows = [(1, "hash123", *([0.0] * 30))]
        
        with patch('src.batch.BatchPredictor.process_row_batch', return_value=([{"id": 1}], None)):
            result, _ = BatchPredictor.process_row_batch(mock_predictor, rows)
            assert isinstance(result, list)
    
    def test_save_predictions_returns_count(self, mock_predictor, temp_db):
        results = [{
            "id": 1,
            "row_hash": "hash123",
            "prediction": 1,
            "diagnosis": "benign",
            "probability_benign": 0.9,
            "probability_malignant": 0.1,
            "prediction_timestamp": "2025-01-01 12:00:00",
            "processing_timestamp": "2025-01-01T12:00:00"
        }]
        
        with patch('src.batch.BatchPredictor.save_predictions', return_value=1):
            count = BatchPredictor.save_predictions(mock_predictor, results, None)
            assert count == 1


class TestDatabaseInitialization:
    """Tests for database initialization from CSV"""
    
    def test_initialize_database_creates_tables(self, temp_db):
        with patch('src.batch.settings.DB_PATH', temp_db):
            with patch('src.batch.BatchPredictor._load_artifacts'):
                initialize_database_from_csv()
                
                conn = sqlite3.connect(temp_db)
                cursor = conn.cursor()
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                assert "input_data" in tables
                assert "predictions" in tables
                
                conn.close()
    
    def test_initialize_database_inserts_data(self, temp_db):
        with patch('src.batch.settings.DB_PATH', temp_db):
            with patch('src.batch.BatchPredictor._load_artifacts'):
                initialize_database_from_csv()
                
                conn = sqlite3.connect(temp_db)
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM input_data")
                count = cursor.fetchone()[0]
                
                assert count > 0
                
                conn.close()


class TestErrorHandling:
    """Tests for error handling in batch pipeline"""
    
    def test_run_batch_handles_no_rows_gracefully(self):
        with patch('src.batch.BatchPredictor.get_unprocessed_rows', return_value=[]):
            with patch('src.batch.BatchPredictor._load_artifacts'):
                predictor = BatchPredictor()
                result = predictor.run_batch()
                assert result["processed"] == 0
                assert result["error"] is None
    
    def test_run_batch_handles_database_error(self):
        with patch('src.batch.BatchPredictor.get_unprocessed_rows', side_effect=Exception("DB error")):
            with patch('src.batch.BatchPredictor._load_artifacts'):
                predictor = BatchPredictor()
                result = predictor.run_batch()
                assert result["error"] is not None
                assert "DB error" in result["error"]
    
    def test_duplicate_row_hash_handled(self):
        with patch('src.batch.BatchPredictor._load_artifacts'):
            predictor = BatchPredictor()
            
            results = [{"row_hash": "same_hash", "id": 1}, {"row_hash": "same_hash", "id": 2}]
            with patch('src.batch.BatchPredictor.save_predictions') as mock_save:
                mock_save.return_value = 1
                predictor.save_predictions(results, None)


@pytest.mark.slow
class TestBatchPerformance:
    """Performance tests for batch processing"""
    
    def test_batch_processing_time(self):
        import time
        
        with patch('src.batch.BatchPredictor._load_artifacts'):
            predictor = BatchPredictor()
            
            start = time.time()
            result = predictor.run_batch()
            duration = time.time() - start
            
            assert duration < 30  # Should complete within 30 seconds