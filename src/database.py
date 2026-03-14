"""
Database management for predictions and input data.
"""
import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

import pandas as pd
import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(settings.DB_PATH)
        self._ensure_db_directory()
        
    def _ensure_db_directory(self):
        """Ensure database directory exists"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def initialize_tables(self):
        """Create all necessary tables if they don't exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS input_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    row_hash TEXT UNIQUE,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    processed INTEGER DEFAULT 0,
                    features TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_id INTEGER,
                    row_hash TEXT UNIQUE,
                    prediction INTEGER,
                    diagnosis TEXT,
                    probability_benign REAL,
                    probability_malignant REAL,
                    prediction_timestamp TEXT,
                    processing_timestamp TEXT,
                    drift_report TEXT,
                    latency_ms REAL,
                    model_version TEXT,
                    FOREIGN KEY (input_id) REFERENCES input_data(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    tags TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drift_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    drift_percentage REAL,
                    features_with_drift INTEGER,
                    total_features INTEGER,
                    drift_report TEXT
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_input_hash ON input_data(row_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_input_processed ON input_data(processed)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_input ON predictions(input_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_prediction_hash ON predictions(row_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics_history(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_drift_timestamp ON drift_history(timestamp)")
            
            conn.commit()
            logger.info("Database tables initialized")
    
    def insert_input_data(self, features: List[float], row_hash: str = None) -> int:
        """Insert new input data for processing"""
        if row_hash is None:
            row_hash = self._generate_hash(features)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT id FROM input_data WHERE row_hash = ?", (row_hash,))
            existing = cursor.fetchone()
            
            if existing:
                return existing["id"]
            
            features_json = json.dumps(features)
            cursor.execute("""
                INSERT INTO input_data (row_hash, features, processed)
                VALUES (?, ?, 0)
            """, (row_hash, features_json))
            
            conn.commit()
            return cursor.lastrowid
    
    def insert_batch_input(self, features_batch: List[List[float]]) -> List[int]:
        """Insert multiple input rows"""
        ids = []
        for features in features_batch:
            row_id = self.insert_input_data(features)
            ids.append(row_id)
        return ids
    
    def get_unprocessed_inputs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get unprocessed inputs"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, row_hash, features, created_at
                FROM input_data
                WHERE processed = 0
                ORDER BY id ASC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            result = []
            for row in rows:
                result.append({
                    "id": row["id"],
                    "row_hash": row["row_hash"],
                    "features": json.loads(row["features"]),
                    "created_at": row["created_at"]
                })
            return result
    
    def save_prediction(self, prediction_data: Dict[str, Any]) -> int:
        """Save prediction result"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if "input_id" in prediction_data:
                cursor.execute("""
                    UPDATE input_data 
                    SET processed = 1 
                    WHERE id = ?
                """, (prediction_data["input_id"],))
            
            cursor.execute("""
                INSERT OR REPLACE INTO predictions (
                    input_id, row_hash, prediction, diagnosis,
                    probability_benign, probability_malignant,
                    prediction_timestamp, processing_timestamp,
                    drift_report, latency_ms, model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction_data.get("input_id"),
                prediction_data.get("row_hash"),
                prediction_data.get("prediction"),
                prediction_data.get("diagnosis"),
                prediction_data.get("probability_benign"),
                prediction_data.get("probability_malignant"),
                prediction_data.get("prediction_timestamp"),
                prediction_data.get("processing_timestamp"),
                prediction_data.get("drift_report"),
                prediction_data.get("latency_ms"),
                prediction_data.get("model_version")
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def log_metric(self, metric_name: str, metric_value: float, tags: Dict[str, str] = None):
        """Log a metric to history"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO metrics_history (timestamp, metric_name, metric_value, tags)
                VALUES (?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                metric_name,
                metric_value,
                json.dumps(tags) if tags else None
            ))
            conn.commit()
    
    def log_drift(self, drift_report: Dict[str, Any]):
        """Log drift detection results"""
        summary = drift_report.get("summary", {})
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO drift_history (
                    timestamp, drift_percentage, features_with_drift,
                    total_features, drift_report
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                summary.get("drift_percentage", 0),
                summary.get("features_with_drift", 0),
                summary.get("total_features", 0),
                json.dumps(drift_report)
            ))
            conn.commit()
    
    def get_prediction_stats(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get prediction statistics for recent period"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN diagnosis = 'benign' THEN 1 ELSE 0 END) as benign_count,
                    SUM(CASE WHEN diagnosis = 'malignant' THEN 1 ELSE 0 END) as malignant_count,
                    AVG(probability_benign) as avg_benign_prob,
                    AVG(probability_malignant) as avg_malignant_prob,
                    AVG(latency_ms) as avg_latency_ms,
                    MAX(latency_ms) as max_latency_ms
                FROM predictions
                WHERE datetime(prediction_timestamp) >= datetime('now', '-' || ? || ' hours')
            """, (hours_back,))
            
            row = cursor.fetchone()
            
            return {
                "total_predictions": row["total"] or 0,
                "benign_predictions": row["benign_count"] or 0,
                "malignant_predictions": row["malignant_count"] or 0,
                "avg_benign_probability": round(row["avg_benign_prob"] or 0, 4),
                "avg_malignant_probability": round(row["avg_malignant_prob"] or 0, 4),
                "avg_latency_ms": round(row["avg_latency_ms"] or 0, 2),
                "max_latency_ms": round(row["max_latency_ms"] or 0, 2)
            }
    
    def get_drift_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent drift history"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, drift_percentage, features_with_drift, drift_report
                FROM drift_history
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            return [
                {
                    "timestamp": row["timestamp"],
                    "drift_percentage": row["drift_percentage"],
                    "features_with_drift": row["features_with_drift"],
                    "drift_report": json.loads(row["drift_report"]) if row["drift_report"] else None
                }
                for row in rows
            ]
    
    def export_predictions(self, output_path: str, format: str = "csv"):
        """Export predictions to file"""
        with self.get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
            
            if format == "csv":
                df.to_csv(output_path, index=False)
            elif format == "json":
                df.to_json(output_path, orient="records", indent=2)
            elif format == "parquet":
                df.to_parquet(output_path, index=False)
            
            logger.info(f"Exported {len(df)} predictions to {output_path}")
            
    @staticmethod
    def _generate_hash(features: List[float]) -> str:
        """Generate unique hash for feature vector"""
        import hashlib
        feature_str = ",".join([str(x) for x in features])
        return hashlib.sha256(feature_str.encode()).hexdigest()


def initialize_database():
    """Initialize database from sklearn dataset"""
    from sklearn.datasets import load_breast_cancer
    
    db = DatabaseManager()
    db.initialize_tables()
    
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    
    for _, row in df.head(100).iterrows():
        features = row.tolist()
        db.insert_input_data(features)
    
    logger.info(f"Database initialized with {min(100, len(df))} input samples")
    return db


db_manager = DatabaseManager()