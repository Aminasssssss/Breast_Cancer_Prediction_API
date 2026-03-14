"""
Batch prediction pipeline for breast cancer classification.
Processes unprocessed input data from database and stores predictions.
"""
import sys
import os
import sqlite3
import hashlib
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.monitoring import detect_drift
from src.alerts import alert_manager

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchPredictor:
    def __init__(
        self,
        db_path: str = None,
        model_path: str = None,
        scaler_path: str = None,
        features_path: str = None
    ):
        self.db_path = db_path or str(settings.DB_PATH)
        self.model_path = model_path or str(settings.MODEL_PATH)
        self.scaler_path = scaler_path or str(settings.SCALER_PATH)
        self.features_path = features_path or str(settings.FEATURES_PATH)
        
        self._load_artifacts()
        self._ensure_database_schema()
        
    def _load_artifacts(self):
        """Load model, scaler and feature names"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.feature_names = joblib.load(self.features_path)
            logger.info(f"Loaded model from {self.model_path}")
            logger.info(f"Loaded {len(self.feature_names)} features")
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            raise
        
    def _ensure_database_schema(self):
        """Ensure database has required tables and columns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("PRAGMA table_info(predictions)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if "row_hash" not in columns:
            logger.info("Adding row_hash column to predictions table")
            cursor.execute("ALTER TABLE predictions ADD COLUMN row_hash TEXT UNIQUE")
            conn.commit()
        
        if "processing_timestamp" not in columns:
            logger.info("Adding processing_timestamp column to predictions table")
            cursor.execute("ALTER TABLE predictions ADD COLUMN processing_timestamp TEXT")
            conn.commit()
        
        if "drift_report" not in columns:
            logger.info("Adding drift_report column to predictions table")
            cursor.execute("ALTER TABLE predictions ADD COLUMN drift_report TEXT")
            conn.commit()
        
        conn.close()
    
    @staticmethod
    def generate_row_hash(row_values: Tuple) -> str:
        """Generate unique hash for a row of input data"""
        row_str = ",".join([str(x) for x in row_values])
        return hashlib.sha256(row_str.encode()).hexdigest()
    
    def get_unprocessed_rows(self, limit: int = None) -> List[Tuple]:
        """Get rows that haven't been processed yet"""
        limit = limit or settings.BATCH_LIMIT
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cols_sql = ", ".join([f'i."{f}"' for f in self.feature_names])
            cursor.execute(f"""
                SELECT i.id, i.row_hash, {cols_sql}
                FROM input_data i
                LEFT JOIN predictions p ON i.row_hash = p.row_hash
                WHERE p.row_hash IS NULL
                LIMIT {limit}
            """)
            rows = cursor.fetchall()
            return rows
        finally:
            conn.close()
    
    def process_row_batch(self, rows: List[Tuple]) -> List[Dict[str, Any]]:
        """Process a batch of rows and return predictions"""
        if not rows:
            return []
        
        ids = []
        hashes = []
        X_rows = []
        
        for row in rows:
            row_id = row[0]
            row_hash = row[1] if row[1] else self.generate_row_hash(row[2:])
            ids.append(row_id)
            hashes.append(row_hash)
            X_rows.append(row[2:])
        
        X = np.array(X_rows, dtype=float)
        X_scaled = self.scaler.transform(X)
        
        drift_report = None
        if settings.ENABLE_DRIFT_MONITORING:
            drift_report = detect_drift(
                X_scaled, 
                self.feature_names, 
                settings.DRIFT_THRESHOLD_KS
            )
            if drift_report and drift_report.get("summary", {}).get("drift_percentage", 0) > settings.DRIFT_ALERT_THRESHOLD_PERCENT:
                logger.warning(f"Data drift detected: {drift_report['summary']['drift_percentage']:.1f}%")
                alert_manager.send(
                    "Batch Data Drift Detected",
                    f"Drift percentage: {drift_report['summary']['drift_percentage']:.1f}%\n"
                    f"Features with drift: {drift_report['summary']['features_with_drift']}",
                    severity="HIGH"
                )
        
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        results = []
        for i in range(len(ids)):
            pred = int(predictions[i])
            diagnosis = "benign" if pred == 1 else "malignant"
            prob_benign = float(probabilities[i][1])
            prob_malignant = float(probabilities[i][0])
            
            result = {
                "id": ids[i],
                "row_hash": hashes[i],
                "prediction": pred,
                "diagnosis": diagnosis,
                "probability_benign": round(prob_benign, 4),
                "probability_malignant": round(prob_malignant, 4),
                "prediction_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "processing_timestamp": datetime.now().isoformat(),
            }
            
            results.append(result)
        
        return results, drift_report
    
    def save_predictions(self, results: List[Dict[str, Any]], drift_report: Optional[Dict] = None) -> int:
        """Save predictions to database"""
        if not results:
            return 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        saved_count = 0
        for result in results:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO predictions
                    (id, row_hash, prediction, diagnosis, probability_benign, 
                     probability_malignant, prediction_timestamp, processing_timestamp, drift_report)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result["id"],
                    result["row_hash"],
                    result["prediction"],
                    result["diagnosis"],
                    result["probability_benign"],
                    result["probability_malignant"],
                    result["prediction_timestamp"],
                    result["processing_timestamp"],
                    json.dumps(drift_report) if drift_report else None
                ))
                saved_count += cursor.rowcount
            except sqlite3.IntegrityError:
                logger.warning(f"Duplicate row hash: {result['row_hash']}")
                continue
        
        conn.commit()
        conn.close()
        
        return saved_count
    
    def run_batch(self, limit: int = None) -> Dict[str, Any]:
        """Run complete batch processing pipeline"""
        start_time = datetime.now()
        
        try:
            rows = self.get_unprocessed_rows(limit)
            
            if not rows:
                logger.info("No unprocessed rows found")
                return {"processed": 0, "skipped": 0, "error": None}
            
            logger.info(f"Found {len(rows)} unprocessed rows")
            
            results, drift_report = self.process_row_batch(rows)
            saved = self.save_predictions(results, drift_report)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return {
                "processed": saved,
                "skipped": len(rows) - saved,
                "duration_seconds": round(duration, 2),
                "drift_detected": drift_report.get("summary", {}).get("drift_percentage", 0) > settings.DRIFT_ALERT_THRESHOLD_PERCENT if drift_report else None,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}", exc_info=True)
            alert_manager.send(
                "Batch Processing Failed",
                f"Error: {str(e)}\nTimestamp: {datetime.now().isoformat()}",
                severity="CRITICAL"
            )
            return {"processed": 0, "skipped": 0, "error": str(e)}

def initialize_database_from_csv(csv_path: str = None):
    """Initialize database from CSV file or create sample data"""
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
    else:
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df = df.head(100)
    
    conn = sqlite3.connect(settings.DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("DROP TABLE IF EXISTS input_data")
    cursor.execute("DROP TABLE IF EXISTS predictions")
    
    feature_cols_sql = ",\n    ".join([f'"{f}" REAL' for f in df.columns])
    cursor.execute(f"""
        CREATE TABLE input_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            row_hash TEXT UNIQUE,
            {feature_cols_sql}
        )
    """)
    
    cursor.execute("""
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY,
            row_hash TEXT UNIQUE,
            prediction INTEGER,
            diagnosis TEXT,
            probability_benign REAL,
            probability_malignant REAL,
            prediction_timestamp TEXT,
            processing_timestamp TEXT,
            drift_report TEXT,
            FOREIGN KEY (id) REFERENCES input_data(id)
        )
    """)
    
    cols = ", ".join([f'"{c}"' for c in df.columns])
    placeholders = ", ".join(["?" for _ in df.columns])
    
    for _, row in df.iterrows():
        row_hash = BatchPredictor.generate_row_hash(tuple(row.values))
        cursor.execute(
            f'INSERT INTO input_data (row_hash, {cols}) VALUES (?, {placeholders})',
            (row_hash,) + tuple(row.values)
        )
    
    conn.commit()
    conn.close()
    
    logger.info(f"Database initialized with {len(df)} rows")

def main():
    """Main entry point for batch processing"""
    predictor = BatchPredictor()
    result = predictor.run_batch()
    
    if result["error"]:
        logger.error(f"Batch processing failed: {result['error']}")
        sys.exit(1)
    
    logger.info(f"Batch processing completed: {result['processed']} processed, {result['skipped']} skipped, {result['duration_seconds']}s")
    
    if result.get("drift_detected"):
        logger.warning("Data drift detected during batch processing")
    
    return result

if __name__ == "__main__":
    main()