import sqlite3
import numpy as np
import joblib
import schedule
import time
import logging
from datetime import datetime
import pandas as pd 

DB_PATH = "predictions.db"
MODEL_PATH = "model.joblib"
SCALER_PATH = "scaler.joblib"
FEATURES_PATH = "feature_names.joblib"
INTERVAL_MIN = 5
BATCH_LIMIT = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)
log.info("Model loaded")


def run_batch():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cols_sql = ", ".join([f'i."{f}"' for f in feature_names])
        cursor.execute(f"""
            SELECT i.id, {cols_sql}
            FROM input_data i
            LEFT JOIN predictions p ON i.id = p.id
            WHERE p.id IS NULL
            LIMIT {BATCH_LIMIT}
        """)
        rows = cursor.fetchall()

        if not rows:
            log.info("No new rows to process")
            return

        log.info(f"Processing {len(rows)} rows")

        ids = [row[0] for row in rows]
        X = pd.DataFrame([row[1:] for row in rows], columns=feature_names)
        X_scaled = scaler.transform(X)

        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        records = [
            (
                ids[i],
                int(preds[i]),
                "benign" if preds[i] == 1 else "malignant",
                round(float(probs[i][1]), 4),
                round(float(probs[i][0]), 4),
                timestamp,
            )
            for i in range(len(ids))
        ]

        cursor.executemany("""
            INSERT INTO predictions
                (id, prediction, diagnosis, probability_benign, probability_malignant, prediction_timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, records)
        conn.commit()

        log.info(f"Saved {len(records)} predictions")

    except Exception as e:
        log.error(f"Error: {e}")
        conn.rollback()
    finally:
        conn.close()


def show_results():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, diagnosis, probability_benign, probability_malignant, prediction_timestamp
        FROM predictions
        ORDER BY prediction_timestamp DESC
        LIMIT 20
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No predictions yet.")
        return

    print(f"\n{'ID':>4}  {'Diagnosis':<12}  {'P(Benign)':>10}  {'P(Malignant)':>13}  Timestamp")
    print("-" * 70)
    for row in rows:
        print(f"{row[0]:>4}  {row[1]:<12}  {row[2]:>10.4f}  {row[3]:>13.4f}  {row[4]}")
    print()


if __name__ == "__main__":
    log.info(f"Batch pipeline started — every {INTERVAL_MIN} min, batch size {BATCH_LIMIT}")

    run_batch()
    show_results()

    schedule.every(INTERVAL_MIN).minutes.do(run_batch)
    log.info("Scheduler running. Press Ctrl+C to stop.")

    while True:
        schedule.run_pending()
        time.sleep(30)