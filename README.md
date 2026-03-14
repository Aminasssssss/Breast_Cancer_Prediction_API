# Breast Cancer Prediction API

Production-grade machine learning system for breast cancer prediction with FastAPI, MLflow, Docker, and batch inference.

## Features

- **FastAPI** — REST API with automatic documentation
- **Random Forest Classifier** — 95.83% F1 score, 94.74% accuracy
- **MLflow** — Experiment tracking and model registry
- **Docker** — Containerized deployment
- **Batch Inference** — Scheduled predictions with SQLite
- **Streamlit** — Interactive web dashboard
- **Monitoring** — Data drift detection and Prometheus metrics
- **CI/CD** — GitHub Actions pipeline with tests

## Tech Stack

| Component | Technology |
|-----------|------------|
| API | FastAPI + Uvicorn |
| ML | Scikit-learn, Random Forest |
| Tracking | MLflow |
| Frontend | Streamlit |
| Database | SQLite |
| Container | Docker, Docker Compose |
| Monitoring | Prometheus |
| Testing | Pytest |
| CI/CD | GitHub Actions |

## Quick Start

### 1. Clone repository
```bash
git clone https://github.com/Aminasssssss/sis3_breast.git
cd sis3_breast
```

### 2. Create virtual environment
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train model
```bash
python train_no_mlflow.py
```

### 5. Start MLflow server
```bash
mlflow server --host 127.0.0.1 --port 5000
```

### 6. Start API server
```bash
uvicorn src.api:app --reload --port 8000
```

### 7. Start Streamlit dashboard
```bash
streamlit run streamlit_app.py
```

### 8. Test prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]}'
```

## Docker

```bash
docker build -t breast-cancer-api .
docker run -p 8000:8000 breast-cancer-api
```

Or with Docker Compose:

```bash
docker-compose up
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Detailed health status |
| GET | `/info` | Model information |
| POST | `/predict` | Make prediction |
| GET | `/docs` | Swagger documentation |

## Project Structure

```
Breast_Cancer/
├── src/
│   ├── api.py          # FastAPI endpoints
│   ├── config.py       # Configuration
│   ├── metrics.py      # Prometheus metrics
│   ├── monitoring.py   # Drift detection
│   └── alerts.py       # Alert system
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   └── test_batch.py
├── .github/workflows/
│   ├── ci.yml          # CI/CD pipeline
│   └── deploy.yml      # Deployment
├── train_no_mlflow.py  # Training script
├── streamlit_app.py    # Web dashboard
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── README.md
```

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 94.74% |
| F1 Score | 95.83% |
| ROC-AUC | 99.37% |


## Author

Amina Zhumatayeva
