.PHONY: help install install-dev test lint format clean run-api run-streamlit run-batch run-mlflow docker-build docker-up docker-down init-db migrate-db train predict health-check

PYTHON := python3
PIP := pip3
PYTEST := pytest
PYTHONPATH := PYTHONPATH=$(shell pwd)

help:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║     Breast Cancer API - Production Makefile                      ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📦 Development:"
	@echo "  make install          - Install production dependencies"
	@echo "  make install-dev      - Install all dependencies including dev tools"
	echo "  make test             - Run all tests with coverage"
	@echo "  make lint             - Run linters (flake8, mypy)"
	@echo "  make format           - Format code with black and isort"
	@echo "  make clean            - Remove cache and temporary files"
	@echo ""
	@echo "🚀 Run services:"
	@echo "  make run-mlflow       - Start MLflow tracking server"
	@echo "  make run-api          - Start FastAPI server (port 8000)"
	@echo "  make run-streamlit    - Start Streamlit dashboard (port 8501)"
	@echo "  make run-batch        - Run batch prediction pipeline once"
	@echo "  make run-batch-daemon - Run batch prediction continuously"
	@echo ""
	@echo "🐳 Docker:"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-up        - Start all services with docker-compose"
	@echo "  make docker-down      - Stop all services"
	@echo "  make docker-logs      - View logs from all services"
	@echo ""
	@echo "🗄️  Database:"
	@echo "  make init-db          - Initialize database schema"
	@echo "  make migrate-db       - Run database migrations"
	@echo "  make reset-db         - Reset database (WARNING: deletes all data)"
	@echo ""
	@echo "🤖 ML:"
	@echo "  make train            - Train model with cross-validation"
	@echo "  make predict          - Make a test prediction"
	@echo "  make monitor          - Run monitoring and drift detection"
	@echo ""
	@echo "🔍 Health:"
	@echo "  make health-check     - Check if API is healthy"
	@echo "  make metrics          - View Prometheus metrics"
	@echo "  make benchmark        - Run performance benchmark"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Production dependencies installed"

install-dev:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov pytest-mock pytest-xdist
	$(PIP) install black isort flake8 mypy pre-commit
	$(PIP) install bandit safety
	$(PIP) install ipython jupyter
	pre-commit install
	@echo "✅ All dependencies installed"

test:
	$(PYTEST) tests/ -v --cov=src/ --cov-report=term-missing --cov-report=html --cov-report=xml -n auto
	@echo "✅ Tests completed"

lint:
	@echo "Running flake8..."
	flake8 src/ tests/ --max-line-length=88 --count --statistics --show-source
	@echo "Running mypy..."
	mypy src/ --ignore-missing-imports --disallow-untyped-defs --warn-return-any --no-implicit-optional
	@echo "Running bandit..."
	bandit -r src/ -ll
	@echo "✅ Linting completed"

format:
	@echo "Formatting with black..."
	black src/ tests/
	@echo "Sorting imports with isort..."
	isort --profile black src/ tests/
	@echo "✅ Formatting completed"

clean:
	@echo "Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/
	@echo "✅ Clean completed"

run-mlflow:
	@echo "Starting MLflow server on http://127.0.0.1:5000"
	mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

run-api:
	@echo "Starting FastAPI server on http://0.0.0.0:8000"
	uvicorn src.api:app --reload --host 0.0.0.0 --port 8000 --log-level info

run-streamlit:
	@echo "Starting Streamlit dashboard on http://localhost:8501"
	streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

run-batch:
	$(PYTHON) -m src.batch

run-batch-daemon:
	@echo "Starting batch prediction daemon (runs every 5 minutes)"
	$(PYTHON) -m src.batch_daemon

docker-build:
	docker build -t breast-cancer-api:latest .
	docker build -f Dockerfile.streamlit -t breast-cancer-streamlit:latest .
	@echo "✅ Docker images built"

docker-up:
	docker-compose up --build -d
	@echo "✅ Services started"
	@echo "  API: http://localhost:8000"
	@echo "  Streamlit: http://localhost:8501"
	@echo "  MLflow: http://localhost:5000"
	@echo "  Metrics: http://localhost:8000/metrics"

docker-down:
	docker-compose down
	@echo "✅ Services stopped"

docker-logs:
	docker-compose logs -f

docker-clean:
	docker-compose down -v
	docker system prune -f
	@echo "✅ Docker cleaned"

init-db:
	$(PYTHON) -m src.database init
	@echo "✅ Database initialized"

migrate-db:
	$(PYTHON) -m src.database migrate
	@echo "✅ Migrations applied"

reset-db:
	@echo "⚠️  WARNING: This will delete all data in the database"
	@read -p "Are you sure? (y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		$(PYTHON) -m src.database reset; \
		echo "✅ Database reset"; \
	else \
		echo "❌ Cancelled"; \
	fi

train:
	@echo "Starting model training..."
	$(PYTHON) train.py
	@echo "✅ Training completed"

predict:
	@echo "Making test prediction..."
	curl -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]}' \
		| python -m json.tool

health-check:
	@echo "Checking API health..."
	curl -f http://localhost:8000/ || echo "❌ API is not running"
	@echo ""
	@echo "Checking MLflow..."
	curl -f http://localhost:5000 || echo "❌ MLflow is not running"

metrics:
	curl -s http://localhost:8000/metrics | head -30

benchmark:
	@echo "Running performance benchmark..."
	@echo "Installing locust..."
	pip install locust
	@echo "Starting locust on http://localhost:8089"
	locust -f tests/locustfile.py --host=http://localhost:8000

monitor:
	$(PYTHON) -m src.monitoring_cli

.PHONY: test-all
test-all: lint test security
	@echo "✅ All checks passed"

security:
	@echo "Running safety check..."
	safety check -r requirements.txt
	@echo "✅ Security check completed"