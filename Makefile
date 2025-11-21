# Makefile for Legal AI System

.PHONY: help install test train clean docs

help:
	@echo "Legal AI System - Available Commands:"
	@echo ""
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run all tests"
	@echo "  make test-cov      - Run tests with coverage"
	@echo "  make train-all     - Train all models"
	@echo "  make train-mamba   - Train Mamba model"
	@echo "  make train-transfer- Train Transfer model"
	@echo "  make train-rag     - Setup RAG system"
	@echo "  make train-rl      - Train RL agent"
	@echo "  make clean         - Clean generated files"
	@echo "  make format        - Format code with black"
	@echo "  make lint          - Run linting checks"
	@echo "  make quickstart    - Run quick start example"
	@echo ""

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	@echo "Installation complete!"

test:
	@echo "Running tests..."
	pytest tests/ -v

test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "Coverage report saved to htmlcov/index.html"

train-mamba:
	@echo "Training Mamba model..."
	python scripts/train_mamba.py --config configs/mamba_config.yaml

train-transfer:
	@echo "Training Transfer model..."
	python scripts/train_transfer.py --config configs/transfer_config.yaml

train-rag:
	@echo "Setting up RAG system..."
	python scripts/train_rag.py --config configs/rag_config.yaml

train-rl:
	@echo "Training RL agent..."
	python scripts/train_rl.py --config configs/rl_config.yaml

train-all: train-mamba train-transfer train-rag train-rl
	@echo "All models trained!"

clean:
	@echo "Cleaning generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage .pytest_cache/
	@echo "Clean complete!"

format:
	@echo "Formatting code..."
	black src/ tests/ scripts/ examples/
	@echo "Formatting complete!"

lint:
	@echo "Running linting checks..."
	flake8 src/ tests/ scripts/ examples/ --max-line-length=100 --ignore=E203,W503
	@echo "Linting complete!"

quickstart:
	@echo "Running quick start example..."
	python examples/quickstart.py

setup-dev:
	@echo "Setting up development environment..."
	pip install -e ".[dev]"
	pre-commit install
	@echo "Development environment ready!"

tensorboard:
	@echo "Starting TensorBoard..."
	tensorboard --logdir runs/

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker build -t legal-ai-system .

docker-run:
	@echo "Running Docker container..."
	docker run -it --gpus all -v $(PWD):/workspace legal-ai-system

# Data preparation
prepare-data:
	@echo "Preparing sample data..."
	mkdir -p data
	python -c "from src.utils.data_loader import LegalDataLoader; LegalDataLoader.create_sample_data(1000, 5, 'data/sample_data.json')"
	@echo "Sample data created in data/sample_data.json"
