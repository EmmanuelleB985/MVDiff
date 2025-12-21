# MVDiff Development Makefile
.PHONY: help install install-dev test test-gpu lint format clean docker run demo benchmark docs

PYTHON := python3
PIP := $(PYTHON) -m pip
PROJECT := mvdiff
DOCKER_IMAGE := mvdiff:latest

# Default target
help:
	@echo "MVDiff Development Commands"
	@echo "═══════════════════════════════════════"
	@echo "Setup:"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make format        - Format code with black & isort"
	@echo "  make lint          - Run linting checks"
	@echo "  make test          - Run unit tests"
	@echo "  make test-gpu      - Run GPU tests"
	@echo "  make coverage      - Generate test coverage report"
	@echo ""
	@echo "Running:"
	@echo "  make demo          - Run quick demo"
	@echo "  make app           - Launch Gradio web interface"
	@echo "  make train         - Start training"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"
	@echo "  make docker-push   - Push to registry"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs          - Build documentation"
	@echo "  make docs-serve    - Serve docs locally"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make benchmark     - Run performance benchmarks"
	@echo "  make check-all     - Run all checks"

# Installation
install:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -e ".[dev]"
	$(PIP) install pytest pytest-cov pytest-xdist pytest-benchmark
	$(PIP) install black flake8 mypy isort pylint
	$(PIP) install sphinx sphinx-rtd-theme
	pre-commit install

# Code Quality
format:
	@echo "Formatting code..."
	black .
	isort .
	@echo "✓ Code formatted"

lint:
	@echo "Running linters..."
	black --check .
	isort --check-only .
	flake8 . --config=.flake8
	mypy --config-file=mypy.ini .
	pylint $(PROJECT)
	@echo "✓ All linting checks passed"

# Testing
test:
	@echo "Running tests..."
	pytest tests/ -v --tb=short

test-gpu:
	@echo "Running GPU tests..."
	pytest tests/ -v -m gpu --tb=short

test-integration:
	@echo "Running integration tests..."
	pytest tests/integration/ -v --tb=short

coverage:
	@echo "Generating coverage report..."
	pytest tests/ --cov=$(PROJECT) --cov-report=html --cov-report=term
	@echo "✓ Coverage report generated in htmlcov/"

test-all: lint test test-integration
	@echo "✓ All tests passed"

# Running
demo:
	$(PYTHON) quick_start.py

app:
	$(PYTHON) app.py --port 7860

train:
	$(PYTHON) -m $(PROJECT).train --config configs/train/shapenet_base.yaml

evaluate:
	$(PYTHON) -m $(PROJECT).evaluate --checkpoint checkpoints/best_model.pth

# Docker
docker-build:
	docker build -t $(DOCKER_IMAGE) .
	@echo "✓ Docker image built: $(DOCKER_IMAGE)"

docker-build-dev:
	docker build --target dev -t $(DOCKER_IMAGE)-dev .
	@echo "✓ Development Docker image built"

docker-run:
	docker run --gpus all -it -p 7860:7860 -v $(PWD):/app $(DOCKER_IMAGE)

docker-run-dev:
	docker run --gpus all -it -p 7860:7860 -p 8888:8888 -v $(PWD):/app $(DOCKER_IMAGE)-dev

docker-push:
	docker tag $(DOCKER_IMAGE) mvdiff/mvdiff:latest
	docker push mvdiff/mvdiff:latest

# Documentation
docs:
	cd docs && make clean && make html
	@echo "✓ Documentation built in docs/_build/html/"

docs-serve:
	cd docs/_build/html && $(PYTHON) -m http.server 8000

# Benchmarking
benchmark:
	@echo "Running benchmarks..."
	$(PYTHON) scripts/benchmark.py --output benchmark-results.json
	@echo "✓ Benchmark results saved"

profile:
	@echo "Profiling model..."
	$(PYTHON) scripts/profile_model.py --checkpoint checkpoints/best_model.pth

# Maintenance
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/
	rm -rf __pycache__ **/__pycache__ **/**/__pycache__
	rm -rf .pytest_cache .coverage htmlcov/
	rm -rf .mypy_cache .ruff_cache
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	@echo "✓ Clean complete"

clean-models:
	@echo "Cleaning model checkpoints..."
	rm -rf checkpoints/*.pth
	rm -rf experiments/*/checkpoints/
	@echo "✓ Model cleanup complete"

# Release
version:
	@$(PYTHON) -c "import $(PROJECT); print($(PROJECT).__version__)"

release-test:
	$(PYTHON) setup.py sdist bdist_wheel
	twine check dist/*
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

release:
	$(PYTHON) setup.py sdist bdist_wheel
	twine upload dist/*

# Combined Commands
check-all: format lint test coverage
	@echo "✓ All checks passed successfully!"

setup: install-dev
	@echo "✓ Development environment ready!"

ci: lint test coverage
	@echo "✓ CI checks passed!"

# Development Workflow Helpers
dev-server:
	@echo "Starting development server with hot reload..."
	watchmedo auto-restart -d $(PROJECT) -p '*.py' -- $(PYTHON) app.py --debug

notebook:
	jupyter lab --port=8888 --no-browser

tensorboard:
	tensorboard --logdir=runs/

# Model Management
download-models:
	$(PYTHON) scripts/download_models.py --all

upload-model:
	$(PYTHON) scripts/upload_to_hub.py --checkpoint $(CHECKPOINT) --repo $(REPO)

convert-model:
	$(PYTHON) scripts/convert_model.py --input $(INPUT) --output $(OUTPUT)

# Performance Optimization
optimize:
	$(PYTHON) scripts/optimize_model.py --checkpoint checkpoints/best_model.pth

quantize:
	$(PYTHON) scripts/quantize_model.py --checkpoint checkpoints/best_model.pth --bits 8

# Dependency Management
deps-update:
	$(PIP) list --outdated
	$(PIP) install --upgrade -r requirements.txt

deps-freeze:
	$(PIP) freeze > requirements-lock.txt

deps-tree:
	pipdeptree

# GPU Management
gpu-check:
	$(PYTHON) -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}')"

gpu-monitor:
	watch -n 1 nvidia-smi

# Database/Cache
cache-clear:
	rm -rf ~/.cache/torch
	rm -rf ~/.cache/huggingface
	@echo "✓ Cache cleared"

# Security
security-check:
	bandit -r $(PROJECT) -f json -o security-report.json
	safety check --json > safety-report.json
	@echo "✓ Security check complete"

# Metrics and Monitoring
metrics:
	$(PYTHON) scripts/compute_metrics.py --results experiments/latest/

visualize:
	$(PYTHON) scripts/visualize_results.py --experiment experiments/latest/

# Help for specific targets
help-test:
	@echo "Testing Commands:"
	@echo "  make test          - Run all unit tests"
	@echo "  make test-gpu      - Run GPU-specific tests"
	@echo "  make test TEST=... - Run specific test file"
	@echo "  make coverage      - Generate coverage report"

.DEFAULT_GOAL := help
