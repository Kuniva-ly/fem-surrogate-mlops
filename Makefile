.PHONY: venv install install-api install-dev lint test \
        build-features train train-mlflow evaluate predict verify-artifacts \
        api dashboard \
        up up-build down logs \
        generate generate-without-hole generate-with-moving-hole \
        validate validate-all

# ─── Detect OS (Windows uses different venv activation paths) ─────────────────
ifeq ($(OS),Windows_NT)
  PYTHON := .venv\Scripts\python
  PIP    := .venv\Scripts\pip
else
  PYTHON := .venv/bin/python
  PIP    := .venv/bin/pip
endif

CLI := $(PYTHON) -m src.cli

# ── Environment ────────────────────────────────────────────────────────────────
venv:
	python -m venv .venv
	$(PIP) install --upgrade pip

install:
	$(PIP) install -r requirements.txt

install-api:
	$(PIP) install -r requirements/api.txt

install-dev:
	$(PIP) install -r requirements/dev.txt

# ── Code quality ───────────────────────────────────────────────────────────────
lint:
	$(PYTHON) -m py_compile \
		src/config.py \
		src/registry.py \
		src/evaluation.py \
		src/cli.py \
		src/api/main.py \
		src/api/schemas.py \
		src/api/model_loader.py \
		src/utils/manifest.py \
		src/utils/integrity.py \
		src/simulations/traction_plate_with_hole.py \
		src/simulations/traction_plate_without_hole.py \
		src/simulations/traction_plate_moving_hole.py \
		src/processing/validate.py \
		src/processing/build_features.py \
		src/training/train_advanced.py

test:
	$(PYTHON) -m unittest discover -s tests -t . -p "test_*.py" -v

# ── Production ML pipeline ─────────────────────────────────────────────────────
## Feature engineering + train/val/test splitting
build-features:
	$(CLI) build-features --input data/raw

## Train LightGBM model (reads configs/training.yaml, registers artifact)
train:
	$(CLI) train

## Train with MLflow logging (requires MLflow service running at MLFLOW_TRACKING_URI)
train-mlflow:
	$(CLI) train --mlflow

## Evaluate latest registered model
evaluate:
	$(CLI) evaluate

## Single-case inference (example payload)
predict:
	$(CLI) predict \
		--case-json "{\"simulation_id\":\"00000000-0000-0000-0000-000000000001\",\"timestamp\":\"2026-03-10T00:00:00Z\",\"material_category\":\"steel\",\"dimension_category\":\"medium\",\"length_m\":1.2,\"height_m\":0.3,\"young_modulus_pa\":2.1e11,\"poisson_ratio\":0.3,\"traction_pa\":1500000,\"mesh_nx\":120,\"mesh_ny\":24,\"geometry_type\":\"with_hole\",\"hole_radius_ratio\":0.1}"

## Verify SHA-256 checksums of latest registered artifacts
verify-artifacts:
	$(CLI) verify-artifacts

# ── Local API & dashboard (without Docker) ────────────────────────────────────
api:
	$(PYTHON) -m uvicorn src.api.main:app --reload --port 8000

dashboard:
	$(PYTHON) -m streamlit run src/dashboard/app.py --server.port 8501

# ── Docker Compose stack ───────────────────────────────────────────────────────
up:
	docker compose up -d

up-build:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

# ── FEM data generation (requires Docker / FEniCS) ────────────────────────────
generate:
	docker compose exec fenics $(PYTHON) -m src.simulations.traction_plate_with_hole \
		--n 10 --seed 42 --out data/raw/sim_v1 --backend fenics

generate-without-hole:
	$(PYTHON) -m src.simulations.traction_plate_without_hole \
		--n 10 --seed 42 --out data/raw/sim_v1_without_hole --backend proxy

generate-with-moving-hole:
	docker compose exec fenics $(PYTHON) -m src.simulations.traction_plate_moving_hole \
		--n 10 --seed 42 --out data/raw/sim_v2_moving_hole --backend fenics \
		--data-version sim_v2_moving_hole

# ── Validation ────────────────────────────────────────────────────────────────
validate:
	$(PYTHON) -m src.processing.validate --input data/raw/sim_v1

validate-all:
	$(PYTHON) -m src.processing.validate --input data/raw/sim_v1
	$(PYTHON) -m src.processing.validate --input data/raw/sim_v1_without_hole
	$(PYTHON) -m src.processing.validate --input data/raw/sim_v2_moving_hole
