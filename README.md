# FEM Surrogate Platform

[![CI](https://github.com/Kuniva-ly/fem-surrogate-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Kuniva-ly/fem-surrogate-mlops/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![LightGBM](https://img.shields.io/badge/model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow%202.12-blue.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/docker-compose-2496ED.svg?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Plateforme ML de substitution pour simuler la réponse structurelle de plaques en traction.
Prédit le **déplacement maximal** et la **contrainte de von Mises maximale** à partir de la géométrie et du chargement — en millisecondes au lieu de plusieurs minutes de calcul FEM.

Géométries supportées : `with_hole` · `without_hole` · `with_hole_moving`

---

## Architecture

```
data/raw/            →  build-features  →  data/processed/{train,val,test}.parquet
                                        →  data/features/advanced/features.parquet
data/processed/      →  train           →  artifacts/models/lgbm_surrogate/<version>/
artifacts/models/    →  evaluate        →  eval_metrics.json + eval_metrics.csv
                     →  predict         →  JSON { predictions: {...} }
                     →  verify-artifacts→  SHA-256 integrity verified
```

All commands read `configs/training.yaml` for their parameters.

---

## Prerequisites

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install pandas pyarrow lightgbm scikit-learn optuna joblib pyyaml numpy
```

---

## Quick start — full pipeline

### 1. (Optional) Generate FEM data

```powershell
# With Docker + FEniCS
docker compose up -d
docker compose exec fenics python -m src.simulations.traction_plate_with_hole `
    --n 5000 --seed 42 --out data/raw/sim_v1 --backend fenics

# Without Docker — fast analytical proxy
.venv\Scripts\python -m src.simulations.traction_plate_without_hole `
    --n 1000 --seed 42 --out data/raw/sim_v1_without_hole --backend proxy
```

### 2. Validate raw data

```powershell
.venv\Scripts\python -m src.processing.validate --input data/raw/sim_v1
```

### 3. Feature engineering + splits

```powershell
.venv\Scripts\python -m src.cli build-features --input data/raw
```

Reads `configs/training.yaml` for seed, ratios, and split strategy.
Produces `data/processed/{train,val,test}.parquet` and `data/features/advanced/`.

### 4. Train the advanced model

```powershell
.venv\Scripts\python -m src.cli train
```

- Separate LightGBM per target (displacement + von Mises)
- Targets in log10 to normalise the 4–6 decades range
- Optuna search (60 trials by default, configurable)
- Physics monotonicity constraints
- Saves to the registry `artifacts/models/lgbm_surrogate/<version>/`
- Generates `manifest.json` (timestamp, git commit, dataset fingerprint)
- Generates `checksums.sha256` + `checksums.json`

### 5. Evaluate

```powershell
.venv\Scripts\python -m src.cli evaluate
```

Produces `eval_metrics.json` and `eval_metrics.csv` in the version directory.

### 6. Predict a single case

```powershell
.venv\Scripts\python -m src.cli predict `
  --case-json '{
    "material_category":"steel","dimension_category":"medium",
    "length_m":1.2,"height_m":0.3,
    "young_modulus_pa":2.1e11,"poisson_ratio":0.3,"traction_pa":1500000,
    "mesh_nx":120,"mesh_ny":24,
    "geometry_type":"with_hole","hole_radius_ratio":0.1
  }'
```

Derived features are computed automatically if absent from the input.

### 7. Verify artifact integrity

```powershell
.venv\Scripts\python -m src.cli verify-artifacts
```

Verifies the SHA-256 checksums of all files in the latest registered version.

---

## Makefile shortcuts

```powershell
make build-features    # feature engineering
make train             # full training
make evaluate          # evaluation
make predict           # inference (example case)
make verify-artifacts  # SHA-256 integrity check
make test              # 61 unit tests
make lint              # syntax check
```

---

## Configuration

All runtime parameters are in `configs/training.yaml`:

```yaml
features:
  seed: 42
  train_ratio: 0.70
  val_ratio: 0.15
  split_strategy: hash   # hash (deterministic) | random

training:
  random_state: 42
  cv_folds: 5
  n_trials: 60           # Optuna trials per target (0 = disable)

artifacts:
  registry_dir: artifacts/models
  model_name: lgbm_surrogate
```

Override via CLI (`--n-trials 0`, `--random-state 99`) or via the `CONFIG_PATH` environment variable.

---

## Model registry

Each training run creates a timestamped version:

```
artifacts/models/lgbm_surrogate/
    v20260310_143022/
        lgbm_max_displacement_m.joblib
        lgbm_max_von_mises_pa.joblib
        advanced_metrics.csv
        advanced_metrics.json     ← nested metrics by target/split
        manifest.json             ← timestamp, git commit, dataset SHA-256, package versions
        checksums.sha256
        checksums.json
    latest.txt                    ← points to the latest version
```

---

## Tests

```powershell
.venv\Scripts\python -m unittest discover -s tests -p "test_*.py" -v
# Ran 61 tests — OK
```

Coverage:
- `test_config.py` — YAML loading and validation
- `test_features.py` — physics formulas, splits, schema
- `test_registry.py` — versioning, copy, model roundtrip
- `test_integrity.py` — SHA-256, tamper detection
- `test_smoke.py` — full E2E pipeline + reproducibility

---

## Inference with the existing model

To use the already-trained models in `data/models/advanced/`:

```powershell
.venv\Scripts\python -m src.cli predict `
  --model-dir data/models/advanced `
  --case-json '{...}'
```

---

## MLflow (optional)

```powershell
$env:MLFLOW_S3_ENDPOINT_URL = "http://localhost:9000"
$env:AWS_ACCESS_KEY_ID      = "minioadmin"
$env:AWS_SECRET_ACCESS_KEY  = "minioadmin"
$env:AWS_DEFAULT_REGION     = "us-east-1"

.venv\Scripts\python -m src.training.train_advanced `
    --data-dir data/processed --out-dir data/models/advanced `
    --n-trials 60 --mlflow --mlflow-run-name lgbm-v2
```

---

## Data contract

See [docs/data_contract.md](docs/data_contract.md).

## Scripts overview

See [docs/scripts_overview.md](docs/scripts_overview.md).
