# Execution Plan (Exact) — Surrogate Modeling Platform (30 Days)

This plan is designed to maximize delivery speed while keeping production-grade quality.
Goal: deliver an end-to-end system (data generation → lake → processing → training → serving → k8s → monitoring) with clean milestones.

Assumptions:
- Local development on a laptop/desktop
- Docker + Kubernetes (Docker Desktop or Minikube/kind)
- MinIO as local S3-compatible Data Lake
- MLflow for experiment tracking and registry
- FastAPI for inference
- Spark OR Ray for processing (choose one; the plan assumes Spark but you can swap)

Key principle:
- Build a minimal working pipeline early, then harden it (tests, validation, monitoring, scale).

Geometry variants convention (applies to all weeks):
- Maintain two explicit dataset tracks: `with_hole` and `without_hole`.
- Keep versioned paths separated in MinIO prefixes to avoid mixing physics regimes.

---

## Week 0 (Day 0) — One-time setup (2–4 hours)

Deliverable: a repo that runs a hello-world stack with `make up`.

1) Install prerequisites
- Python 3.11+
- Docker + docker compose
- kubectl
- kind or minikube
- make (or use a PowerShell equivalent on Windows)

2) Create repository skeleton
- `src/`, `docker/`, `kubernetes/`, `orchestration/`, `mlflow/`, `tests/`, `docs/`, `data/`
- add `.gitignore`, `README.md`, `LICENSE`, `.env.example`

3) Add Makefile targets (minimum)
- `make venv`
- `make lint`
- `make test`
- `make up` (docker compose)
- `make down`
- `make kind-up` (optional)
- `make k8s-apply` (optional)

4) Define coding conventions
- black + ruff + mypy (optional)
- pre-commit hooks (optional but strong)

Acceptance criteria:
- `make up` starts at least one container (even a placeholder)
- repo is clean and readable

---

## Week 1 (Days 1–7) — Data Generation + Data Lake (MVP first)

Goal: generate real simulation-like data and store it in MinIO with versioned folders.
You will not touch Kubernetes yet.

### Day 1 — Data contract and dataset design
Deliverable: a written schema + folder layout

1) Define the “simulation record” contract (minimum fields)
- `simulation_id` (uuid)
- `timestamp`
- geometry params: `length`, `width`, `thickness`, plus optional `hole_radius` (can be null)
- material params: `young_modulus`, `poisson_ratio`, `yield_strength`
- boundary/load params: `force_x`, `force_y`, `fixed_edges` (encoded)
- outputs: `max_von_mises`, `max_displacement`, `safety_margin`
- metadata: `solver_name`, `solver_version`, `mesh_resolution`

2) Define the storage layout in the Data Lake
- `s3://raw-simulations/sim_v1/date=YYYY-MM-DD/`
- `s3://processed-simulations/sim_v1/date=YYYY-MM-DD/`
- `s3://features/sim_v1/date=YYYY-MM-DD/`

3) Define acceptance tests for data
- null checks
- value ranges (positive thickness, realistic modulus)
- uniqueness (simulation_id unique)

Acceptance criteria:
- `docs/data_contract.md` exists
- you can point to exact fields and ranges

---

### Day 2 — MinIO stack in docker compose
Deliverable: MinIO up + bucket creation

1) Implement `docker-compose.yml` with:
- `minio` + `mc` (minio client) init job that creates buckets:
  - `raw-simulations`
  - `processed-simulations`
  - `features`
  - `ml-artifacts`

2) Add `.env` variables:
- `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `MINIO_REGION`

Acceptance criteria:
- MinIO console accessible
- buckets exist after `make up`

---

### Day 3 — Simulation generator (MVP)
Deliverable: generator script producing 1k records (parquet)

1) Create `src/simulations/generate.py`
- generate param sets (geometry/material/load)
- compute outputs with a baseline physics-inspired proxy
  - you can start with a simplified analytical approximation
  - later you can swap to a real FEM solver
- write parquet locally: `data/raw/sim_v1/date=.../part-000.parquet`

2) Add deterministic seeds
- `--seed` parameter

Acceptance criteria:
- `python -m src.simulations.generate --n 1000 --out data/raw/...`
- parquet file produced with correct schema

---

### Day 4 — Upload raw data to MinIO (ingestion)
Deliverable: raw parquet stored in MinIO

1) Create `src/ingestion/upload_to_minio.py`
- upload local parquet to `raw-simulations/...`

2) Add a CLI:
- `python -m src.ingestion.upload_to_minio --local_path ... --s3_path ...`

Acceptance criteria:
- files visible in MinIO bucket under versioned path
- you can download and validate the file

---

### Day 5 — Data validation and quality checks (must-have)
Deliverable: quality checks runnable locally

1) Implement `src/processing/validate.py`
- schema validation
- value ranges
- duplicates
- basic descriptive stats
- fail fast with clear errors

2) Add unit tests in `tests/`
- test schema
- test ranges

Acceptance criteria:
- `make test` runs validation tests
- invalid data fails

---

### Day 6 — Create a “daily batch” pipeline (manual trigger)
Deliverable: a single command to generate+upload+validate

1) Create `src/pipelines/daily_batch.py`
- generate → validate → upload raw
- log outputs

Acceptance criteria:
- one command produces a new daily partition in MinIO

---

### Day 7 — Week 1 review and documentation
Deliverable: docs that show the system runs

1) Update README with:
- how to start MinIO
- how to generate a batch
- where data lives in S3 paths

Acceptance criteria:
- a newcomer can run Week 1 in under 30 minutes

---

## Week 2 (Days 8–14) — Processing + Feature Engineering (Distributed-ready)

Goal: transform raw data to processed + feature datasets. Start with Spark locally in Docker.

### Day 8 — Spark stack in docker compose
Deliverable: spark master + worker + job runner

1) Add services:
- `spark-master`
- `spark-worker`
- optional `spark-jupyter` (only if needed)

2) Ensure spark can read from MinIO (S3A config)
- set Hadoop AWS jars
- configure `fs.s3a.endpoint`, creds, path-style access

Acceptance criteria:
- spark job can list objects in MinIO

---

### Day 9 — Processing job: clean + normalize + enrich
Deliverable: `processed` dataset written to MinIO

1) Implement `src/processing/spark_job.py`
- read raw parquet
- cast types, handle nulls
- normalize numeric fields (optional at this stage)
- write parquet to `processed-simulations/...`

Acceptance criteria:
- processed parquet exists in bucket
- row count matches raw

---

### Day 10 — Feature engineering job
Deliverable: `features` dataset written to MinIO

1) Implement `src/feature_engineering/build_features.py`
- compute derived features:
  - load magnitude
  - aspect ratios
  - stiffness proxies
  - safety-related ratios
- write to `features/...`

Acceptance criteria:
- features table has only model-ready columns + target(s)

---

### Day 11 — Dataset splitting strategy
Deliverable: reproducible train/val/test split

1) Implement split by:
- hash of `simulation_id` (stable)
or
- time-based (if simulating real drift)

Acceptance criteria:
- same record always ends in same split for a given seed/version

---

### Day 12 — Data versioning and lineage
Deliverable: a dataset manifest for each run

1) Create `docs/datasets_manifest.md` and optionally JSON:
- dataset version
- code git commit hash
- counts per split
- min/max ranges

Acceptance criteria:
- you can tell exactly what data produced a model

---

### Day 13 — Week 2 pipeline command
Deliverable: “process raw to features” one command

1) Create:
- `python -m src.pipelines.process_batch --date YYYY-MM-DD`

Acceptance criteria:
- one command generates processed + features in MinIO

---

### Day 14 — Week 2 review
Deliverable: minimal EDA summary in docs
- distribution plots (optional)
- anomalies detected
- feature list and definitions

Acceptance criteria:
- feature list is documented and defensible

---

## Week 3 (Days 15–21) — Model Training + MLflow (Production signals)

Goal: train surrogate model, track experiments, register best model.

### Day 15 — MLflow stack in docker compose
Deliverable: MLflow tracking server + artifact store on MinIO

1) Add:
- `mlflow-server`
- backend store (Postgres or SQLite for MVP)
- artifacts in `ml-artifacts` bucket

Acceptance criteria:
- MLflow UI accessible
- logs show successful connection to artifact store

---

### Day 16 — Baseline training (fast)
Deliverable: baseline model + metrics logged

1) Implement `src/training/train_baseline.py`
- read features from MinIO
- train simple model (LinearRegression / RandomForest)
- log RMSE/MAE/R2
- log model artifact to MLflow

Acceptance criteria:
- one MLflow run with metrics + model artifact exists

---

### Day 17 — Strong model candidate
Deliverable: best candidate model trained and logged

1) Train XGBoost/LightGBM or GradientBoostingRegressor
2) Add hyperparameter config file `configs/train.yaml`

Acceptance criteria:
- MLflow shows improvement vs baseline

---

### Day 18 — Model evaluation report
Deliverable: evaluation artifact saved

1) Save:
- residual plots (optional)
- error by geometry/material bins
- worst-case analysis

Acceptance criteria:
- you can explain where the model fails

---

### Day 19 — Register model in MLflow registry
Deliverable: a versioned model in registry

1) Register best run
2) Tag with dataset version and git commit hash

Acceptance criteria:
- MLflow registry shows model version + tags

---

### Day 20 — Training pipeline command
Deliverable: one command to train end-to-end

1) Create:
- `python -m src.pipelines.train --dataset_version sim_v1 --date YYYY-MM-DD`

Acceptance criteria:
- end-to-end training runnable without manual steps

---

### Day 21 — Week 3 review
Deliverable: README section “Model” updated
- features used
- targets
- metrics
- limitations

Acceptance criteria:
- clear narrative for interviews

---

## Week 4 (Days 22–30) — Serving + Kubernetes + Monitoring (Portfolio-grade)

Goal: deploy the inference API, monitor it, and run the stack in Kubernetes.

### Day 22 — FastAPI inference service (local)
Deliverable: API running locally in Docker

1) Implement `src/api/main.py`
- `/health`
- `/predict`
- `/metrics` (Prometheus)

2) Load model from MLflow registry or from artifact path

Acceptance criteria:
- `curl /health` returns ok
- `/predict` works for a sample payload

---

### Day 23 — Containerize API
Deliverable: production Dockerfile

1) Add `docker/api/Dockerfile`
- slim base
- non-root user
- pinned dependencies
- healthcheck

Acceptance criteria:
- `docker build` succeeds
- container runs and serves endpoints

---

### Day 24 — Kubernetes manifests (API first)
Deliverable: k8s deployment for API + service

1) Create:
- `kubernetes/api-deployment.yaml`
- `kubernetes/api-service.yaml`
- config via ConfigMap/Secret

Acceptance criteria:
- `kubectl apply` deploys
- service accessible via port-forward

---

### Day 25 — Kubernetes for MinIO and MLflow (minimal)
Deliverable: core services running on k8s

1) Deploy MinIO with PV/PVC
2) Deploy MLflow (backend store + artifact store config)

Acceptance criteria:
- MinIO and MLflow reachable in cluster
- API can pull model artifacts

---

### Day 26 — Monitoring stack
Deliverable: Prometheus + Grafana running, scraping API metrics

1) Deploy Prometheus with scrape config
2) Deploy Grafana and import a dashboard:
- request count
- latency
- error rate
- CPU/memory

Acceptance criteria:
- dashboard shows live metrics when calling `/predict`

---

### Day 27 — End-to-end run on Kubernetes
Deliverable: full demo script

1) Provide `docs/demo.md`:
- deploy stack
- generate sample input
- call predict
- show metrics dashboard

Acceptance criteria:
- someone can reproduce the demo

---

### Day 28 — CI/CD (minimal but credible)
Deliverable: GitHub Actions pipeline

1) Add:
- lint + tests
- build docker image
- (optional) push image
- (optional) apply manifests (not required for portfolio)

Acceptance criteria:
- PR triggers pipeline checks

---

### Day 29 — Hardening
Deliverable: quality and reliability upgrades

1) Add:
- request validation (pydantic)
- model input schema enforcement
- structured logging
- rate limiting (optional)
- caching (optional)

Acceptance criteria:
- API rejects invalid payloads cleanly
- logs are readable and useful

---

### Day 30 — Final packaging for portfolio
Deliverable: “portfolio-ready” repo

1) Final README:
- problem
- architecture diagram
- how to run locally
- how to deploy to k8s
- key metrics
- limitations and roadmap

2) Add screenshots (Grafana, MLflow) in `docs/`

Acceptance criteria:
- recruiter can understand the project in 2 minutes and run it in 15–30 minutes

---

# Daily Work Pattern (Recommended)

Each day:
1) Implement the smallest working change
2) Add validation/tests
3) Update docs
4) Commit with clear messages

---

# Minimum Success Criteria (Portfolio)

By Day 30 you must have:
- versioned datasets in MinIO
- processing to features
- MLflow tracked training + registry model
- FastAPI inference in Docker
- Kubernetes deployment
- Prometheus metrics + Grafana dashboard
- clean README and reproducible commands

If you want to extend beyond 30 days:
- real FEM solver integration
- streaming with Kafka
- feature store with Feast
- full IaC with Terraform
