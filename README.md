# Surrogate Stress Platform (MVP)

This repository contains a first MVP for Week 1:
- Parametric FEM simulation batch generation (FEniCS/dolfinx)
- Local parquet dataset generation
- Data quality validation
- One-command daily batch pipeline
- Three geometry variants: `with_hole`, `without_hole`, and `with_hole_moving`

## Quick start

1. Start infra:
`docker compose up -d`
Spark UI is available at `http://localhost:8080` (worker UI: `http://localhost:8081`).
MLflow UI is available at `http://localhost:5000` (start only MLflow with `docker compose up -d mlflow`).

2. Generate sample data:
`docker compose exec fenics python fenics_projet/traction_plate_with_hole.py --n 50 --seed 42 --out data/raw/sim_v1 --backend fenics`

Generate sample data (without hole):
`python fenics_projet/traction_plate_without_hole.py --n 50 --seed 42 --out data/raw/sim_v1_without_hole --backend proxy`

Generate sample data (with moving hole):
`docker compose exec fenics python fenics_projet/traction_plate_moving_hole.py --n 50 --seed 42 --out data/raw/sim_v2_moving_hole --backend fenics --data-version sim_v2_moving_hole`

Note: output is parquet only. Install `pyarrow` (recommended) or `fastparquet`.

3. Validate generated data:
`python -m src.processing.validate --input data/raw/sim_v1`
`python -m src.processing.validate --input data/raw/sim_v1_without_hole`
`python -m src.processing.validate --input data/raw/sim_v2_moving_hole`

4. Run daily batch (generate + validate):
`python -m src.pipelines.daily_batch --n 50 --seed 42 --base data/raw/sim_v1`

5. Build ML-ready dataset (processed splits + feature artifacts):
`python -m src.processing.build_features --input data/raw/sim_v1/date=YYYY-MM-DD --out-dir data/processed/date=YYYY-MM-DD --features-out-dir data/features/stress_model/v1/with_hole/date=YYYY-MM-DD`

6. Upload daily partitions to MinIO:
`python -m src.ingestion.upload_to_minio --local-path data/raw/sim_v1/date=YYYY-MM-DD --bucket raw-simulations --prefix with_hole/sim_v1/date=YYYY-MM-DD`
`python -m src.ingestion.upload_to_minio --local-path data/raw/sim_v1_without_hole/date=YYYY-MM-DD --bucket raw-simulations --prefix without_hole/sim_v1_without_hole/date=YYYY-MM-DD`
`python -m src.ingestion.upload_to_minio --local-path data/raw/sim_v2_moving_hole/date=YYYY-MM-DD --bucket raw-simulations --prefix with_hole_moving/sim_v2_moving_hole/date=YYYY-MM-DD`

Upload processed splits to MinIO:
`python -m src.ingestion.upload_to_minio --local-path data/processed/date=YYYY-MM-DD --bucket processed-simulations --prefix with_hole/date=YYYY-MM-DD`

Upload feature-store artifacts to MinIO:
`python -m src.ingestion.upload_to_minio --local-path data/features/stress_model/v1/with_hole/date=YYYY-MM-DD --bucket features --prefix stress_model/v1/with_hole/date=YYYY-MM-DD`

7. Run daily batch with direct MinIO upload:
`python -m src.pipelines.daily_batch --n 50 --seed 42 --base data/raw/sim_v1 --backend proxy --chunk-size 5000 --upload-minio --feature-group stress_model --feature-version v1`

8. Load parquet data into PostgreSQL:
`python -m src.ingestion.load_to_postgres --input data/raw/sim_v1/date=YYYY-MM-DD --geometry-type with_hole`
`python -m src.ingestion.load_to_postgres --input data/raw/sim_v1_without_hole/date=YYYY-MM-DD --geometry-type without_hole`
Requires: `pip install psycopg2-binary`

## Dataset schema
See `docs/data_contract.md`.

```powershell
python fenics_projet/traction_plate_with_hole_wide.py --n 10000 --backend fenics --mesh-nx 120 --mesh-ny 24
python fenics_projet/traction_plate_without_hole_wide.py --n 10000 --backend fenics --mesh-nx 120 --mesh-ny 24
python fenics_projet/traction_plate_moving_hole_wide.py --n 10000 --backend fenics --mesh-nx 120 --mesh-ny 24

```
