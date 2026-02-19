# A quoi sert chaque script

Le projet gere trois variantes de geometrie:
- `with_hole`
- `without_hole`
- `with_hole_moving`

## Simulation
```bash
docker compose exec fenics python -m pip install pandas pyarrow
```

- `fenics_projet/traction_plate_with_hole.py`
  Genere les donnees de simulation plaque avec trou (FEniCS reel ou proxy), puis ecrit des fichiers parquet par chunks.
- `fenics_projet/traction_plate_without_hole.py`
  Genere les donnees de simulation plaque sans trou (FEniCS ou proxy), avec le meme format de sortie parquet.
- `fenics_projet/traction_plate_moving_hole.py`
  Genere les donnees de simulation plaque avec trou mobile (centre du trou variable), en FEniCS reel ou proxy, avec sorties parquet par chunks.

## Validation qualite
- `src/processing/validate.py`
  Verifie la qualite des datasets (colonnes requises, nulls, doublons, plages de valeurs) avant traitement Spark/ML.

## Feature engineering
- `src/processing/build_features.py`
  Construit un dataset ML pret a l'emploi depuis les parquet raw:
  - features derivees (`area_m2`, `aspect_ratio`, `traction_over_E`, `mesh_density`, etc.)
  - splits `train.parquet`, `val.parquet`, `test.parquet` (zone `processed`)
  - fichier `features.parquet` + `feature_columns.txt` (zone `features`, option `--features-out-dir`)

## Ingestion Data Lake (MinIO)
- `src/ingestion/upload_to_minio.py`
  Upload les fichiers parquet d'un dossier local vers MinIO/S3 (bucket + prefix).
  Prefix recommandes:
  - `with_hole/sim_v1/date=YYYY-MM-DD`
  - `without_hole/sim_v1_without_hole/date=YYYY-MM-DD`
  - `with_hole_moving/sim_v2_moving_hole/date=YYYY-MM-DD`
  - `with_hole/date=YYYY-MM-DD` (processed)
  - `stress_model/v1/with_hole/date=YYYY-MM-DD` (features pseudo feature store)

## Ingestion PostgreSQL
- `src/ingestion/load_to_postgres.py`
  Charge les parquet locaux dans PostgreSQL (`simulation_records`) avec upsert sur `simulation_id`.
  Permet de forcer le routage via `--geometry-type with_hole|without_hole|with_hole_moving`.

## Pipeline batch
- `src/pipelines/daily_batch.py`
  Orchestration locale en une commande: generation -> validation -> build processed/features -> upload MinIO (optionnel), pour le flux `with_hole`.

## Initialisation Python package
- `src/__init__.py`
  Marque `src` comme package Python.
- `src/processing/__init__.py`
  Marque `src.processing` comme package Python.
- `src/ingestion/__init__.py`
  Marque `src.ingestion` comme package Python.
- `src/pipelines/__init__.py`
  Marque `src.pipelines` comme package Python.

## Infra et base de donnees
- `docker-compose.yml`
  Definit les services locaux: MinIO, init MinIO, Postgres, FEniCS, Spark master/worker, MLflow.
- `db/init/001_schema.sql`
  Cree le schema SQL (tables simulations/ML + table dataset `simulation_records`) au demarrage de Postgres.

## Automatisation locale
- `Makefile`
  Raccourcis de commandes (`up`, `generate`, `generate-without-hole`, `validate`, `batch`, etc.).

## Documentation
- `README.md`
  Guide de demarrage et commandes principales.
- `docs/data_contract.md`
  Contrat de donnees (schema et regles de validation).
- `docs/generate_large_dataset.md`
  Commandes pour generer un gros volume de donnees.
- `docs/use_case_et_utilite.md`
  Cas d'usage et utilite du projet.
- `roadmap.md`
  Plan d'execution du projet.
- `projet_file_rouge.md`
  Document de cadrage et vision globale du projet.

## Commandes training MLflow
```bash
.\.venv\Scripts\pip.exe install boto3
$env:MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"
$env:AWS_ACCESS_KEY_ID="minioadmin"
$env:AWS_SECRET_ACCESS_KEY="minioadmin"
$env:AWS_DEFAULT_REGION="us-east-1"
.\.venv\Scripts\python.exe src\training\train_baseline.py --mlflow --mlflow-run-name rf-baseline
```
