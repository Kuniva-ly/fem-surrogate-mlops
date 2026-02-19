.PHONY: venv lint test up down generate generate-without-hole generate-with-moving-hole validate validate-all build-features batch batch-upload load-postgres

venv:
	python -m venv .venv
	.venv\Scripts\python -m pip install --upgrade pip

lint:
	python -m py_compile fenics_projet/traction_plate_with_hole.py fenics_projet/traction_plate_without_hole.py fenics_projet/traction_plate_moving_hole.py src/processing/validate.py src/processing/build_features.py src/pipelines/daily_batch.py src/ingestion/load_to_postgres.py src/ingestion/upload_to_minio.py src/training/train_baseline.py src/inference/predict_baseline.py

test:
	python -m unittest discover -s tests -p "test_*.py"

up:
	docker compose up -d

down:
	docker compose down

generate:
	docker compose exec fenics python fenics_projet/traction_plate_with_hole.py --n 10 --seed 42 --out data/raw/sim_v1 --backend fenics

generate-without-hole:
	python fenics_projet/traction_plate_without_hole.py --n 10 --seed 42 --out data/raw/sim_v1_without_hole --backend proxy

generate-with-moving-hole:
	docker compose exec fenics python fenics_projet/traction_plate_moving_hole.py --n 10 --seed 42 --out data/raw/sim_v2_moving_hole --backend fenics --data-version sim_v2_moving_hole

validate:
	python -m src.processing.validate --input data/raw/sim_v1

validate-all:
	python -m src.processing.validate --input data/raw/sim_v1
	python -m src.processing.validate --input data/raw/sim_v1_without_hole
	python -m src.processing.validate --input data/raw/sim_v2_moving_hole

build-features:
	python -m src.processing.build_features --input data/raw/sim_v1 --out-dir data/processed

batch:
	python -m src.pipelines.daily_batch --n 10 --seed 42 --base data/raw/sim_v1

batch-upload:
	python -m src.pipelines.daily_batch --n 10 --seed 42 --base data/raw/sim_v1 --backend proxy --chunk-size 5000 --upload-minio

load-postgres:
	python -m src.ingestion.load_to_postgres --input data/raw/sim_v1 --geometry-type with_hole
