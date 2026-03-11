@echo off
set PYTHON=.venv\Scripts\python
set CLI=%PYTHON% -m src.cli

if "%1"=="test" (
    %PYTHON% -m unittest discover -s tests -t . -p "test_*.py" -v
) else if "%1"=="lint" (
    %PYTHON% -m py_compile src/config.py src/registry.py src/evaluation.py src/cli.py src/utils/manifest.py src/utils/integrity.py src/processing/build_features.py src/training/train_advanced.py
) else if "%1"=="build-features" (
    %CLI% build-features --input data/raw
) else if "%1"=="train" (
    set MLFLOW_TRACKING_URI=http://localhost:5000
    set AWS_ACCESS_KEY_ID=minioadmin
    set AWS_SECRET_ACCESS_KEY=minioadmin
    set MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
    %CLI% train --mlflow
) else if "%1"=="evaluate" (
    %CLI% evaluate
) else if "%1"=="verify-artifacts" (
    %CLI% verify-artifacts
) else if "%1"=="validate" (
    %PYTHON% -m src.processing.validate --input data/raw/sim_v1
) else if "%1"=="validate-all" (
    %PYTHON% -m src.processing.validate --input data/raw/sim_v1
    %PYTHON% -m src.processing.validate --input data/raw/sim_v1_without_hole
    %PYTHON% -m src.processing.validate --input data/raw/sim_v2_moving_hole
) else if "%1"=="upload-artifacts" (
    echo Upload des artifacts vers MinIO ^(bucket: ml-artifacts^)...
    docker run --rm --entrypoint sh --network projet_fil_rouge_fem-net -v "%cd%/artifacts:/artifacts:ro" minio/mc:latest -c "mc alias set local http://minio:9000 minioadmin minioadmin && mc cp --recursive /artifacts/ local/ml-artifacts/artifacts/"
    echo Done.
) else if "%1"=="upload-raw" (
    echo Upload des donnees brutes vers MinIO ^(bucket: raw-simulations^)...
    docker run --rm --entrypoint sh --network projet_fil_rouge_fem-net -v "%cd%/data/raw:/raw:ro" minio/mc:latest -c "mc alias set local http://minio:9000 minioadmin minioadmin && mc cp --recursive /raw/ local/raw-simulations/"
    echo Done.
) else if "%1"=="upload-processed" (
    echo Upload des donnees traitees vers MinIO ^(bucket: processed-simulations^)...
    docker run --rm --entrypoint sh --network projet_fil_rouge_fem-net -v "%cd%/data/processed:/processed:ro" minio/mc:latest -c "mc alias set local http://minio:9000 minioadmin minioadmin && mc cp --recursive /processed/ local/processed-simulations/"
    echo Done.
) else if "%1"=="upload-features" (
    echo Upload des features vers MinIO ^(bucket: features^)...
    docker run --rm --entrypoint sh --network projet_fil_rouge_fem-net -v "%cd%/data/features:/features:ro" minio/mc:latest -c "mc alias set local http://minio:9000 minioadmin minioadmin && mc cp --recursive /features/ local/features/"
    echo Done.
) else if "%1"=="upload-all" (
    echo === Upload artifacts ===
    docker run --rm --entrypoint sh --network projet_fil_rouge_fem-net -v "%cd%/artifacts:/artifacts:ro" minio/mc:latest -c "mc alias set local http://minio:9000 minioadmin minioadmin && mc cp --recursive /artifacts/ local/ml-artifacts/artifacts/"
    echo === Upload raw data ===
    docker run --rm --entrypoint sh --network projet_fil_rouge_fem-net -v "%cd%/data/raw:/raw:ro" minio/mc:latest -c "mc alias set local http://minio:9000 minioadmin minioadmin && mc cp --recursive /raw/ local/raw-simulations/"
    echo === Upload processed data ===
    docker run --rm --entrypoint sh --network projet_fil_rouge_fem-net -v "%cd%/data/processed:/processed:ro" minio/mc:latest -c "mc alias set local http://minio:9000 minioadmin minioadmin && mc cp --recursive /processed/ local/processed-simulations/"
    echo === Upload features ===
    docker run --rm --entrypoint sh --network projet_fil_rouge_fem-net -v "%cd%/data/features:/features:ro" minio/mc:latest -c "mc alias set local http://minio:9000 minioadmin minioadmin && mc cp --recursive /features/ local/features/"
    echo Done.
) else (
    echo Commandes disponibles: test, lint, build-features, train, evaluate, verify-artifacts, validate, validate-all, upload-artifacts, upload-raw, upload-processed, upload-features, upload-all
)
