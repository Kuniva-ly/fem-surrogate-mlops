@echo off
setlocal enabledelayedexpansion

set PYTHON=.venv\Scripts\python
set CLI=%PYTHON% -m src.cli

:: Charger SLACK_WEBHOOK_URL depuis .env
for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
    if "%%a"=="SLACK_WEBHOOK_URL" set SLACK_WEBHOOK_URL=%%b
)

if "%1"=="" goto :usage
if "%1"=="test"             goto :test
if "%1"=="lint"             goto :lint
if "%1"=="build-features"         goto :build_features
if "%1"=="build-features-minio"   goto :build_features_minio
if "%1"=="train"            goto :train
if "%1"=="evaluate"         goto :evaluate
if "%1"=="verify-artifacts" goto :verify_artifacts
if "%1"=="validate"         goto :validate
if "%1"=="validate-all"     goto :validate_all
if "%1"=="upload-artifacts" goto :upload_artifacts
if "%1"=="upload-raw"       goto :upload_raw
if "%1"=="upload-processed" goto :upload_processed
if "%1"=="upload-features"  goto :upload_features
if "%1"=="upload-all"       goto :upload_all
if "%1"=="start"            goto :start
if "%1"=="spark-ingest"     goto :spark_ingest
if "%1"=="spark-etl"        goto :spark_etl
if "%1"=="up"               goto :up
if "%1"=="up-build"         goto :up_build
if "%1"=="down"             goto :down
if "%1"=="reset"            goto :reset
if "%1"=="logs"             goto :logs
if "%1"=="monitor"          goto :monitor
if "%1"=="api"              goto :api
if "%1"=="dashboard"        goto :dashboard
goto :usage

:: ─────────────────────────────────────────────────────────────────────────────

:test
%PYTHON% -m pytest tests/ -v
goto :eof

:lint
%PYTHON% -m py_compile src/config.py src/registry.py src/evaluation.py src/cli.py src/api/main.py src/api/schemas.py src/api/model_loader.py src/utils/manifest.py src/utils/integrity.py src/processing/build_features.py src/training/train_advanced.py
goto :eof

:start
echo === Demarrage du projet FEM Surrogate ===
echo.
echo [1/5] Build et demarrage des services Docker...
docker compose up -d --build
if errorlevel 1 goto :err
echo.
echo [2/5] Attente de MinIO (buckets)...
%PYTHON% scripts/wait_for_minio.py
if errorlevel 1 goto :err
echo.
echo [3/5] Upload des donnees brutes vers MinIO...
%PYTHON% scripts/upload_raw_to_minio.py
if errorlevel 1 goto :err
echo.
echo [4/5] Pipeline ETL Spark (MinIO raw -^> processed)...
docker compose --profile etl run --rm spark-etl
if errorlevel 1 goto :err
echo.
echo [5/5] Feature engineering depuis MinIO...
%CLI% build-features --use-minio
if errorlevel 1 goto :err
echo.
echo === Projet demarre ===
echo   API       : http://localhost:8000
echo   Dashboard : http://localhost:8501
echo   MLflow    : http://localhost:5000
echo   MinIO     : http://localhost:9001
echo   Grafana   : http://localhost:3000
goto :eof

:err
echo ERREUR — arret du demarrage.
exit /b 1

:spark_ingest
%CLI% spark-ingest
goto :eof

:spark_etl
echo Upload des donnees brutes vers MinIO...
%PYTHON% scripts/upload_raw_to_minio.py
echo Lancement du pipeline ETL Spark dans Docker...
docker compose --profile etl run --rm spark-etl
goto :eof

:build_features
%CLI% build-features --input data/raw
goto :eof

:build_features_minio
%CLI% build-features --use-minio
goto :eof

:train
%CLI% train --mlflow --use-minio
set _TRAIN_ERR=!errorlevel!
if !_TRAIN_ERR!==0 (
    if defined SLACK_WEBHOOK_URL if not "!SLACK_WEBHOOK_URL!"=="" (
        powershell -Command "Invoke-RestMethod -Uri '!SLACK_WEBHOOK_URL!' -Method Post -ContentType 'application/json' -Body '{\"text\":\"Training termine - modele enregistre dans MLflow http://localhost:5000\"}'" >nul 2>&1
    )
) else (
    if defined SLACK_WEBHOOK_URL if not "!SLACK_WEBHOOK_URL!"=="" (
        powershell -Command "Invoke-RestMethod -Uri '!SLACK_WEBHOOK_URL!' -Method Post -ContentType 'application/json' -Body '{\"text\":\"Training echoue - verifier les logs\"}'" >nul 2>&1
    )
)
goto :eof

:evaluate
%CLI% evaluate
goto :eof

:verify_artifacts
%CLI% verify-artifacts
goto :eof

:validate
%PYTHON% -m src.processing.validate --input data/raw/sim_v1
goto :eof

:validate_all
%PYTHON% -m src.processing.validate --input data/raw/sim_v1
%PYTHON% -m src.processing.validate --input data/raw/sim_v1_without_hole
%PYTHON% -m src.processing.validate --input data/raw/sim_v2_moving_hole
goto :eof

:upload_artifacts
echo Upload des artifacts vers MinIO (bucket: ml-artifacts)...
docker run --rm --entrypoint sh --network fem-surrogate_fem-net -v "%cd%/artifacts:/artifacts:ro" minio/mc:latest -c "mc alias set local http://minio:9000 minioadmin minioadmin && mc cp --recursive /artifacts/ local/ml-artifacts/artifacts/"
echo Done.
goto :eof

:upload_raw
echo Upload des donnees brutes vers MinIO (bucket: raw-simulations)...
docker run --rm --entrypoint sh --network fem-surrogate_fem-net -v "%cd%/data/raw:/raw:ro" minio/mc:latest -c "mc alias set local http://minio:9000 minioadmin minioadmin && mc cp --recursive /raw/ local/raw-simulations/"
echo Done.
goto :eof

:upload_processed
echo Upload des donnees traitees vers MinIO (bucket: processed-simulations)...
docker run --rm --entrypoint sh --network fem-surrogate_fem-net -v "%cd%/data/processed:/processed:ro" minio/mc:latest -c "mc alias set local http://minio:9000 minioadmin minioadmin && mc cp --recursive /processed/ local/processed-simulations/"
echo Done.
goto :eof

:upload_features
echo Upload des features vers MinIO (bucket: features)...
docker run --rm --entrypoint sh --network fem-surrogate_fem-net -v "%cd%/data/features:/features:ro" minio/mc:latest -c "mc alias set local http://minio:9000 minioadmin minioadmin && mc cp --recursive /features/ local/features/"
echo Done.
goto :eof

:upload_all
echo === Upload artifacts ===
docker run --rm --entrypoint sh --network fem-surrogate_fem-net -v "%cd%/artifacts:/artifacts:ro" minio/mc:latest -c "mc alias set local http://minio:9000 minioadmin minioadmin && mc cp --recursive /artifacts/ local/ml-artifacts/artifacts/"
echo === Upload raw data ===
docker run --rm --entrypoint sh --network fem-surrogate_fem-net -v "%cd%/data/raw:/raw:ro" minio/mc:latest -c "mc alias set local http://minio:9000 minioadmin minioadmin && mc cp --recursive /raw/ local/raw-simulations/"
echo === Upload processed data ===
docker run --rm --entrypoint sh --network fem-surrogate_fem-net -v "%cd%/data/processed:/processed:ro" minio/mc:latest -c "mc alias set local http://minio:9000 minioadmin minioadmin && mc cp --recursive /processed/ local/processed-simulations/"
echo === Upload features ===
docker run --rm --entrypoint sh --network fem-surrogate_fem-net -v "%cd%/data/features:/features:ro" minio/mc:latest -c "mc alias set local http://minio:9000 minioadmin minioadmin && mc cp --recursive /features/ local/features/"
echo Done.
goto :eof

:up
docker compose up -d
goto :eof

:up_build
docker compose up -d --build
goto :eof

:down
docker compose down
goto :eof

:reset
echo ATTENTION: supprime tous les volumes (MLflow, MinIO, Postgres) !
choice /c ON /m "Continuer"
if errorlevel 2 (
    echo Annule.
) else (
    docker compose down -v
    echo Volumes supprimes.
)
goto :eof

:logs
docker compose logs -f
goto :eof

:monitor
echo Surveillance API http://localhost:8000/health (Ctrl+C pour arreter)
set _WH=!SLACK_WEBHOOK_URL!
powershell -NoProfile -Command ^
  "$wh = $env:SLACK_WEBHOOK_URL; $wasDown = $false;" ^
  "while ($true) {" ^
  "  try {" ^
  "    $r = Invoke-RestMethod -Uri 'http://localhost:8000/health' -TimeoutSec 5;" ^
  "    if ($wasDown -and $wh) { Invoke-RestMethod -Uri $wh -Method Post -ContentType 'application/json' -Body '{\"text\":\"API revenue en ligne\"}' }" ^
  "    $wasDown = $false;" ^
  "    Write-Host \"$([datetime]::Now.ToString('HH:mm:ss')) OK - model=$($r.model_loaded)\"" ^
  "  } catch {" ^
  "    if (-not $wasDown) {" ^
  "      Write-Host \"$([datetime]::Now.ToString('HH:mm:ss')) DOWN\";" ^
  "      if ($wh) { Invoke-RestMethod -Uri $wh -Method Post -ContentType 'application/json' -Body '{\"text\":\"API DOWN - http://localhost:8000 ne repond plus\"}' }" ^
  "    }" ^
  "    $wasDown = $true" ^
  "  };" ^
  "  Start-Sleep 30" ^
  "}"
goto :eof

:api
%PYTHON% -m uvicorn src.api.main:app --reload --port 8000
goto :eof

:dashboard
%PYTHON% -m streamlit run src/dashboard/app.py --server.port 8501
goto :eof

:usage
echo Usage: .\run ^<commande^>
echo.
echo  Demarrage complet (1 commande):
echo   run start              -- build Docker + MinIO + ETL Spark (tout en 1)
echo.
echo  Pipeline Spark / Data Lake:
echo   run spark-ingest       -- ETL Spark local (data/raw -^> data/processed)
echo   run spark-etl          -- upload MinIO + ETL Spark dans Docker
echo.
echo  Pipeline ML (local):
echo   run build-features     -- feature engineering
echo   run train              -- entrainer le modele (avec MLflow)
echo   run evaluate           -- evaluer le modele
echo   run verify-artifacts   -- verifier les checksums
echo.
echo  Stack Docker:
echo   run up-build           -- build + demarrer tous les services
echo   run up                 -- demarrer (images deja buildees)
echo   run down               -- arreter la stack (volumes preserves)
echo   run reset              -- arreter + supprimer tous les volumes (DESTRUCTIF)
echo   run logs               -- suivre les logs en direct
echo.
echo  Dev local (sans Docker):
echo   run monitor            -- surveiller l'API et notifier Slack si down
echo   run api                -- FastAPI sur :8000
echo   run dashboard          -- Streamlit sur :8501
echo.
echo  Upload donnees vers MinIO:
echo   run upload-all         -- tout uploader
echo   run upload-artifacts   -- artifacts seuls
echo   run upload-raw         -- donnees brutes seules
goto :eof
