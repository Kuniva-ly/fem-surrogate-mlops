@echo off
set PYTHON=.venv\Scripts\python
set CLI=%PYTHON% -m src.cli

:: Charger SLACK_WEBHOOK_URL depuis .env
for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
    if "%%a"=="SLACK_WEBHOOK_URL" set SLACK_WEBHOOK_URL=%%b
)

if "%1"=="test" (
    %PYTHON% -m pytest tests/ -v
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
    if %errorlevel%==0 (
        if defined SLACK_WEBHOOK_URL if not "%SLACK_WEBHOOK_URL%"=="" (
            powershell -Command "Invoke-RestMethod -Uri '%SLACK_WEBHOOK_URL%' -Method Post -ContentType 'application/json' -Body '{\"text\":\"✅ *Training termine* — modele enregistre dans MLflow http://localhost:5000\"}'" >nul 2>&1
        )
    ) else (
        if defined SLACK_WEBHOOK_URL if not "%SLACK_WEBHOOK_URL%"=="" (
            powershell -Command "Invoke-RestMethod -Uri '%SLACK_WEBHOOK_URL%' -Method Post -ContentType 'application/json' -Body '{\"text\":\"❌ *Training echoue* — verifier les logs\"}'" >nul 2>&1
        )
    )
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
) else if "%1"=="up" (
    docker compose up -d
) else if "%1"=="up-build" (
    docker compose up -d --build
) else if "%1"=="down" (
    docker compose down
) else if "%1"=="reset" (
    echo ATTENTION: supprime tous les volumes ^(MLflow, MinIO, Postgres^) !
    choice /c ON /m "Continuer"
    if errorlevel 2 (
        echo Annule.
    ) else (
        docker compose down -v
        echo Volumes supprimes.
    )
) else if "%1"=="logs" (
    docker compose logs -f
) else if "%1"=="monitor" (
    echo Surveillance API http://localhost:8000/health ^(Ctrl+C pour arreter^)
    powershell -Command "$webhook='%SLACK_WEBHOOK_URL%'; $wasDown=$false; while($true) { try { $r=Invoke-RestMethod -Uri 'http://localhost:8000/health' -TimeoutSec 5; if($wasDown) { if($webhook) { Invoke-RestMethod -Uri $webhook -Method Post -ContentType 'application/json' -Body '{\"text\":\"✅ *API revenue en ligne*\"}' } }; $wasDown=$false; Write-Host \"[$(Get-Date -f HH:mm:ss)] OK - modele=$($r.model_loaded)\" } catch { if(-not $wasDown) { Write-Host \"[$(Get-Date -f HH:mm:ss)] DOWN\"; if($webhook) { Invoke-RestMethod -Uri $webhook -Method Post -ContentType 'application/json' -Body '{\"text\":\"🚨 *API DOWN* — http://localhost:8000 ne repond plus\"}' } }; $wasDown=$true }; Start-Sleep 30 }"
) else if "%1"=="api" (
    %PYTHON% -m uvicorn src.api.main:app --reload --port 8000
) else if "%1"=="dashboard" (
    %PYTHON% -m streamlit run src/dashboard/app.py --server.port 8501
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
    echo Usage: .\run ^<commande^>
    echo.
    echo  Pipeline ML (local^):
    echo   run build-features     ^-^- feature engineering
    echo   run train              ^-^- entrainer le modele ^(avec MLflow^)
    echo   run evaluate           ^-^- evaluer le modele
    echo   run verify-artifacts   ^-^- verifier les checksums
    echo.
    echo  Stack Docker:
    echo   run up-build           ^-^- build + demarrer tous les services
    echo   run up                 ^-^- demarrer ^(images deja buildees^)
    echo   run down               ^-^- arreter la stack ^(volumes preserves^)
  echo   run reset              ^-^- arreter + supprimer tous les volumes ^(DESTRUCTIF^)
    echo   run logs               ^-^- suivre les logs en direct
    echo.
    echo  Dev local ^(sans Docker^):
    echo   run monitor            ^-^- surveiller l'API et notifier Slack si down
  echo   run api                ^-^- FastAPI sur :8000
    echo   run dashboard          ^-^- Streamlit sur :8501
    echo.
    echo  Upload donnees vers MinIO:
    echo   run upload-all         ^-^- tout uploader
    echo   run upload-artifacts   ^-^- artifacts seuls
    echo   run upload-raw         ^-^- donnees brutes seules
)
