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
    %CLI% train
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
) else (
    echo Commandes disponibles: test, lint, build-features, train, evaluate, verify-artifacts, validate, validate-all
)
