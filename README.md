# Surrogate Stress Platform

Pipeline ML local pour prédire la contrainte et le déplacement maximal de plaques sous traction,
à partir de simulations FEM (FEniCS) ou d'un proxy analytique rapide.

Géométries supportées : `with_hole` · `without_hole` · `with_hole_moving`

---

## Architecture

```
data/raw/            →  build-features  →  data/processed/{train,val,test}.parquet
                                        →  data/features/advanced/features.parquet
data/processed/      →  train           →  artifacts/models/lgbm_surrogate/<version>/
artifacts/models/    →  evaluate        →  eval_metrics.json + eval_metrics.csv
                     →  predict         →  JSON { predictions: {...} }
                     →  verify-artifacts→  intégrité SHA-256 vérifiée
```

Toutes les commandes lisent `configs/training.yaml` pour leurs paramètres.

---

## Prérequis

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install pandas pyarrow lightgbm scikit-learn optuna joblib pyyaml numpy
```

---

## Quick start — pipeline complet

### 1. (Optionnel) Générer des données FEM

```powershell
# Avec Docker + FEniCS
docker compose up -d
docker compose exec fenics python -m src.simulations.traction_plate_with_hole `
    --n 5000 --seed 42 --out data/raw/sim_v1 --backend fenics

# Sans Docker — proxy analytique rapide
.venv\Scripts\python -m src.simulations.traction_plate_without_hole `
    --n 1000 --seed 42 --out data/raw/sim_v1_without_hole --backend proxy
```

### 2. Valider les données brutes

```powershell
.venv\Scripts\python -m src.processing.validate --input data/raw/sim_v1
```

### 3. Feature engineering + splits

```powershell
.venv\Scripts\python -m src.cli build-features --input data/raw
```

Lit `configs/training.yaml` pour seed, ratios et stratégie de split.
Produit `data/processed/{train,val,test}.parquet` et `data/features/advanced/`.

### 4. Entraîner le modèle avancé

```powershell
.venv\Scripts\python -m src.cli train
```

- LightGBM séparé par cible (displacement + von Mises)
- Targets en log10 pour uniformiser les 4–6 décades
- Recherche Optuna (60 essais par défaut, configurable)
- Contraintes de monotonicité physiques
- Sauvegarde dans le registre `artifacts/models/lgbm_surrogate/<version>/`
- Génère `manifest.json` (timestamp, git commit, fingerprint dataset)
- Génère `checksums.sha256` + `checksums.json`

### 5. Évaluer

```powershell
.venv\Scripts\python -m src.cli evaluate
```

Produit `eval_metrics.json` et `eval_metrics.csv` dans le répertoire de la version.

### 6. Prédire un cas unique

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

Les features dérivées sont calculées automatiquement si absentes de l'entrée.

### 7. Vérifier l'intégrité des artefacts

```powershell
.venv\Scripts\python -m src.cli verify-artifacts
```

Vérifie les SHA-256 de tous les fichiers de la dernière version enregistrée.

---

## Raccourcis Makefile

```powershell
make build-features    # feature engineering
make train             # entraînement complet
make evaluate          # évaluation
make predict           # inférence (cas exemple)
make verify-artifacts  # intégrité SHA-256
make test              # 61 tests unitaires
make lint              # vérification syntaxe
```

---

## Configuration

Tous les paramètres runtime sont dans `configs/training.yaml` :

```yaml
features:
  seed: 42
  train_ratio: 0.70
  val_ratio: 0.15
  split_strategy: hash   # hash (déterministe) | random

training:
  random_state: 42
  cv_folds: 5
  n_trials: 60           # essais Optuna par cible (0 = désactiver)

artifacts:
  registry_dir: artifacts/models
  model_name: lgbm_surrogate
```

Surcharger via CLI (`--n-trials 0`, `--random-state 99`) ou via la variable d'environnement `CONFIG_PATH`.

---

## Registre de modèles

Chaque entraînement crée une version horodatée :

```
artifacts/models/lgbm_surrogate/
    v20260310_143022/
        lgbm_max_displacement_m.joblib
        lgbm_max_von_mises_pa.joblib
        advanced_metrics.csv
        advanced_metrics.json     ← métriques imbriquées par cible/split
        manifest.json             ← timestamp, git commit, SHA-256 dataset, versions packages
        checksums.sha256
        checksums.json
    latest.txt                    ← pointe vers la dernière version
```

---

## Tests

```powershell
.venv\Scripts\python -m unittest discover -s tests -p "test_*.py" -v
# Ran 61 tests — OK
```

Couverture :
- `test_config.py` — chargement et validation YAML
- `test_features.py` — formules physiques, splits, schéma
- `test_registry.py` — versionnement, copie, roundtrip modèle
- `test_integrity.py` — SHA-256, détection de falsification
- `test_smoke.py` — pipeline E2E complet + reproductibilité

---

## Inférence avec le modèle existant

Pour utiliser les modèles déjà entraînés dans `data/models/advanced/` :

```powershell
.venv\Scripts\python -m src.cli predict `
  --model-dir data/models/advanced `
  --case-json '{...}'
```

Ou avec l'ancien script de prédiction baseline :

```powershell
.venv\Scripts\python -m src.inference.predict_baseline `
  --model-path data/processed/baseline_model.joblib `
  --case-json '{...}'
```

---

## MLflow (optionnel)

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

## Contrat de données

Voir [docs/data_contract.md](docs/data_contract.md).

## Vue d'ensemble des scripts

Voir [docs/scripts_overview.md](docs/scripts_overview.md).
