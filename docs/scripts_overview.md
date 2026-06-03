# Vue d'ensemble des scripts

Le projet supporte trois variantes de géométrie :
- `with_hole` — plaque rectangulaire avec trou centré
- `without_hole` — plaque pleine
- `with_hole_moving` — plaque avec trou à position variable

---

## CLI principal

### `src/cli.py`

Point d'entrée unifié pour toute la production. Toutes les commandes lisent
`configs/training.yaml` et acceptent des surcharges CLI.

```powershell
.venv\Scripts\python -m src.cli <commande> [--config path/config.yaml] [options]
```

| Commande | Description |
|---|---|
| `build-features` | Feature engineering + splits train/val/test |
| `train` | Entraînement LightGBM + Optuna + enregistrement registre |
| `evaluate` | Évaluation sur val/test (dernière version du registre) |
| `predict` | Inférence cas unique (JSON inline ou fichier) |
| `verify-artifacts` | Vérification SHA-256 des artefacts enregistrés |

---

## Configuration

### `src/config.py`

Charge et valide `configs/training.yaml` avec messages d'erreur clairs.
Retourne un `PipelineConfig` typé (dataclasses).

```python
from src.config import load_config
cfg = load_config()              # configs/training.yaml par défaut
cfg = load_config("mon.yaml")    # chemin explicite
# ou : CONFIG_PATH=mon.yaml python -m src.cli train
```

Sections validées : `data`, `features`, `training`, `artifacts`, `evaluation`.

---

## Simulation FEM

### `src/simulations/traction_plate_with_hole.py`

Génère des simulations de plaque rectangulaire avec trou centré.

```powershell
# FEniCS réel (Docker requis)
docker compose exec fenics python -m src.simulations.traction_plate_with_hole `
    --n 5000 --seed 42 --out data/raw/sim_v1 --backend fenics

# Proxy analytique rapide (sans Docker)
.venv\Scripts\python -m src.simulations.traction_plate_with_hole `
    --n 1000 --seed 42 --out data/raw/sim_v1 --backend proxy
```

Paramètres clés : `--n`, `--seed`, `--out`, `--backend` (fenics|proxy|auto),
`--sampling-mode` (categorical|continuous), `--chunk-size`.

### `src/simulations/traction_plate_without_hole.py`

Idem pour plaques pleines (sans trou). Même API.

### `src/simulations/traction_plate_moving_hole.py`

Plaque avec trou à position variable (`hole_cx_ratio`, `hole_cy_ratio`).
Inclut la validation que le trou reste dans les limites de la plaque.

---

## Utilitaires partagés

### `src/utils/s3_client.py`

Point d'entrée unique pour toutes les connexions MinIO/S3. Centralise endpoint,
access key et secret key lus depuis les variables d'environnement.
**Ne jamais instancier `boto3.client()` directement dans le code applicatif.**

```python
from src.utils.s3_client import get_s3_client, BUCKET_FEATURES, BUCKET_PROCESSED

s3 = get_s3_client()
s3.upload_file(str(local_path), BUCKET_FEATURES, key)
```

Variables d'environnement lues :
`MLFLOW_S3_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`MINIO_BUCKET_RAW`, `MINIO_BUCKET_PROCESSED`, `MINIO_BUCKET_FEATURES`, `MINIO_BUCKET_ARTIFACTS`.

---

## Ingestion MinIO

### `scripts/wait_for_minio.py`

Attend que MinIO soit accessible avant de lancer des scripts locaux.
**Utile uniquement hors Docker** : quand MinIO tourne dans un conteneur et qu'on exécute
`upload_raw_to_minio.py` depuis la machine hôte. Dans Docker Compose, l'attente est gérée
nativement via `healthcheck` + `depends_on: condition: service_healthy`.

```powershell
.venv\Scripts\python scripts/wait_for_minio.py   # attend jusqu'à 60s, exit 0 si prêt
```

### `scripts/upload_raw_to_minio.py`

Upload les fichiers `data/raw/*.parquet` vers le bucket MinIO `raw-simulations` (boto3).
Étape préalable au Spark ETL quand les simulations ont été générées en local.

```powershell
# S'assurer que MinIO est prêt (si hors Docker)
.venv\Scripts\python scripts/wait_for_minio.py

.venv\Scripts\python scripts/upload_raw_to_minio.py
.venv\Scripts\python scripts/upload_raw_to_minio.py --raw-dir data/raw --endpoint http://localhost:9000
```

### `src/processing/spark_ingest.py`

ETL PySpark complet : lit `s3a://raw-simulations` (ou local), transforme, écrit dans `s3a://processed-simulations`.

```powershell
# Local
.venv\Scripts\python -m src.processing.spark_ingest --raw-dir data/raw --wh-dir data/processed

# MinIO (Docker stack requis)
.venv\Scripts\python -m src.processing.spark_ingest --use-minio
```

---

## Validation qualité

### `src/processing/validate.py`

Vérifie un dossier de parquets bruts avant traitement ML :
- colonnes requises (16 champs)
- pas de nulls
- unicité `simulation_id`
- plages numériques (poisson ∈ (0,0.5), mesh_nx ≥ 8, etc.)
- cohérence géométrique (with_hole doit avoir `hole_radius_ratio`)

```powershell
.venv\Scripts\python -m src.processing.validate --input data/raw/sim_v1
```

---

## Feature engineering

### `src/processing/build_features.py`

Construit 42 features physiques depuis les colonnes brutes.

**Features principales :**
- Géométrie : `area_m2`, `aspect_ratio`, `radius_abs`, `d_over_W`
- Concentration de contrainte (Peterson) : `Kt_theory`, `net_section_ratio`, `sigma_net`
- Déformation élastique : `epsilon`, `delta_theory`, `biaxial_factor`
- Ligaments (trou mobile) : `lig_left/right/top/bottom`, `lig_min`, `edge_ratio`
- Excentricité : `eccentricity_x`, `eccentricity_y`, `eccentricity`
- Log-espace : `logE`, `logS`, `log_epsilon`, `log_delta_th`, `log_sigma_net`, `log_Kt`, etc.
- Indicateurs : `has_hole`, `has_moving_hole`

**Colonnes exclues des features modèle :**
- `mesh_nx`, `mesh_ny` — constantes dans toutes les lignes (variance nulle)
- `simulation_id`, `timestamp`, métadonnées solver — non prédictifs

**Split déterministe :**
- Stratégie `hash` (défaut) : SHA-256(simulation_id|seed) → bucket [0,1)
- Reproduit exactement le même découpage sur des relances identiques
- Stratégie `random` disponible pour expérimentations

```powershell
# Via CLI (recommandé)
.venv\Scripts\python -m src.cli build-features --input data/raw

# Via module direct (paramètres explicites)
.venv\Scripts\python -m src.processing.build_features `
    --input data/raw --out-dir data/processed `
    --features-out-dir data/features/advanced `
    --split-strategy hash --seed 42
```

---

## Entraînement

### `src/training/train_advanced.py` ← modèle principal

LightGBM avec transformation log des cibles + tuning Optuna.

**Points clés :**
- Cibles transformées : `log10(max_displacement_m)` et `log10(max_von_mises_pa)`
  (compresse 4–6 décades en distribution uniforme)
- Un modèle séparé par cible (physics différente)
- Recherche Optuna : 60 essais par cible, CV 5-fold sur train
- Contraintes de monotonicité physiques (ex. traction_pa → +1, young_modulus_pa → -1)
- Encodeur catégoriel `OrdinalEncoder` fitté **uniquement sur train** (pas de fuite val/test)
- Seed global propagé à `random`, `numpy.random`, `PYTHONHASHSEED`

**Artefacts produits :**
- `lgbm_max_displacement_m.joblib` — modèle + feature_cols + encoder
- `lgbm_max_von_mises_pa.joblib` — idem
- `advanced_metrics.csv` — R²(log), RMSE(log), MAE(log), R²(orig), RMSE(orig), MAPE
- `manifest.json` — timestamp UTC, commit git, versions packages, fingerprint dataset
- `checksums.sha256` + `checksums.json` — intégrité SHA-256

```powershell
# Via CLI (recommandé — enregistre dans le registre)
.venv\Scripts\python -m src.cli train --n-trials 60

# Via module direct (sans registre)
.venv\Scripts\python -m src.training.train_advanced `
    --data-dir data/processed --out-dir data/models/advanced --n-trials 60
```

**Espace de recherche Optuna :**

| Hyperparamètre | Plage | Note |
|---|---|---|
| `n_estimators` | 400–2000 | |
| `learning_rate` | 0.01–0.2 (log) | |
| `num_leaves` | 31–255 | |
| `max_depth` | **8–16** | Élargi (était 4–12) |
| `min_child_samples` | 10–100 | |
| `subsample` | 0.6–1.0 | |
| `colsample_bytree` | 0.5–1.0 | |
| `reg_alpha`, `reg_lambda` | 1e-8–10 (log) | |

**Paramètres par défaut (`n_trials=0`)** — chargés depuis `configs/default_hyperparameters.yaml`
(remplace les valeurs hardcodées, configurable via `DEFAULT_PARAMS_PATH`).

**Feature importance** — exportée automatiquement pour chaque cible :
- `feature_importance_split_<target>.csv`
- `feature_importance_gain_<target>.csv`

Enregistrées dans le registre et loguées dans MLflow.

**Performances actuelles (dataset complet ~50k lignes) :**

| Cible | Split | R²(log) | RMSE(log) | R²(orig) | MAPE |
|---|---|---|---|---|---|
| max_displacement_m | val | 0.9908 | 0.0529 | 0.9538 | 5.4% |
| max_displacement_m | test | 0.9917 | 0.0493 | 0.9710 | 5.1% |
| max_von_mises_pa | val | 0.9925 | 0.0320 | 0.7422 | 3.8% |
| max_von_mises_pa | test | 0.9939 | 0.0292 | 0.9407 | 3.8% |

---

## Évaluation (protocole figé)

### `src/evaluation.py`

Centralise le calcul des métriques — toutes les évaluations utilisent la même fonction.
Produit `metrics.json` (imbriqué par cible/split) et `metrics.csv` (tidy).

Métriques canoniques : `r2_log`, `rmse_log`, `mae_log`, `r2_orig`, `rmse_orig`, `mape`.

---

## Registre de modèles

### `src/registry.py`

Registre local versionné avec pointeur `latest` (écriture atomique via rename).

```
artifacts/models/lgbm_surrogate/
    v20260310_143022/       ← horodatage UTC
        *.joblib
        advanced_metrics.csv
        manifest.json
        checksums.sha256
        checksums.json
    latest.txt              ← contient "v20260310_143022"
```

```python
from src.registry import ModelRegistry
reg = ModelRegistry(Path("artifacts/models"), "lgbm_surrogate")
print(reg.latest_version())   # "v20260310_143022"
print(reg.list_versions())    # ["v20260310_143022", ...]
```

---

## Reproductibilité

### `src/utils/manifest.py`

Génère un manifest JSON pour chaque run :
- `timestamp_utc` — horodatage ISO-8601
- `git_commit` — hash HEAD (ou "unavailable")
- `package_versions` — lightgbm, scikit-learn, optuna, pandas, numpy, etc.
- `config` — snapshot des paramètres utilisés
- `dataset_fingerprint` — SHA-256 agrégé de tous les fichiers d'entrée
- `dataset_files` — liste des fichiers consommés

### `src/utils/integrity.py`

Génère et vérifie des checksums SHA-256 pour tous les artefacts.

```powershell
.venv\Scripts\python -m src.cli verify-artifacts
# → "All artifacts OK — checksums match."
```

---

## Inférence

### `src/cli.py predict` ← interface principale

```powershell
.venv\Scripts\python -m src.cli predict `
  --case-json '{"length_m":1.2,"height_m":0.3,"young_modulus_pa":2.1e11,
               "poisson_ratio":0.3,"traction_pa":1500000,
               "mesh_nx":120,"mesh_ny":24,
               "geometry_type":"with_hole","hole_radius_ratio":0.1}'
```

Calcule automatiquement les 42 features si seuls les paramètres bruts sont fournis.
Utilise la dernière version enregistrée dans le registre par défaut.

---

## Tests

```powershell
.\run test
# ou : .venv\Scripts\python -m pytest tests/ -v
# Ran 64 tests — OK
```

| Fichier | Tests | Couverture |
|---|---|---|
| `tests/test_config.py` | 13 | Chargement YAML, validation, env var |
| `tests/test_features.py` | 21 | Formules physiques, splits, nulls |
| `tests/test_registry.py` | 14 | Versionnement, copie, roundtrip joblib |
| `tests/test_integrity.py` | 9 | SHA-256, falsification, fichier manquant |
| `tests/test_smoke.py` | 6 | Pipeline E2E + reproductibilité |

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

## Infrastructure Docker (optionnel)

`docker-compose.yml` (à recréer si nécessaire) définit :
MinIO, PostgreSQL, FEniCS, MLflow.

```powershell
make up    # docker compose up -d
make down  # docker compose down
```
