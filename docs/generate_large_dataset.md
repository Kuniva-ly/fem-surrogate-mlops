# Générer un grand dataset

Depuis la racine du projet `c:\Users\Administrateur\Desktop\fem-surrogate`.

---

## 1. Démarrer l'infra Docker

Requis uniquement pour le backend FEniCS (calcul EF réel).
Sans Docker, utiliser `--backend proxy` pour toutes les commandes.

```powershell
docker compose up -d
```

---

## 2. Générer — with_hole

```powershell
# FEniCS réel (Docker requis)
docker compose exec fenics python -m src.simulations.traction_plate_with_hole `
    --n 10000 --seed 42 --out data/raw/sim_v1 `
    --backend fenics --chunk-size 2000 --sampling-mode continuous `
    --mesh-nx 120 --mesh-ny 24

# Proxy analytique (sans Docker)
.venv\Scripts\python -m src.simulations.traction_plate_with_hole `
    --n 10000 --seed 42 --out data/raw/sim_v1 `
    --backend proxy --chunk-size 2000 --sampling-mode continuous
```

---

## 3. Générer — without_hole

```powershell
.venv\Scripts\python -m src.simulations.traction_plate_without_hole `
    --n 10000 --seed 42 --out data/raw/sim_v1_without_hole `
    --backend proxy --chunk-size 2000 --sampling-mode continuous
```

---

## 4. Générer — with_hole_moving

```powershell
# FEniCS réel (Docker requis)
docker compose exec fenics python -m src.simulations.traction_plate_moving_hole `
    --n 10000 --seed 42 --out data/raw/sim_v2_moving_hole `
    --backend fenics --chunk-size 2000 --sampling-mode continuous `
    --data-version sim_v2_moving_hole --mesh-nx 120 --mesh-ny 24

# Proxy analytique (sans Docker)
.venv\Scripts\python -m src.simulations.traction_plate_moving_hole `
    --n 10000 --seed 42 --out data/raw/sim_v2_moving_hole `
    --backend proxy --chunk-size 2000 --sampling-mode continuous `
    --data-version sim_v2_moving_hole
```

---

## 5. Validation des données brutes

```powershell
.venv\Scripts\python -m src.processing.validate --input data/raw/sim_v1
.venv\Scripts\python -m src.processing.validate --input data/raw/sim_v1_without_hole
.venv\Scripts\python -m src.processing.validate --input data/raw/sim_v2_moving_hole

# Ou via Makefile (les trois en une commande)
make validate-all
```

---

## 6. Feature engineering + splits ML

```powershell
# Via CLI — lit configs/training.yaml (seed, ratios, stratégie)
.venv\Scripts\python -m src.cli build-features --input data/raw

# Paramètres explicites (surcharge le config)
.venv\Scripts\python -m src.processing.build_features `
    --input data/raw --out-dir data/processed `
    --features-out-dir data/features/advanced `
    --split-strategy hash --seed 42
```

---

## 7. Entraînement

```powershell
# Complet avec Optuna (60 essais par défaut — lit configs/training.yaml)
.venv\Scripts\python -m src.cli train

# Sans Optuna (paramètres par défaut LightGBM, rapide)
.venv\Scripts\python -m src.cli train --n-trials 0

# Paramètres personnalisés
.venv\Scripts\python -m src.cli train --n-trials 30 --cv-folds 3 --random-state 99
```

---

## 8. Upload des données brutes vers MinIO

Copie les fichiers `data/raw/*.parquet` générés localement vers le bucket `raw-simulations`.

```powershell
# Valeurs par défaut (minioadmin / localhost:9000)
.venv\Scripts\python scripts/upload_raw_to_minio.py

# Paramètres explicites
.venv\Scripts\python scripts/upload_raw_to_minio.py `
    --raw-dir data/raw --endpoint http://localhost:9000
```

Variables d'environnement (optionnelles) :
```powershell
$env:MLFLOW_S3_ENDPOINT_URL = "http://localhost:9000"
$env:AWS_ACCESS_KEY_ID      = "minioadmin"
$env:AWS_SECRET_ACCESS_KEY  = "minioadmin"
```

---

## 9. ETL Spark + ingestion MinIO

`src/processing/spark_ingest.py` gère l'ETL complet et l'écriture vers MinIO.

```powershell
# Mode local (data/raw → data/processed/warehouse.parquet)
.venv\Scripts\python -m src.processing.spark_ingest `
    --raw-dir data/raw --wh-dir data/processed

# Mode MinIO (s3a://raw-simulations → s3a://processed-simulations)
# Nécessite le Docker stack actif (make up)
.venv\Scripts\python -m src.processing.spark_ingest --use-minio
```

Variables d'environnement (optionnelles, sinon valeurs par défaut `minioadmin`) :
```powershell
$env:MLFLOW_S3_ENDPOINT_URL = "http://localhost:9000"
$env:AWS_ACCESS_KEY_ID      = "minioadmin"
$env:AWS_SECRET_ACCESS_KEY  = "minioadmin"
```

---

## 10. Convention feature store

| Dossier | Contenu | Décision |
|---|---|---|
| `data/processed/` | splits train/val/test nettoyés | data engineering |
| `data/features/` | features retenues pour un modèle précis | ML / science |

Structure recommandée orientée modèle/version :

```
data/features/
  stress_model/
    v1/
      with_hole/date=YYYY-MM-DD/features.parquet
      without_hole/date=YYYY-MM-DD/features.parquet
    v2/
      ...
```

**Principe clé :** `training = serving` — même feature, même transformation,
pour éviter le training-serving skew.

Solutions feature store en production : Feast · Tecton · Hopsworks.
