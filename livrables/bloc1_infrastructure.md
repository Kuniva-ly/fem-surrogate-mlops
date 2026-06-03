# Livrable Bloc 1 — Infrastructure & Data Lake

**Notebook :** [01_infrastructure_bloc1.ipynb](../notebooks/01_infrastructure_bloc1.ipynb)
**Retour au sommaire :** [Sommaire](./sommaire.md)

---

## Compétences RNCP couvertes

| Code | Intitulé |
|------|---------|
| C1.1 | Concevoir et déployer une architecture de traitement de données massives |
| C1.2 | Mettre en œuvre un pipeline d'ingestion de données (ETL) |
| C1.3 | Stocker et organiser les données dans un Data Lake |
| C1.4 | Garantir la qualité et la traçabilité des données |

---

## Objectif

Mettre en place un Data Lake scalable pour stocker et traiter les simulations FEM
(Finite Element Method) de plaques métalliques sous traction.

---

## Données

- **Source :** simulations Python (FEniCS-like) générées localement
- **Volume :** 51 550 simulations, 3 géométries (`with_hole`, `without_hole`, `with_hole_moving`)
- **Format :** Parquet partitionné par source et type de géométrie
- **Raw Lake :** 37 fichiers — 8.3 MB (bucket `raw-simulations`)

---

## Architecture

```
data/raw/
  sim_v1/date=2026-01-01/with_hole/*.parquet
  sim_v1/date=2026-01-01/without_hole/*.parquet
  sim_v2/...

         Spark ETL (PySpark) — 20 cœurs
              ↓
data/processed/warehouse.parquet   ← Data Warehouse consolidé
              ↓
   MinIO S3 (bucket: processed-simulations)
```

## Stack technique

| Outil | Rôle |
|-------|------|
| **PySpark 3.5.7** | ETL — lecture, transformation, validation, écriture |
| **MinIO** | Stockage objet S3-compatible (Data Lake) |
| **Docker Compose** | Orchestration infrastructure (MinIO + PostgreSQL) |
| **Parquet (Snappy)** | Format columnar compressé |

---

## Résultats mesurés

| Indicateur | Valeur |
|-----------|--------|
| Simulations chargées | **61 550** |
| Colonnes warehouse | 21 |
| Partitions Spark | 19 |
| Doublons sur `simulation_id` | **0** |
| Valeurs manquantes (colonnes critiques) | **0** |
| Taille warehouse MinIO | **9.0 MB** (10 fichiers Parquet) |
| Pseudonymisation | SHA-256 (64 car.) — UUID original absent du warehouse ✓ |

### Sources ingérées

| Source | Simulations |
|--------|------------|
| sim_v1 | 20 000 |
| sim_v2_moving_hole_wide | 10 000 |
| sim_v1_wide | 10 000 |
| sim_v1_without_hole_wide | 10 000 |
| sim_v2_moving_hole | 10 000 |
| Autres (low, etc.) | 1 550 |

---

## Réalisations

- Pipeline Spark `spark_ingest.py` : lecture multi-sources, déduplication, validation schéma
- Stockage dans MinIO via protocole S3A (`s3a://processed-simulations/`)
- Partitionnement par `geometry_type` et `date`
- Validation qualité : contraintes de types, valeurs nulles, plages physiques
- Pseudonymisation des identifiants simulés (SHA-256)

---

## Commande

```bash
python -m src.cli spark-ingest --use-minio
```
