# Sommaire des Livrables — FEM Surrogate ML

Modèle de substitution machine learning pour simulations éléments finis (FEM)
de plaques métalliques sous traction.

---

## Livrables techniques

| Thème | Titre | Livrable |
|-------|-------|---------|
| 1 | Infrastructure & Data Lake | [Bloc 1](./bloc1_infrastructure.md) |
| 2 | Analyse Exploratoire (EDA) | [Bloc 2](./bloc2_eda.md) |
| 3 | Machine Learning | [Bloc 3](./bloc3_ml.md) |
| 4 | Deep Learning | [Bloc 4](./bloc4_dl.md) |
| 5 | Industrialisation | [Bloc 5](./bloc5_industria.md) |

---

## Notebooks

| Notebook | Description | Lien |
|----------|-------------|------|
| `01_infrastructure_bloc1.ipynb` | Spark ETL, Data Lake MinIO, schéma Parquet | [../notebooks/01_infrastructure_bloc1.ipynb](../notebooks/01_infrastructure_bloc1.ipynb) |
| `02_eda_bloc2.ipynb` | Analyse exploratoire, distributions, corrélations | [../notebooks/02_eda_bloc2.ipynb](../notebooks/02_eda_bloc2.ipynb) |
| `03_ml_bloc3.ipynb` | LightGBM, clustering, analyse statistique | [../notebooks/03_ml_bloc3.ipynb](../notebooks/03_ml_bloc3.ipynb) |
| `04_dl_bloc4.ipynb` | CNN scratch, MobileNetV2, régression image | [../notebooks/04_dl_bloc4.ipynb](../notebooks/04_dl_bloc4.ipynb) |
| `05_industria_bloc5.ipynb` | FastAPI, Docker, MLflow, Grafana | [../notebooks/05_industria_bloc5.ipynb](../notebooks/05_industria_bloc5.ipynb) |

---

## Gestion de Projet

| Document | Description |
|----------|-------------|
| [bloc6_gestion_projet.md](./bloc6_gestion_projet.md) | Vue d'ensemble, métriques de succès |
| [bloc6_cahier_des_charges.md](./bloc6_cahier_des_charges.md) | Contexte, objectifs SMART, périmètre, contraintes |
| [bloc6_planning_sprints.md](./bloc6_planning_sprints.md) | Backlog, 5 sprints, Gantt, vélocité |
| [bloc6_gestion_risques.md](./bloc6_gestion_risques.md) | 7 risques identifiés, niveaux, mitigations |
| [bloc6_retrospective.md](./bloc6_retrospective.md) | Bilan, adaptations, enseignements, résultats finaux |

---

## Données

| Fichier | Description |
|---------|-------------|
| `data/raw/` | Simulations FEM brutes (Parquet partitionné) |
| `data/processed/` | Splits train / val / test après feature engineering |
| `data/unstructured/images/` | 51 550 images PNG 64×64 (champs Von Mises) |
| `artifacts/models/` | Modèles LightGBM versionnés (registre local) |
