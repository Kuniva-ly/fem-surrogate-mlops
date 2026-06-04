# Livrable Bloc 5 — Industrialisation du Modèle ML

**Notebook :** [05_industria_bloc5.ipynb](../notebooks/05_industria_bloc5.ipynb)
**Retour au sommaire :** [Sommaire](./sommaire.md)

---

## Compétences RNCP couvertes

| Code | Intitulé |
|------|---------|
| C5.1 | Assurer la traçabilité des expériences (MLflow) |
| C5.2 | Conteneuriser et déployer un modèle ML (Docker) |
| C5.3 | Exposer un modèle via une API REST (FastAPI) |
| C5.4 | Créer un tableau de bord interactif (Streamlit) |
| C5.5 | Mettre en place un monitoring (Prometheus / Grafana) |

---

## Objectif

Déployer le surrogate LightGBM en production via une stack industrielle complète :
API REST, dashboard interactif, suivi des expériences, monitoring et alertes.

---

## Architecture

```
Streamlit :8501  ──►  FastAPI :8000  ──►  LightGBM (registry)
                           │
                      /metrics
                           ↓
                   Prometheus :9090  ──►  Grafana :3000  ──►  Slack
                           │
                      MLflow :5000  ──►  MinIO (artefacts S3)
                                 ──►  PostgreSQL (metadata)
```

---

## Section 1 — MLflow (C5.1)

- Tracking des expériences : hyperparamètres, métriques, artefacts
- Registre de modèles : versionnage `v{YYYYMMDD}_{HHMMSS}`
- Artefacts stockés dans MinIO (`s3://ml-artifacts/`)
- SDK Python : `mlflow.log_params()`, `mlflow.log_metric()`, `mlflow.log_artifact()`

Expériences enregistrées : **2** (`fem-surrogate`, `Default`)
Runs dans `fem-surrogate` : **13 runs** (tous FINISHED)

Meilleur run (60 essais Optuna) :

| Métrique | Valeur |
|----------|--------|
| R² déplacement (log) | **0.9258** |
| R² Von Mises (log) | **0.9829** |
| RMSE déplacement (log) | 0.0420 |

---

## Section 2 — Docker Compose (C5.2)

| Service | Image | Port | Rôle |
|---------|-------|------|------|
| `api` | `fem-surrogate-api` | 8000 | FastAPI |
| `dashboard` | `fem-surrogate-dash` | 8501 | Streamlit |
| `mlflow` | `fem-surrogate-mlflow` | 5000 | Tracking |
| `minio` | `minio/minio` | 9000/9001 | Artefacts S3 + console |
| `postgres` | `postgres:15` | 5432 (interne) | MLflow DB |
| `prometheus` | `prom/prometheus` | 9090 | Métriques |
| `grafana` | `grafana/grafana` | 3000 | Dashboards |

Persistance via **bind mounts** (pas de volumes Docker) :
- `./data/minio-storage` → MinIO
- `./data/postgres` → PostgreSQL
- `./artifacts/models` → modèles

---

## Section 3 — FastAPI (C5.3)

| Endpoint | Méthode | Auth | Description |
|----------|---------|------|-------------|
| `/health` | GET | Non | Sonde disponibilité + état modèle |
| `/version` | GET | Basic | Version API + nb features |
| `/predict` | POST | Basic | Inférence surrogate |
| `/metrics` | GET | Non | Métriques Prometheus |

Exemple de réponse `/predict` (acier, with_hole, 50 MPa) :
```json
{
  "model_version": "v20260603_102835",
  "predictions": {
    "max_displacement_m": 5.083e-05,
    "max_von_mises_pa":   1.617e+08
  },
  "input_summary": {
    "geometry_type": "with_hole",
    "length_m": 0.2,
    "height_m": 0.05,
    "young_modulus_pa": 210000000000.0,
    "traction_pa": 50000000.0,
    "hole_radius_ratio": 0.2
  }
}
```

Prédictions batch (3 géométries) :

| Cas | Déplacement max (mm) | Von Mises max (MPa) |
|-----|----------------------|---------------------|
| with_hole / acier | 0.0508 | 161.66 |
| without_hole / aluminium | 0.0799 | 33.15 |
| with_hole_moving / titane | 0.0745 | 100.67 |

Latence mesurée sur 30 appels (CPU only, 1 worker) :

| Métrique | Valeur |
|----------|--------|
| Médiane | **20.1 ms** |
| Moyenne | 26.4 ms |
| P95 | **40.3 ms** |
| Max | 42.4 ms |

---

## Section 4 — Streamlit (C5.4)

- Formulaire paramètres géométriques et matériaux
- Prédiction en temps réel via `POST /predict`
- Comparaison surrogate vs estimation analytique (Kirsch, Hooke)
- Analyse de sensibilité 1-D (sweep paramétrique)
- Surface de réponse 2-D (traction × rayon trou)

---

## Section 5 — Prometheus / Grafana (C5.5)

- Scraping `/metrics` toutes les 15 secondes
- Dashboards : latence P95, requêtes/s, taux erreurs 5xx
- Alertes : latence P95 > 500 ms → notification Slack

Cibles Prometheus (état mesuré) :

| Cible | État |
|-------|------|
| `fem_api` | **up** |
| `prometheus` | **up** |
| `minio` | **up** |
| `mlflow` | down (endpoint `/metrics` non exposé) |

> MLflow ne scrape pas nativement Prometheus — sa cible est down, ce qui n'affecte pas le fonctionnement de la stack. À corriger via un exporter dédié si le monitoring MLflow est requis.

---

## Démarrage de la stack

```bash
docker compose up -d
```

URLs :
- API Swagger : http://localhost:8000/docs
- Dashboard : http://localhost:8501
- MLflow : http://localhost:5000
- Grafana : http://localhost:3000
