# Livrable Bloc 5 — Industrialisation du Modèle ML

**Notebook :** [industria_bloc5.ipynb](../notebooks/industria_bloc5.ipynb)
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
- Artefacts stockés dans MinIO (`s3://mlflow/`)
- SDK Python : `mlflow.log_params()`, `mlflow.log_metric()`, `mlflow.log_artifact()`

---

## Section 2 — Docker Compose (C5.2)

| Service | Image | Port | Rôle |
|---------|-------|------|------|
| `api` | `fem-surrogate-api` | 8000 | FastAPI |
| `dashboard` | `fem-surrogate-dash` | 8501 | Streamlit |
| `mlflow` | `fem-surrogate-mlflow` | 5000 | Tracking |
| `minio` | `minio/minio` | 9000 | Artefacts S3 |
| `db` | `postgres:15` | 5432 | MLflow DB |
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

Exemple de réponse `/predict` :
```json
{
  "model_version": "v20260519_143022",
  "predictions": {
    "max_displacement_m": 1.23e-05,
    "max_von_mises_pa": 4.56e+07
  }
}
```

Latence mesurée : **médiane < 10 ms**, P95 < 50 ms

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
