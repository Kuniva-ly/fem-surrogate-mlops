# Planning & Sprints — FEM Surrogate ML

**Retour :** [Gestion de projet](./bloc6_gestion_projet.md) | [Sommaire](./sommaire.md)

---

## Méthodologie

Projet conduit en **Agile adapté** : découpage en 5 sprints thématiques alignés sur les blocs
thèmes techniques. Chaque sprint produit un livrable fonctionnel et indépendant, validé avant de passer
au suivant.

**Durée totale du projet :** Février 2026 → Juin 2026 (4 mois)

---

## Backlog général (User Stories)

| ID | En tant que… | Je veux… | Pour… | Priorité | Sprint |
|----|-------------|---------|-------|---------|--------|
| US-01 | Ingénieur | Stocker toutes les simulations FEM dans un Data Lake | Centraliser et versionner les données | Haute | 1 |
| US-02 | Data scientist | Avoir un pipeline ETL automatisé | Ne pas retraiter les données manuellement | Haute | 1 |
| US-03 | Ingénieur | Comprendre la distribution des simulations | Identifier les paramètres influents | Haute | 2 |
| US-04 | Data scientist | Détecter les valeurs aberrantes | Garantir la qualité du dataset | Haute | 2 |
| US-05 | Ingénieur | Prédire σ_VM en < 1 seconde | Accélérer les décisions de conception | Critique | 3 |
| US-06 | Data scientist | Évaluer l'overfitting du modèle | Garantir la généralisation | Haute | 3 |
| US-07 | Ingénieur | Classifier automatiquement le type de géométrie depuis une image | Identifier la géométrie sans métadonnées | Moyenne | 4 |
| US-08 | Data scientist | Comparer CNN vs LightGBM | Choisir la meilleure approche | Moyenne | 4 |
| US-09 | Ingénieur | Appeler l'API depuis n'importe quel outil | Intégrer le surrogate dans une chaîne CAO | Haute | 5 |
| US-10 | Opérateur | Surveiller la latence de l'API en temps réel | Détecter les dégradations de performance | Haute | 5 |
| US-11 | Responsable technique | Reproduire l'ensemble du pipeline | Valider la reproductibilité avant mise en production | Critique | 5 |

---

## Planning par Sprint

### Sprint 1 — Infrastructure & Data Lake
**Durée :** Février 2026 (3 semaines)
**Objectif :** Mettre en place le Data Lake et le pipeline ETL Spark

| Tâche | Effort (j) | Statut |
|-------|-----------|--------|
| Simulateur FEniCS — 9 variantes géométriques | 5 | ✅ Done |
| Upload brut vers MinIO (boto3) | 1 | ✅ Done |
| Pipeline ETL Apache Spark (Extract → Transform → Load) | 3 | ✅ Done |
| Pseudonymisation RGPD SHA-256 | 1 | ✅ Done |
| Validation Data Warehouse (0 doublon, 0 null critique) | 1 | ✅ Done |
| Notebook `01_infrastructure_bloc1.ipynb` | 2 | ✅ Done |

**Critère de validation :** Data Warehouse contient 61 550 simulations propres dans MinIO

---

### Sprint 2 — Analyse Exploratoire (EDA)
**Durée :** Mars 2026 (2 semaines)
**Objectif :** Comprendre les données et identifier les variables influentes

| Tâche | Effort (j) | Statut |
|-------|-----------|--------|
| Statistiques descriptives (pandas) | 1 | ✅ Done |
| Détection valeurs aberrantes (IQR) | 1 | ✅ Done |
| Matrice de corrélations Pearson | 1 | ✅ Done |
| Analyses Spark distribuées (C2.3) | 2 | ✅ Done |
| Visualisations Plotly interactives | 2 | ✅ Done |
| Notebook `02_eda_bloc2.ipynb` | 1 | ✅ Done |

**Critère de validation :** Les paramètres influents identifiés (traction → Von Mises, E → déplacement)

---

### Sprint 3 — Machine Learning
**Durée :** Mars–Avril 2026 (3 semaines)
**Objectif :** Entraîner et valider le modèle surrogate LightGBM

| Tâche | Effort (j) | Statut |
|-------|-----------|--------|
| Feature engineering physique (42 features) | 3 | ✅ Done |
| Encodage OrdinalEncoder + StandardScaler | 1 | ✅ Done |
| LightGBM + Optuna 60 trials | 3 | ✅ Done |
| K-Fold Cross-Validation 5 folds | 1 | ✅ Done |
| Test sur-entraînement/sous-entraînement | 1 | ✅ Done |
| Clustering KMeans + DBSCAN sur images Kirsch | 2 | ✅ Done |
| Analyse résidus + KS test + IC 95% | 2 | ✅ Done |
| Notebook `03_ml_bloc3.ipynb` | 1 | ✅ Done |

**Critère de validation :** R² > 0.95 sur test set pour les 2 cibles

---

### Sprint 4 — Deep Learning
**Durée :** Avril–Mai 2026 (3 semaines)
**Objectif :** Classifier les géométries par images et comparer avec LightGBM

| Tâche | Effort (j) | Statut |
|-------|-----------|--------|
| Génération 51 550 images PNG 64×64 (Kirsch) | 2 | ✅ Done |
| CNN scratch (Conv2D × 3 + GAP + Dense) | 2 | ✅ Done |
| Transfer Learning MobileNetV2 + fine-tuning 2 phases | 3 | ✅ Done |
| Augmentation données (flip, rotation, zoom) | 1 | ✅ Done |
| Évaluation : accuracy, F1, matrice confusion | 1 | ✅ Done |
| CNN Regression + comparaison LightGBM | 2 | ✅ Done |
| Notebook `04_dl_bloc4.ipynb` | 1 | ✅ Done |

**Critère de validation :** Accuracy classification > 90% (réalisé : 98.5%)

---

### Sprint 5 — Industrialisation
**Durée :** Mai–Juin 2026 (3 semaines)
**Objectif :** Déployer la stack complète en production

| Tâche | Effort (j) | Statut |
|-------|-----------|--------|
| FastAPI + authentification Basic Auth | 2 | ✅ Done |
| Métriques Prometheus (`/metrics` endpoint) | 1 | ✅ Done |
| Docker Compose 7 services | 2 | ✅ Done |
| MLflow tracking + Model Registry | 2 | ✅ Done |
| Dashboard Streamlit + analyse sensibilité | 3 | ✅ Done |
| Grafana alertes Slack (latence P95 > 500 ms) | 1 | ✅ Done |
| Manifeste d'intégrité (git hash + SHA-256) | 1 | ✅ Done |
| Notebook `05_industria_bloc5.ipynb` | 1 | ✅ Done |

**Critère de validation :** Stack opérationnelle — tous services `UP` dans `docker compose ps`

---

## Jalons

```
Fév 2026                 Mar 2026            Avr 2026       Mai 2026        Juin 2026
    │                       │                    │              │               │
    ▼                       ▼                    ▼              ▼               ▼
[S1 Data Lake]         [S2 EDA]           [S3 ML ✓]      [S4 DL ✓]      [S5 Prod ✓]
61 550 sims            Corrélations       R²=0.99         Acc=98.5%       API<100ms
MinIO + Spark          Visualisations     LightGBM        CNN+MobileNet   Docker×7
```

---

## Vélocité et coûts par sprint

| Sprint | Effort estimé (j) | Effort réel (j) | Écart | Coût réel (550 €/j) |
|--------|------------------|-----------------|-------|---------------------|
| S1 Infrastructure | 13 | 14 | +1 (config Hadoop Windows) | 7 700 € |
| S2 EDA | 8 | 7 | -1 | 3 850 € |
| S3 ML | 14 | 15 | +1 (Optuna 60 trials) | 8 250 € |
| S4 DL | 12 | 13 | +1 (CPU training lent) | 7 150 € |
| S5 Industrialisation | 13 | 13 | 0 | 7 150 € |
| **Total** | **60** | **62** | **+2 j (+3%)** | **34 100 €** |
