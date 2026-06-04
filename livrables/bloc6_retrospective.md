# Rétrospective — FEM Surrogate ML

**Retour :** [Gestion de projet](./bloc6_gestion_projet.md) | [Sommaire](./sommaire.md)

---

## Bilan financier

| Poste | Prévu | Réalisé | Écart |
|-------|-------|---------|-------|
| Développement (j/h) | 60 j × 550 € = 33 000 € | 62 j × 550 € = 34 100 € | +1 100 € (+3%) |
| Compute (électricité locale) | 15 € | 17 € | +2 € |
| Licences logicielles | 0 € | 0 € | 0 |
| **Total** | **33 015 €** | **34 117 €** | **+1 102 € (+3,3%)** |

Dépassement de +2 jours dû à la configuration Hadoop/Windows (S1) et au temps d'entraînement CNN sur CPU (S4).
Aucun dépassement sur les outils : stack 100% open source maintenue.

---

## Bilan global

| Indicateur | Prévu | Réalisé | Écart |
|-----------|-------|---------|-------|
| Durée totale | 4 mois | 4 mois | 0 |
| Effort total | 60 j | 62 j | +2 j (+3%) |
| Notebooks d'analyse | 5 | 5 | 0 |
| Simulations FEM générées | 50 000 | 61 550 | +23% |
| R² von Mises (log) | > 0.95 | **0.9829** | +3.3pts |
| R² displacement (orig) | > 0.95 | **0.9798** | +3.0pts |
| Latence API P95 | < 200 ms | **< 100 ms** | ×2 mieux |
| CNN accuracy | > 90% | **98.5%** | +8.5pts |

---

## Ce qui a bien fonctionné

### Architecture des données
La décision de normaliser les cibles par la traction (`Kt_eff = σ_VM / σ_applied`) avant
l'entraînement a été déterminante. En exploitant la linéarité de l'élasticité, le modèle
prédit un ratio adimensionnel indépendant de la magnitude de la charge — ce qui le rend
valide pour toute traction dans le régime élastique, bien au-delà des 2.2 MPa d'entraînement.

### Feature engineering physique
Les 42 features calculées (Peterson Kt, contrainte nette, distances ligaments, ratios
géométriques) ont donné à LightGBM les "bonnes variables" pour apprendre. Le modèle
atteint R² = 0.99 avec seulement 400 arbres — très peu pour une telle précision.

### Stack Docker modulaire
Le découpage en 7 services indépendants (api, dashboard, mlflow, minio, postgres,
prometheus, grafana) a permis de reconstruire un service sans toucher aux autres.
`docker compose up -d --build dashboard` en 30 secondes.

### Découpage Agile par thème technique
Aligner les sprints sur les thèmes techniques (infrastructure, EDA, ML, DL, industrialisation)
a simplifié le reporting : chaque sprint produit un livrable fonctionnel indépendant et
un notebook d'analyse documenté.

---

## Ce qui a été difficile

### CNN sans GPU (Windows natif)
TensorFlow ne supporte pas le GPU sur Windows natif (>=2.11). L'entraînement du CNN
(51 550 images × 40 epochs) a pris plusieurs heures sur CPU. Décision prise : accepter
le surcoût et utiliser EarlyStopping agressif.

**Enseignement :** Pour les prochains projets DL sur Windows, utiliser WSL2 ou un
conteneur Docker avec CUDA.

### Configuration Spark sur Windows
La configuration de Hadoop sur Windows (HADOOP_HOME, hadoop.dll, winutils.exe) a coûté
une journée supplémentaire au sprint 1. Problème spécifique à l'environnement de
développement, pas au code.

**Enseignement :** Prévoir +1 jour buffer pour la configuration des outils Big Data
sur environnement non-Linux.

### Régression CNN (R² = 0.28)
La régression directe depuis les pixels d'images Kirsch a donné des résultats très faibles
(R² = 0.28 vs 0.99 pour LightGBM). Ce n'est pas un échec : c'est un résultat pédagogique
attendu qui démontre pourquoi les features physiques sont indispensables.

**Enseignement :** Les données non-structurées (images) seules ne suffisent pas pour
la régression physique — les features physiques calculées sont irremplaçables.

---

## Adaptations en cours de projet

| Moment | Problème rencontré | Adaptation |
|--------|-------------------|------------|
| Sprint 1 | Hadoop non configuré sur Windows | +1j de configuration, ajout d'instructions dans le notebook |
| Sprint 3 | Optuna 60 trials trop long sur CPU | Parallélisation Optuna avec `n_jobs=-1` |
| Sprint 4 | TF GPU indisponible | EarlyStopping patience=5, batch_size=64 |
| Sprint 5 | Limite slider traction 2.2 MPa trop restrictive | Remplacement par number_input libre + warning physique |
| Sprint 5 | Sliders E et ν non contraints par matériau | Ajout de `_MAT_PROPS` avec plages par matériau |

---

## Résultats finaux par bloc

### Bloc 1 — Infrastructure
- **61 550 simulations** dans MinIO `raw-simulations` (9 sources)
- Pipeline ETL Spark : 19 partitions, 20 cœurs, 61 550 lignes → 0 doublon, 0 null critique
- Data Warehouse : `processed-simulations/warehouse.parquet` (9.0 MB, 10 fichiers Parquet)

### Bloc 2 — EDA
- Corrélation Pearson la plus forte : `traction_pa` → `max_von_mises_pa` (r ≈ 0.38)
- `young_modulus_pa` → `max_displacement_m` inversement corrélé (r ≈ -0.38)
- Valeurs aberrantes IQR : faible proportion, physiquement valides

### Bloc 3 — Machine Learning
| Métrique | max_displacement_m | max_von_mises_pa |
|---------|-------------------|-----------------|
| R²(log) test | 0.9258 | 0.9829 |
| R²(orig) test | 0.9798 | 0.8652 |
| MAPE | 3.96% | 2.80% |
| IC 95% couverture | 96.8% | 96.6% |
| Diagnostic | OK (pas d'overfitting) | OK |

### Bloc 4 — Deep Learning
| Modèle | Tâche | Résultat |
|--------|-------|---------|
| CNN scratch | Classification 3 classes | **98.5% accuracy** |
| MobileNetV2 | Classification 3 classes | 97.0% accuracy |
| CNN Regression | Régression σ_VM depuis pixels | R² = 0.28 (attendu bas) |

### Bloc 5 — Industrialisation
- 7 services Docker tous `UP` et `healthy`
- API latence P95 : **< 100 ms** (objectif < 200 ms)
- MLflow : 7 runs tracés, modèle `v20260603_102835` en production
- Grafana : alertes Slack configurées sur latence P95 > 500 ms

---

## Enseignements clés

1. **La normalisation physique des targets est la décision la plus impactante** du projet.
   Elle a permis d'atteindre R²(log) = 0.98 et de rendre le modèle valide hors plage d'entraînement.

2. **LightGBM + features physiques >> CNN + pixels bruts** pour la régression mécanique.
   Les connaissances domaine (formules de Peterson, déformation linéaire) remplacent
   avantageusement les représentations apprises automatiquement.

3. **Le monitoring n'est pas optionnel.** Sans Prometheus + Grafana, une dégradation de
   performance en production est invisible jusqu'à ce qu'un utilisateur se plaigne.

4. **La reproductibilité se conçoit dès le début**, pas en fin de projet. MLflow + manifest.json
   + Docker compose permettent de recréer l'environnement exact de n'importe quel run passé.
