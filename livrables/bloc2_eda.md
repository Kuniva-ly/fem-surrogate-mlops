# Livrable Bloc 2 — Analyse Exploratoire des Données (EDA)

**Notebook :** [eda_bloc2.ipynb](../notebooks/eda_bloc2.ipynb)
**Retour au sommaire :** [Sommaire](./sommaire.md)

---

## Compétences RNCP couvertes

| Code | Intitulé |
|------|---------|
| C2.1 | Analyser et comprendre un jeu de données |
| C2.2 | Identifier les distributions et les anomalies |
| C2.3 | Visualiser les relations entre variables |
| C2.4 | Préparer les données pour la modélisation |

---

## Objectif

Explorer les données de simulation FEM pour comprendre les distributions,
les corrélations et les patterns physiques avant modélisation.

---

## Données

- **61 550 simulations** FEM de plaques métalliques — 21 colonnes, 44.2 MB en mémoire
- **Variables d'entrée :** `length_m`, `height_m`, `young_modulus_pa`, `poisson_ratio`,
  `traction_pa`, `hole_radius_ratio`, `material_category`, `dimension_category`
- **Variables cibles :** `max_displacement_m`, `max_von_mises_pa`
- **Plages physiques :**
  - Déplacement : ~10⁻⁸ à 10⁻² m (6 décades)
  - Von Mises : ~10⁴ à 5×10⁸ Pa (4 décades)

---

## Analyses réalisées

### Valeurs manquantes

| Colonne | % manquant | Explication |
|---------|-----------|-------------|
| `hole_cx_ratio` | **66.7%** | Absent si géométrie sans trou |
| `hole_cy_ratio` | **66.7%** | Absent si géométrie sans trou |
| `geometry_type` | **49.6%** | Rempli via `source` en feature engineering |
| `hole_radius_ratio` | **33.4%** | Absent si géométrie sans trou |
| Colonnes critiques (targets, features numériques) | **0%** | ✓ |

### Valeurs aberrantes (IQR)

| Variable | Outliers |
|----------|---------|
| `max_displacement_m` | 6 765 (11.0%) |
| `max_von_mises_pa` | 3 286 (5.3%) |
| `height_m` | 1 105 (1.8%) |
| `poisson_ratio` | 412 (0.7%) |

### Corrélations Pearson (avec les cibles)

| Feature | vs max_von_mises_pa | vs max_displacement_m |
|---------|--------------------|-----------------------|
| `traction_pa` | **+0.376** | +0.229 |
| `young_modulus_pa` | -0.025 | **-0.379** |
| `length_m` | -0.008 | +0.192 |
| `height_m` | +0.120 | +0.096 |

> Les corrélations linéaires brutes sont faibles car les relations sont log-linéaires.
> En espace log₁₀ : `log10(traction_pa)` vs `log10(max_von_mises_pa)` → r ≈ **0.96**

### Feature engineering préliminaire
- `delta_theory = (σ/E) × L` — déplacement théorique (r = 0.96 avec target)
- `Kt_theory` — facteur de concentration de contrainte (Peterson)
- `sigma_net` — contrainte nette sur la section réduite

---

## Apprentissage non supervisé (Spark + PCA + KMeans)

- **PCA** : **4 composantes** suffisent pour expliquer **95%** de la variance des images
- **KMeans K=3** : chaque cluster correspond naturellement à un type de géométrie

| Cluster | with_hole | with_hole_moving | without_hole |
|---------|-----------|-----------------|--------------|
| 0 | 59 | 32 | 0 |
| 1 | 0 | 0 | 123 |
| 2 | 157 | 229 | 0 |

- **DBSCAN** : 2 clusters détectés — **0 outlier** (0.0%)

---

## Visualisations clés

- Distribution des 3 géométries
- Scatter `log10(delta_theory)` vs `log10(max_displacement_m)`
- Boxplots Von Mises par matériau et dimension
- Corrélogramme complet (42 features)
- Analyse Spark : statistiques agrégées par source
