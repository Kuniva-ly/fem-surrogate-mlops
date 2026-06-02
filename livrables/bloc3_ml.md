# Livrable Bloc 3 — Machine Learning Supervisé & Non Supervisé

**Notebook :** [ml_bloc3.ipynb](../notebooks/ml_bloc3.ipynb)
**Retour au sommaire :** [Sommaire](./sommaire.md)

---

## Compétences RNCP couvertes

| Code | Intitulé |
|------|---------|
| C3.1 | Préparer les données pour l'apprentissage automatique |
| C3.2 | Entraîner un modèle supervisé (régression/classification) |
| C3.3 | Mettre en œuvre un apprentissage non supervisé |
| C3.4 | Évaluer et interpréter les résultats d'un modèle ML |
| C3.5 | Analyser les résidus et valider les hypothèses statistiques |

---

## Objectif

Entraîner un **modèle de substitution** (surrogate) LightGBM capable de prédire
les sorties FEM (`max_displacement_m`, `max_von_mises_pa`) en quelques millisecondes
au lieu de plusieurs minutes de simulation numérique.

---

## Données

| Split | Lignes | % |
|-------|--------|---|
| Train | 36 086 | 70% |
| Val   | 7 758  | 15% |
| Test  | 7 706  | 15% |
| **Total** | **51 550** | 100% |

- **42 features physiques** après feature engineering (52 colonnes brutes → sélection)
- **3 catégorielles** encodées : `geometry_type`, `material_category`, `dimension_category`
- **2 targets** transformées en espace log₁₀

---

## Apprentissage supervisé (C3.1 – C3.2)

### Préparation
- `OrdinalEncoder` pour les variables catégorielles
- `StandardScaler` sur les features numériques
- Transformation `log10` des targets (6 décades → distribution quasi-normale)

### Modèle : LightGBM + Optuna
- **60 trials** Optuna (TPE Sampler)
- **5-fold cross-validation** sur le train set
- Objectif : minimiser RMSE(log₁₀)
- Contraintes de monotonie physique (ex: traction ↑ → stress ↑)

### Résultats LightGBM (test set)

| Target | R²(log) | RMSE(log) | MAPE | R² CV moyen |
|--------|---------|-----------|------|-------------|
| max_displacement_m | **0.9917** | 0.0493 | **5.1%** | 0.9871 |
| max_von_mises_pa | **0.9939** | 0.0292 | **3.8%** | 0.9904 |

### Contrôle du sur-apprentissage

| Target | R² train | R² val | R² test | Diagnostic |
|--------|---------|--------|---------|------------|
| max_displacement_m | 0.9966 | 0.9908 | 0.9917 | OK (écart < 0.01) |
| max_von_mises_pa | 0.9978 | 0.9925 | 0.9939 | OK (écart < 0.01) |

---

## Apprentissage non supervisé (C3.3)

### Génération d'images Kirsch
- 600 images 64×64 px (champs Von Mises analytiques)
- Colormap `hot` : noir = contrainte faible, blanc = contrainte maximale

### Réduction dimensionnelle
- **PCA** : **4 composantes** suffisent pour **95%** de variance expliquée

### Clustering
- **KMeans** (méthode du coude → K=3) → retrouve parfaitement les 3 géométries
- **DBSCAN** (eps=5, min_samples=5) → 2 clusters, **0 point aberrant** (0.0%)

---

## Analyse statistique des résidus (C3.5)

- **Test de Kolmogorov-Smirnov** : résidus **non-normaux** (p-value < 0.001)
  → queue lourde due aux simulations extrêmes (outliers physiques)
- **IC 95%** basé sur ±1.96 × σ_résidus :

| Target | σ résidus | IC 95% | Couverture réelle |
|--------|----------|--------|------------------|
| max_displacement_m | 0.0493 | ±0.097 | **96.8%** |
| max_von_mises_pa | 0.0292 | ±0.057 | **96.6%** |

- **Test de sur-apprentissage** : |R²_train − R²_val| < 0.01 ✓

---

## Commande d'entraînement

```bash
python -m src.cli train --n-trials 60 --mlflow
```
