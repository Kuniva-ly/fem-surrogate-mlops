# Livrable Bloc 3 — Machine Learning Supervisé & Non Supervisé

**Notebook :** [03_ml_bloc3.ipynb](../notebooks/03_ml_bloc3.ipynb)
**Retour au sommaire :** [Sommaire](./sommaire.md)

---

## Compétences RNCP couvertes

| Code | Intitulé |
|------|---------|
| C3.1 | Préparer les données pour l'apprentissage automatique |
| C3.2 | Entraîner un modèle supervisé (régression/classification) |
| C3.3 | Mettre en œuvre un apprentissage non supervisé |
| C3.4 | Évaluer et interpréter les résultats d'un modèle ML — R², RMSE, MAPE, résidus, IC |

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

### Modèle : LightGBM
- **5-fold cross-validation** sur le train set
- Early stopping (patience=30) sur le val set pour le modèle final
- Objectif : minimiser RMSE(log₁₀)

### Résultats LightGBM (test set)

| Target | R²(log) | RMSE(log) | MAPE | R² CV moyen |
|--------|---------|-----------|------|-------------|
| max_displacement_m | **0.9258** | 0.0420 | **3.96%** | 0.8823 ± 0.0206 |
| max_von_mises_pa | **0.9829** | 0.0282 | **2.80%** | 0.9784 ± 0.0004 |

### Comparaison des modèles (test set)

| Modèle | R² displacement | MAPE displacement | R² von Mises | MAPE von Mises |
|--------|----------------|-------------------|-------------|----------------|
| Moyenne (baseline) | 0.00 | — | 0.00 | — |
| RandomForest | 0.8508 | 6.00% | 0.9561 | 4.49% |
| **LightGBM** | **0.9258** | **3.96%** | **0.9829** | **2.80%** |

### Contrôle du sur-apprentissage

| Target | R² train | R² val | R² test | Diagnostic |
|--------|---------|--------|---------|------------|
| max_displacement_m | 0.9686 | 0.8880 | 0.9258 | OK (gap train-test < 0.05) |
| max_von_mises_pa | 0.9951 | 0.9802 | 0.9829 | OK (gap train-test < 0.05) |

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

- **Test de Kolmogorov-Smirnov** : résidus **non-normaux** (p-value = 0.0000)
  → rejet attendu sur 7 700 points (KS très sensible) ; résidus quasi-gaussiens visuellement
- μ résidus ≈ 0 (pas de biais systématique)
- σ résidus : 0.0420 (displacement) / 0.0282 (von Mises)
- **IC 95%** basé sur ±1.96 × σ_résidus (voir notebook pour couverture réelle)
- **Test de sur-apprentissage** : gap train-test < 0.05 ✓

---

## Commande d'entraînement

```bash
python -m src.cli train --n-trials 60 --mlflow
```
