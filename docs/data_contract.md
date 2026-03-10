# Data Contract — sim_v1 family

Chaque enregistrement correspond à une simulation FEM.

---

## Schema des données brutes

### Champs requis

| Champ | Type | Contrainte |
|---|---|---|
| `simulation_id` | string (UUID) | unique par batch |
| `timestamp` | string ISO-8601 UTC | non null |
| `material_category` | string | non vide (ex. steel/aluminum/titanium) |
| `dimension_category` | string | non vide (ex. small/medium/large) |
| `length_m` | float | > 0 |
| `height_m` | float | > 0 |
| `young_modulus_pa` | float | > 0 |
| `poisson_ratio` | float | ∈ (0, 0.5) |
| `traction_pa` | float | ≥ 0 |
| `mesh_nx` | int | ≥ 8 |
| `mesh_ny` | int | ≥ 4 |
| `max_displacement_m` | float | ≥ 0 — **cible ML** |
| `max_von_mises_pa` | float | ≥ 0 — **cible ML** |
| `solver_name` | string | ex. dolfinx, proxy |
| `solver_version` | string | non vide |
| `data_version` | string | ex. sim_v1, sim_v2_moving_hole |

### Champs optionnels

| Champ | Type | Contrainte | Présent pour |
|---|---|---|---|
| `hole_radius_ratio` | float | ∈ (0, 0.5) | `with_hole`, `with_hole_moving` |
| `hole_cx_ratio` | float | ∈ (0, 1) | `with_hole_moving` |
| `hole_cy_ratio` | float | ∈ (0, 1) | `with_hole_moving` |
| `geometry_type` | string | `with_hole` \| `without_hole` \| `with_hole_moving` | recommandé pour tous |

**Note :** `mesh_nx` et `mesh_ny` sont présents dans les données brutes mais **exclus des features ML**
(constantes à 120×24 sur tout le dataset — variance nulle, biais potentiel).

---

## Règles de validation (`src/processing/validate.py`)

1. Pas de nulls sur les champs requis
2. `simulation_id` unique dans le batch
3. Plages numériques conformes (voir tableau ci-dessus)
4. Labels catégoriels non vides
5. Cohérence géométrique :
   - `with_hole` → doit avoir `hole_radius_ratio`, pas de `hole_cx/cy`
   - `without_hole` → pas de champs `hole_*`
   - `with_hole_moving` → doit avoir `hole_radius_ratio`, `hole_cx_ratio`, `hole_cy_ratio`

---

## Disposition des fichiers

### Données brutes

```
data/raw/
    sim_v1/date=YYYY-MM-DD/part-*.parquet           with_hole
    sim_v1_without_hole/date=YYYY-MM-DD/part-*.parquet
    sim_v2_moving_hole/date=YYYY-MM-DD/part-*.parquet
    sim_v1_wide/...                                  plages étendues
    sim_v1_without_hole_wide/...
    sim_v2_moving_hole_wide/...
```

### Données traitées

```
data/processed/
    train.parquet
    val.parquet
    test.parquet
data/features/advanced/
    features.parquet            dataset ML complet (42 features)
    feature_columns.txt         liste ordonnée des 42 features
```

### MinIO (optionnel)

```
raw-simulations/with_hole/sim_v1/date=YYYY-MM-DD/part-*.parquet
raw-simulations/without_hole/sim_v1_without_hole/date=YYYY-MM-DD/part-*.parquet
raw-simulations/with_hole_moving/sim_v2_moving_hole/date=YYYY-MM-DD/part-*.parquet
processed-simulations/with_hole/date=YYYY-MM-DD/{train,val,test}.parquet
features/stress_model/v1/with_hole/date=YYYY-MM-DD/features.parquet
```

---

## Features ML (42 colonnes)

Produites par `src/processing/build_features.py::engineer_features()`.

### Colonnes brutes conservées (11)
`material_category`, `dimension_category`, `length_m`, `height_m`,
`young_modulus_pa`, `poisson_ratio`, `traction_pa`, `hole_radius_ratio`,
`geometry_type`, `hole_cx_ratio`, `hole_cy_ratio`

### Features dérivées (31)

| Feature | Formule / Description |
|---|---|
| `area_m2` | L × H |
| `aspect_ratio` | L / H |
| `radius_abs` | r × min(L, H) |
| `d_over_W` | 2r_abs / H (ratio Peterson, clip [0, 0.95]) |
| `Kt_theory` | 3 − 3.13d/W + 3.66(d/W)² − 1.53(d/W)³ (Peterson) |
| `net_section_ratio` | 1 − d/W |
| `sigma_net` | traction / net_section_ratio |
| `epsilon` | traction / E (déformation élastique) |
| `delta_theory` | epsilon × L (élongation théorique) |
| `biaxial_factor` | 1 − ν² |
| `lig_left/right/top/bottom` | distances normalisées trou→bord |
| `lig_min` | min des 4 ligaments |
| `edge_ratio` | r / lig_min (proximité bord) |
| `eccentricity_x/y/` | distance centre trou → centre plaque |
| `eccentricity` | √(exc_x² + exc_y²) |
| `stress_amp_proxy` | Kt × sigma_net |
| `has_hole`, `has_moving_hole` | indicateurs binaires |
| `logE`, `logS` | log10(E), log10(traction) |
| `log_epsilon`, `log_delta_th` | transformés log pour tree models |
| `log_sigma_net`, `log_Kt` | idem |
| `log_lig_min`, `log_edge_ratio` | idem |
| `traction_over_E` | alias epsilon (compatibilité) |

---

## Politique de split reproductible

- Stratégie `hash` (défaut) : `SHA-256(simulation_id | seed)` → bucket [0, 1)
- Ratios par défaut : 70% train · 15% val · 15% test
- Le seed agit comme sel — même seed = même découpage sur toute relance
- Pas de fuite : l'encodeur catégoriel est fitté sur train uniquement

Configuré dans `configs/training.yaml` (section `features`).

---

## Artefacts du modèle avancé

Produits par `src/training/train_advanced.py` et enregistrés dans le registre.

### Structure d'une version

```
artifacts/models/lgbm_surrogate/v20260310_143022/
    lgbm_max_displacement_m.joblib   model + feature_cols + encoder + target
    lgbm_max_von_mises_pa.joblib     idem
    advanced_metrics.csv             tidy (target, split, r2_log, rmse_log, ...)
    advanced_metrics.json            imbriqué {target: {split: {metric: value}}}
    manifest.json                    reproductibilité
    checksums.sha256                 intégrité artefacts
    checksums.json                   idem (format JSON)
```

### Métriques d'évaluation (protocole figé)

| Métrique | Espace | Description |
|---|---|---|
| `r2_log` | log10 | Objectif d'optimisation (entraînement) |
| `rmse_log` | log10 | Erreur quadratique en log |
| `mae_log` | log10 | Erreur absolue en log |
| `r2_orig` | physique | Interprétation pratique |
| `rmse_orig` | physique | En unités réelles (m ou Pa) |
| `mape` | physique | Erreur relative moyenne |

### Manifest de run

```json
{
  "timestamp_utc": "2026-03-10T14:30:22+00:00",
  "git_commit": "06b9899...",
  "package_versions": { "lightgbm": "4.x", "scikit-learn": "1.x", ... },
  "config": { "n_trials": 60, "random_state": 42, ... },
  "dataset_fingerprint": "sha256hex...",
  "dataset_files": ["data/processed/train.parquet", ...]
}
```

---

## Mapping PostgreSQL (optionnel)

Table cible : `simulation_records`

| Colonne source | Colonne SQL | Remarque |
|---|---|---|
| `simulation_id` | `simulation_id` | clé upsert |
| `timestamp` | `ts` | colonne temporelle |
| `geometry_type` | `geometry_type` | `with_hole` \| `without_hole` \| `with_hole_moving` |
| Autres champs | identique | |
