# Livrable Bloc 4 — Deep Learning sur Données Non Structurées

**Notebook :** [04_dl_bloc4.ipynb](../notebooks/04_dl_bloc4.ipynb)
**Retour au sommaire :** [Sommaire](./sommaire.md)

---

## Compétences RNCP couvertes

| Code | Intitulé |
|------|---------|
| C4.1 | Traiter des données non structurées (images) en tenseurs |
| C4.2 | Concevoir et entraîner un CNN from scratch |
| C4.3 | Appliquer le transfer learning (MobileNetV2) |
| C4.4 | Augmenter les données pour améliorer la généralisation |
| C4.5 | Évaluer un modèle de classification et de régression |

---

## Objectif

Entraîner des réseaux de neurones convolutifs sur les **images de champs de contrainte**
Von Mises pour deux tâches :
1. **Classification** : identifier le type de géométrie depuis l'image
2. **Régression** : prédire `max_von_mises_pa` et `max_displacement_m` depuis l'image

---

## Données images

| Split | Images |
|-------|--------|
| Train | 36 086 |
| Val | 7 758 |
| Test | 7 706 |
| **Total** | **51 550** |

- **Format :** PNG 64×64, 3 canaux RGB
- **Normalisation globale** : `pixel = kirsch_vm / global_vm_max`
  → même échelle de couleur absolue pour toutes les images (max pixel ≠ 1.0 confirmé)
- **Targets encodées dans les noms de fichiers** : `{sim_id}_V{vm_int}_D{disp_int}.png`
  → chargement 100% image, sans CSV

---

## Section 1 — Préparation des données (C4.1)

- Chargement via `image_dataset_from_directory` (classification)
- Chargement via `tf.data.Dataset.from_tensor_slices` + parse filename (régression)
- Normalisation pixels `[0, 255] → [0, 1]`
- Tenseurs : forme `(64, 64, 64, 3)` — batch=64

---

## Section 2 — Augmentation des données (C4.4)

Transformations appliquées sur le train set :
- `RandomFlip` horizontal et vertical
- `RandomRotation` ±10%
- `RandomZoom` ±10%
- `RandomTranslation` ±10% (x et y)

---

## Section 3 — CNN from Scratch (C4.2)

Architecture :
```
Input (64, 64, 3)
  → Conv2D(32) + BN + MaxPool(2)
  → Conv2D(64) + BN + MaxPool(2)
  → Conv2D(128) + BN + MaxPool(2)
  → GlobalAveragePooling2D
  → Dense(256, relu) + Dropout(0.4)
  → Dense(3, softmax)   [classification]
```

| Métrique | Valeur |
|----------|--------|
| Test Accuracy | **98.5%** |
| Test Loss | 0.0536 |
| Val Accuracy (best epoch) | 98.2% |
| Val Loss (best epoch) | 0.0622 |
| Paramètres totaux | 127 939 |
| Epochs | 29 entraînées, meilleur à l'époque 24 (EarlyStopping patience=5) |
| ReduceLROnPlateau | déclenché aux époques 11 (→ 5e-4), 19 (→ 2.5e-4), 27 (→ 1.25e-4) |

---

## Section 4 — Transfer Learning MobileNetV2 (C4.3)

- Base MobileNetV2 pré-entraînée ImageNet — 2 422 339 paramètres
- Phase 1 : tête seule entraînée (164 355 paramètres)
- Phase 2 : fine-tuning 30 dernières couches (lr=1e-4) — 1 690 755 paramètres

| Métrique | Phase 1 (tête seule) | Phase 2 (fine-tuning) |
|----------|----------------------|-----------------------|
| Test Accuracy | — | **97.0%** |
| Test Loss | — | 0.1021 |
| Val Accuracy (best) | 94.1% (époque 15) | 96.8% (époque 10) |
| Val Loss (best) | 0.1749 | 0.0962 |
| Epochs | 15 / 15 | 15 entraînées, best époque 10 (EarlyStopping) |

**Meilleur modèle : CNN from scratch** (98.5% > 97.0%)

---

## Section 5 — Évaluation Classification (C4.5)

### Rapport de classification (CNN scratch, test set)

| Classe | Précision | Rappel | F1 | Support |
|--------|-----------|--------|-----|---------|
| with_hole | 0.97 | 1.00 | 0.98 | 3 096 |
| with_hole_moving | 0.99 | 0.97 | 0.98 | 3 044 |
| without_hole | 1.00 | 1.00 | 1.00 | 1 566 |
| **macro avg** | **0.99** | **0.99** | **0.99** | **7 706** |
| **weighted avg** | **0.98** | **0.98** | **0.98** | **7 706** |

Erreurs de classification : **119 / 7 706 (1.5%)**

---

## Section 6 — CNN Régression (C4.5)

Architecture : même backbone CNN scratch, tête `Dense(2, linear)` (multi-output)

| Métrique entraînement | Valeur |
|-----------------------|--------|
| Epochs | 29 entraînées, meilleur à l'époque 23 (EarlyStopping patience=6) |
| ReduceLROnPlateau | déclenché aux époques 8, 12, 17, 26, 29 |
| Val loss (best) | 0.1894 | Val MAE (best) | 0.3277 |

### Résultats test set

| Target | R²(log) | RMSE(log) | MAPE |
|--------|---------|-----------|------|
| max_von_mises_pa | **0.279** | 0.322 | 83.2% |
| max_displacement_m | **0.083** | 0.537 | 170.2% |

### Comparaison CNN vs LightGBM

| Modèle | Target | R²(log) | RMSE(log) | MAPE |
|--------|--------|---------|-----------|------|
| CNN Regression (pixels) | max_von_mises_pa | 0.279 | 0.322 | 83.2% |
| CNN Regression (pixels) | max_displacement_m | 0.083 | 0.537 | 170.2% |
| **LightGBM (42 features)** | max_von_mises_pa | **0.9829** | **0.0282** | **2.80%** |
| **LightGBM (42 features)** | max_displacement_m | **0.9258** | **0.0420** | **3.96%** |

> Le CNN prédit depuis les couleurs absolues (normalisation globale). Sa faible
> performance sur le déplacement s'explique par l'ambiguïté : deux images identiques
> peuvent avoir des déplacements très différents si `young_modulus_pa` diffère —
> information absente des pixels. Ce résultat justifie le choix du LightGBM en production (Bloc 5).

---

## Commande génération images

```bash
python scripts/generate_images.py --size 64
```
