# Livrable Bloc 4 — Deep Learning sur Données Non Structurées

**Notebook :** [dl_bloc4.ipynb](../notebooks/dl_bloc4.ipynb)
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
| Test Accuracy | **97.6%** |
| Test Loss | 0.0897 |
| Paramètres totaux | 127 939 |
| Epochs | early stopping (patience=5) |

---

## Section 4 — Transfer Learning MobileNetV2 (C4.3)

- Base MobileNetV2 pré-entraînée ImageNet — 2 422 339 paramètres
- Phase 1 : tête seule entraînée (164 355 paramètres)
- Phase 2 : fine-tuning 30 dernières couches (lr=1e-4) — 1 690 755 paramètres

| Métrique | Valeur |
|----------|--------|
| Test Accuracy | **96.4%** |
| Test Loss | 0.1109 |

**Meilleur modèle : CNN from scratch** (97.6% > 96.4%)

---

## Section 5 — Évaluation Classification (C4.5)

### Rapport de classification (CNN scratch, test set)

| Classe | Précision | Rappel | F1 | Support |
|--------|-----------|--------|-----|---------|
| with_hole | 0.94 | 1.00 | 0.97 | 3 096 |
| with_hole_moving | 1.00 | 0.94 | 0.97 | 3 044 |
| without_hole | 1.00 | 1.00 | 1.00 | 1 566 |
| **weighted avg** | **0.98** | **0.98** | **0.98** | **7 706** |

Erreurs de classification : **187 / 7 706 (2.4%)**

---

## Section 6 — CNN Régression (C4.5)

Architecture : même backbone CNN scratch, tête `Dense(2, linear)` (multi-output)

| Target | R²(log) | RMSE(log) | MAPE |
|--------|---------|-----------|------|
| max_von_mises_pa | **0.255** | 0.328 | 76.6% |
| max_displacement_m | **0.076** | 0.539 | 185.6% |

### Comparaison CNN vs LightGBM

| Modèle | Target | R²(log) | MAPE |
|--------|--------|---------|------|
| CNN Regression (pixels) | max_von_mises_pa | 0.255 | 76.6% |
| CNN Regression (pixels) | max_displacement_m | 0.076 | 185.6% |
| **LightGBM (42 features)** | max_von_mises_pa | **0.993** | **3.9%** |
| **LightGBM (42 features)** | max_displacement_m | **0.990** | **6.1%** |

> Le CNN prédit depuis les couleurs absolues (normalisation globale). Sa faible
> performance sur le déplacement s'explique par l'ambiguïté : deux images identiques
> peuvent avoir des déplacements très différents si `young_modulus_pa` diffère —
> information absente des pixels. Ce résultat justifie le choix du LightGBM en production (Bloc 5).

---

## Commande génération images

```bash
python scripts/generate_images.py --size 64
```
