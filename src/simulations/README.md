# src/simulations — Générateurs de simulations FEM

Ce module contient les générateurs de données de simulation pour plaques sous traction.
Chaque script supporte deux backends : `fenics` (calcul réel via dolfinx) et `proxy` (formule analytique rapide).

---

## Scripts disponibles

| Script | Géométrie | Description |
|---|---|---|
| `traction_plate_with_hole.py` | `with_hole` | Plaque rectangulaire avec trou centré |
| `traction_plate_without_hole.py` | `without_hole` | Plaque pleine (sans trou) |
| `traction_plate_moving_hole.py` | `with_hole_moving` | Plaque avec trou à position variable |
| `traction_plate_with_hole_wide.py` | `with_hole` | Plages de paramètres élargies |
| `traction_plate_without_hole_wide.py` | `without_hole` | Plages de paramètres élargies |

---

## Usage

```powershell
# Proxy rapide (sans FEniCS)
.venv\Scripts\python -m src.simulations.traction_plate_with_hole `
    --n 1000 --seed 42 --out data/raw/sim_v1 --backend proxy

# FEniCS réel (Docker requis)
docker compose exec fenics python -m src.simulations.traction_plate_with_hole `
    --n 5000 --seed 42 --out data/raw/sim_v1 --backend fenics --chunk-size 2000

# Trou mobile
.venv\Scripts\python -m src.simulations.traction_plate_moving_hole `
    --n 1000 --seed 42 --out data/raw/sim_v2_moving_hole --backend proxy
```

## Paramètres communs

| Paramètre | Défaut | Description |
|---|---|---|
| `--n` | 1 | Nombre de simulations |
| `--seed` | 42 | Seed aléatoire (reproductibilité) |
| `--out` | — | Dossier de sortie (active le mode batch) |
| `--backend` | auto | `fenics` \| `proxy` \| `auto` |
| `--sampling-mode` | categorical | `categorical` \| `continuous` |
| `--chunk-size` | 5000 | Lignes par fichier parquet |
| `--mesh-nx` | 120 | Résolution maillage horizontal |
| `--mesh-ny` | 24 | Résolution maillage vertical |

## Sorties

Fichiers parquet partitionnés par date :

```
data/raw/sim_v1/
    date=2026-03-10/
        part-00000.parquet
        part-00001.parquet
        ...
```

Chaque ligne = une simulation. Colonnes conformes au contrat de données (`docs/data_contract.md`).

## Backends

**`proxy`** — Formule analytique (Peterson + corrections) :
- ~10 000 simulations/seconde
- Adapté au prototypage, tests, génération rapide

**`fenics`** — Solveur éléments finis réel (dolfinx + gmsh) :
- Précision physique complète (EF P1 Lagrange, solveur direct PETSc/MUMPS)
- Nécessite le conteneur Docker `fenics`
- ~1–10 simulations/seconde selon la résolution du maillage
