# Commandes pour generer plus de donnees

Depuis la racine du projet `c:\Users\Administrateur\Desktop\Projet_fil_rouge`.

## 1) Demarrer l'infra
```powershell
docker compose up -d
```

## 2) Generation with_hole (calcul reel FEniCS)
Exemple 10 000 simulations:
```powershell
docker compose exec fenics python fenics_projet/traction_plate_with_hole.py --n 10000 --seed 42 --out data/raw/sim_v1 --backend fenics --chunk-size 2000 --sampling-mode continuous --mesh-nx 120 --mesh-ny 24
```

## 3) Generation without_hole (proxy rapide)
Exemple 10 000 simulations:
```powershell
python fenics_projet/traction_plate_without_hole.py --n 10000 --seed 42 --out data/raw/sim_v1_without_hole --backend proxy --chunk-size 2000 --sampling-mode continuous --mesh-nx 120 --mesh-ny 24
```

## 4) Generation with_hole_moving (calcul reel FEniCS)
Exemple 10 000 simulations:
```powershell
docker compose exec fenics python fenics_projet/traction_plate_moving_hole.py --n 10000 --seed 42 --out data/raw/sim_v2_moving_hole --backend fenics --chunk-size 2000 --sampling-mode continuous --data-version sim_v2_moving_hole --mesh-nx 120 --mesh-ny 24
```

## 5) Validation
Remplace la date si besoin.
```powershell
python -m src.processing.validate --input data/raw/sim_v1/date=2026-02-11
python -m src.processing.validate --input data/raw/sim_v1_without_hole/date=2026-02-11
python -m src.processing.validate --input data/raw/sim_v2_moving_hole/date=2026-02-11
```

## 6) Upload vers MinIO
Remplace la date si besoin.
```powershell
python -m src.ingestion.upload_to_minio --local-path data/raw/sim_v1/date=2026-02-11 --bucket raw-simulations --prefix with_hole/sim_v1/date=2026-02-11
python -m src.ingestion.upload_to_minio --local-path data/raw/sim_v1_without_hole/date=2026-02-11 --bucket raw-simulations --prefix without_hole/sim_v1_without_hole/date=2026-02-11
python -m src.ingestion.upload_to_minio --local-path data/raw/sim_v2_moving_hole/date=2026-02-11 --bucket raw-simulations --prefix with_hole_moving/sim_v2_moving_hole/date=2026-02-11
```

## 7) Option batch (generer + valider + preparation + features + uploader)
Batch existant (with_hole):
```powershell
python -m src.pipelines.daily_batch --n 10000 --seed 42 --base data/raw/sim_v1 --backend fenics --chunk-size 2000 --upload-minio
```
Ce batch:
- genere et valide le lot `raw`
- construit `data/processed/date=YYYY-MM-DD` (nettoyage + agregation + preparation train/val/test)
- construit des artefacts `features` selectionnes pour les modeles dans `data/features/stress_model/v1/<geometry>/date=YYYY-MM-DD`
- upload le raw vers `raw-simulations`
- upload `processed` vers `processed-simulations/<geometry>/date=YYYY-MM-DD`
- upload `features` vers `features/stress_model/v1/<geometry>/date=YYYY-MM-DD` (pseudo feature store)

Options utiles du batch:
```powershell
python -m src.pipelines.daily_batch --n 10000 --seed 42 --base data/raw/sim_v1 --upload-minio --feature-group stress_model --feature-version v1
```

## 8) Convention recommandee: pseudo feature store (bucket `features`)
Ne pas confondre:
- `processed`: donnees preparees (pipeline data engineering)
- `features`: variables retenues pour un modele (decision ML/science)

Exemple orienté modele/version/date:
```text
features/
  stress_model/
    v1/
      date=2026-02-11/
    v2/
      date=2026-02-11/
```

Exemple orienté domaines de features:
```text
features/
  geometry_features/
  physics_features/
  derived_features/
```

But:
- garantir `training = serving` (meme feature, meme transformation)
- eviter le training-serving skew

Solutions "feature store" classiques en production:
- Feast
- Tecton
- Hopsworks
