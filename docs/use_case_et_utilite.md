# Use Case et utilite du projet

## Resume en une phrase
Cette plateforme remplace une partie des simulations mecaniques FEM longues et couteuses par un modele de Machine Learning de substitution, capable de predire rapidement les contraintes mecaniques pour accelerer les decisions d'ingenierie.

## Use case principal
Une equipe d'ingenierie concoit des pieces mecaniques (plaques, supports, composants structurels) et doit verifier leur resistance avant fabrication.

Flux classique (sans plateforme):
1. Definir une geometrie
2. Lancer une simulation FEM
3. Analyser les contraintes
4. Modifier la piece
5. Recommencer

Ce cycle prend souvent des heures, voire des jours, pour explorer suffisamment de variantes.

Flux cible (avec la plateforme):
1. Generer des jeux de donnees via simulations automatisees
   - variante `with_hole`
   - variante `without_hole`
2. Entrainer un modele surrogate sur ces donnees
3. Exposer le modele via API
4. Interroger l'API pour obtenir une prediction quasi immediate

## Problemes resolus
- Temps de calcul trop eleve pour iterer vite
- Cout infra important quand le volume de simulations augmente
- Difficultes a industrialiser un workflow notebook-only
- Manque de tracabilite sans pipeline, validation et suivi des experiments

## Utilite metier
- Acceleration du cycle de conception produit
- Reduction du time-to-validation sur les variantes de design
- Capacite a tester davantage d'options a cout controle
- Aide a la decision plus rapide pour les equipes R&D/industrialisation

## Utilite technique
- Pipeline reproductible: generation -> validation -> ingestion
- Data Lake S3-compatible (MinIO) pour centraliser les datasets
- Validation de la qualite des donnees avant usage ML
- Base solide pour training, suivi d'experiences (MLflow) et deploiement API
- Architecture compatible production (conteneurs, orchestration, observabilite)

## Ce que ce projet demontre
- Capacite a relier simulation scientifique et ML applique
- Maitrise Data Engineering + MLOps + API
- Vision "plateforme" et non simple script local

## Exemples concrets dans ce repo
- `fenics_projet/traction_plate_with_hole.py`: generation des donnees `with_hole`
- `fenics_projet/traction_plate_without_hole.py`: generation des donnees `without_hole`
- `src/processing/validate.py`: controle qualite des datasets
- `src/ingestion/upload_to_minio.py`: ingestion vers MinIO/S3
- `src/pipelines/daily_batch.py`: orchestration batch locale
- `src/training/train_baseline.py`: entrainement baseline et tracking MLflow

## Limites a garder en tete
- Le surrogate model ne remplace pas totalement la simulation FEM de reference
- La qualite de prediction depend fortement de la couverture des donnees d'entrainement
- Une validation physique/ingenierie reste necessaire pour les cas critiques

## Conclusion
L'utilite centrale du projet est de transformer un processus de simulation lent en une capacite de prediction rapide, industrialisable et tracable, afin de gagner du temps, reduire les couts et augmenter le nombre d'iterations de conception possibles.
