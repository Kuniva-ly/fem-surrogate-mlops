# Gouvernance du projet

## Objectif
Definir les regles de pilotage du projet pour garantir qualite, tracabilite et reproductibilite.

## Portee
Cette gouvernance couvre:
- generation des donnees (`with_hole`, `without_hole`)
- validation des datasets
- entrainement des modeles
- exposition API et visualisation Dash

## Roles (MVP)
- Product/Tech Lead: priorites, arbitrages, roadmap
- Data/ML Engineer: data pipeline, entrainement, suivi metriques
- Platform Engineer (ou role combine): MinIO, Docker, MLflow, deployment

## Regles de donnees
- Separation stricte des flux `with_hole` et `without_hole`
- Pas d'ecrasement des partitions (versioning par date et version)
- Validation obligatoire avant upload vers zone validee
- Separation stricte des zones:
  - `raw-simulations`: donnees brutes
  - `processed-simulations`: donnees preparees
  - `features`: artefacts pseudo feature store
- Conventions de prefixes MinIO:
  - raw: `<geometry>/<data_version>/date=YYYY-MM-DD`
  - processed: `<geometry>/date=YYYY-MM-DD`
  - features: `<feature_group>/<feature_version>/<geometry>/date=YYYY-MM-DD`

## Regles de modeles
- 1 modele par geometrie (`model_with_hole`, `model_without_hole`)
- Tracking des runs dans MLflow (params, metriques, artefacts)
- Promotion d'un modele seulement apres verification val/test
- Reference explicite a la version de donnees utilisee

## Qualite et controle
- Validation automatique des schemas/ranges avant training
- Journalisation des executions batch
- Conservation des metriques de baseline et des features utilisees

## Gestion du changement
- Toute evolution de schema doit etre documentee dans `docs/data_contract.md`
- Toute nouvelle convention (bucket, endpoint, variable d'env) doit etre synchronisee dans:
  - `README.md`
  - `.env.example`
  - `docs/scripts_overview.md`

## Risques principaux
- Melange des flux `with_hole` et `without_hole`
- Drift des donnees non suivi
- Incoherence entre doc et implementation

## Revue minimale par iteration
- Donnees: validateur passe
- Modele: metriques val/test acceptables et stables
- API: endpoint fonctionnel + payload valide
- Doc: commandes et chemins a jour
