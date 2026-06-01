# Cahier des Charges — FEM Surrogate ML

**Retour :** [Gestion de projet](./bloc6_gestion_projet.md) | [Sommaire](./sommaire.md)

---

## 1. Contexte et problématique métier

### Contexte

Les bureaux d'études en mécanique structurale utilisent la méthode des éléments finis (FEM)
pour évaluer la tenue mécanique de pièces sous chargement. Le solveur **FEniCS/DOLFINx**
résout les équations de l'élasticité linéaire sur un maillage 2D discrétisé.

### Problème

| Aspect | Situation actuelle | Impact |
|--------|-------------------|--------|
| Durée d'une simulation FEM | 2 à 10 minutes | Impossible d'explorer > 100 configurations/jour |
| Ressources nécessaires | CPU multi-cœur, mémoire > 4 Go | Coût infrastructure élevé |
| Expertise requise | FEniCS, PETSc, maillage | Barrière à l'entrée forte |
| Itérations de conception | Lentes (jours) | Prise de décision ralentie |

### Opportunité

Un modèle de substitution (surrogate) entraîné sur des simulations préalables peut
**reproduire les sorties FEM en quelques millisecondes** sans nécessiter le solveur.

---

## 2. Objectifs SMART

| # | Objectif | Spécifique | Mesurable | Atteignable | Réaliste | Temporel |
|---|---------|------------|-----------|-------------|----------|---------|
| O1 | Précision prédictive | Prédire σ_VM et δ_max | R² > 0.95 sur test set | Oui (LightGBM) | Oui (données FEM disponibles) | Sprint 3 |
| O2 | Vitesse d'inférence | Réponse API | < 200 ms (P95) | Oui (modèle tabulaire) | Oui | Sprint 5 |
| O3 | Couverture géométrique | 3 types de géométries | with_hole, without_hole, with_hole_moving | Oui | Oui (3 simulateurs) | Sprint 1 |
| O4 | Traçabilité ML | Reproduire n'importe quel run | MLflow + manifest.json | Oui | Oui | Sprint 5 |
| O5 | Accessibilité | Interface non-technique | Dashboard Streamlit | Oui | Oui | Sprint 5 |

---

## 3. Périmètre du projet

### Inclus

- Génération de données par simulation FEM (FEniCS/DOLFINx) sur 3 géométries
- Pipeline ETL distribué (Apache Spark + MinIO)
- Feature engineering physique (42 features : Peterson Kt, déformations, rapports géométriques)
- Entraînement LightGBM avec optimisation Optuna (60 trials)
- Classification d'images de champs de contrainte (CNN scratch + MobileNetV2)
- API REST de prédiction (FastAPI) avec authentification Basic Auth
- Dashboard interactif (Streamlit) avec comparaison analytique
- Stack de monitoring (Prometheus + Grafana + alertes Slack)
- Traçabilité complète (MLflow + registre de modèles)
- Conteneurisation Docker Compose (7 services)

### Exclus

- Simulation FEM 3D (hors périmètre : coût computationnel)
- Régime plastique (modèle entraîné uniquement en élasticité linéaire)
- Interface multi-utilisateurs avec authentification avancée
- Déploiement cloud public (AWS / GCP / Azure)

---

## 4. Parties prenantes

| Partie prenante | Rôle | Attentes |
|----------------|------|---------|
| Ingénieur mécanique | Utilisateur final | Prédictions rapides, interface intuitive |
| Data scientist | Développeur | Code propre, reproductibilité, MLflow |
| Responsable technique | Décideur | Validation des performances, go/no-go déploiement |
| Équipe DevOps | Opérateur | Stack Docker stable, monitoring |

---

## 5. Contraintes

### Techniques
- Environnement Windows (pas de GPU TensorFlow natif → CNN entraîné sur CPU)
- Élasticité linéaire uniquement (FEM incompatible avec la plasticité)
- Données issues de calculs numériques FEM (FEniCS/DOLFINx) — résultats de solveur, non d'essais physiques en laboratoire

### Réglementaires
- Pseudonymisation RGPD des `simulation_id` (SHA-256 irréversible) avant stockage Data Warehouse
- Aucune donnée à caractère personnel dans les simulations

### Qualité
- Tests unitaires sur les features engineered (`pytest`)
- Manifeste d'intégrité (git hash + SHA-256 dataset) à chaque artefact modèle
- Validation croisée 5-fold obligatoire avant déploiement

---

## 6. Budget prévisionnel

### Hypothèses

- **TJM data scientist / ML engineer** : 550 €/j (profil senior, marché France 2026)
- **Outils** : 100% open source (Python, LightGBM, TensorFlow, Spark, Docker, MLflow, MinIO, Prometheus, Grafana) → **0 € de licence**
- **Infrastructure** : machine locale 20 cœurs (pas de cloud en développement)

### Coûts de développement

| Sprint | Thème | Durée (j) | Coût (€) |
|--------|-------|-----------|---------|
| S1 | Infrastructure & Data Lake | 14 | 7 700 |
| S2 | Analyse Exploratoire | 7 | 3 850 |
| S3 | Machine Learning | 15 | 8 250 |
| S4 | Deep Learning | 13 | 7 150 |
| S5 | Industrialisation | 13 | 7 150 |
| **Total** | | **62** | **34 100 €** |

### Coûts de calcul (compute)

| Poste | Volume | Temps machine | Coût local | Équivalent AWS |
|-------|--------|--------------|------------|----------------|
| Simulations FEM (61 550 × ~3 min, 20 cœurs) | 61 550 sims | ~154 h wall time | ~15 € (électricité) | ~105 € (c5.4xlarge) |
| Entraînement LightGBM + Optuna (60 trials) | 60 runs | ~2 h | < 1 € | ~1 € |
| Entraînement CNN (CPU, ~80s/epoch) | ~70 epochs total | ~5 h | < 1 € | ~3 € |
| **Total compute** | | **~161 h** | **~17 €** | **~109 €** |

### Coûts récurrents (production)

| Service | Outil | Coût mensuel local | Coût mensuel AWS |
|---------|-------|-------------------|-----------------|
| API serving (FastAPI) | Docker local | 0 € | ~33 € (t3.medium) |
| Stockage artefacts (MinIO) | Docker local | 0 € | ~0,23 € (S3, 10 GB) |
| Monitoring (Prometheus + Grafana) | Docker local | 0 € | ~10 € |
| **Total mensuel** | | **0 €** | **~43 €/mois** |

### Budget total projet

| Poste | Montant |
|-------|---------|
| Développement (62 j × 550 €) | 34 100 € |
| Compute (local) | 17 € |
| Licences logicielles | 0 € |
| **Total** | **34 117 €** |

### Retour sur investissement (ROI)

Un ingénieur lançant 100 simulations FEM/jour économise :
- 100 × 3 min = 5 h/jour de temps machine libéré
- Accélération exploration : ×3 600 (3 min → 50 ms)
- **À 550 €/j** : économie ≈ 275 €/jour dès que le surrogate remplace les simulations exploratoires
- **Point mort** : 34 100 / 275 ≈ **124 jours ouvrés (~6 mois)**

---

## 7. Livrables attendus

| Livrable | Format | Critère d'acceptation |
|---------|--------|----------------------|
| Notebooks d'analyse | `.ipynb` exécutés | Toutes cellules sans erreur |
| API de prédiction | FastAPI Docker | `/health` retourne `status: ok` |
| Dashboard | Streamlit Docker | Prédiction < 200 ms |
| Modèle versionné | MLflow registry | R² test > 0.95 sur les 2 cibles |
| Stack monitoring | Prometheus + Grafana | P95 latence visible en temps réel |
| Documentation | Markdown (`livrables/`) | Tous les blocs documentés |
