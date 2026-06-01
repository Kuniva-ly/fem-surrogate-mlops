# Gestion des Risques — FEM Surrogate ML

**Retour :** [Gestion de projet](./bloc6_gestion_projet.md) | [Sommaire](./sommaire.md)

---

## Grille d'évaluation

| Probabilité \ Impact | Faible | Moyen | Élevé |
|---------------------|--------|-------|-------|
| **Élevée** | 🟡 Moyen | 🔴 Critique | 🔴 Critique |
| **Moyenne** | 🟢 Faible | 🟡 Moyen | 🔴 Critique |
| **Faible** | 🟢 Faible | 🟢 Faible | 🟡 Moyen |

---

## Registre des risques

### R1 — Extrapolation hors plage d'entraînement
| | |
|---|---|
| **Catégorie** | Qualité modèle |
| **Description** | Le modèle est appelé avec des paramètres hors des plages d'entraînement (traction > 2.2 MPa pour acier, E hors plage matériau) |
| **Probabilité** | Élevée (utilisateur libre de saisir des valeurs) |
| **Impact** | Élevé (prédictions physiquement incorrectes sans avertissement) |
| **Niveau** | 🔴 Critique |
| **Mitigation** | Warning orange dans le dashboard si traction > max entraînement par matériau ; sliders E et ν contraints par plage physique du matériau sélectionné |
| **Statut** | ✅ Mitigé (implémenté en dashboard) |

---

### R2 — Dépassement de la limite élastique
| | |
|---|---|
| **Catégorie** | Validité physique |
| **Description** | L'utilisateur applique une traction telle que σ_VM > σ_yield → régime plastique, modèle invalide |
| **Probabilité** | Moyenne (cas extrêmes intentionnels) |
| **Impact** | Élevé (résultat affiché sans avertissement = décision erronée) |
| **Niveau** | 🔴 Critique |
| **Mitigation** | Calcul du facteur de sécurité (σ_yield / σ_VM) ; badge rouge "Plastic regime" si SF < 1 ; jauge colorée |
| **Statut** | ✅ Mitigé (implémenté en dashboard) |

---

### R3 — Écart solveur / essais physiques
| | |
|---|---|
| **Catégorie** | Qualité données |
| **Description** | Les données d'entraînement sont des résultats de calculs FEM (FEniCS/DOLFINx), qui peuvent présenter un écart par rapport à des mesures expérimentales réelles (tolérance du maillage, hypothèses de modélisation 2D) |
| **Probabilité** | Faible (FEM est la référence industrielle, validé sur des milliers de cas) |
| **Impact** | Moyen (biais résiduel lié aux hypothèses du solveur : élasticité linéaire, 2D plan) |
| **Niveau** | 🟡 Moyen |
| **Mitigation** | Comparaison avec formules analytiques de Peterson dans le dashboard ; documentation des hypothèses du solveur (plan stress, maillage 120×24) |
| **Statut** | ✅ Documenté |

---

### R4 — Absence de GPU (entraînement CNN lent)
| | |
|---|---|
| **Catégorie** | Infrastructure |
| **Description** | TensorFlow natif Windows ne supporte pas le GPU → CNN entraîné sur CPU uniquement |
| **Probabilité** | Certaine (contrainte Windows) |
| **Impact** | Faible (entraînement plus lent mais résultat identique) |
| **Niveau** | 🟡 Moyen |
| **Mitigation** | Acceptance du temps d'entraînement plus long ; EarlyStopping pour limiter les epochs inutiles ; LightGBM (pas de GPU requis) pour le modèle de production |
| **Statut** | ✅ Accepté |

---

### R5 — Dérive du modèle en production (model drift)
| | |
|---|---|
| **Catégorie** | Production |
| **Description** | Si de nouvelles géométries ou matériaux sont utilisés, les prédictions se dégradent silencieusement |
| **Probabilité** | Faible (domaine physique stable) |
| **Impact** | Élevé (décisions de conception erronées) |
| **Niveau** | 🟡 Moyen |
| **Mitigation** | Prometheus collecte les métriques de latence et volume de requêtes ; Grafana alerte si anomalie ; versionnage MLflow pour rollback rapide |
| **Statut** | ✅ Monitoring en place |

---

### R6 — Non-conformité RGPD
| | |
|---|---|
| **Catégorie** | Réglementaire |
| **Description** | Stockage d'identifiants de simulations non pseudonymisés |
| **Probabilité** | Faible (données purement physiques) |
| **Impact** | Moyen (traçabilité et conformité réglementaire) |
| **Niveau** | 🟢 Faible |
| **Mitigation** | Pseudonymisation SHA-256 irréversible des `simulation_id` dans le pipeline ETL Spark avant tout chargement dans le Data Warehouse |
| **Statut** | ✅ Implémenté (bloc 1) |

---

### R7 — Régression de performance lors d'un réentraînement
| | |
|---|---|
| **Catégorie** | Qualité modèle |
| **Description** | Un nouveau run d'entraînement produit un modèle moins bon que la version en production |
| **Probabilité** | Moyenne |
| **Impact** | Élevé (dégradation silencieuse de la qualité) |
| **Niveau** | 🟡 Moyen |
| **Mitigation** | MLflow compare automatiquement les métriques des runs ; promotion manuelle vers "Production" uniquement si R² > seuil ; manifest.json trace la version déployée |
| **Statut** | ✅ Processus défini |

---

## Synthèse

| Risque | Niveau initial | Niveau résiduel | Action |
|--------|---------------|-----------------|--------|
| R1 Extrapolation | 🔴 Critique | 🟢 Faible | Warning dashboard |
| R2 Plasticité | 🔴 Critique | 🟢 Faible | Badge + jauge SF |
| R3 Écart solveur / essais | 🟡 Moyen | 🟢 Faible | Documentation |
| R4 Sans GPU | 🟡 Moyen | 🟢 Faible | EarlyStopping |
| R5 Model drift | 🟡 Moyen | 🟢 Faible | Monitoring Grafana |
| R6 RGPD | 🟢 Faible | 🟢 Faible | SHA-256 ETL |
| R7 Régression perf | 🟡 Moyen | 🟢 Faible | MLflow registry |
