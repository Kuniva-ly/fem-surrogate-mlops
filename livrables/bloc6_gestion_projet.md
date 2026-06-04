# Gestion de Projet — FEM Surrogate ML

**Retour au sommaire :** [Sommaire](./sommaire.md)

---

## Documents

| Document | Description | Lien |
|----------|-------------|------|
| Cahier des charges | Contexte, besoins, objectifs SMART, périmètre | [bloc6_cahier_des_charges.md](./bloc6_cahier_des_charges.md) |
| Planning & Sprints | Découpage Agile, Gantt, jalons, livrables | [bloc6_planning_sprints.md](./bloc6_planning_sprints.md) |
| Gestion des risques | Identification, évaluation, mitigation | [bloc6_gestion_risques.md](./bloc6_gestion_risques.md) |
| Rétrospective | Bilan, écarts, enseignements, résultats finaux | [bloc6_retrospective.md](./bloc6_retrospective.md) |

---

## Résumé exécutif

**Projet :** FEM Surrogate ML — Modèle de substitution pour simulations éléments finis

**Problème métier :** Les simulations numériques FEM (FEniCS/DOLFINx) d'une plaque métallique
sous traction prennent plusieurs **minutes** par cas. Un bureau d'études qui explore des milliers
de configurations géométriques ne peut pas se permettre ce coût computationnel en phase de
conception préliminaire.

**Solution apportée :** Un modèle surrogate LightGBM entraîné sur 61 550 simulations FEM
qui prédit la contrainte de Von Mises maximale et le déplacement maximal en **moins de 50 ms**,
avec une précision R²(log) ≥ 0.98 — soit un gain de vitesse de l'ordre de **×1000 à ×10 000**.

**Stack déployée :**

```
Données FEM (FEniCS) → Data Lake MinIO → ETL Spark → Feature Engineering
→ LightGBM + Optuna → FastAPI :8000 → Streamlit :8501 → Monitoring Grafana
```

---

## Métriques de succès atteintes

| Indicateur | Cible | Réalisé | Statut |
|-----------|-------|---------|--------|
| R² max_von_mises_pa (log) | > 0.95 | **0.9829** | ✅ |
| R² max_displacement_m (orig) | > 0.95 | **0.9798** | ✅ |
| MAPE von Mises | < 10% | **2.80%** | ✅ |
| Latence API P95 | < 200 ms | **< 100 ms** | ✅ |
| CNN classification accuracy | > 90% | **98.5%** | ✅ |
| Couverture données IC 95% | > 95% | **96.7%** | ✅ |
| Pipeline reproductible (Docker) | Oui | **7 services** | ✅ |
