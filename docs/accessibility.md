# Accessibilite (MVP)

## Objectif
Assurer une interface API/Dash utilisable par un maximum d'utilisateurs, avec un niveau d'accessibilite pragmatique des la phase MVP.

## Perimetre
- API FastAPI (documentation et messages d'erreur)
- Dashboard Dash (formulaires, resultats, navigation)

## Principes
- Clarte: labels explicites, termes metier comprehensibles
- Robustesse: comportements predictibles en cas d'erreur
- Lisibilite: contraste suffisant et structure visuelle simple
- Navigation: parcours utilisable sans souris autant que possible

## Publics concernes (handicap)
- Handicap visuel: basse vision, daltonisme, usage lecteur d'ecran
- Handicap moteur: difficultes de precision souris, navigation clavier
- Handicap auditif: information non dependante du son
- Handicap cognitif: besoin de parcours simple, messages clairs, charge mentale reduite

## Exigences API
- Champs d'entree explicites (`geometry_type`, dimensions, materiau, charge)
- Messages d'erreur comprehensibles et actionnables
- Endpoint de sante `/health`
- Exemples de requetes dans la doc API

## Exigences Dash
- Champs avec labels visibles (pas seulement placeholders)
- Ordre de tabulation coherent
- Couleurs avec contraste suffisant
- Etat de chargement visible
- Affichage clair du modele utilise (`with_hole` ou `without_hole`)
- Aucun message critique transmis uniquement par couleur
- Focus clavier visible sur les composants interactifs

## Exigences specifiques par type de handicap
- Visuel:
  - Contraste texte/fond suffisant
  - Tailles de police lisibles
  - Compatibilite lecteur d'ecran (labels, titres, ordre logique)
- Moteur:
  - Utilisation complete au clavier
  - Zones cliquables suffisantes
  - Eviter interactions exigeant une precision fine
- Auditif:
  - Pas de dependance a des alertes sonores
  - Toute information importante existe en version textuelle
- Cognitif:
  - Formulaires courts et structures stables
  - Messages d'erreur explicites avec action attendue
  - Unites et termes metier systematiquement indiques

## Validation accessibilite (checklist MVP)
- Navigation clavier sur les champs principaux
- Verification de contraste texte/fond
- Messages d'erreur visibles et comprehensibles
- Titres et sections coherents

## Bonnes pratiques de contenu
- Eviter jargon non explique
- Utiliser des unites explicites (m, Pa, etc.)
- Afficher date/heure et version modele dans les resultats

## Limites MVP
- Audit WCAG complet non inclus a ce stade
- Objectif actuel: niveau "usable by default" avant durcissement final

## Evolutions fin de projet
- Audit accessibilite plus formel (WCAG 2.1 AA)
- Tests utilisateurs cibles
- Correctifs de contrastes, focus, et composants non conformes
