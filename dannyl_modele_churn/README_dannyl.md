# Dannyl MOUHEB – Modèle churn ClientCo

Le CSV doit être dans `data/dataset_sales_2026.csv` (à la racine du dépôt).

python dannyl_modele_churn/dannyl_1_modele_mensuel.py               # le modèle et les chiffres de la présentation
python dannyl_modele_churn/dannyl_2_importance_variables.py          # quelles features comptent (après le 1)
python dannyl_modele_churn/dannyl_3_comparaison_mensuel_bimensuel.py # liste tous les mois ou toutes les 2 semaines ?
python dannyl_modele_churn/dannyl_4_frequence_optimisee.py          # la fréquence comme hyperparamètre


Le premier lancement lit le dataset et crée `cache/agregats.pkl`

| Fichier | Rôle |
|---|---|
| `dannyl_*.ipynb` | Les 4 notebooks : explications, lancement et affichage des résultats |
| `dannyl_commun.py` | Lecture du CSV, photos clients, features, cibles, modèles, métriques, simulation des relances |
| `dannyl_1_modele_mensuel.py` | Modèle retenu : liste le 1er du mois, résultats de la présentation (`resultats/mensuel/`) |
| `dannyl_2_importance_variables.py` | Matrice d'importance des features, corrélations (`resultats/importance/`) |
| `dannyl_3_comparaison_mensuel_bimensuel.py` | Mensuel vs bimensuel à capacité égale (`resultats/frequences/`) |
| `dannyl_4_frequence_optimisee.py` | Pas de 7 / 14 / 21 / 28 jours choisi sur validation, mesuré sur test |

## Le modèle en bref

- **Photo** : à une date T, une ligne par client régulier (≥ 3 jours d'achat sur 12 mois, pas déjà parti).
  Les features n'utilisent que le passé (avant T), les cibles que le futur (après T).
- **Cible** : départ total (aucun achat pendant 3 × son rythme habituel, entre 90 et 180 jours) **ou** forte
  baisse (CA du trimestre < 50 % du même trimestre N-1 et du trimestre moyen).
- **28 features** en 7 familles : montants, fréquence, tendances, saisonnalité du client, rythme d'achat,
  comportement (produits, canaux, retours), taille de l'agence. Liste et définitions : `DESCRIPTION_VARIABLES`
  dans `dannyl_commun.py`.
- **Modèles comparés** : règle simple (silence vs rythme habituel), régression logistique, XGBoost.
- **Découpage** : 80 % des clients pour apprendre (photos d'oct. 2023 à janv. 2024), 20 % jamais vus
  pour tester (photo du 1er mars 2024).
- **Liste d'appels** : clients classés par probabilité × CA de référence, coupés à la capacité
  (200 commerciaux × 20 contacts par mois, rapportés à la part du groupe couverte par le dataset).
- **Métriques** : CA à risque capté par la liste (principale), précision, rappel sur les gros clients,
  AUC, PR-AUC, score de Brier, calibration, matrice de confusion, intervalle de confiance à 90 %.

## Corrections par rapport à la version du CTM

1. Le mois de l'année n'est plus une feature : l'apprentissage ne voit qu'octobre à janvier et le test
   est en mars, le modèle ne pouvait rien en tirer. La saisonnalité reste portée par des features propres
   à chaque client (même trimestre N-1, indice saisonnier).
2. Un jour avec uniquement des retours (CA net ≤ 0) ne compte plus comme un jour d'achat (rythme,
   récence, prochain achat), comme dans le code d'Elouan.
3. Les tendances et l'indice saisonnier sont bornés entre 0 et 10 (certains ratios dépassaient 25 000).
4. Les fenêtres sont en jours glissants, ce qui permet de prendre une photo à n'importe quelle date
   (indispensable pour comparer les fréquences).

## Limites connues

- Pas de séparation stricte dans le temps entre les cibles d'apprentissage et le test (2 ans de données
  ne le permettent pas avec 12 mois d'historique et jusqu'à 180 jours de cible) : compensé par des
  clients de test jamais vus.
- La taille de l'agence est calculée sur toute la période (attribut quasi fixe).
- La valeur annuelle suppose que le churn d'un trimestre est représentatif de l'année (× 4).
