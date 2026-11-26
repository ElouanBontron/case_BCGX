# Modèle churn ClientCo

Prédit, chaque mois, les clients réguliers qui vont partir ou fortement réduire leurs achats, et produit
la liste d'appels à transmettre aux commerciaux.

Le CSV doit être dans `data/dataset_sales_2026.csv` (à la racine du dépôt). Le premier lancement lit le
dataset et crée `cache/agregats.pkl` ; les suivants réutilisent ce cache.

## Lancer

Ouvrir les notebooks dans l'ordre et faire « Run All », ou depuis la racine du dépôt :

```bash
pip install pandas numpy scikit-learn matplotlib xgboost
python modele_churn/churn_1_modele_mensuel.py               # modèle et chiffres de la présentation
python modele_churn/churn_2_importance_variables.py          # features qui comptent (après le 1)
python modele_churn/churn_3_comparaison_mensuel_bimensuel.py # liste mensuelle ou toutes les 2 semaines
python modele_churn/churn_4_frequence_optimisee.py          # fréquence de scoring comme hyperparamètre
```

| Fichier | Rôle |
|---|---|
| `churn_commun.py` | Lecture du CSV, photos clients, features, cibles, modèles, métriques, simulation des relances |
| `churn_1_modele_mensuel` | Modèle retenu : liste le 1er du mois, résultats de la présentation (`resultats/mensuel/`) |
| `churn_2_importance_variables` | Matrice d'importance des features et corrélations (`resultats/importance/`) |
| `churn_3_comparaison_mensuel_bimensuel` | Mensuel vs toutes les 2 semaines, à capacité égale (`resultats/frequences/`) |
| `churn_4_frequence_optimisee` | Pas de 7, 14, 21 ou 28 jours choisi sur validation, mesuré sur test |

## Le modèle en bref

- **Photo** : à une date T, une ligne par client régulier (≥ 3 jours d'achat sur 12 mois, pas déjà parti).
  Les features n'utilisent que le passé (avant T), les cibles que le futur (après T).
- **Cible** : départ total (aucun achat pendant 3 × son rythme habituel, entre 90 et 180 jours) **ou** forte
  baisse (CA du trimestre < 50 % du même trimestre N-1 et du trimestre moyen).
- **28 features** : montants, fréquence, tendances, saisonnalité, rythme d'achat, produits, canaux, retours,
  taille de l'agence. Définitions : `DESCRIPTION_VARIABLES` dans `churn_commun.py`.
- **Modèles comparés** : règle simple (silence vs rythme habituel), régression logistique, XGBoost.
- **Découpage** : 80 % des clients pour apprendre (photos d'oct. 2023 à janv. 2024), 20 % jamais vus
  pour tester (photo du 1er mars 2024).
- **Liste d'appels** : clients classés par probabilité × CA de référence, coupés à la capacité
  (200 commerciaux × 20 contacts par mois, rapportés à la part du groupe couverte par le dataset).
- **Métriques** : CA à risque capté par la liste (principale), précision, rappel sur les gros clients,
  AUC, PR-AUC, score de Brier, calibration, matrice de confusion, intervalle de confiance à 90 %.

## Choix de modélisation

- Le mois de l'année n'est pas une feature : l'apprentissage couvre octobre à janvier et le test mars.
  La saisonnalité passe par des features propres à chaque client (même trimestre N-1, indice saisonnier).
- Un jour avec uniquement des retours (CA net ≤ 0) ne compte pas comme un jour d'achat.
- Les tendances et l'indice saisonnier sont bornés entre 0 et 10.
- Les fenêtres sont en jours glissants, ce qui permet une photo à n'importe quelle date.

## Limites

- Pas de séparation stricte dans le temps entre les cibles d'apprentissage et le test (2 ans de données
  ne le permettent pas avec 12 mois d'historique et jusqu'à 180 jours de cible) : compensé par des
  clients de test jamais vus.
- La taille de l'agence est calculée sur toute la période (attribut quasi fixe).
- La valeur annuelle suppose que le churn d'un trimestre est représentatif de l'année (× 4).
