# Data not included on the github (file too heavy), to use it dowload the csv file locally and put it in a data folder to the root

## XGBoost final et simulation des relances

Depuis la racine du projet, lancer le programme avec le CSV des ventes dans `data/`.

```shell
python churn_final.py
```

Le programme lit le CSV avec Polars, entraîne trois configurations XGBoost et écrit le backtest par boutique dans `data/churn_final/boutiques/`. Les options `--ventes` et `--sortie` permettent de changer les chemins. Le libellé magasin du fichier fourni est `at the store`. Pour une autre nomenclature, répéter `--canal-magasin` pour chaque libellé concerné.

Les valeurs par défaut sont définies dans `ConfigurationRelances` (`churn_boutiques.py`). `coefficient_relance` fixe la capacité et `taux_churn_evite_par_relance` fixe l’effet commercial supposé. Pour les modifier le temps d’une exécution, utiliser par exemple `python churn_final.py --coefficient-relance 0.30 --taux-churn-evite-par-relance 0.12`.

Les sommes de chiffre d’affaires utilisent le millionième d’euro pour rendre les agrégations parallèles reproductibles. Le réglage retenu maximise les détections à 14 jours en validation avec les budgets par boutique, avec l’average precision pour départager une égalité. Le seuil des matrices de confusion maximise le F1 sur cette même validation.

- **Population.** Chaque lundi matin, il faut au moins deux jours d’achat antérieurs, une médiane des intervalles d’au plus 30 jours et aucun achat magasin déjà connu. Un panier mixte compte comme un jour d’achat en magasin. Le canal et les montants sont lus sur les lignes produit, avec les mêmes critères d’achat positif que dans le notebook.
- **Classification.** Pour un lundi T, tout rachat entre T et T + 90 jours, bornes incluses, donne une cible de 0. Les deux classes exigent 90 jours de recul. Les 26 lundis de validation et les 13 derniers lundis de test à cible observable sont positionnés depuis la fin du fichier. L’entraînement initial, puis le réentraînement final, utilisent uniquement les labels entièrement connus avant leur coupure respective.
- **Budgets.** Pour chaque boutique, le programme compte les clients distincts qui ont commandé via `by phone` chaque semaine complète du train initial (du lundi au dimanche). La moyenne porte sur les semaines avec au moins une vente positive, tous clients et canaux confondus. Une semaine active sans commande téléphonique compte pour zéro. Le budget hebdomadaire est l’entier le plus proche de la moyenne multipliée par le coefficient (0,25 par défaut, entier pair en cas d’égalité). Une boutique sans semaine active dans ce périmètre figure dans le CSV avec une moyenne vide et un budget nul.
- **Relances.** Le client est affecté chaque lundi à la boutique ayant le plus fort CA cumulé avant ce lundi. À égalité, l’achat le plus récent puis le plus petit identifiant boutique départagent les boutiques. Chaque boutique utilise son propre budget fixe en validation et en test. XGBoost classe les clients par `proba_churn × ca_90j` décroissant et la règle de référence par retard relatif décroissant. Le plus petit identifiant client départage les ex æquo. Chaque méthode contacte un épisode au plus une fois, tant que son échéance est strictement future. Un rachat le lundi même ou le jour de l’échéance est comptabilisé.
- **Anticipation.** Le dénominateur comprend tous les épisodes de churn échus dont chaque lundi de relance possible a été scoré. L’avance médiane porte sur les épisodes détectés de cette cohorte. Le taux de réachat porte sur l’ensemble des relances d’issue connue, y compris celles en bordure du test.
- **Gain de CA.** Pour chaque relance XGBoost suivie d’un churn avéré, le calcul retient le `ca_90j` du lundi de l’alerte. C’est la somme des achats positifs de ce client, toutes boutiques confondues, pendant les 90 jours précédents, lundi exclu. Un seul jour d’avance suffit et les épisodes en bordure du test sont inclus dès que l’issue est connue. Le gain estimé est la somme de ces CA multipliée par le taux de churn évité par tentative, appels sans réponse inclus. Les relances d’issue inconnue consomment un créneau et sont comptées séparément. Les scénarios à 5 %, 10 %, 20 % et le taux paramétré sont présentés. L’effet réel reste à mesurer sur le terrain.

| Fichier dans `data/churn_final/boutiques/` | Contenu |
|---|---|
| `budgets_boutiques.csv` | Boutique, moyenne des clients téléphoniques par semaine active de train, budget hebdomadaire fixe |
| `gain_ca_test.csv` | Nombre de relances XGBoost, issues inconnues, épisodes détectés avant échéance, CA cumulé et scénarios de gain |
| `canaux.csv`, `periodes.csv` | Inventaire des canaux, dates, clients et couples client-lundi |
| `validation_xgboost.csv`, `classification.csv` | Choix du réglage à 14 jours, performances au seuil et cellules des matrices de confusion |
| `comparaison.csv`, `courbe_detection.png`, `courbe_detection.csv` | Bilan métier et courbes d’anticipation |
| `relances.csv` | Une ligne par contact et par méthode, avec boutique, budget, CA du lundi, probabilité de churn, score de classement, avance et issue |
| `episodes_test.parquet`, `scores_test.parquet` | Registre des épisodes et scores de tous les clients éligibles du test |
| `modele_churn.joblib` | Modèle final, variables dans l’ordre du modèle, configuration et seuil |
