from datetime import date

import polars as pl
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier


def entrainer_churn(
    lf_churn: pl.LazyFrame,
    debut_validation: date = date(2024, 6, 1),
    debut_test: date = date(2024, 8, 1),
    fraction_alertes: float = 0.05,
    seed: int = 42,
) -> dict:
    """Entraîne XGBoost et évalue ses alertes sur des clients isolés."""
    if debut_validation >= debut_test or not 0 < fraction_alertes <= 1:
        raise ValueError("Vérifie les dates de découpage et la fraction d'alertes.")

    variables = ["nb_jours_achat", "intervalle_median_j", "recence_j", "retard_relatif"]
    if "ca_median_jour_achat" in lf_churn.collect_schema():
        variables.append("ca_median_jour_achat")

    data = lf_churn.select(
        "client_id",
        "date_score",
        pl.col("echeance").dt.date(),
        "churn",
        pl.col(variables).cast(pl.Float32),
    ).collect()

    clients_test = (
        data.select("client_id")
        .unique()
        .sort("client_id")
        .sample(fraction=0.20, seed=seed)
    )
    developpement = data.join(clients_test, on="client_id", how="anti").drop_nulls("churn")
    debut = pl.col("date_score")
    echeance = pl.col("echeance")

    # Une cible d'entraînement doit être observable avant la période suivante.
    train = developpement.filter((debut < debut_validation) & (echeance < debut_validation))
    validation = developpement.filter(
        (debut >= debut_validation) & (debut < debut_test) & (echeance < debut_test)
    )
    train_final = developpement.filter((debut < debut_test) & (echeance < debut_test))
    test_complet = data.filter(debut >= debut_test).join(clients_test, on="client_id", how="semi")
    test = test_complet.drop_nulls("churn")

    for nom, table in (("train", train), ("validation", validation), ("test", test)):
        if set(table["churn"].unique().to_list()) != {0, 1}:
            raise ValueError(f"Le split {nom} doit contenir les deux classes. Vérifie les dates.")

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "max_depth": 5,
        "learning_rate": 0.05,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "random_state": seed,
    }
    choix = XGBClassifier(**params, n_estimators=500, early_stopping_rounds=30)
    choix.fit(
        train.select(variables).to_numpy(),
        train["churn"].to_numpy(),
        eval_set=[(validation.select(variables).to_numpy(), validation["churn"].to_numpy())],
        verbose=False,
    )

    # Réentraîner avec le nombre d'arbres choisi sur la validation.
    nb_arbres = choix.best_iteration + 1
    modele = XGBClassifier(**params, n_estimators=nb_arbres)
    modele.fit(train_final.select(variables).to_numpy(), train_final["churn"].to_numpy())

    proba = modele.predict_proba(test.select(variables).to_numpy())[:, 1]
    predictions = test.select("client_id", "date_score", "churn", "retard_relatif").with_columns(
        pl.Series("proba_churn", proba)
    )
    mesures = []
    for score in ("proba_churn", "retard_relatif"):
        alertes = predictions.sort(
            ["date_score", score, "client_id"], descending=[False, True, False]
        ).filter(
            pl.int_range(pl.len()).over("date_score")
            < (pl.len().over("date_score") * fraction_alertes).ceil()
        )
        mesures.append({
            "score": score,
            "average_precision": average_precision_score(
                predictions["churn"].to_numpy(), predictions[score].to_numpy()
            ),
            "precision_alertes": alertes["churn"].mean(),
            "nb_alertes": alertes.height,
        })

    couverture_test = test_complet.group_by("date_score").agg(
        pl.len().alias("nb_clients"),
        pl.col("churn").is_not_null().mean().alias("part_cibles_connues"),
    ).sort("date_score")

    return {
        "modele": modele,
        "variables": variables,
        "nb_arbres": nb_arbres,
        "tailles": {
            "train": train.height, "validation": validation.height,
            "train_final": train_final.height, "test": test.height,
        },
        "metriques_test": pl.DataFrame(mesures),
        "couverture_test": couverture_test,
        "predictions_test": predictions,
    }
