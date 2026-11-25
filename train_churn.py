from datetime import date

import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier


def parametres_xgboost(seed: int = 42) -> dict:
    """Réglage de référence utilisé dans la première version."""
    return {
        "objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist",
        "max_depth": 5, "learning_rate": 0.05, "min_child_weight": 10,
        "subsample": 0.8, "colsample_bytree": 0.9, "random_state": seed, "n_jobs": 4,
    }


def _decouper_periodes(
    data: pl.DataFrame | pl.LazyFrame,
    debut_validation: date,
    fin_validation: date,
    debut_test: date,
    fin_test: date,
) -> dict[str, pl.DataFrame | pl.LazyFrame]:
    """Sépare les scores et attend la disponibilité complète de leurs cibles."""
    data = data.with_columns(
        pl.when(pl.col("fin_prediction") <= pl.col("fin_observation"))
        .then(pl.col("churn"))
        .otherwise(None)
        .alias("churn")
    )
    connues = data.drop_nulls("churn")
    score = pl.col("date_score")
    fin = pl.col("fin_prediction")
    test_complet = data.filter((score >= debut_test) & (score < fin_test))
    return {
        "train": connues.filter((score < debut_validation) & (fin < debut_validation)),
        "validation": connues.filter(
            (score >= debut_validation) & (score < fin_validation) & (fin < debut_test)
        ),
        # Les scores écartés du premier train redeviennent utilisables en avril.
        "train_final": connues.filter((score < debut_test) & (fin < debut_test)),
        "test": test_complet.drop_nulls("churn"),
        "test_complet": test_complet,
    }


def _choisir_seuil(cible: np.ndarray, proba: np.ndarray) -> float:
    """Maximise le F1 sur la validation ; à égalité, retient le seuil le plus haut."""
    precision, rappel, seuils = precision_recall_curve(cible, proba)
    somme = precision[:-1] + rappel[:-1]
    f1 = np.divide(
        2 * precision[:-1] * rappel[:-1], somme,
        out=np.zeros_like(somme), where=somme > 0,
    )
    indice = np.flatnonzero(f1 == f1.max())[-1]
    return float(seuils[indice])


def _mesures_classification(
    jeu: str, cible: np.ndarray, proba: np.ndarray, seuil: float
) -> dict:
    predictions = proba >= seuil
    return {
        "jeu": jeu,
        "seuil": seuil,
        "precision": precision_score(cible, predictions, zero_division=0),
        "rappel": recall_score(cible, predictions, zero_division=0),
        "f1": f1_score(cible, predictions, zero_division=0),
        "nb_alertes": int(predictions.sum()),
    }


def entrainer_churn(
    lf_churn: pl.LazyFrame,
    debut_validation: date = date(2023, 11, 1),
    debut_test: date = date(2024, 4, 1),
    fraction_alertes: float = 0.05,
    seed: int = 42,
    *,
    fin_validation: date = date(2024, 1, 1),
    fin_test: date = date(2024, 7, 1),
) -> dict:
    """Entraîne XGBoost et évalue un portefeuille sur des dates futures.

    Par défaut : train initial à cibles connues avant novembre 2023, validation
    sur les scores de novembre-décembre 2023, test sur avril-juin 2024. Une cible
    de train/validation doit être observable avant la période suivante.

    Les variables numériques et booléennes sont utilisées, hors identifiant,
    cible et métadonnées. Les dates brutes sont exclues ; mois_score est conservé.
    Les valeurs manquantes sont laissées à XGBoost, les valeurs non finies sont
    traitées comme manquantes. Les clients peuvent être communs aux périodes.

    Le nombre d'arbres et le seuil maximisant le F1 sont choisis uniquement sur
    la validation. Le modèle final réapprend ensuite sur toutes les cibles
    disponibles avant debut_test. Les métriques au seuil et celles des listes
    d'alertes de taille fraction_alertes sont retournées séparément.
    """
    if (
        not debut_validation < fin_validation <= debut_test < fin_test
        or not 0 < fraction_alertes <= 1
    ):
        raise ValueError("Vérifie les dates de découpage et la fraction d'alertes.")

    schema = lf_churn.collect_schema()
    requises = {"client_id", "date_score", "fin_prediction", "fin_observation", "churn"}
    manquantes = requises.difference(schema.names())
    if manquantes:
        raise ValueError(
            f"Colonnes manquantes : {', '.join(sorted(manquantes))}. "
            "Reconstruis lf_churn avec construire_lf_churn."
        )
    metadonnees = requises | {"cible_observable", "prochain_achat"}
    variables = [
        nom for nom, dtype in schema.items()
        if nom not in metadonnees and (dtype.is_numeric() or dtype == pl.Boolean)
    ]
    if not variables:
        raise ValueError("Aucune variable prédictive numérique disponible.")

    data = (
        lf_churn.filter(pl.col("date_score") < fin_test)
        .select(
            "client_id", "churn",
            pl.col("date_score", "fin_prediction", "fin_observation").cast(pl.Date),
            pl.col(variables).cast(pl.Float32),
        )
        .with_columns(
            pl.when(pl.col(variables).is_finite())
            .then(pl.col(variables)).otherwise(None)
        )
        .collect(engine="streaming")
    )
    periodes = _decouper_periodes(
        data, debut_validation, fin_validation, debut_test, fin_test
    )
    train = periodes["train"]
    validation = periodes["validation"]
    train_final = periodes["train_final"]
    test = periodes["test"]

    for nom, table in (("train", train), ("validation", validation), ("test", test)):
        if set(table["churn"].unique().to_list()) != {0, 1}:
            raise ValueError(f"Le split {nom} doit contenir les deux classes. Vérifie les dates.")

    params = parametres_xgboost(seed)
    choix = XGBClassifier(**params, n_estimators=500, early_stopping_rounds=30)
    x_validation = validation.select(variables).to_numpy()
    y_validation = validation["churn"].to_numpy()
    choix.fit(
        train.select(variables).to_numpy(),
        train["churn"].to_numpy(),
        eval_set=[(x_validation, y_validation)],
        verbose=False,
    )
    proba_validation = choix.predict_proba(x_validation)[:, 1]
    seuil = _choisir_seuil(y_validation, proba_validation)

    # Les décisions sont figées avant de réentraîner et de consulter le test.
    nb_arbres = choix.best_iteration + 1
    del choix, x_validation
    modele = XGBClassifier(**params, n_estimators=nb_arbres)
    modele.fit(train_final.select(variables).to_numpy(), train_final["churn"].to_numpy())

    proba = modele.predict_proba(test.select(variables).to_numpy())[:, 1]
    y_test = test["churn"].to_numpy()
    references = [nom for nom in ("recence_j", "retard_relatif") if nom in variables]
    predictions = test.select(
        "client_id", "date_score", "fin_prediction", "churn", *references
    ).with_columns(
        pl.Series("proba_churn", proba),
        pl.Series("prediction_churn", (proba >= seuil).astype(np.int8)),
    )
    mesures = []
    for score in ("proba_churn", *references):
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
            "rappel_alertes": float(alertes["churn"].sum() / predictions["churn"].sum()),
            "nb_alertes": alertes.height,
        })

    couverture_test = periodes["test_complet"].group_by("date_score").agg(
        pl.len().alias("nb_clients"),
        pl.col("churn").count().alias("nb_cibles_connues"),
        pl.col("churn").is_not_null().mean().alias("part_cibles_connues"),
    ).sort("date_score")
    bilan_periodes = pl.DataFrame([
        {
            "jeu": nom,
            "nb_lignes": table.height,
            "nb_clients": table["client_id"].n_unique(),
            "premier_score": table["date_score"].min(),
            "dernier_score": table["date_score"].max(),
            "derniere_fin_prediction": table["fin_prediction"].max(),
            "taux_churn": table["churn"].mean(),
        }
        for nom, table in periodes.items() if nom != "test_complet"
    ])

    return {
        "modele": modele,
        "variables": variables,
        "nb_arbres": nb_arbres,
        "seuil_decision": seuil,
        "tailles": {nom: table.height for nom, table in periodes.items() if nom != "test_complet"},
        "periodes": bilan_periodes,
        "metriques_classification": pl.DataFrame([
            _mesures_classification("validation", y_validation, proba_validation, seuil),
            _mesures_classification("test", y_test, proba, seuil),
        ]),
        # Lignes = classes réelles, colonnes = classes prédites, ordre [0, 1].
        "matrice_confusion": confusion_matrix(y_test, proba >= seuil, labels=[0, 1]),
        "fraction_alertes": fraction_alertes,
        "metriques_test": pl.DataFrame(mesures),
        "couverture_test": couverture_test,
        "predictions_test": predictions,
    }
