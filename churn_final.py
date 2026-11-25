"""XGBoost final, budgets de relance par boutique et gain de CA estimé.

Exécution depuis le dépôt : python churn_final.py
"""

import argparse
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, confusion_matrix
from xgboost import XGBClassifier

from churn_boutiques import ConfigurationRelances, estimer_budgets, semaines_completes
from churn_cible import HORIZON, inventorier_ventes, preparer_observations, variables_modele
from churn_episodes import estimer_gain_ca, evaluer_relances, issues_episodes, selectionner_relances
from train_churn import _choisir_seuil, _decouper_periodes, _mesures_classification, parametres_xgboost


REFERENCE = "Réglage V1"
CONFIGURATIONS = {
    REFERENCE: {},
    "Arbres peu profonds": {"max_depth": 3, "min_child_weight": 20},
    "Arbres régularisés": {"max_depth": 7, "min_child_weight": 30, "reg_lambda": 5},
}
META_SCORES = [
    "client_id", "date_score", "dernier_achat", "echeance", "branch_id", "ca_90j",
    "recence_j", "intervalle_median_j", "retard_relatif", "churn",
]


def lundi_precedent_ou_egal(jour: date) -> date:
    return jour - timedelta(days=jour.weekday())


def decoupage(observations: pl.LazyFrame, fin: date):
    """Réserve 13 lundis de test mûrs et score ensuite jusqu'à la fin de la source."""
    dernier_test_mur = lundi_precedent_ou_egal(fin - timedelta(days=HORIZON))
    debut_test = dernier_test_mur - timedelta(weeks=12)
    dernier_val = lundi_precedent_ou_egal(debut_test - timedelta(days=HORIZON + 1))
    debut_val = dernier_val - timedelta(weeks=25)
    jeux = _decouper_periodes(
        observations.with_columns(pl.lit(fin).alias("fin_observation")),
        debut_val, dernier_val + timedelta(days=1), debut_test, fin + timedelta(days=1),
    )
    return jeux, debut_val, debut_test


def bilan_periodes(observations, jeux) -> pl.DataFrame:
    lignes = []
    relancables = jeux["test_complet"].filter(pl.col("date_score") < pl.col("echeance"))
    for nom, table in {"population": observations, **jeux, "simulation_avant_echeance": relancables}.items():
        resume = table.select(
            pl.lit(nom).alias("jeu"), pl.col("client_id").n_unique().alias("clients"),
            pl.len().alias("couples_client_lundi"), pl.col("date_score").n_unique().alias("lundis"),
            pl.min("date_score").alias("premier_lundi"), pl.max("date_score").alias("dernier_lundi"),
            pl.col("fin_prediction").filter(pl.col("churn").is_not_null()).max().alias("derniere_fin_label"),
        ).collect(engine="streaming")
        lignes.append(resume)
    return pl.concat(lignes)


def matrice(table: pl.DataFrame, variables: list[str]) -> np.ndarray:
    x = pl.col(variables).cast(pl.Float32)
    return table.select(pl.when(x.is_finite()).then(x).otherwise(None)).to_numpy(order="c")


def charger_train(table, variables):
    donnees = table.sort("date_score", "client_id").select("churn", *variables).collect(engine="streaming")
    return matrice(donnees, variables), donnees["churn"].to_numpy()


def ajouter_risque_et_priorite(scores, probabilites):
    return scores.with_columns(pl.Series("proba_churn", probabilites)).with_columns(
        (pl.col("proba_churn").cast(pl.Float64) * pl.col("ca_90j")).alias("score_priorite")
    )


def choisir_configuration(jeux, registre, variables, fin_suivi, budgets):
    x_train, y_train = charger_train(jeux["train"], variables)
    validation = jeux["validation"].sort("date_score", "client_id").collect(engine="streaming")
    x_val, y_val = matrice(validation, variables), validation["churn"].to_numpy()
    if any(set(np.unique(y)) != {0, 1} for y in (y_train, y_val)):
        raise ValueError("Le train et la validation doivent chacun contenir les deux classes.")
    scores = validation.select(META_SCORES)
    candidats = []
    episodes = None
    for nom, ajustements in CONFIGURATIONS.items():
        params = parametres_xgboost() | ajustements
        modele = XGBClassifier(**params, n_estimators=500, early_stopping_rounds=30)
        modele.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        proba = modele.predict_proba(x_val)[:, 1]
        decisions = selectionner_relances(ajouter_risque_et_priorite(scores, proba), "score_priorite", budgets)
        if episodes is None:
            episodes = issues_episodes(registre, scores, fin_suivi)
        bilan, _, _ = evaluer_relances(decisions, episodes, nom)
        if not bilan["episodes_churn_evaluables"]:
            raise ValueError("La validation ne couvre aucun épisode de churn en entier.")
        seuil = _choisir_seuil(y_val, proba)
        mesures = _mesures_classification("validation", y_val, proba, seuil)
        candidats.append({
            "configuration": nom, "max_depth": params["max_depth"], "min_child_weight": params["min_child_weight"],
            "reg_lambda": params.get("reg_lambda", 1), "arbres": modele.best_iteration + 1, "seuil": seuil,
            "episodes_churn_evaluables": bilan["episodes_churn_evaluables"],
            "detectes_14j": bilan["detectes_14j"], "pct_14j": bilan["pct_14j"],
            "average_precision": float(average_precision_score(y_val, proba)),
            "precision": mesures["precision"], "rappel": mesures["rappel"],
        })
    # Même dénominateur pour les candidats. Départage à l'average precision de validation.
    retenu = max(candidats, key=lambda c: (c["detectes_14j"], c["average_precision"]))
    return retenu, pl.DataFrame(candidats)


def classification_test(scores, modele, configuration, seuil):
    connus = scores.drop_nulls("churn")
    y, p = connus["churn"].to_numpy(), connus["proba_churn"].to_numpy()
    mesures = _mesures_classification("test", y, p, seuil)
    vn, fp, fn, vp = confusion_matrix(y, p >= seuil, labels=[0, 1]).ravel().tolist()
    return {
        "modele": modele, "configuration": configuration, "couples_client_lundi": len(y),
        **mesures, "vrais_negatifs": vn, "faux_positifs": fp, "faux_negatifs": fn, "vrais_positifs": vp,
    }


def entrainer_et_evaluer(
    observations: pl.LazyFrame, registre: pl.LazyFrame, fin: date, *,
    ventes: pl.LazyFrame, config: ConfigurationRelances = ConfigurationRelances(),
) -> dict:
    variables = variables_modele(observations)
    jeux, debut_val, debut_test = decoupage(observations, fin)
    periodes = bilan_periodes(observations, jeux)
    for nom, coupure in (("train", debut_val), ("validation", debut_test), ("train_final", debut_test)):
        derniere = periodes.filter(pl.col("jeu") == nom)["derniere_fin_label"].item()
        if derniere is None or derniere >= coupure:
            raise ValueError(f"Fenêtres de labels absentes ou incomplètes avant la coupure du jeu {nom}.")
    test_mur = periodes.filter(pl.col("jeu") == "test").row(0, named=True)
    if not test_mur["lundis"] or test_mur["derniere_fin_label"] > fin:
        raise ValueError("Aucun test avec 90 jours complets de recul.")

    train = periodes.filter(pl.col("jeu") == "train").row(0, named=True)
    budgets = estimer_budgets(ventes, train["premier_lundi"], train["dernier_lundi"], config)
    bornes_budgets = semaines_completes(train["premier_lundi"], train["dernier_lundi"])
    retenu, validation = choisir_configuration(jeux, registre, variables, debut_test - timedelta(days=1), budgets)
    # Les réglages, le nombre d'arbres et les seuils sont figés avant le réapprentissage.
    x_train, y_train = charger_train(jeux["train_final"], variables)
    test = jeux["test_complet"].sort("date_score", "client_id").collect(engine="streaming")
    x_test = matrice(test, variables)
    meta = test.select(META_SCORES)
    del test
    reference = validation.filter(pl.col("configuration") == REFERENCE).row(0, named=True)
    ajustements = {}
    for candidat in (reference, retenu):
        nom = candidat["configuration"]
        if nom in ajustements:
            continue
        params = parametres_xgboost() | CONFIGURATIONS[nom]
        modele = XGBClassifier(**params, n_estimators=candidat["arbres"])
        modele.fit(x_train, y_train)
        scores = ajouter_risque_et_priorite(meta, modele.predict_proba(x_test)[:, 1])
        ajustements[nom] = modele, scores
    modele_final, scores = ajustements[retenu["configuration"]]
    alertes = {
        nom: selectionner_relances(scores, colonne, budgets)
        for nom, colonne in (("XGBoost", "score_priorite"), ("Règle de rythme", "retard_relatif"))
    }
    episodes = issues_episodes(registre, scores, fin)
    classification = pl.DataFrame([
        classification_test(ajustements[c["configuration"]][1], nom, c["configuration"], c["seuil"])
        for nom, c in (("XGBoost V1 réentraîné", reference), ("XGBoost final", retenu))
    ])
    bilans, courbes, details = zip(*(
        evaluer_relances(relances, episodes, nom) for nom, relances in alertes.items()
    ))
    relances = pl.concat(details)
    gain = estimer_gain_ca(
        relances.filter(pl.col("methode") == "XGBoost"), config.taux_churn_evite_par_relance,
    ).with_columns(pl.lit(config.coefficient_relance).alias("coefficient_relance"))
    return {
        "modele": {"modele": modele_final, "variables": variables, "seuil": retenu["seuil"],
                   "configuration": retenu["configuration"], "debut_test": debut_test},
        "periodes": periodes, "validation_xgboost": validation, "classification": classification,
        "comparaison": pl.DataFrame(bilans), "courbe_detection": pl.concat(courbes),
        "relances": relances, "episodes_test": episodes, "scores_test": scores,
        "budgets_boutiques": budgets, "gain_ca_test": gain, "bornes_budgets": bornes_budgets,
        "configuration_relances": config,
    }


def tracer_courbe(courbe: pl.DataFrame, destination: Path) -> None:
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    fig = Figure(figsize=(8.6, 4.8), layout="constrained")
    ax = fig.subplots()
    ax.axvline(14, color="0.55", linestyle=":", linewidth=1)
    for serie, couleur in zip(courbe.partition_by("methode", maintain_order=True), ("#126b85", "#b14b23")):
        taux = serie.filter(pl.col("avance_min_j") == 14)["pct_detectes"].item()
        legende = serie["methode"][0]
        if taux is not None:
            legende += f" · {taux:.2f} % à 14 jours".replace(".", ",")
            ax.scatter(14, taux, color=couleur, zorder=3)
        ax.step(serie["avance_min_j"], serie["pct_detectes"], where="pre", label=legende, color=couleur)
    ax.set(title="Détection du churn avec les budgets hebdomadaires des boutiques",
           xlabel="Avance minimale exigée (jours)", ylabel="Part des épisodes de churn évaluables")
    ax.set_xlim(0, HORIZON)
    ax.set_ylim(bottom=0)
    ax.set_xticks([0, 7, 14, 30, 45, 60, 75, 90])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(destination, dpi=160)


def enregistrer(resultats: dict, canaux: pl.DataFrame, sortie: Path) -> None:
    sortie.mkdir(parents=True, exist_ok=True)
    canaux.write_csv(sortie / "canaux.csv")
    for nom in ("periodes", "validation_xgboost", "classification", "comparaison", "courbe_detection",
                "relances", "budgets_boutiques", "gain_ca_test"):
        resultats[nom].write_csv(sortie / f"{nom}.csv")
    for nom in ("scores_test", "episodes_test"):
        resultats[nom].write_parquet(sortie / f"{nom}.parquet")
    joblib.dump(resultats["modele"], sortie / "modele_churn.joblib", compress=3)
    tracer_courbe(resultats["courbe_detection"], sortie / "courbe_detection.png")


def afficher(resultats: dict) -> None:
    debut, fin_exclue = resultats["bornes_budgets"]
    budgets = resultats["budgets_boutiques"]
    print(f"Budgets calibrés du {debut} au {fin_exclue - timedelta(days=1)} inclus, "
          f"coefficient {resultats['configuration_relances'].coefficient_relance:g}. "
          f"{budgets.height} boutiques, {budgets['budget_relances_par_semaine'].sum()} créneaux par semaine.")
    for nom in ("periodes", "validation_xgboost", "classification", "comparaison"):
        print(f"\n{nom}\n{resultats[nom].write_csv()}")
    xgb, regle = resultats["comparaison"].iter_rows(named=True)
    n = xgb["episodes_churn_evaluables"]
    if n:
        print(f"À 14 jours sur {n} épisodes de churn évaluables, "
              f"XGBoost détecte {xgb['detectes_14j']} épisodes ({xgb['pct_14j']:.2f} %) "
              f"et la règle {regle['detectes_14j']} ({regle['pct_14j']:.2f} %). "
              f"Écart XGBoost moins règle {xgb['pct_14j'] - regle['pct_14j']:+.2f} points.")
    else:
        print("Aucun épisode de churn entièrement couvert en test. Le taux à 14 jours est indéfini.")
    gain = resultats["gain_ca_test"]
    central = gain.filter(pl.col("scenario_configure")).row(0, named=True)
    print(f"\nXGBoost a simulé {central['relances_simulees']} relances sur le test, "
          f"{central['episodes_churn_detectes_a_temps']} épisodes de churn détectés avant l'échéance, "
          f"{central['issues_inconnues']} issues inconnues. CA de référence cumulé "
          f"{central['ca_reference_cumule_eur']:.2f} EUR. Au taux configuré par tentative de "
          f"{central['taux_churn_evite_par_relance']:.2%}, gain de CA estimé {central['gain_ca_estime_eur']:.2f} EUR.")
    print(gain.select("taux_churn_evite_par_relance", "gain_ca_estime_eur").write_csv())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ventes", type=Path, default=Path("data/dataset_sales_2026.csv"))
    parser.add_argument("--sortie", type=Path, default=Path("data/churn_final/boutiques"))
    parser.add_argument("--canal-magasin", action="append", help="Libellé magasin, option répétable.")
    defauts = ConfigurationRelances()
    parser.add_argument("--coefficient-relance", type=float, default=defauts.coefficient_relance,
                        help="Multiplicateur de la fréquentation téléphonique hebdomadaire.")
    parser.add_argument("--taux-churn-evite-par-relance", type=float, default=defauts.taux_churn_evite_par_relance,
                        help="Fraction supposée des churns évités par tentative d'appel.")
    args = parser.parse_args()
    config = ConfigurationRelances(args.coefficient_relance, args.taux_churn_evite_par_relance)
    ventes = pl.scan_csv(args.ventes, separator=";", infer_schema_length=10_000, low_memory=True)
    lignes, canaux, fin = inventorier_ventes(ventes, tuple(args.canal_magasin or ["at the store"]))
    print(f"Commandes observées jusqu'au {fin}.\n{canaux.write_csv()}", flush=True)
    args.sortie.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.sortie, prefix="calcul_") as temporaire:
        observations, registre = preparer_observations(lignes, fin, Path(temporaire))
        resultats = entrainer_et_evaluer(observations, registre, fin, ventes=ventes, config=config)
    enregistrer(resultats, canaux, args.sortie)
    afficher(resultats)


if __name__ == "__main__":
    main()
