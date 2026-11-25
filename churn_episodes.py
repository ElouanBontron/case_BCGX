"""Simulation des relances et mesure de l'anticipation par épisode d'achat."""

from datetime import date, timedelta

import polars as pl

from churn_cible import CLE_EPISODE, HORIZON, UNITES_PAR_EURO


def selectionner_relances(scores: pl.DataFrame, score: str, budget: int | pl.DataFrame = 50) -> pl.DataFrame:
    """Sélection du lundi, à budget fixe par boutique ou à ancien budget global."""
    par_boutique = isinstance(budget, pl.DataFrame)
    if not par_boutique and budget < 1:
        raise ValueError("Le budget hebdomadaire doit être positif.")
    contexte = ["branch_id", "ca_90j", "proba_churn"] if par_boutique else []
    candidats = (
        scores.select(*CLE_EPISODE, "date_score", "echeance", "recence_j", "intervalle_median_j",
                      *contexte, pl.col(score).cast(pl.Float64).alias("score"))
        .filter(pl.col("date_score") < pl.col("echeance"))
    )
    if par_boutique:
        capacites = budget.filter(pl.col("budget_relances_par_semaine") > 0)
        candidats = candidats.join(capacites.select("branch_id", "budget_relances_par_semaine"),
                                   on="branch_id", validate="m:1")
    contactes = candidats.select(CLE_EPISODE).clear()
    listes = []
    for semaine in candidats.sort("date_score").partition_by("date_score", maintain_order=True):
        disponibles = (
            semaine.join(contactes, on=CLE_EPISODE, how="anti")
            .sort("score", "client_id", descending=[True, False])
        )
        if par_boutique:
            choix = disponibles.filter(pl.int_range(pl.len()).over("branch_id") < pl.col("budget_relances_par_semaine"))
        else:
            choix = disponibles.head(budget)
        listes.append(choix)
        contactes = pl.concat([contactes, choix.select(CLE_EPISODE)])
    appels = pl.concat(listes) if listes else candidats.clear()
    return appels.rename({"date_score": "date_alerte"}).with_columns(
        (pl.col("echeance") - pl.col("date_alerte")).dt.total_days().alias("avance_avant_echeance_j"),
        (pl.col("recence_j") - pl.col("intervalle_median_j")).alias("retard_sur_rythme_habituel_j"),
    )


def issues_episodes(registre: pl.LazyFrame, scores: pl.DataFrame, fin_observation: date) -> pl.DataFrame:
    """Registre indépendant des relances, avec la couverture de tous les lundis possibles."""
    debut = scores["date_score"].min()
    fin = min(scores["date_score"].max() + timedelta(days=6), fin_observation)
    comptes = (
        scores.lazy().filter(pl.col("date_score") < pl.col("echeance"))
        .group_by(CLE_EPISODE).len("lundis_scores")
    )
    fin_detection = pl.min_horizontal(pl.col("echeance") - pl.duration(days=1), "prochain_achat")
    jours = (fin_detection - pl.col("premier_lundi")).dt.total_days()
    return (
        registre.filter(
            (pl.col("dernier_achat") <= fin) & (pl.col("echeance") > debut)
            & (pl.col("prochain_achat").is_null() | (pl.col("prochain_achat") >= debut))
        )
        .with_columns(pl.when(pl.col("prochain_achat") <= fin_observation).then(pl.col("prochain_achat"))
                      .alias("prochain_achat"))
        .join(comptes, on=CLE_EPISODE, how="left")
        .with_columns(pl.col("lundis_scores").fill_null(0))
        .with_columns(
            (pl.col("lundis_scores") == pl.max_horizontal(0, 1 + jours // 7)).alias("couverture_complete"),
            pl.when(pl.col("prochain_achat") <= pl.col("echeance")).then(pl.lit("rachat"))
            .when(pl.col("echeance") <= fin_observation).then(pl.lit("churn"))
            .otherwise(pl.lit("inconnue")).alias("issue"),
        )
        .collect(engine="streaming")
    )


def evaluer_relances(
    relances: pl.DataFrame, episodes: pl.DataFrame, methode: str,
) -> tuple[dict, pl.DataFrame, pl.DataFrame]:
    """Rapporte les détections à toute la cohorte, et les rachats aux relances d'issue connue."""
    details = relances.join(
        episodes.select(*CLE_EPISODE, "issue", "couverture_complete"), on=CLE_EPISODE, how="left",
        validate="1:1",
    ).with_columns(pl.lit(methode).alias("methode"))
    if details["issue"].null_count():
        raise ValueError("Une relance ne correspond à aucun épisode du registre.")
    cohorte = episodes.filter(pl.col("couverture_complete") & (pl.col("issue") == "churn"))
    avances = details.filter(pl.col("couverture_complete") & (pl.col("issue") == "churn"))[
        "avance_avant_echeance_j"
    ]
    inconnues = (details["issue"] == "inconnue").sum()
    connues = details.height - inconnues
    rachats = (details["issue"] == "rachat").sum()
    seuils = list(range(HORIZON + 1))
    detectes = [(avances >= j).sum() for j in seuils]
    taux = [100 * n / cohorte.height if cohorte.height else None for n in detectes]
    courbe = pl.DataFrame({
        "methode": [methode] * len(seuils), "avance_min_j": seuils,
        "churns_detectes": detectes, "pct_detectes": taux,
    }, schema_overrides={"pct_detectes": pl.Float64})
    bilan = {
        "methode": methode, "relances": details.height, "episodes_churn_evaluables": cohorte.height,
        "churns_detectes": len(avances), "detectes_14j": detectes[14],
        "pct_7j": taux[7], "pct_14j": taux[14], "pct_30j": taux[30],
        "avance_mediane_j": avances.median(), "issues_connues": connues, "issues_inconnues": inconnues,
        "rachats": rachats, "pct_relances_avec_rachat": 100 * rachats / connues if connues else None,
    }
    return bilan, courbe, details


def estimer_gain_ca(relances_xgb: pl.DataFrame, taux_churn_evite_par_relance: float) -> pl.DataFrame:
    """Valorise au CA du lundi chaque épisode appelé, échu et sans rachat à l'échéance."""
    detectes = (
        relances_xgb.filter((pl.col("issue") == "churn") & (pl.col("date_alerte") < pl.col("echeance")))
        .sort("date_alerte").unique(subset=CLE_EPISODE, keep="first")
    )
    ca_micro = (pl.col("ca_90j").cast(pl.Float64) * UNITES_PAR_EURO).round(0).cast(pl.Int64)
    ca_reference = detectes.select(ca_micro.sum()).item() / UNITES_PAR_EURO
    inconnues = (relances_xgb["issue"] == "inconnue").sum()
    taux = sorted({0.05, 0.10, 0.20, taux_churn_evite_par_relance})
    return pl.DataFrame([{
        "taux_churn_evite_par_relance": t, "scenario_configure": t == taux_churn_evite_par_relance,
        "relances_simulees": relances_xgb.height, "issues_inconnues": inconnues,
        "episodes_churn_detectes_a_temps": detectes.height,
        "ca_reference_cumule_eur": round(ca_reference, 2),
        "gain_ca_estime_eur": round(ca_reference * t, 2),
    } for t in taux])
