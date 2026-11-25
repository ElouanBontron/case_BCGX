"""Plan d'appels par boutique et scénarios de CA additionnel sur le test.

L'affectation et le classement utilisent uniquement les achats antérieurs au
score. La boutique responsable est celle du plus gros CA cumulé historique.
Le classement utilise proba_churn * CA des 365 derniers jours par défaut,
toutes boutiques confondues. Le quota de 5 % porte sur le nombre de clients
scorés et affectés à chaque boutique, arrondi à l'entier inférieur.

Le gain est une simulation, pas un effet mesuré des appels. Pour les clients
appelés dont churn == 1, on applique un taux de récupération supposé au CA
manquant : max(CA habituel projeté - CA observé dans l'historique de référence, 0).
Le CA habituel est la moyenne calendaire de la fenêtre passée, projetée sur
les 90 jours suivant chaque appel. Les retours spontanés observés sont donc
déjà retranchés. La récupération suppose un retour au niveau habituel pendant
cette période ; le taux est incrémental par rapport à la politique historique.

Un client ne peut être rappelé pendant au moins la durée de projection, même
s'il change de boutique responsable : ses horizons de gain ne se chevauchent
pas. Les gains sont attribués au portefeuille de la boutique responsable,
pas nécessairement au lieu des transactions futures. Ils ne sont ni une marge
ni des gains annuels. Les probabilités et comportements historiques ne sont
pas modifiés par les interventions simulées. L'historique sert de référence
sans cette nouvelle campagne, sans supposer qu'aucun appel n'avait jamais eu lieu.

Exemple :
    simulation = simuler_appels(
        lf, resultats["predictions_test"], taux_recuperation=0.10
    )
    display(simulation["scenarios"])
    display(simulation["bilan_par_boutique"])
    display(simulation["appels"])
"""

from datetime import timedelta
from math import isfinite

import polars as pl

from churn_boutiques import affecter_boutiques


def _lazy(data: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    return data.lazy() if isinstance(data, pl.DataFrame) else data


def _preparer_predictions(predictions: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    source = _lazy(predictions)
    requises = {"client_id", "date_score", "proba_churn", "churn"}
    manquantes = requises.difference(source.collect_schema().names())
    if manquantes:
        raise ValueError(f"Prédictions : colonnes manquantes {sorted(manquantes)}.")
    data = source.select(
        "client_id", pl.col("date_score").cast(pl.Date),
        pl.col("proba_churn").cast(pl.Float64), "churn",
    ).collect()
    if data.is_empty():
        raise ValueError("Aucune prédiction à simuler.")
    invalides = data.filter(
        pl.col("client_id").is_null() | pl.col("date_score").is_null()
        | pl.col("proba_churn").is_null() | ~pl.col("proba_churn").is_finite()
        | ~pl.col("proba_churn").is_between(0, 1)
        | (pl.col("date_score").dt.weekday() != 1)
        | (pl.col("churn").is_not_null() & ~pl.col("churn").is_in([0, 1]))
    )
    if not invalides.is_empty():
        raise ValueError("Scores invalides : identifiants, lundis, probabilités ou cibles.")
    if data.select(pl.struct("client_id", "date_score").n_unique()).item() != data.height:
        raise ValueError("Il faut une seule prédiction par client et par lundi.")
    return data.with_columns(pl.col("churn").cast(pl.Int8)).sort(["date_score", "client_id"])


def _achats_journaliers(
    ventes: pl.DataFrame | pl.LazyFrame, predictions: pl.DataFrame, horizon: int
) -> tuple[pl.DataFrame, dict]:
    source = _lazy(ventes)
    schema = source.collect_schema()
    requises = {"client_id", "branch_id", "date_order", "sales_net", "quantity"}
    manquantes = requises.difference(schema.names())
    if manquantes:
        raise ValueError(f"Ventes : colonnes manquantes {sorted(manquantes)}.")
    jour = pl.col("date_order")
    if schema["date_order"] == pl.String:
        jour = jour.str.to_date(format="%Y-%m-%d")
    borne_fin = predictions["date_score"].max() + timedelta(days=horizon)
    journalier = (
        source.select(
            pl.col("client_id").cast(predictions.schema["client_id"]), "branch_id",
            jour.cast(pl.Date).alias("date"),
            pl.col("sales_net").cast(pl.Float64), "quantity",
        )
        .drop_nulls(subset=["client_id", "date"])
        # Bornes de la source avant le filtrage sur clients et achats positifs.
        .with_columns(
            pl.col("date").min().alias("debut_donnees"),
            pl.col("date").max().alias("fin_donnees"),
        )
        .join(predictions.select("client_id").unique().lazy(), on="client_id", how="semi")
        .filter(
            (pl.col("date") < borne_fin)
            & pl.col("quantity").is_finite() & (pl.col("quantity") > 0)
            & pl.col("sales_net").is_finite() & (pl.col("sales_net") > 0)
        )
        .group_by("client_id", "branch_id", "date")
        .agg(
            pl.col("sales_net").sum().alias("ca_achats"),
            pl.col("debut_donnees").first(), pl.col("fin_donnees").first(),
        )
        .collect(engine="streaming")
    )
    if journalier.is_empty():
        raise ValueError("Aucun achat positif trouvé pour les clients des prédictions.")
    calendrier = journalier.select("debut_donnees", "fin_donnees").row(0, named=True)
    return journalier.drop("debut_donnees", "fin_donnees"), calendrier


def _ca_avant_date(
    requetes: pl.DataFrame, cumuls: pl.DataFrame, borne: str, nom_ca: str
) -> pl.DataFrame:
    """Somme des achats strictement antérieurs à la borne, pour chaque score."""
    return (
        requetes.select("client_id", "date_score", pl.col(borne).alias("_borne"))
        .sort(["client_id", "_borne"])
        .join_asof(
            cumuls, left_on="_borne", right_on="date", by="client_id",
            strategy="backward", allow_exact_matches=False, check_sortedness=False,
        )
        .select("client_id", "date_score", pl.col("ca_cumule").fill_null(0).alias(nom_ca))
    )


def _enrichir_portefeuille(
    predictions: pl.DataFrame, journalier: pl.DataFrame, fenetre: int, debut_donnees
) -> tuple[pl.DataFrame, pl.DataFrame]:
    cles = ["client_id", "date_score"]
    cumuls_clients = (
        journalier.group_by("client_id", "date").agg(pl.col("ca_achats").sum())
        .sort(["client_id", "date"])
        .with_columns(pl.col("ca_achats").cum_sum().over("client_id").alias("ca_cumule"))
        .select("client_id", "date", "ca_cumule")
    )
    affectations = (
        affecter_boutiques(predictions.lazy(), journalier.lazy())
        .rename({"branch_id": "boutique_responsable"}).collect()
    )
    portefeuille = predictions.with_columns(
        (pl.col("date_score") - pl.duration(days=fenetre)).alias("_debut_reference")
    )
    portefeuille = (
        portefeuille.join(affectations, on=cles, how="left")
        .join(_ca_avant_date(portefeuille, cumuls_clients, "date_score", "ca_historique_client"), on=cles)
        .join(_ca_avant_date(portefeuille, cumuls_clients, "_debut_reference", "_ca_avant_reference"), on=cles)
        .with_columns(
            (pl.col("ca_historique_client") - pl.col("_ca_avant_reference"))
            .clip(lower_bound=0).alias("ca_reference_client"),
            (pl.col("_debut_reference") >= pl.lit(debut_donnees)).alias("reference_complete"),
        )
        .with_columns(
            (pl.col("ca_reference_client") * pl.col("proba_churn")).alias("score_priorite")
        )
        .drop("_debut_reference", "_ca_avant_reference")
    )
    return portefeuille, cumuls_clients


def _planifier(
    portefeuille: pl.DataFrame, fraction: float, delai_rappel: int, capacite_max: int | None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    derniers_appels = {}
    plans, bilans = [], []
    cle_boutique = ["boutique_responsable"]
    for lundi in portefeuille["date_score"].unique().sort().to_list():
        jour = portefeuille.filter(pl.col("date_score") == lundi).drop_nulls("boutique_responsable")
        capacites = jour.group_by(cle_boutique).agg(pl.len().alias("nb_clients_scores"))
        capacites = capacites.with_columns(
            (pl.col("nb_clients_scores") * fraction).floor().cast(pl.UInt32).alias("quota_clients")
        ).with_columns(
            (pl.min_horizontal("quota_clients", pl.lit(capacite_max)) if capacite_max is not None
             else pl.col("quota_clients")).alias("capacite_appels")
        )
        indisponibles = [
            client for client, dernier in derniers_appels.items()
            if (lundi - dernier).days < delai_rappel
        ]
        candidats = jour.filter(pl.col("score_priorite") > 0)
        if indisponibles:
            candidats = candidats.filter(~pl.col("client_id").is_in(indisponibles))
        disponibles = candidats.group_by(cle_boutique).agg(pl.len().alias("nb_clients_disponibles"))
        choisis = (
            candidats.sort(
                ["boutique_responsable", "score_priorite", "proba_churn", "client_id"],
                descending=[False, True, True, False],
            )
            .with_columns(pl.int_range(1, pl.len() + 1).over(cle_boutique).alias("rang_appel"))
            .join(capacites, on=cle_boutique)
            .filter(pl.col("rang_appel") <= pl.col("capacite_appels"))
        )
        for client in choisis["client_id"].to_list():
            derniers_appels[client] = lundi
        plans.append(choisis)
        bilans.append(
            capacites.join(disponibles, on=cle_boutique, how="left")
            .join(choisis.group_by(cle_boutique).agg(pl.len().alias("nb_appels")), on=cle_boutique, how="left")
            .with_columns(
                pl.lit(lundi).alias("date_score"),
                pl.col("nb_clients_disponibles", "nb_appels").fill_null(0),
            )
        )
    return pl.concat(plans).sort(["date_score", "boutique_responsable", "rang_appel"]), pl.concat(bilans)


def _synthese(evaluation: pl.DataFrame, groupes: tuple[str, ...] = ()) -> pl.DataFrame:
    mesures = [
        pl.len().alias("nb_appels"),
        pl.col("client_id").n_unique().alias("nb_clients_distincts"),
        pl.col("evaluation_complete").sum().alias("nb_appels_evaluables"),
        pl.when(pl.col("evaluation_complete")).then(pl.col("churn"))
        .sum().alias("nb_churners_appeles_evaluables"),
        pl.when(pl.len() == 0).then(0.0)
        .when(pl.col("ca_manquant_churn").count() > 0)
        .then(pl.col("ca_manquant_churn").sum()).otherwise(None)
        .alias("ca_manquant_churn_evaluable"),
    ]
    if groupes:
        return evaluation.group_by(list(groupes)).agg(mesures)
    return evaluation.select(mesures)


def simuler_appels(
    ventes: pl.DataFrame | pl.LazyFrame,
    predictions: pl.DataFrame | pl.LazyFrame,
    *,
    fraction_appels: float = 0.05,
    taux_recuperation: float = 0.10,
    scenarios_recuperation: tuple[float, ...] = (0.05, 0.10, 0.20),
    fenetre_ca_j: int = 365,
    horizon_gain_j: int = 90,
    delai_rappel_j: int | None = None,
    capacite_max_appels: int | None = None,
) -> dict:
    """Simule les appels, sans réentraîner ni modifier les prédictions.

    Les quotas sont calculés avant exclusion des clients récemment appelés,
    puis remplis avec les autres clients par score décroissant. Un plafond
    absolu facultatif s'applique à chaque boutique/lundi. Une petite boutique
    de moins de 20 clients scorés a un quota nul avec fraction_appels=0.05.

    Le taux de récupération est une hypothèse parmi les vrais churners appelés,
    pas un taux de réponse téléphonique. Dans le bilan rétrospectif, on utilise
    leur churn observé : on ne multiplie pas une seconde fois par proba_churn.

    Un gain est évaluable seulement si toute la fenêtre de référence passée,
    toute la fenêtre future de CA et le churn sont connus. Les bilans indiquent
    cette couverture. Les gains des scénarios portent sur cette partie évaluable.
    Aucun coût d'appel, remise, marge ou effet causal mesuré n'est inclus.
    """
    delai = horizon_gain_j if delai_rappel_j is None else delai_rappel_j
    taux = sorted(set((0.0, taux_recuperation, *scenarios_recuperation)))
    if not isfinite(fraction_appels) or not 0 < fraction_appels <= 1:
        raise ValueError("fraction_appels doit appartenir à ]0, 1].")
    if any(not isfinite(t) or not 0 <= t <= 1 for t in taux):
        raise ValueError("Les taux de récupération doivent appartenir à [0, 1].")
    if (
        not isinstance(fenetre_ca_j, int) or fenetre_ca_j < 1
        or not isinstance(horizon_gain_j, int) or horizon_gain_j < 1
        or not isinstance(delai, int) or delai < horizon_gain_j
        or (capacite_max_appels is not None and
            (not isinstance(capacite_max_appels, int) or capacite_max_appels < 0))
    ):
        raise ValueError("Délais positifs, délai de rappel >= horizon de gain et capacité entière >= 0 requis.")

    scores = _preparer_predictions(predictions)
    journalier, calendrier = _achats_journaliers(ventes, scores, horizon_gain_j)
    portefeuille, cumuls = _enrichir_portefeuille(
        scores, journalier, fenetre_ca_j, calendrier["debut_donnees"]
    )
    selection, capacites = _planifier(portefeuille, fraction_appels, delai, capacite_max_appels)
    cles = ["client_id", "date_score"]
    bornes = selection.with_columns(
        (pl.col("date_score") + pl.duration(days=horizon_gain_j)).alias("_fin_gain")
    )
    evaluation = (
        selection.join(_ca_avant_date(bornes, cumuls, "_fin_gain", "_ca_fin_gain"), on=cles)
        .with_columns(
            (pl.col("ca_reference_client") * horizon_gain_j / fenetre_ca_j)
            .alias("ca_habituel_horizon"),
            (pl.col("_ca_fin_gain") - pl.col("ca_historique_client"))
            .clip(lower_bound=0).alias("ca_observe_horizon"),
            (
                pl.col("reference_complete") & pl.col("churn").is_not_null()
                & ((pl.col("date_score") + pl.duration(days=horizon_gain_j - 1))
                   <= pl.lit(calendrier["fin_donnees"]))
            ).alias("evaluation_complete"),
        )
        .with_columns(
            (pl.col("ca_habituel_horizon") - pl.col("ca_observe_horizon"))
            .clip(lower_bound=0).alias("ca_manquant_horizon")
        )
        .with_columns(
            pl.when(pl.col("evaluation_complete"))
            .then(pl.col("ca_manquant_horizon") * pl.col("churn"))
            .otherwise(None).alias("ca_manquant_churn")
        )
        .with_columns(
            (taux_recuperation * pl.col("ca_manquant_churn")).alias("ca_incremental_estime")
        )
        .drop("_ca_fin_gain")
    )
    global_ = _synthese(evaluation)
    scenarios = pl.DataFrame({"taux_recuperation": taux}).join(global_, how="cross").with_columns(
        (pl.col("taux_recuperation") * pl.col("nb_churners_appeles_evaluables"))
        .alias("nb_recuperations_estimees"),
        (pl.col("taux_recuperation") * pl.col("ca_manquant_churn_evaluable"))
        .alias("ca_incremental_estime"),
    )

    def bilan(groupes):
        return _synthese(evaluation, groupes).with_columns(
            (taux_recuperation * pl.col("nb_churners_appeles_evaluables"))
            .alias("nb_recuperations_estimees"),
            (taux_recuperation * pl.col("ca_manquant_churn_evaluable"))
            .alias("ca_incremental_estime"),
        ).sort(list(groupes))

    sans_boutique = portefeuille.filter(pl.col("boutique_responsable").is_null())
    return {
        "appels": selection.drop("churn"),
        "evaluation_appels": evaluation,
        "capacites": capacites.sort(["date_score", "boutique_responsable"]),
        "scenarios": scenarios,
        "bilan_hebdomadaire": bilan(("date_score",)),
        "bilan_par_boutique": bilan(("boutique_responsable",)),
        "clients_sans_boutique": sans_boutique.select("client_id", "date_score", "proba_churn"),
        "parametres": {
            "fraction_appels": fraction_appels, "taux_recuperation": taux_recuperation,
            "fenetre_ca_j": fenetre_ca_j, "horizon_gain_j": horizon_gain_j,
            "delai_rappel_j": delai, "capacite_max_appels": capacite_max_appels,
            **calendrier,
            "premier_lundi": scores["date_score"].min(),
            "dernier_lundi": scores["date_score"].max(),
            "fin_horizon_gain": scores["date_score"].max() + timedelta(days=horizon_gain_j - 1),
        },
    }
