"""Construction des observations client et des cibles de rupture d'achat."""

from datetime import date
from math import ceil, isfinite

import polars as pl


def normaliser_commandes(ventes: pl.LazyFrame) -> pl.LazyFrame:
    jour = pl.col("date_order")
    if ventes.collect_schema()["date_order"] == pl.String:
        jour = jour.str.to_date("%Y-%m-%d")
    return ventes.with_columns(
        jour.cast(pl.Date), pl.col("order_channel").str.strip_chars().str.to_lowercase(),
    )


def achat_positif() -> pl.Expr:
    return (
        pl.col("quantity").is_finite() & (pl.col("quantity") > 0)
        & pl.col("sales_net").is_finite() & (pl.col("sales_net") > 0)
    )


def construire_quotidien(ventes: pl.LazyFrame, *, montant: pl.Expr | None = None) -> pl.LazyFrame:
    """Agrège les achats positifs par client et jour, sans compter les retours.

    Les listes de produits, d'agences et de canaux restent facultatives : les
    anciens parquets quotidiens peuvent toujours alimenter construire_lf_churn.
    """
    schema = ventes.collect_schema()
    date_achat = pl.col("date_order")
    if schema["date_order"] == pl.String:
        date_achat = date_achat.str.to_date(format="%Y-%m-%d")

    valeur = pl.col("sales_net") if montant is None else montant
    aggregations = [
        valeur.sum().alias("ca_achats"),
        pl.len().alias("nb_lignes"),
    ]
    for source, destination in (("product_id", "produits"), ("branch_id", "agences")):
        if source in schema:
            aggregations.append(pl.col(source).drop_nulls().unique().alias(destination))
    if "order_channel" in schema:
        canal = pl.col("order_channel").str.strip_chars().str.to_lowercase()
        aggregations.extend([
            canal.drop_nulls().unique().alias("canaux"),
            pl.when(canal == "online").then(valeur).otherwise(0)
            .sum().alias("ca_online"),
        ])

    return (
        ventes.with_columns(date_achat.cast(pl.Date).alias("date"))
        .filter(
            pl.col("client_id").is_not_null()
            & pl.col("date").is_not_null()
            & achat_positif()
        )
        .group_by(["client_id", "date"])
        .agg(aggregations)
    )


def _fenetre_achats(
    achats: pl.LazyFrame, jours: int, decalage: int = 0
) -> pl.LazyFrame:
    """Agrège [T - jours - decalage, T - decalage[ pour chaque lundi T."""
    suffixe = f"{jours}j" + ("_precedents" if decalage else "")
    aggregations = [
        pl.len().alias(f"nb_jours_achat_{suffixe}"),
        pl.col("ca_achats").sum().alias(f"ca_{suffixe}"),
    ]
    schema = achats.collect_schema()
    if jours == 90:
        for source, destination in (
            ("produits", "nb_produits"),
            ("agences", "nb_agences"),
            ("canaux", "nb_canaux"),
        ):
            if source in schema:
                aggregations.append(
                    pl.col(source).explode().drop_nulls().n_unique()
                    .alias(f"{destination}_{suffixe}")
                )
        if "ca_online" in schema:
            aggregations.append(pl.col("ca_online").sum().alias(f"ca_online_{suffixe}"))

    # Décaler les achats permet d'étiqueter les fenêtres par le lundi du score.
    # L'offset, multiple de 7, inclut aussi les premières fenêtres incomplètes.
    return (
        achats.with_columns(
            (pl.col("date") + pl.duration(days=jours + decalage)).alias("date_score")
        )
        .group_by_dynamic(
            "date_score", every="1w", period=f"{jours}d",
            offset=f"-{7 * ceil(jours / 7)}d", closed="left", label="left",
            group_by="client_id",
        )
        .agg(aggregations)
    )


def profils_historiques(achats: pl.LazyFrame) -> pl.LazyFrame:
    """Un profil au soir de chaque achat, sur des jours uniques triés par client."""
    return (
        achats.with_columns(
            pl.col("date").first().over("client_id").alias("premier_achat"),
            pl.col("date").cum_count().over("client_id").alias("nb_jours_achat"),
            pl.col("date").diff().over("client_id").dt.total_days().alias("intervalle_j"),
            pl.col("date").shift(-1).over("client_id").alias("prochain_achat"),
        )
        .with_columns(
            pl.col("intervalle_j").cumulative_eval(pl.element().median(), min_samples=1)
            .over("client_id").alias("intervalle_median_j"),
            pl.col("intervalle_j").cumulative_eval(pl.element().std(), min_samples=2)
            .over("client_id").alias("intervalle_std_j"),
            pl.col("intervalle_j").rolling_median(window_size=5, min_samples=3)
            .over("client_id").alias("intervalle_recent_median_j"),
            pl.col("ca_achats").cumulative_eval(pl.element().median(), min_samples=1)
            .over("client_id").alias("ca_median_jour_achat"),
            ((pl.col("date") - pl.col("premier_achat")).dt.total_days()
             / (pl.col("nb_jours_achat").cast(pl.Int64) - 1))
            .alias("intervalle_moyen_j"),
        )
        .with_columns(
            (pl.col("intervalle_std_j") / pl.col("intervalle_moyen_j"))
            .alias("cv_intervalles"),
            (pl.col("intervalle_recent_median_j") / pl.col("intervalle_median_j") - 1)
            .alias("evolution_intervalle"),
        )
    )


def construire_lf_churn(
    quotidien: pl.LazyFrame,
    *,
    min_jours_achat: int = 6,
    facteur_intervalle: float = 3.0,
    min_inactivite_j: int = 90,
    horizon_prediction_j: int = 90,
    fin_observation: date | None = None,
) -> pl.LazyFrame:
    """Produit une ligne par client éligible et lundi, avant les achats du jour.

    Le seuil d'inactivité est max(min_inactivite_j, facteur_intervalle * médiane),
    arrondi au jour supérieur et figé à chaque score. Seuls les clients ayant
    assez d'achats et une échéance courante dans [T, T + horizon] sont retenus.
    La cible vaut 1 si aucun rachat ne survient au plus tard à cette échéance.
    Elle décrit la rupture du cycle courant, pas un départ définitif.

    Les deux classes restent nulles tant que T + horizon n'est pas observé.
    fin_prediction, fin_observation et cible_observable sont des métadonnées
    pour les découpages futurs, à exclure des variables prédictives.
    Si fin_observation est omise, la dernière date source disponible est utilisée.
    """
    if (
        min_jours_achat < 2 or min_inactivite_j < 1 or horizon_prediction_j < 1
        or not isfinite(facteur_intervalle) or facteur_intervalle <= 0
    ):
        raise ValueError("Les délais doivent être positifs et min_jours_achat >= 2.")

    schema = quotidien.collect_schema()
    date_achat = pl.col("date")
    if schema["date"] == pl.String:
        date_achat = date_achat.str.to_date(format="%Y-%m-%d")
    colonnes_optionnelles = [
        c for c in ("produits", "agences", "canaux", "ca_online") if c in schema
    ]
    source = quotidien.select(
        "client_id", date_achat.cast(pl.Date),
        pl.col("ca_achats").cast(pl.Float64), *colonnes_optionnelles,
    ).drop_nulls(subset=["client_id", "date"])
    if fin_observation is not None:
        source = source.filter(pl.col("date") <= pl.lit(fin_observation))
    fin = pl.lit(fin_observation, dtype=pl.Date) if fin_observation else pl.col("date").max()

    aggregations = [
        pl.col("ca_achats").sum(), pl.col("fin_observation").first(),
    ]
    for colonne in colonnes_optionnelles:
        expression = pl.col(colonne)
        aggregations.append(
            expression.sum() if colonne == "ca_online"
            else expression.explode().drop_nulls().unique()
        )
    achats = (
        source.with_columns(fin.alias("fin_observation"))
        .filter(pl.col("ca_achats").is_finite() & (pl.col("ca_achats") > 0))
        # Une répétition de clé s'agrège : aucun montant n'est perdu avec unique().
        .group_by(["client_id", "date"]).agg(aggregations)
        .sort(["client_id", "date"])
    )
    profils = (
        profils_historiques(achats)
        .filter(pl.col("nb_jours_achat") >= min_jours_achat)
        .with_columns(
            pl.max_horizontal(
                pl.lit(min_inactivite_j),
                (facteur_intervalle * pl.col("intervalle_median_j")).ceil(),
            ).cast(pl.Int64).alias("delai_churn_j"),
        )
        # Garde l'ordre historique des variables pour les modèles de la première version.
        .select(pl.exclude("cv_intervalles", "evolution_intervalle"), "cv_intervalles", "evolution_intervalle")
        .rename({"date": "dernier_achat"})
        .with_columns(
            (pl.col("dernier_achat") + pl.duration(days=pl.col("delai_churn_j")))
            .alias("echeance"),
            (pl.col("dernier_achat")
             + pl.duration(days=8 - pl.col("dernier_achat").dt.weekday()))
            .alias("premier_lundi"),
        )
        .with_columns(
            pl.min_horizontal("prochain_achat", "echeance", "fin_observation")
            .alias("fin_scoring")
        )
        .filter(pl.col("premier_lundi") <= pl.col("fin_scoring"))
        .with_columns(
            pl.date_ranges("premier_lundi", "fin_scoring", interval="1w").alias("date_score")
        )
        .explode("date_score")
        .with_columns(
            (pl.col("date_score") + pl.duration(days=horizon_prediction_j))
            .alias("fin_prediction")
        )
        .filter(pl.col("echeance") <= pl.col("fin_prediction"))
        .with_columns(
            (pl.col("date_score") - pl.col("dernier_achat")).dt.total_days()
            .alias("recence_j"),
            (pl.col("date_score") - pl.col("premier_achat")).dt.total_days()
            .alias("anciennete_j"),
            (pl.col("echeance") - pl.col("date_score")).dt.total_days()
            .alias("delai_avant_echeance_j"),
            (pl.col("fin_prediction") <= pl.col("fin_observation"))
            .alias("cible_observable"),
            pl.col("date_score").dt.month().cast(pl.Int8).alias("mois_score"),
        )
        .with_columns(
            (pl.col("recence_j") / pl.col("intervalle_median_j")).alias("retard_relatif"),
            (30 * pl.col("nb_jours_achat") / pl.col("anciennete_j"))
            .alias("frequence_historique_30j"),
            (pl.col("anciennete_j") >= 180).alias("historique_180j_complet"),
            pl.when(~pl.col("cible_observable")).then(None)
            .otherwise(
                (pl.col("prochain_achat").is_null()
                 | (pl.col("prochain_achat") > pl.col("echeance"))).cast(pl.Int8)
            ).alias("churn"),
        )
        .drop(
            "ca_achats", *colonnes_optionnelles, "intervalle_j", "prochain_achat",
            "premier_lundi", "fin_scoring",
        )
    )

    return ajouter_fenetres_achats(profils, achats).sort(["date_score", "client_id"])


def ajouter_fenetres_achats(profils: pl.LazyFrame, achats: pl.LazyFrame) -> pl.LazyFrame:
    """Ajoute les agrégats des fenêtres se terminant avant le lundi matin."""
    for jours, decalage in ((30, 0), (90, 0), (90, 90), (180, 0), (365, 0)):
        fenetre = _fenetre_achats(achats, jours, decalage)
        colonnes = [c for c in fenetre.collect_schema() if c not in ("client_id", "date_score")]
        profils = profils.join(fenetre, on=["client_id", "date_score"], how="left")
        profils = profils.with_columns(pl.col(colonnes).fill_null(0))

    def evolution(colonne: str) -> pl.Expr:
        precedent = pl.col(f"{colonne}_90j_precedents")
        return pl.when(pl.col("historique_180j_complet") & (precedent > 0)).then(
            pl.col(f"{colonne}_90j") / precedent - 1
        )

    profils = profils.with_columns(
        evolution("ca").alias("evolution_ca_90j"),
        evolution("nb_jours_achat").alias("evolution_frequence_90j"),
        pl.when(pl.col("nb_jours_achat_90j") > 0)
        .then(pl.col("ca_90j") / pl.col("nb_jours_achat_90j"))
        .alias("ca_moyen_jour_achat_90j"),
        pl.when(pl.col("nb_jours_achat_90j_precedents") > 0)
        .then(pl.col("ca_90j_precedents") / pl.col("nb_jours_achat_90j_precedents"))
        .alias("ca_moyen_jour_achat_90j_precedents"),
    )
    profils = profils.with_columns(
        evolution("ca_moyen_jour_achat").alias("evolution_ca_jour_achat_90j")
    )
    if "ca_online" in achats.collect_schema():
        profils = profils.with_columns(
            pl.when(pl.col("ca_90j") > 0)
            .then(pl.col("ca_online_90j") / pl.col("ca_90j"))
            .alias("part_ca_online_90j"),
            pl.when(pl.col("ca_90j_precedents") > 0)
            .then(pl.col("ca_online_90j_precedents") / pl.col("ca_90j_precedents"))
            .alias("part_ca_online_90j_precedents"),
        ).with_columns(
            pl.when(pl.col("historique_180j_complet"))
            .then(pl.col("part_ca_online_90j") - pl.col("part_ca_online_90j_precedents"))
            .alias("evolution_part_ca_online_90j")
        )
    return profils
