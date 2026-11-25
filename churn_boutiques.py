"""Capacité d'appel issue des commandes téléphoniques et rattachement des clients."""

from dataclasses import dataclass
from datetime import date, timedelta
from math import isfinite

import polars as pl

from churn_features import achat_positif, normaliser_commandes


@dataclass(frozen=True)
class ConfigurationRelances:
    coefficient_relance: float = 0.25
    taux_churn_evite_par_relance: float = 0.10

    def __post_init__(self):
        if not isfinite(self.coefficient_relance) or self.coefficient_relance < 0:
            raise ValueError("Le coefficient de relance doit être fini et positif ou nul.")
        if not 0 <= self.taux_churn_evite_par_relance <= 1:
            raise ValueError("Le taux de churn évité doit être compris entre 0 et 1.")


def semaines_completes(debut_train: date, fin_train: date) -> tuple[date, date]:
    """Bornes [lundi, lundi[ des semaines entièrement incluses dans le train (fin incluse)."""
    premier_lundi = debut_train + timedelta(days=(-debut_train.weekday()) % 7)
    fin_exclue = fin_train + timedelta(days=1)
    return premier_lundi, fin_exclue - timedelta(days=fin_exclue.weekday())


def estimer_budgets(
    ventes: pl.LazyFrame, debut_train: date, fin_train: date, config: ConfigurationRelances,
) -> pl.DataFrame:
    """Compte les clients téléphoniques distincts par semaine active de chaque boutique."""
    debut, fin = semaines_completes(debut_train, fin_train)
    jour = pl.col("date_order")
    dans_train = jour.is_between(debut, fin, closed="left") & achat_positif()
    moyenne = "clients_telephone_par_semaine_moy"
    # La semaine nulle sert uniquement à conserver tous les branch_id dans l'annuaire.
    # Hors train, aucun client ni aucune semaine ne contribue à une moyenne.
    clients_semaines = (
        normaliser_commandes(ventes).drop_nulls("branch_id").select(
            "branch_id", pl.when(dans_train).then(jour.dt.truncate("1w")).alias("semaine"),
            pl.when(dans_train & (pl.col("order_channel") == "by phone"))
            .then(pl.col("client_id")).alias("client_telephone"),
        ).unique()
    )
    return (
        clients_semaines.group_by("branch_id", "semaine").agg(pl.col("client_telephone").count().alias("clients"))
        .group_by("branch_id").agg(pl.col("clients").filter(pl.col("semaine").is_not_null()).mean().alias(moyenne))
        .with_columns(
            (pl.col(moyenne) * config.coefficient_relance).round(0, mode="half_to_even")
            .fill_null(0).cast(pl.Int64).alias("budget_relances_par_semaine")
        )
        .sort("branch_id").collect(engine="streaming")
    )


def affecter_boutiques(lundis: pl.LazyFrame, journalier: pl.LazyFrame) -> pl.LazyFrame:
    """Boutique au plus fort CA cumulé strictement avant le score, puis achat récent et ID."""
    cles = ["client_id", "date_score"]
    cumuls = (
        journalier.drop_nulls("branch_id").sort("client_id", "branch_id", "date")
        .with_columns(pl.col("ca_achats").cum_sum().over("client_id", "branch_id").alias("ca_historique_boutique"))
        .select("client_id", "branch_id", "date", "ca_historique_boutique")
    )
    return (
        lundis.select(cles).join(cumuls.select("client_id", "branch_id").unique(), on="client_id")
        .sort("client_id", "branch_id", "date_score")
        .join_asof(
            cumuls, left_on="date_score", right_on="date", by=["client_id", "branch_id"],
            strategy="backward", allow_exact_matches=False, check_sortedness=False,
        )
        .drop_nulls("ca_historique_boutique")
        .sort([*cles, "ca_historique_boutique", "date", "branch_id"],
              descending=[False, False, True, True, False])
        .unique(subset=cles, keep="first", maintain_order=True)
        .select(*cles, "branch_id", "ca_historique_boutique")
    )
