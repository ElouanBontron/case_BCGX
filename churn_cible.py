"""Population et observations du modèle final de churn."""

from datetime import date
from pathlib import Path

import polars as pl

from churn_boutiques import affecter_boutiques
from churn_features import (
    achat_positif, ajouter_fenetres_achats, construire_quotidien, normaliser_commandes, profils_historiques,
)


HORIZON = 90
CLE_EPISODE = ["client_id", "dernier_achat"]
UNITES_PAR_EURO = 1_000_000


def inventorier_ventes(ventes: pl.LazyFrame, canaux_magasin: tuple[str, ...]):
    """Identifie le premier achat magasin sur les lignes produit, tous paniers confondus."""
    canaux_magasin = tuple(c.strip().lower() for c in canaux_magasin)
    source = normaliser_commandes(ventes)
    jour_valide = pl.when(achat_positif()).then(pl.col("date_order"))
    stats = (
        source.group_by("client_id", "order_channel").agg(
            jour_valide.min().alias("premier"), jour_valide.max().alias("dernier"),
            pl.col("date_order").max().alias("fin_source"), pl.len().alias("nb_lignes"),
        )
        .collect(engine="streaming")
        .with_columns(pl.col("order_channel").is_in(canaux_magasin).fill_null(False).alias("magasin"))
    )
    canaux = stats.group_by("order_channel", "magasin").agg(pl.sum("nb_lignes")).sort("order_channel")
    fin = stats["fin_source"].max()
    if fin is None:
        raise ValueError("Aucune date de commande dans la source.")
    clients = (
        stats.drop_nulls("client_id").group_by("client_id").agg(
            pl.col("premier").filter(pl.col("magasin")).min().alias("premier_magasin"),
            pl.col("premier").min(), pl.col("dernier").max(),
        )
        .filter(
            (pl.col("premier") < pl.col("dernier"))
            & (pl.col("premier_magasin").is_null() | (pl.col("premier") < pl.col("premier_magasin")))
        )
        .select("client_id", "premier_magasin")
    )
    # Le jour du premier achat magasin peut clore l'épisode précédent, y compris un lundi.
    lignes = source.join(clients.lazy(), on="client_id").filter(
        achat_positif() & pl.col("date_order").is_not_null()
        & (pl.col("premier_magasin").is_null() | (pl.col("date_order") <= pl.col("premier_magasin")))
    )
    return lignes, canaux, fin


def construire_lot(lignes: pl.LazyFrame, fin: date) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Construit les lundis éligibles et le registre des épisodes, même sans lundi."""
    # Matérialiser une fois le quotidien du lot pour les cinq fenêtres glissantes.
    # Les sommes entières rendent les montants et leurs ratios indépendants de l'ordre
    # de réduction parallèle. L'éligibilité utilise toujours les montants bruts.
    montant = (pl.col("sales_net") * UNITES_PAR_EURO).round(0).cast(pl.Int64)
    quotidien = construire_quotidien(lignes, montant=montant).sort("client_id", "date").collect(engine="streaming")
    achats = quotidien.lazy()
    magasins = lignes.select("client_id", "premier_magasin").unique()
    profils = (
        profils_historiques(achats)
        .join(magasins, on="client_id")
        .filter(
            (pl.col("nb_jours_achat") >= 2) & (pl.col("intervalle_median_j") <= 30)
            & (pl.col("premier_magasin").is_null() | (pl.col("date") < pl.col("premier_magasin")))
        )
        .drop("ca_achats", "nb_lignes", "intervalle_j", "premier_magasin",
              *[c for c in ("produits", "agences", "canaux", "ca_online") if c in quotidien])
        .rename({"date": "dernier_achat"})
        .with_columns(
            (pl.col("dernier_achat") + pl.duration(days=HORIZON)).alias("echeance"),
            (pl.col("dernier_achat") + pl.duration(days=8 - pl.col("dernier_achat").dt.weekday()))
            .alias("premier_lundi"),
        )
        .collect(engine="streaming").lazy()
    )
    episodes = profils.select(*CLE_EPISODE, "premier_lundi", "echeance", "prochain_achat")
    lundis = (
        profils.with_columns(pl.min_horizontal("prochain_achat", pl.lit(fin)).alias("fin_scoring"))
        .filter(pl.col("premier_lundi") <= pl.col("fin_scoring"))
        .with_columns(pl.date_ranges("premier_lundi", "fin_scoring", interval="1w").alias("date_score"))
        .explode("date_score")
        .with_columns(
            (pl.col("date_score") + pl.duration(days=HORIZON)).alias("fin_prediction"),
            (pl.col("date_score") - pl.col("dernier_achat")).dt.total_days().alias("recence_j"),
            (pl.col("date_score") - pl.col("premier_achat")).dt.total_days().alias("anciennete_j"),
            pl.col("date_score").dt.month().alias("mois_score"),
        )
        .with_columns(
            pl.when(pl.col("fin_prediction") <= fin).then(
                (pl.col("prochain_achat").is_null() | (pl.col("prochain_achat") > pl.col("fin_prediction")))
                .cast(pl.Int8)
            ).alias("churn"),
            (pl.col("recence_j") / pl.col("intervalle_median_j")).alias("retard_relatif"),
            (30 * pl.col("nb_jours_achat") / pl.col("anciennete_j")).alias("frequence_historique_30j"),
            (pl.col("anciennete_j") >= 180).alias("historique_180j_complet"),
        )
        .drop("fin_scoring", "premier_achat", "premier_lundi", "prochain_achat")
    )
    table = ajouter_fenetres_achats(lundis, achats).with_columns(pl.col("^ca_.*$") / UNITES_PAR_EURO)
    if "branch_id" in lignes.collect_schema():
        journalier = (
            lignes.group_by("client_id", "branch_id", pl.col("date_order").alias("date"))
            .agg(montant.sum().alias("ca_achats"))
        )
        boutiques = affecter_boutiques(lundis, journalier).select("client_id", "date_score", "branch_id")
        table = table.join(boutiques, on=["client_id", "date_score"], how="left")
    return table, episodes


def preparer_observations(
    lignes: pl.LazyFrame, fin: date, dossier: Path,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Partitionne par client pour traiter le CSV complet avec une mémoire bornée."""
    # Une seule seconde lecture du CSV. Tous les achats d'un client restent dans le même lot.
    colonnes = ["date_order", "client_id", "quantity", "sales_net", "order_channel", "premier_magasin"]
    colonnes += [c for c in ("product_id", "branch_id") if c in lignes.collect_schema()]
    partitions = pl.PartitionBy(
        dossier / "lignes", key={"lot": pl.col("client_id").hash(seed=42) % 16}, include_key=False
    )
    lignes.select(colonnes).sink_parquet(partitions, mkdir=True, engine="streaming")
    lots = sorted({f.parent for f in (dossier / "lignes").rglob("*.parquet")})
    if not lots:
        raise ValueError("Aucun client avec deux jours d'achat et un historique avant le magasin.")
    for i, lot in enumerate(lots):
        source = pl.scan_parquet(lot / "*.parquet", hive_partitioning=False)
        table, episodes = construire_lot(source, fin)
        # Le CA servant au classement et au chiffrage garde sa précision monétaire.
        variables = [c for c in variables_modele(table) if c != "ca_90j"]
        table.with_columns(pl.col(variables).cast(pl.Float32)).sink_parquet(
            dossier / f"lundis_{i:02}.parquet", engine="streaming"
        )
        episodes.sink_parquet(dossier / f"episodes_{i:02}.parquet", engine="streaming")
    return pl.scan_parquet(dossier / "lundis_*.parquet"), pl.scan_parquet(dossier / "episodes_*.parquet")


def variables_modele(table: pl.LazyFrame) -> list[str]:
    return [
        nom for nom, dtype in table.collect_schema().items()
        if nom not in {"client_id", "branch_id", "churn"} and (dtype.is_numeric() or dtype == pl.Boolean)
    ]
