"""Frontières du lundi, censure et dénominateur métier du programme final."""

from datetime import date, timedelta
import unittest

import polars as pl
from polars.testing import assert_frame_equal

from churn_cible import CLE_EPISODE, construire_lot, inventorier_ventes, variables_modele
from churn_episodes import evaluer_relances, issues_episodes, selectionner_relances
from churn_final import decoupage


T = date(2024, 4, 1)
J = timedelta(days=1)


def preparer(lignes, fin):
    source = pl.LazyFrame(
        [*lignes, (999, fin, "other", -1, -1)],
        schema=["client_id", "date_order", "order_channel", "sales_net", "quantity"], orient="row",
    )
    ventes, canaux, borne = inventorier_ventes(source, ("at the store",))
    scores, registre = construire_lot(ventes, borne)
    return scores.collect().sort("client_id", "date_score"), registre, canaux


def achats(client, decalages):
    return [(client, T + j * J, "online", 10, 1) for j in decalages]


class PopulationTests(unittest.TestCase):
    def test_panier_mixte_premier_magasin_et_mediane_historique(self):
        lignes = (
            achats(1, [-40, -20, -20, 0]) + [(1, T, " AT the store ", 1, 1)]
            + [(1, T - 24 * J, "at the store", -1, -1)]
            + achats(2, [-40, -10]) + achats(3, [-20, -20]) + achats(4, [-42, -11])
            + achats(5, [-40, -20]) + [(5, T - 40 * J, "at the store", 1, 1)]
        )
        scores, registre, canaux = preparer(lignes, T + 120 * J)
        lundi = scores.filter(pl.col("date_score") == T)
        self.assertEqual(lundi["client_id"].to_list(), [1, 2])
        ligne = lundi.row(0, named=True)
        self.assertEqual((ligne["nb_jours_achat"], ligne["intervalle_median_j"]), (2, 20))
        self.assertEqual((ligne["nb_jours_achat_30j"], ligne["ca_30j"], ligne["churn"]), (1, 20, 0))
        self.assertEqual(lundi["intervalle_median_j"][1], 30)
        self.assertEqual(scores.filter(pl.col("client_id") == 1)["date_score"].max(), T)
        self.assertEqual(canaux.filter(pl.col("magasin"))["order_channel"].to_list(), ["at the store"])
        self.assertEqual(registre.filter(pl.col("client_id") == 1).collect()["prochain_achat"].item(), T)
        # Les clients restent dans la population de classification après leur échéance.
        self.assertGreater(scores.filter(
            (pl.col("client_id") == 2) & (pl.col("recence_j") > 90)
        ).height, 0)
        self.assertEqual(scores.select(pl.struct("client_id", "date_score").n_unique()).item(), scores.height)

    def test_le_futur_ne_change_aucune_variable_du_lundi(self):
        base = achats(1, [-75, -55, -35, -15])
        avant, _, _ = preparer(base, T + 100 * J)
        apres, _, _ = preparer(base + [(1, T, "at the store", 500000, 10)], T + 100 * J)
        variables = variables_modele(avant.lazy())
        self.assertFalse({"prochain_achat", "churn", "client_id"} & set(variables))
        assert_frame_equal(
            avant.filter(pl.col("date_score") == T).select(variables),
            apres.filter(pl.col("date_score") == T).select(variables),
        )

    def test_deux_horizons_bornes_inclusives_et_censure(self):
        lignes = sum((achats(i, [-55, -35, suivant]) for i, suivant in [(1, 90), (2, 91), (3, 55)]), [])
        scores, registre, _ = preparer(lignes, T + 100 * J)
        lundi = scores.filter(pl.col("date_score") == T)
        self.assertEqual(lundi["churn"].to_list(), [0, 1, 0])
        episodes = issues_episodes(registre, lundi, T + 100 * J).sort("client_id")
        self.assertEqual(episodes["issue"].to_list(), ["churn", "churn", "rachat"])
        alertes = selectionner_relances(lundi, "retard_relatif")
        self.assertEqual(alertes["avance_avant_echeance_j"].to_list(), [55] * 3)
        self.assertEqual(alertes["retard_sur_rythme_habituel_j"].to_list(), [15] * 3)
        incomplets, registre, _ = preparer(achats(1, [-55, -35, 3]) + achats(2, [-55, -35]), T + 4 * J)
        lundi = incomplets.filter(pl.col("date_score") == T)
        self.assertEqual(lundi["churn"].null_count(), 2)
        episodes = issues_episodes(registre, lundi, T + 4 * J).filter(pl.col("dernier_achat") < T)
        self.assertEqual(episodes.sort("client_id")["issue"].to_list(), ["rachat", "inconnue"])

    def test_redevient_eligible_quand_la_mediane_redescend(self):
        scores, _, _ = preparer(achats(1, [-108, -98, -18, -8]), T + 90 * J)
        self.assertFalse(scores.filter(pl.col("date_score") == T - 14 * J).height)
        self.assertEqual(scores.filter(pl.col("date_score") == T)["intervalle_median_j"].item(), 10)

    def test_sommes_reproductibles_et_petit_achat_positif(self):
        lignes = [(1, T - 20 * J, "online", 0.0000001, 1)]
        lignes += [(1, T - 10 * J, "online", m, 1) for m in (0.000001, 200000.001, 3.123456)]
        a, _, _ = preparer(lignes, T + 90 * J)
        b, _, _ = preparer(list(reversed(lignes)), T + 90 * J)
        assert_frame_equal(a, b)
        ligne = a.filter(pl.col("date_score") == T).row(0, named=True)
        self.assertEqual(ligne["nb_jours_achat"], 2)
        self.assertAlmostEqual(ligne["ca_30j"], 200003.124457)


class EpisodesTests(unittest.TestCase):
    def test_un_episode_deja_en_cours_consomme_le_budget_avant_evaluation(self):
        lignes = achats(1, [-18, -8]) + achats(2, [-11, -1]) + achats(3, [-11, -1])
        scores, registre, _ = preparer(lignes, T + 100 * J)
        scores = scores.filter(pl.col("date_score").is_between(T, T + 84 * J))
        appels = selectionner_relances(scores, "retard_relatif", budget=1)
        episodes = issues_episodes(registre, scores, T + 100 * J)
        bilan, _, details = evaluer_relances(appels, episodes, "Règle")
        self.assertEqual(details.row(0, named=True)["client_id"], 1)
        self.assertFalse(details["couverture_complete"][0])
        self.assertEqual((bilan["relances"], bilan["episodes_churn_evaluables"], bilan["detectes_14j"]),
                         (3, 2, 2))

    def test_budget_independant_episodes_non_detectes_et_couverture(self):
        premier = date(2024, 1, 8)
        achat = premier - J
        echeance = achat + 90 * J
        lignes = [
            (c, achat, premier + 7 * i * J, echeance, 1 + 7 * i, float(c), (1 + 7 * i) / c, c / 20)
            for c in range(1, 17) for i in range(13)
        ]
        scores = pl.DataFrame(lignes, schema=[*CLE_EPISODE, "date_score", "echeance", "recence_j",
                                               "intervalle_median_j", "retard_relatif", "proba"], orient="row")
        registre = scores.select(*CLE_EPISODE, "echeance").unique().with_columns(
            pl.lit(premier).alias("premier_lundi"), pl.lit(None, dtype=pl.Date).alias("prochain_achat"),
        ).lazy()
        episodes = issues_episodes(registre, scores, echeance)
        model = selectionner_relances(scores, "proba", budget=1)
        regle = selectionner_relances(scores, "retard_relatif", budget=1)
        self.assertEqual(model["client_id"].to_list(), list(range(16, 3, -1)))
        self.assertEqual(regle["client_id"].to_list(), list(range(1, 14)))
        bilan, courbe, _ = evaluer_relances(model, episodes, "XGBoost")
        self.assertEqual((bilan["episodes_churn_evaluables"], bilan["churns_detectes"]), (16, 13))
        self.assertEqual(bilan["pct_14j"], 100 * 11 / 16)
        self.assertTrue(courbe["pct_detectes"].is_sorted(descending=True))
        trou = scores.filter(~((pl.col("client_id") == 10) & (pl.col("date_score") == premier + 7 * J)))
        avec_trou = issues_episodes(registre, trou, echeance)
        self.assertFalse(avec_trou.filter(pl.col("client_id") == 10)["couverture_complete"].item())

    def test_selection_sans_issue_budget_50_et_nouveau_panier(self):
        lignes = sum((achats(i, [-30, -15]) for i in range(1, 56)), []) + achats(1, [0, 1, 2])
        scores, registre, _ = preparer(lignes, T + 9 * J)
        scores = scores.filter(pl.col("date_score") >= T).with_columns(
            (-pl.col("client_id")).alias("priorite")
        )
        appels = selectionner_relances(scores, "priorite")
        self.assertEqual(appels.filter(pl.col("date_alerte") == T).height, 50)
        self.assertEqual(appels.filter(pl.col("client_id") == 1)["dernier_achat"].to_list(),
                         [T - 15 * J, T + 2 * J])
        issues = issues_episodes(registre, scores, T + 9 * J)
        # L'épisode du lundi au mardi n'a jamais offert de date de score.
        sans_lundi = issues.filter((pl.col("client_id") == 1) & (pl.col("dernier_achat") == T))
        self.assertEqual(sans_lundi.select("lundis_scores", "issue").row(0), (0, "rachat"))
        bilan, _, _ = evaluer_relances(appels, issues, "XGBoost")
        self.assertEqual((bilan["rachats"], bilan["issues_connues"], bilan["pct_relances_avec_rachat"]), (1, 1, 100))
        self.assertEqual(bilan["issues_inconnues"], appels.height - 1)
        self.assertIsNone(bilan["pct_14j"])
        echeance_aujourdhui = scores.with_columns(pl.col("date_score").alias("echeance"))
        self.assertTrue(selectionner_relances(echeance_aujourdhui, "priorite").is_empty())

    def test_decoupage_purge_et_derniers_lundis(self):
        fin = date(2024, 9, 22)
        lundis = pl.date_range(date(2022, 9, 26), fin, interval="1w", eager=True)
        scores = pl.DataFrame({"date_score": lundis, "churn": 1}).lazy().with_columns(
            (pl.col("date_score") + pl.duration(days=90)).alias("fin_prediction")
        )
        jeux, debut_val, debut_test = decoupage(scores, fin)
        for nom, limite in (("train", debut_val), ("validation", debut_test), ("train_final", debut_test)):
            self.assertLess(jeux[nom].select(pl.max("fin_prediction")).collect().item(), limite)
        self.assertEqual(jeux["validation"].select(pl.len()).collect().item(), 26)
        self.assertEqual(jeux["test"].select(pl.len()).collect().item(), 13)
        self.assertEqual(jeux["test"].select(pl.max("date_score")).collect().item(), date(2024, 6, 24))
        self.assertEqual(jeux["test_complet"].select(pl.max("date_score")).collect().item(), date(2024, 9, 16))


if __name__ == "__main__":
    unittest.main()
